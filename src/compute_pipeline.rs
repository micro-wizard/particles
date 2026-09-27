use crate::{
    config,
    ecs::Particle,
    gpu_context::GpuContext,
    gpu_timing::{PassTimer, Span},
    materials::{Globals, MaterialParams, MATERIAL_COUNT},
};
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc,
};
use wgpu::{util::DeviceExt, BindGroup, Buffer, ComputePipeline};

const WORKGROUP_SIZE: u32 = 64;
const PARAMS_STRIDE: u64 = 256;
const SCAN_WORKGROUP: u32 = 256;

fn padded_grid() -> [u32; 2] {
    let span =
        |extent: f32| ((extent + 2.0 * config::HASH_MARGIN) / config::CELL_SIZE).ceil() as u32;
    [span(config::WORLD_WIDTH), span(config::WORLD_HEIGHT)]
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Params {
    world: [f32; 2],
    grid: [u32; 2],
    inv_cell_size: f32,
    dt: f32,
    particle_count: u32,
    shelter_reach: f32,
    max_contacts: u32,
    material_count: u32,
    gravity: f32,
    max_speed: f32,
    render_hysteresis: f32,
    smoothing_radius: f32,
    hash_margin: f32,
    sweep: i32,
    min_temperature: f32,
    max_temperature: f32,
    temperature_reference: f32,
    ambient_density: f32,
    wind: f32,
    air_drag: f32,
    gust_centre: [f32; 2],
    gust_velocity: [f32; 2],
    gust_radius: f32,
    gust_outflow: f32,
    contact_parity: u32,
    rest_temperature: f32,
}

unsafe impl bytemuck::Zeroable for Params {}
unsafe impl bytemuck::Pod for Params {}

const _: () = assert!(
    std::mem::size_of::<Params>() == 120,
    "Params must stay the 120 bytes compute.wgsl declares — see its doc comment",
);
const _: () = assert!(
    std::mem::size_of::<Params>() as u64 <= PARAMS_STRIDE,
    "the two copies of Params must not overlap",
);

#[derive(Copy, Clone, Debug, Default, PartialEq)]
pub struct Gust {
    pub centre: [f32; 2],
    pub radius: f32,
    pub velocity: [f32; 2],
    pub outflow: f32,
}

#[repr(C)]
#[derive(Copy, Clone)]
struct ContactRecord {
    partner: u32,
    _padding: u32,
    tangent: [f32; 2],
}

unsafe impl bytemuck::Zeroable for ContactRecord {}
unsafe impl bytemuck::Pod for ContactRecord {}

#[repr(C)]
#[derive(Copy, Clone)]
struct CellEntry {
    position: [f32; 2],
    velocity: [f32; 2],
    radius: f32,
    packing: f32,
    temperature: f32,
    packed: u32,
}

const PACKED_MATERIAL_SHIFT: u32 = 20;

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
pub struct SpawnRequest {
    pub position: [f32; 2],
    pub radius: f32,
    pub material: u32,
    pub temperature: f32,
    pub budget: u32,
}

unsafe impl bytemuck::Zeroable for SpawnRequest {}
unsafe impl bytemuck::Pod for SpawnRequest {}

#[repr(C)]
#[derive(Copy, Clone, Debug, Default)]
struct BrushStroke {
    centre: [f32; 2],
    radius: f32,
    delta: f32,
    slot_bound: u32,
    protected_below: u32,
}

#[derive(Copy, Clone)]
enum StrokeKind {
    Heat,
    Erase,
}

unsafe impl bytemuck::Zeroable for BrushStroke {}
unsafe impl bytemuck::Pod for BrushStroke {}

#[repr(C)]
#[derive(Copy, Clone)]
struct SpawnHeader {
    count: u32,
    slot_bound: u32,
    max_contacts: u32,
    _padding: u32,
    lo: [f32; 2],
    hi: [f32; 2],
}

unsafe impl bytemuck::Zeroable for SpawnHeader {}
unsafe impl bytemuck::Pod for SpawnHeader {}

const _: () = assert!(std::mem::size_of::<SpawnHeader>().is_multiple_of(8));

const NO_CONTACT: u32 = 0xFFFF_FFFF;

const _: () = assert!(
    config::MAX_PARTICLES < (1 << PACKED_MATERIAL_SHIFT),
    "MAX_PARTICLES no longer fits below the material bits of a CellEntry",
);
const _: () = assert!(
    MATERIAL_COUNT <= (u32::MAX >> PACKED_MATERIAL_SHIFT),
    "too many materials to pack above the index bits of a CellEntry",
);
const _: () = assert!(
    std::mem::size_of::<CellEntry>() == 32,
    "CellEntry must stay 32 bytes — see its doc comment",
);

pub struct Solver {
    particle_buffer: Buffer,
    bind_group: BindGroup,
    count_pipeline: ComputePipeline,
    scan_pipeline: ComputePipeline,
    scatter_pipeline: ComputePipeline,
    solve_pipeline: ComputePipeline,
    plants_pipeline: ComputePipeline,
    grow_pipeline: ComputePipeline,
    growing: bool,
    seeded: u32,
    slot_bound: u32,
    cell_counts: Buffer,
    counts_stale: bool,
    accumulator: f32,
    contact_parity: u32,
    params_buffer: Buffer,
    materials_buffer: Buffer,
    globals: Globals,
    materials: [MaterialParams; MATERIAL_COUNT as usize],
    gust: Gust,
    next_gust: Gust,
    last_substeps: u32,
    spawn: Spawner,
    timer: Option<PassTimer>,
}

struct Spawner {
    batch_buffer: Buffer,
    allocator: Buffer,
    blocked_buffer: Buffer,
    bind_group: BindGroup,
    check_pipeline: ComputePipeline,
    commit_pipeline: ComputePipeline,
    heat_pipeline: ComputePipeline,
    erase_pipeline: ComputePipeline,
    forget_pipeline: ComputePipeline,
    stroke_buffer: Buffer,
    pending_stroke: Option<(StrokeKind, BrushStroke)>,
    pending: Vec<SpawnRequest>,
    readback: Buffer,
    readback_in_flight: bool,
    readback_ready: Arc<AtomicBool>,
    count_stale: bool,
    live_count: u32,
    requested_since_copy: u32,
}

fn filled_to_capacity(seed: &[Particle]) -> Vec<Particle> {
    let mut particles = seed.to_vec();
    particles.resize(config::MAX_PARTICLES as usize, Particle::dead());
    particles
}

fn initial_allocator(seeded: u32, capacity: u32) -> Vec<u32> {
    [capacity - seeded, seeded]
        .into_iter()
        .chain((seeded..capacity).rev())
        .collect()
}

impl Solver {
    pub fn new(gpu_context: &GpuContext<'_>, seed: &[Particle]) -> Self {
        let capacity = config::MAX_PARTICLES;
        assert!(seed.len() as u32 <= capacity, "seed exceeds MAX_PARTICLES");
        let slot_bound = seed.len() as u32;
        let particles = filled_to_capacity(seed);
        let grid = padded_grid();
        let cell_count = grid[0] * grid[1];
        let padded_cells = cell_count.div_ceil(4) * 4;
        assert!(
            padded_cells.is_multiple_of(4) && SCAN_WORKGROUP == 256,
            "scan_cells reads counts four at a time from one 256-thread workgroup",
        );
        let device = &gpu_context.device;

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("DEM Solver"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/compute.wgsl").into()),
        });
        let scan_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Cell Scan"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/scan.wgsl").into()),
        });

        let particle_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Particles"),
            contents: bytemuck::cast_slice(&particles),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::VERTEX
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let empty_history = vec![
            ContactRecord {
                partner: NO_CONTACT,
                _padding: 0,
                tangent: [0.0, 0.0]
            };
            (capacity * 2 * config::MAX_CONTACTS) as usize
        ];
        let contact_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Contact History"),
            contents: bytemuck::cast_slice(&empty_history),
            usage: wgpu::BufferUsages::STORAGE,
        });

        let cell_array = |label: &str, len: u64, usage: wgpu::BufferUsages| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size: len * 4,
                usage: wgpu::BufferUsages::STORAGE | usage,
                mapped_at_creation: false,
            })
        };
        let cell_counts = cell_array(
            "Cell Counts",
            padded_cells as u64,
            wgpu::BufferUsages::COPY_DST,
        );
        let cell_start = cell_array(
            "Cell Starts",
            padded_cells as u64 + 4,
            wgpu::BufferUsages::empty(),
        );
        let cell_cursor = cell_array(
            "Cell Cursors",
            padded_cells as u64,
            wgpu::BufferUsages::empty(),
        );
        let cell_entries = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Cell Entries"),
            size: capacity as u64 * std::mem::size_of::<CellEntry>() as u64,
            usage: wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let materials = MaterialParams::defaults();
        let globals = Globals::default();
        let materials_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Materials"),
            contents: bytemuck::cast_slice(&materials),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });

        let params_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Solver Params"),
            size: 2 * PARAMS_STRIDE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        for parity in 0..2 {
            gpu_context.queue.write_buffer(
                &params_buffer,
                parity as u64 * PARAMS_STRIDE,
                bytemuck::bytes_of(&Self::params(
                    slot_bound,
                    &globals,
                    &Gust::default(),
                    parity,
                )),
            );
        }

        let spawn = Spawner::new(
            gpu_context,
            &particle_buffer,
            &contact_buffer,
            capacity,
            slot_bound,
        );

        let storage = |read_only: bool| wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Solver Bind Group Layout"),
            entries: &[
                entry(0, storage(false)),
                entry(1, storage(false)),
                entry(2, storage(false)),
                entry(3, storage(false)),
                entry(4, storage(false)),
                entry(
                    5,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: true,
                        min_binding_size: None,
                    },
                ),
                entry(6, storage(false)),
                entry(7, storage(true)),
                entry(8, storage(false)),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Solver Bind Group"),
            layout: &bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: cell_counts.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: cell_start.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: cell_cursor.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: cell_entries.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &params_buffer,
                        offset: 0,
                        size: wgpu::BufferSize::new(std::mem::size_of::<Params>() as u64),
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: contact_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: materials_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: spawn.allocator.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Solver Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });
        let pipeline = |label: &str, module: &wgpu::ShaderModule, entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            particle_buffer,
            bind_group,
            count_pipeline: pipeline("Count Particles", &shader, "count_particles"),
            scan_pipeline: pipeline("Scan Cells", &scan_shader, "scan_cells"),
            scatter_pipeline: pipeline("Scatter Particles", &shader, "scatter_particles"),
            solve_pipeline: pipeline("Solve Contacts", &shader, "solve"),
            plants_pipeline: pipeline("Plant Forces", &shader, "plant_forces"),
            grow_pipeline: pipeline("Grow Plants And Emit Flames", &shader, "grow_and_emit"),
            growing: false,
            seeded: slot_bound,
            slot_bound,
            cell_counts,
            counts_stale: true,
            accumulator: 0.0,
            contact_parity: 0,
            params_buffer,
            materials_buffer,
            globals,
            materials,
            gust: Gust::default(),
            next_gust: Gust::default(),
            last_substeps: 0,
            timer: None,
            spawn,
        }
    }

    fn params(slot_bound: u32, globals: &Globals, gust: &Gust, contact_parity: u32) -> Params {
        Params {
            world: [config::WORLD_WIDTH, config::WORLD_HEIGHT],
            grid: padded_grid(),
            inv_cell_size: 1.0 / config::CELL_SIZE,
            dt: config::SUBSTEP,
            particle_count: slot_bound,
            shelter_reach: config::SHELTER_REACH,
            max_contacts: config::MAX_CONTACTS,
            material_count: MATERIAL_COUNT,
            gravity: globals.gravity,
            max_speed: globals.max_speed,
            render_hysteresis: globals.render_hysteresis,
            smoothing_radius: config::SMOOTHING_RADIUS,
            hash_margin: config::HASH_MARGIN,
            sweep: config::SWEEP_RADIUS,
            min_temperature: config::MIN_TEMPERATURE,
            max_temperature: config::MAX_TEMPERATURE,
            temperature_reference: config::TEMPERATURE_REFERENCE,
            ambient_density: globals.ambient_density,
            wind: globals.wind,
            air_drag: globals.air_drag,
            gust_centre: gust.centre,
            gust_velocity: gust.velocity,
            gust_radius: gust.radius,
            gust_outflow: gust.outflow,
            contact_parity,
            rest_temperature: globals.rest_temperature,
        }
    }

    pub fn materials(&self) -> &[MaterialParams; MATERIAL_COUNT as usize] {
        &self.materials
    }

    pub fn materials_buffer(&self) -> &Buffer {
        &self.materials_buffer
    }

    pub fn globals(&self) -> &Globals {
        &self.globals
    }

    pub fn last_substeps(&self) -> u32 {
        self.last_substeps
    }

    pub fn set_parameters(
        &mut self,
        gpu_context: &GpuContext<'_>,
        materials: &[MaterialParams; MATERIAL_COUNT as usize],
        globals: &Globals,
    ) {
        if *materials != self.materials {
            self.materials = *materials;
            gpu_context.queue.write_buffer(
                &self.materials_buffer,
                0,
                bytemuck::cast_slice(&self.materials),
            );
        }
        if *globals != self.globals {
            self.globals = *globals;
            self.write_params(gpu_context);
        }
    }

    fn write_params(&self, gpu_context: &GpuContext<'_>) {
        for parity in 0..2 {
            gpu_context.queue.write_buffer(
                &self.params_buffer,
                parity as u64 * PARAMS_STRIDE,
                bytemuck::bytes_of(&Self::params(
                    self.slot_bound,
                    &self.globals,
                    &self.gust,
                    parity,
                )),
            );
        }
    }

    /// Grows plants by one node per ready shoot tip and sheds a flame from
    /// each ready piece of burning fuel. New particles land in slots the cell
    /// counts and `slot_bound` do not know about yet, so both are refreshed:
    /// the counts next frame, the bound once the readback lands.
    fn record_grow(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Grow And Emit Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[self.params_offset()]);
        pass.set_pipeline(&self.grow_pipeline);
        pass.dispatch_workgroups(self.slot_bound.div_ceil(WORKGROUP_SIZE), 1, 1);
        drop(pass);
        self.counts_stale = true;
        self.spawn.count_stale = true;
    }

    fn params_offset(&self) -> u32 {
        self.contact_parity * PARAMS_STRIDE as u32
    }

    fn record_count(&mut self, encoder: &mut wgpu::CommandEncoder) {
        encoder.clear_buffer(&self.cell_counts, 0, None);
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Count Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[self.params_offset()]);
        pass.set_pipeline(&self.count_pipeline);
        pass.dispatch_workgroups(self.slot_bound.div_ceil(WORKGROUP_SIZE), 1, 1);
        drop(pass);
        self.counts_stale = false;
    }

    pub fn particle_buffer(&self) -> &Buffer {
        &self.particle_buffer
    }

    pub fn slot_bound(&self) -> u32 {
        self.slot_bound
    }

    pub fn pass_timings(&self) -> Option<[f32; Span::ALL.len()]> {
        self.timer.as_ref().map(PassTimer::averages)
    }

    pub fn set_timing(&mut self, gpu_context: &GpuContext<'_>, enabled: bool) {
        if enabled != self.timer.is_some() {
            self.timer = if enabled {
                PassTimer::new(gpu_context)
            } else {
                None
            };
        }
    }

    #[allow(dead_code)]
    pub fn read_materials(
        &self,
        gpu_context: &GpuContext<'_>,
    ) -> [MaterialParams; MATERIAL_COUNT as usize] {
        let bytes = std::mem::size_of::<[MaterialParams; MATERIAL_COUNT as usize]>() as u64;
        let staging = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Materials Readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Materials Readback"),
                });
        encoder.copy_buffer_to_buffer(&self.materials_buffer, 0, &staging, 0, bytes);
        gpu_context.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu_context.device.poll(wgpu::Maintain::Wait);
        let view = staging.slice(..).get_mapped_range();
        let out = *bytemuck::from_bytes::<[MaterialParams; MATERIAL_COUNT as usize]>(&view);
        drop(view);
        staging.unmap();
        out
    }

    #[allow(dead_code)]
    pub fn read_particles(&self, gpu_context: &GpuContext<'_>, count: u32) -> Vec<Particle> {
        let bytes = count as u64 * std::mem::size_of::<Particle>() as u64;
        let staging = gpu_context.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Particle Readback"),
            size: bytes,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Particle Readback"),
                });
        encoder.copy_buffer_to_buffer(&self.particle_buffer, 0, &staging, 0, bytes);
        gpu_context.queue.submit(Some(encoder.finish()));
        staging.slice(..).map_async(wgpu::MapMode::Read, |_| {});
        gpu_context.device.poll(wgpu::Maintain::Wait);
        let view = staging.slice(..).get_mapped_range();
        let out = bytemuck::cast_slice::<u8, Particle>(&view).to_vec();
        drop(view);
        staging.unmap();
        out
    }

    pub fn live_count(&self) -> u32 {
        self.spawn.live_count
    }

    pub fn placed_count(&self) -> u32 {
        self.live_count().saturating_sub(self.seeded)
    }

    pub fn heat(&mut self, centre: [f32; 2], radius: f32, delta: f32) {
        self.spawn.pending_stroke = Some((StrokeKind::Heat, self.stroke(centre, radius, delta)));
    }

    pub fn erase(&mut self, centre: [f32; 2], radius: f32) {
        self.spawn.pending_stroke = Some((StrokeKind::Erase, self.stroke(centre, radius, 0.0)));
    }

    fn stroke(&self, centre: [f32; 2], radius: f32, delta: f32) -> BrushStroke {
        BrushStroke {
            centre,
            radius,
            delta,
            slot_bound: self.slot_bound,
            protected_below: self.seeded,
        }
    }

    pub fn reset(&mut self, gpu_context: &GpuContext<'_>, seed: &[Particle]) {
        let particles = filled_to_capacity(seed);
        gpu_context
            .queue
            .write_buffer(&self.particle_buffer, 0, bytemuck::cast_slice(&particles));
        self.seeded = seed.len() as u32;
        self.spawn.reset(gpu_context, self.seeded);
        self.slot_bound = self.seeded;
        self.counts_stale = true;
        self.growing = false;
        self.write_params(gpu_context);
    }

    pub fn gust(&self) -> Gust {
        self.gust
    }

    pub fn blow(&mut self, gust: Gust) {
        self.next_gust = gust;
    }

    pub fn spawn(&mut self, request: SpawnRequest) {
        if self.spawn.pending.len() < config::MAX_SPAWNS_PER_FRAME as usize {
            self.growing |= request.budget > 0;
            self.spawn.pending.push(request);
        }
    }

    fn dispatch(&self, span: Span) -> (&ComputePipeline, [u32; 3]) {
        let particle_groups = [self.slot_bound.div_ceil(WORKGROUP_SIZE), 1, 1];
        match span {
            Span::Scan => (&self.scan_pipeline, [1, 1, 1]),
            Span::Scatter => (&self.scatter_pipeline, particle_groups),
            Span::Plants if !self.growing => (&self.plants_pipeline, [0, 1, 1]),
            Span::Plants => (&self.plants_pipeline, particle_groups),
            Span::Solve => (&self.solve_pipeline, particle_groups),
        }
    }

    pub fn step(&mut self, gpu_context: &GpuContext<'_>, frame_dt: f32) {
        let substeps = self.take_due_substeps(frame_dt);
        self.apply_slot_bound_readback(gpu_context);
        if let Some(timer) = &mut self.timer {
            timer.poll(gpu_context);
        }
        self.apply_next_gust(gpu_context);

        let spawn_count = self
            .spawn
            .pending
            .len()
            .min(config::MAX_SPAWNS_PER_FRAME as usize);
        let stroking = self.spawn.pending_stroke.is_some();
        if substeps == 0 && spawn_count == 0 && !stroking && !self.spawn.wants_readback() {
            return;
        }

        let mut encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Solver Encoder"),
                });
        self.record_spawns(gpu_context, &mut encoder, spawn_count);
        if self.spawn.record_stroke(gpu_context, &mut encoder) {
            self.counts_stale = true;
        }
        let timing = self.timer.as_ref().is_some_and(PassTimer::is_idle);
        self.record_simulation(&mut encoder, substeps, timing);

        let reading = self.spawn.record_readback(&mut encoder);
        gpu_context.queue.submit(Some(encoder.finish()));
        if reading {
            self.spawn.begin_readback();
        }
        if timing {
            self.timer
                .as_ref()
                .expect("timing implies a timer")
                .begin_readback();
        }
    }

    fn take_due_substeps(&mut self, frame_dt: f32) -> u32 {
        let ceiling = config::SUBSTEP * config::MAX_SUBSTEPS as f32;
        self.accumulator = (self.accumulator + frame_dt).min(ceiling);

        let mut substeps = 0u32;
        while self.accumulator >= config::SUBSTEP && substeps < config::MAX_SUBSTEPS {
            self.accumulator -= config::SUBSTEP;
            substeps += 1;
        }
        self.last_substeps = substeps;
        substeps
    }

    fn apply_slot_bound_readback(&mut self, gpu_context: &GpuContext<'_>) {
        let Some(bound) = self.spawn.poll_readback(gpu_context) else {
            return;
        };
        let bound = bound.min(config::MAX_PARTICLES);
        if bound != self.slot_bound {
            self.slot_bound = bound;
            self.counts_stale = true;
            self.write_params(gpu_context);
        }
    }

    fn apply_next_gust(&mut self, gpu_context: &GpuContext<'_>) {
        let gust = std::mem::take(&mut self.next_gust);
        if gust != self.gust {
            self.gust = gust;
            self.write_params(gpu_context);
        }
    }

    fn record_spawns(
        &mut self,
        gpu_context: &GpuContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
        spawn_count: usize,
    ) {
        if spawn_count == 0 {
            return;
        }
        self.slot_bound = (self.slot_bound + spawn_count as u32).min(config::MAX_PARTICLES);
        self.write_params(gpu_context);
        self.spawn
            .record(gpu_context, encoder, spawn_count, self.slot_bound);
        self.counts_stale = true;
    }

    fn record_simulation(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        substeps: u32,
        timing: bool,
    ) {
        if substeps > 0 && self.counts_stale {
            self.record_count(encoder);
        }
        if timing {
            self.record_timed_substeps(encoder, substeps);
        } else {
            self.record_substeps(encoder, substeps);
        }
        if substeps > 0 && self.growing {
            self.record_grow(encoder);
        }
    }

    fn record_substeps(&mut self, encoder: &mut wgpu::CommandEncoder, substeps: u32) {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Solver Pass"),
            timestamp_writes: None,
        });
        for _ in 0..substeps {
            pass.set_bind_group(0, &self.bind_group, &[self.params_offset()]);
            for span in Span::ALL {
                self.record_span(&mut pass, span);
            }
            self.contact_parity ^= 1;
        }
    }

    fn record_timed_substeps(&mut self, encoder: &mut wgpu::CommandEncoder, substeps: u32) {
        self.timer
            .as_mut()
            .expect("timing implies a timer")
            .begin_frame();
        for _ in 0..substeps {
            for span in Span::ALL {
                self.record_timed_span(encoder, span);
            }
            self.contact_parity ^= 1;
        }
        self.timer
            .as_mut()
            .expect("timing implies a timer")
            .resolve(encoder);
    }

    fn record_timed_span(&mut self, encoder: &mut wgpu::CommandEncoder, span: Span) {
        let pair = self.timer.as_mut().and_then(PassTimer::next_pair);
        let timer = self.timer.as_ref().expect("timing implies a timer");
        let writes = pair.map(|(begin, end)| wgpu::ComputePassTimestampWrites {
            query_set: timer.query_set(),
            beginning_of_pass_write_index: Some(begin),
            end_of_pass_write_index: Some(end),
        });
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some(span.label()),
            timestamp_writes: writes,
        });
        pass.set_bind_group(0, &self.bind_group, &[self.params_offset()]);
        self.record_span(&mut pass, span);
    }

    fn record_span(&self, pass: &mut wgpu::ComputePass<'_>, span: Span) {
        let (pipeline, groups) = self.dispatch(span);
        if groups[0] == 0 {
            return;
        }
        pass.set_pipeline(pipeline);
        pass.dispatch_workgroups(groups[0], groups[1], groups[2]);
    }
}

impl Spawner {
    fn new(
        gpu_context: &GpuContext<'_>,
        particle_buffer: &Buffer,
        contact_buffer: &Buffer,
        capacity: u32,
        seeded: u32,
    ) -> Self {
        let device = &gpu_context.device;
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Spawner"),
            source: wgpu::ShaderSource::Wgsl(include_str!("shaders/spawn.wgsl").into()),
        });

        let allocator = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Slot Allocator"),
            contents: bytemuck::cast_slice(&initial_allocator(seeded, capacity)),
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
        });
        let batch_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn Batch"),
            size: (std::mem::size_of::<SpawnHeader>()
                + config::MAX_SPAWNS_PER_FRAME as usize * std::mem::size_of::<SpawnRequest>())
                as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let stroke_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Brush Stroke"),
            size: std::mem::size_of::<BrushStroke>() as u64,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let blocked_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn Blocked"),
            size: config::MAX_SPAWNS_PER_FRAME as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let readback = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("Spawn State Readback"),
            size: 8,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let storage = wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only: false },
            has_dynamic_offset: false,
            min_binding_size: None,
        };
        let entry = |binding: u32, ty: wgpu::BindingType| wgpu::BindGroupLayoutEntry {
            binding,
            visibility: wgpu::ShaderStages::COMPUTE,
            ty,
            count: None,
        };
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Spawn Bind Group Layout"),
            entries: &[
                entry(0, storage),
                entry(1, storage),
                entry(
                    2,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(3, storage),
                entry(
                    5,
                    wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                ),
                entry(6, storage),
            ],
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Spawn Bind Group"),
            layout: &layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: particle_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: contact_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: batch_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: allocator.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: stroke_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: blocked_buffer.as_entire_binding(),
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Spawn Pipeline Layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = |label: &str, entry_point: &str| {
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some(entry_point),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            })
        };

        Self {
            batch_buffer,
            allocator,
            blocked_buffer,
            bind_group,
            check_pipeline: pipeline("Check Spawns", "check_spawns"),
            commit_pipeline: pipeline("Commit Spawns", "commit_spawns"),
            heat_pipeline: pipeline("Heat Brush", "heat_brush"),
            erase_pipeline: pipeline("Erase Brush", "erase_brush"),
            forget_pipeline: pipeline("Forget Erased Relatives", "forget_erased_relatives"),
            stroke_buffer,
            pending_stroke: None,
            pending: Vec::new(),
            readback,
            readback_in_flight: false,
            readback_ready: Arc::new(AtomicBool::new(false)),
            count_stale: false,
            live_count: seeded,
            requested_since_copy: 0,
        }
    }

    fn record(
        &mut self,
        gpu_context: &GpuContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
        count: usize,
        slot_bound: u32,
    ) {
        let requests: Vec<SpawnRequest> = self.pending.drain(..count).collect();
        let (lo, hi) = requests
            .iter()
            .fold(([f32::MAX; 2], [f32::MIN; 2]), |(lo, hi), r| {
                (
                    [
                        lo[0].min(r.position[0] - r.radius),
                        lo[1].min(r.position[1] - r.radius),
                    ],
                    [
                        hi[0].max(r.position[0] + r.radius),
                        hi[1].max(r.position[1] + r.radius),
                    ],
                )
            });
        let header = SpawnHeader {
            count: count as u32,
            slot_bound,
            max_contacts: config::MAX_CONTACTS,
            _padding: 0,
            lo,
            hi,
        };
        let queue = &gpu_context.queue;
        queue.write_buffer(&self.batch_buffer, 0, bytemuck::bytes_of(&header));
        queue.write_buffer(
            &self.batch_buffer,
            std::mem::size_of::<SpawnHeader>() as u64,
            bytemuck::cast_slice(&requests),
        );
        queue.write_buffer(
            &self.blocked_buffer,
            0,
            bytemuck::cast_slice(&vec![0u32; count]),
        );

        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Spawn Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        pass.set_pipeline(&self.check_pipeline);
        pass.dispatch_workgroups(slot_bound.div_ceil(WORKGROUP_SIZE), 1, 1);
        pass.set_pipeline(&self.commit_pipeline);
        pass.dispatch_workgroups((count as u32).div_ceil(WORKGROUP_SIZE), 1, 1);
        self.count_stale = true;
        self.requested_since_copy += count as u32;
    }

    fn record_stroke(
        &mut self,
        gpu_context: &GpuContext<'_>,
        encoder: &mut wgpu::CommandEncoder,
    ) -> bool {
        let Some((kind, stroke)) = self.pending_stroke.take() else {
            return false;
        };
        gpu_context
            .queue
            .write_buffer(&self.stroke_buffer, 0, bytemuck::bytes_of(&stroke));
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Brush Stroke Pass"),
            timestamp_writes: None,
        });
        pass.set_bind_group(0, &self.bind_group, &[]);
        let groups = stroke.slot_bound.div_ceil(WORKGROUP_SIZE);
        let pipelines = match kind {
            StrokeKind::Heat => vec![&self.heat_pipeline],
            StrokeKind::Erase => vec![&self.erase_pipeline, &self.forget_pipeline],
        };
        for pipeline in pipelines {
            pass.set_pipeline(pipeline);
            pass.dispatch_workgroups(groups, 1, 1);
        }
        let erased = matches!(kind, StrokeKind::Erase);
        self.count_stale |= erased;
        erased
    }

    fn reset(&mut self, gpu_context: &GpuContext<'_>, seeded: u32) {
        let allocator = initial_allocator(seeded, config::MAX_PARTICLES);
        gpu_context
            .queue
            .write_buffer(&self.allocator, 0, bytemuck::cast_slice(&allocator));
        self.pending.clear();
        self.pending_stroke = None;
        self.live_count = seeded;
        self.requested_since_copy = 0;
        self.count_stale = true;
    }

    fn wants_readback(&self) -> bool {
        self.count_stale && !self.readback_in_flight
    }

    fn record_readback(&mut self, encoder: &mut wgpu::CommandEncoder) -> bool {
        if !self.wants_readback() {
            return false;
        }
        encoder.copy_buffer_to_buffer(&self.allocator, 0, &self.readback, 0, 8);
        self.count_stale = false;
        self.readback_in_flight = true;
        self.requested_since_copy = 0;
        true
    }

    fn begin_readback(&self) {
        let ready = Arc::clone(&self.readback_ready);
        self.readback
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                if result.is_ok() {
                    ready.store(true, Ordering::Release);
                }
            });
    }

    fn poll_readback(&mut self, gpu_context: &GpuContext<'_>) -> Option<u32> {
        if !self.readback_in_flight {
            return None;
        }
        gpu_context.device.poll(wgpu::Maintain::Poll);
        if !self.readback_ready.swap(false, Ordering::Acquire) {
            return None;
        }
        let [free, high_water] =
            *bytemuck::from_bytes::<[u32; 2]>(&self.readback.slice(..).get_mapped_range());
        self.readback.unmap();
        self.live_count = config::MAX_PARTICLES - free;
        self.readback_in_flight = false;
        Some(high_water + self.requested_since_copy)
    }
}
