use crate::{
    compute_pipeline::{Gust, Solver, SpawnRequest},
    config, ecs,
    gpu_context::GpuContext,
    gpu_timing::Span,
    materials::{Globals, MaterialParams, MATERIAL_COUNT, NO_TRANSITION},
};

pub struct Model {
    solver: Solver,
    seeded: u32,
}

impl Model {
    pub fn new(gpu_context: &GpuContext<'_>) -> Self {
        let particles = ecs::seed_world(config::PARTICLE_COUNT);
        Self {
            solver: Solver::new(gpu_context, &particles),
            seeded: particles.len() as u32,
        }
    }

    pub fn update(&mut self, dt: f32, gpu_context: &GpuContext<'_>) {
        self.solver.step(gpu_context, dt);
    }

    pub fn particle_buffer(&self) -> &wgpu::Buffer {
        self.solver.particle_buffer()
    }

    pub fn materials_buffer(&self) -> &wgpu::Buffer {
        self.solver.materials_buffer()
    }

    pub fn materials(&self) -> [MaterialParams; MATERIAL_COUNT as usize] {
        *self.solver.materials()
    }

    pub fn globals(&self) -> Globals {
        *self.solver.globals()
    }

    pub fn last_substeps(&self) -> u32 {
        self.solver.last_substeps()
    }

    pub fn pass_timings(&self) -> Option<[f32; Span::ALL.len()]> {
        self.solver.pass_timings()
    }

    pub fn set_timing(&mut self, gpu_context: &GpuContext<'_>, enabled: bool) {
        self.solver.set_timing(gpu_context, enabled);
    }

    pub fn set_parameters(
        &mut self,
        gpu_context: &GpuContext<'_>,
        materials: &[MaterialParams; MATERIAL_COUNT as usize],
        globals: &Globals,
    ) {
        self.solver.set_parameters(gpu_context, materials, globals);
    }

    pub fn slot_bound(&self) -> u32 {
        self.solver.slot_bound()
    }

    pub fn live_count(&self) -> u32 {
        self.solver.live_count()
    }

    pub fn placed_count(&self) -> u32 {
        self.live_count().saturating_sub(self.seeded)
    }

    pub fn paint(&mut self, centre: [f32; 2], material: u32, radius: f32) {
        let spacing = self.solver.materials()[material as usize].rest_spacing();
        for point in ecs::brush_points(centre, radius, spacing) {
            self.spawn(point, material);
        }
    }

    pub fn heat(&mut self, centre: [f32; 2], radius: f32, delta: f32) {
        self.solver.heat(centre, radius, delta);
    }

    pub fn blow(&mut self, centre: [f32; 2], radius: f32, velocity: [f32; 2], outflow: f32) {
        self.solver.blow(Gust {
            centre,
            radius,
            velocity,
            outflow,
        });
    }

    pub fn gust(&self) -> Gust {
        self.solver.gust()
    }

    pub fn spawn(&mut self, position: [f32; 2], material: u32) {
        let params = self.solver.materials()[material as usize];

        let radius = (params.radius
            * ecs::random_range(1.0 - ecs::SIZE_SPREAD, 1.0 + ecs::SIZE_SPREAD))
        .min(config::RADIUS_LIMIT);

        let position = [
            position[0].clamp(radius, config::WORLD_WIDTH - radius),
            position[1].clamp(radius, config::WORLD_HEIGHT - radius),
        ];
        self.solver.spawn(SpawnRequest {
            position,
            radius,
            material,

            temperature: params.default_temperature,
            budget: if params.sprouts == NO_TRANSITION {
                0
            } else {
                (config::SHOOT_LENGTH as f32 * ecs::random_range(0.6, 1.0)) as u32
            },
        });
    }
}
