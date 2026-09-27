use wgpu::util::DeviceExt;
use wgpu::PresentMode;
use winit::{event::WindowEvent, window::Window};

use crate::{compute_pipeline::Gust, config, gpu_context::GpuContext, overlay::Overlay};

const QUAD: &[f32] = &[
    -1.0, -1.0, 1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0, 1.0,
];

const PIXEL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8Unorm;

#[repr(C)]
#[derive(Copy, Clone)]
struct Placement {
    offset: [f32; 2],
    scale: f32,
    _padding: f32,
}

unsafe impl bytemuck::Zeroable for Placement {}
unsafe impl bytemuck::Pod for Placement {}

impl Placement {
    fn for_surface(width: u32, height: u32, reserved: f32) -> Self {
        let surface = [width.max(1) as f32, height.max(1) as f32];
        let pixels = [config::PIXEL_WIDTH as f32, config::PIXEL_HEIGHT as f32];
        let room = (surface[1] - reserved).max(1.0);
        let scale = (surface[0] / pixels[0]).min(room / pixels[1]);
        Self {
            offset: [
                ((surface[0] - pixels[0] * scale) * 0.5).max(0.0).floor(),
                0.0,
            ],
            scale,
            _padding: 0.0,
        }
    }
}

#[repr(C)]
#[derive(Copy, Clone)]
struct Air {
    gust_centre: [f32; 2],
    gust_velocity: [f32; 2],
    gust_radius: f32,
    gust_outflow: f32,
    wind: f32,

    time: f32,
}

unsafe impl bytemuck::Zeroable for Air {}
unsafe impl bytemuck::Pod for Air {}

const AIR_CLOCK_WRAP: f32 = 600.0;

#[repr(C)]
#[derive(Copy, Clone)]
struct RenderParams {
    world: [f32; 2],
    pixels: [f32; 2],
}

unsafe impl bytemuck::Zeroable for RenderParams {}
unsafe impl bytemuck::Pod for RenderParams {}

pub struct View<'a> {
    pub size: winit::dpi::PhysicalSize<u32>,
    pub surface_format: wgpu::TextureFormat,
    config: wgpu::SurfaceConfiguration,
    window: &'a Window,
    render_pipeline: wgpu::RenderPipeline,
    bind_group: wgpu::BindGroup,
    quad_buffer: wgpu::Buffer,

    pixel_view: wgpu::TextureView,
    blit_pipeline: wgpu::RenderPipeline,
    blit_bind_group: wgpu::BindGroup,

    placement: Placement,

    menu_height: f32,
    placement_buffer: wgpu::Buffer,

    air_buffer: wgpu::Buffer,
    air_clock: f32,
}

impl<'a> View<'a> {
    pub fn new(
        window: &'a Window,
        gpu_context: &GpuContext<'_>,
        materials: &wgpu::Buffer,
    ) -> View<'a> {
        let size = window.inner_size();

        let surface_caps = gpu_context.surface.get_capabilities(&gpu_context.adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .find(|f| !f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: PresentMode::Fifo,
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let shader = gpu_context
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: Some("Particle Shader"),
                source: wgpu::ShaderSource::Wgsl(include_str!("shaders/shader.wgsl").into()),
            });

        let params_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Render Params"),
                    contents: bytemuck::bytes_of(&RenderParams {
                        world: [config::WORLD_WIDTH, config::WORLD_HEIGHT],
                        pixels: [config::PIXEL_WIDTH as f32, config::PIXEL_HEIGHT as f32],
                    }),
                    usage: wgpu::BufferUsages::UNIFORM,
                });

        let bind_group_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Render Bind Group Layout"),
                    entries: &[wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::VERTEX,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    }],
                });
        let bind_group = gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Render Bind Group"),
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: params_buffer.as_entire_binding(),
                }],
            });

        let render_pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Render Pipeline Layout"),
                    bind_group_layouts: &[&bind_group_layout],
                    push_constant_ranges: &[],
                });

        let render_pipeline =
            gpu_context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Render Pipeline"),
                    layout: Some(&render_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("vs_main"),
                        buffers: &[
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<[f32; 2]>()
                                    as wgpu::BufferAddress,
                                step_mode: wgpu::VertexStepMode::Vertex,
                                attributes: &[wgpu::VertexAttribute {
                                    offset: 0,
                                    shader_location: 0,
                                    format: wgpu::VertexFormat::Float32x2,
                                }],
                            },
                            wgpu::VertexBufferLayout {
                                array_stride: std::mem::size_of::<crate::ecs::Particle>()
                                    as wgpu::BufferAddress,
                                step_mode: wgpu::VertexStepMode::Instance,
                                attributes: &[
                                    wgpu::VertexAttribute {
                                        offset: 24,
                                        shader_location: 1,
                                        format: wgpu::VertexFormat::Float32x2,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: 16,
                                        shader_location: 2,
                                        format: wgpu::VertexFormat::Float32,
                                    },
                                    wgpu::VertexAttribute {
                                        offset: 20,
                                        shader_location: 3,
                                        format: wgpu::VertexFormat::Uint32,
                                    },
                                ],
                            },
                        ],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("fs_main"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: PIXEL_FORMAT,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,

                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        let quad_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Quad Buffer"),
                    contents: bytemuck::cast_slice(QUAD),
                    usage: wgpu::BufferUsages::VERTEX,
                });

        let pixel_texture = gpu_context.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Pixel Buffer"),
            size: wgpu::Extent3d {
                width: config::PIXEL_WIDTH,
                height: config::PIXEL_HEIGHT,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: PIXEL_FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let pixel_view = pixel_texture.create_view(&wgpu::TextureViewDescriptor::default());

        let pixel_sampler = gpu_context.device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("Pixel Sampler"),
            address_mode_u: wgpu::AddressMode::ClampToEdge,
            address_mode_v: wgpu::AddressMode::ClampToEdge,
            address_mode_w: wgpu::AddressMode::ClampToEdge,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            ..Default::default()
        });

        let blit_layout =
            gpu_context
                .device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Blit Bind Group Layout"),
                    entries: &[
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Texture {
                                multisampled: false,
                                view_dimension: wgpu::TextureViewDimension::D2,
                                sample_type: wgpu::TextureSampleType::Float { filterable: false },
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::NonFiltering),
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::FRAGMENT,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });

        let placement = Placement::for_surface(size.width, size.height, 0.0);
        let placement_buffer =
            gpu_context
                .device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Placement"),
                    contents: bytemuck::bytes_of(&placement),
                    usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                });

        let air_buffer = gpu_context
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Air"),
                contents: bytemuck::bytes_of(&<Air as bytemuck::Zeroable>::zeroed()),
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            });
        let blit_bind_group = gpu_context
            .device
            .create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("Blit Bind Group"),
                layout: &blit_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: wgpu::BindingResource::TextureView(&pixel_view),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::Sampler(&pixel_sampler),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: placement_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 3,
                        resource: materials.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 4,
                        resource: air_buffer.as_entire_binding(),
                    },
                ],
            });
        let blit_pipeline_layout =
            gpu_context
                .device
                .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("Blit Pipeline Layout"),
                    bind_group_layouts: &[&blit_layout],
                    push_constant_ranges: &[],
                });
        let blit_pipeline =
            gpu_context
                .device
                .create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                    label: Some("Blit Pipeline"),
                    layout: Some(&blit_pipeline_layout),
                    vertex: wgpu::VertexState {
                        module: &shader,
                        entry_point: Some("blit_vs"),
                        buffers: &[],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    },
                    fragment: Some(wgpu::FragmentState {
                        module: &shader,
                        entry_point: Some("blit_fs"),
                        targets: &[Some(wgpu::ColorTargetState {
                            format: surface_format,
                            blend: Some(wgpu::BlendState::REPLACE),
                            write_mask: wgpu::ColorWrites::ALL,
                        })],
                        compilation_options: wgpu::PipelineCompilationOptions::default(),
                    }),
                    primitive: wgpu::PrimitiveState {
                        topology: wgpu::PrimitiveTopology::TriangleList,
                        strip_index_format: None,
                        front_face: wgpu::FrontFace::Ccw,
                        cull_mode: None,
                        polygon_mode: wgpu::PolygonMode::Fill,
                        unclipped_depth: false,
                        conservative: false,
                    },
                    depth_stencil: None,
                    multisample: wgpu::MultisampleState {
                        count: 1,
                        mask: !0,
                        alpha_to_coverage_enabled: false,
                    },
                    multiview: None,
                    cache: None,
                });

        if size.width > 0 && size.height > 0 {
            gpu_context.surface.configure(&gpu_context.device, &config);
        }

        Self {
            size,
            surface_format,
            config,
            window,
            render_pipeline,
            bind_group,
            quad_buffer,
            pixel_view,
            blit_pipeline,
            blit_bind_group,
            placement,
            menu_height: 0.0,
            placement_buffer,
            air_buffer,
            air_clock: 0.0,
        }
    }

    pub fn is_configured(&self) -> bool {
        self.config.width > 0 && self.config.height > 0
    }

    pub fn window(&self) -> &Window {
        self.window
    }

    pub fn resize(
        &mut self,
        new_size: winit::dpi::PhysicalSize<u32>,
        gpu_context: &GpuContext<'_>,
    ) {
        if new_size.width > 0 && new_size.height > 0 {
            self.size = new_size;
            self.config.width = new_size.width;
            self.config.height = new_size.height;
            gpu_context
                .surface
                .configure(&gpu_context.device, &self.config);
            self.place(gpu_context);
        }
    }

    pub fn set_menu_height(&mut self, height: f32, gpu_context: &GpuContext<'_>) {
        if height != self.menu_height {
            self.menu_height = height;
            self.place(gpu_context);
        }
    }

    fn place(&mut self, gpu_context: &GpuContext<'_>) {
        self.placement =
            Placement::for_surface(self.config.width, self.config.height, self.menu_height);
        gpu_context.queue.write_buffer(
            &self.placement_buffer,
            0,
            bytemuck::bytes_of(&self.placement),
        );
    }

    pub fn set_air(&mut self, gpu_context: &GpuContext<'_>, wind: f32, gust: &Gust, dt: f32) {
        self.air_clock = (self.air_clock + dt) % AIR_CLOCK_WRAP;

        let per_unit = config::PIXEL_WIDTH as f32 / config::WORLD_WIDTH;
        let air = Air {
            gust_centre: gust.centre.map(|c| c * per_unit),
            gust_velocity: gust.velocity.map(|v| v * per_unit),
            gust_radius: gust.radius * per_unit,
            gust_outflow: gust.outflow * per_unit,
            wind: wind * per_unit,
            time: self.air_clock,
        };
        gpu_context
            .queue
            .write_buffer(&self.air_buffer, 0, bytemuck::bytes_of(&air));
    }

    pub fn world_rect(&self) -> [f32; 4] {
        let p = &self.placement;
        [
            p.offset[0],
            p.offset[1],
            config::PIXEL_WIDTH as f32 * p.scale,
            config::PIXEL_HEIGHT as f32 * p.scale,
        ]
    }

    pub fn screen_to_world(&self, position: [f32; 2]) -> Option<[f32; 2]> {
        let dims = [config::PIXEL_WIDTH as f32, config::PIXEL_HEIGHT as f32];
        let world = [config::WORLD_WIDTH, config::WORLD_HEIGHT];

        let mut out = [0.0; 2];
        for axis in 0..2 {
            let local = (position[axis] - self.placement.offset[axis]) / self.placement.scale;
            if !(0.0..dims[axis]).contains(&local) {
                return None;
            }
            out[axis] = local * world[axis] / dims[axis];
        }
        Some(out)
    }

    pub fn input(&mut self, _event: &WindowEvent) -> bool {
        false
    }

    pub fn render(
        &mut self,
        gpu_context: &GpuContext<'_>,
        particles: &wgpu::Buffer,
        particle_count: u32,
        overlay: &mut Overlay,
        scale: f32,
    ) -> Result<(), wgpu::SurfaceError> {
        let surface_texture = gpu_context.surface.get_current_texture()?;
        let texture_view = surface_texture
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut command_encoder =
            gpu_context
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("Render encoder"),
                });

        {
            let mut grain_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Grain pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &self.pixel_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            grain_pass.set_pipeline(&self.render_pipeline);
            grain_pass.set_bind_group(0, &self.bind_group, &[]);
            grain_pass.set_vertex_buffer(0, self.quad_buffer.slice(..));
            grain_pass.set_vertex_buffer(1, particles.slice(..));
            grain_pass.draw(0..6, 0..particle_count);
        }

        {
            let mut blit_pass = command_encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Upscale pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &texture_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });

            blit_pass.set_pipeline(&self.blit_pipeline);
            blit_pass.set_bind_group(0, &self.blit_bind_group, &[]);
            blit_pass.draw(0..3, 0..1);
        }

        overlay.render(
            &gpu_context.device,
            &gpu_context.queue,
            &mut command_encoder,
            &texture_view,
            self.config.width,
            self.config.height,
            scale,
        );

        gpu_context
            .queue
            .submit(std::iter::once(command_encoder.finish()));
        surface_texture.present();

        Ok(())
    }
}
