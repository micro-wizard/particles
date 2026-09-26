use winit::window::Window;

pub struct GpuContext<'a> {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    pub adapter: wgpu::Adapter,
    pub surface: wgpu::Surface<'a>,
}

impl<'a> GpuContext<'a> {
    pub async fn new(window: &'a Window) -> GpuContext<'a> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            #[cfg(not(target_arch = "wasm32"))]
            backends: wgpu::Backends::PRIMARY,
            #[cfg(target_arch = "wasm32")]
            backends: wgpu::Backends::BROWSER_WEBGPU,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window)
            .expect("Failed to create surface");
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::default(),
                compatible_surface: Some(&surface),
                force_fallback_adapter: false,
            })
            .await
            .expect(
                "No suitable GPU adapter. The web build requires WebGPU \
                 (navigator.gpu); WebGL2 cannot run this engine's compute shaders.",
            );
        let info = adapter.get_info();
        log::info!(
            "adapter: {} ({:?}, {:?})",
            info.name,
            info.device_type,
            info.backend
        );
        let timestamps = adapter.features() & wgpu::Features::TIMESTAMP_QUERY;
        log::info!(
            "GPU timestamp queries: {}",
            if timestamps.is_empty() {
                "unavailable"
            } else {
                "available"
            }
        );
        let (device, queue) = adapter
            .request_device(
                &wgpu::DeviceDescriptor {
                    required_features: timestamps,
                    required_limits: wgpu::Limits::default(),
                    label: None,
                    memory_hints: Default::default(),
                },
                None,
            )
            .await
            .expect("Failed to request device and queue");

        Self {
            device,
            queue,
            adapter,
            surface,
        }
    }
}
