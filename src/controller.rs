use crate::{
    config,
    gpu_context::GpuContext,
    model::Model,
    overlay::{self, BrushMode, Overlay},
    view::View,
};
use instant::Instant;
use winit::{
    event::*,
    event_loop::EventLoop,
    keyboard::{KeyCode, PhysicalKey},
    window::{Fullscreen, Window},
};

const AIM_STEP: f32 = 2.0;

struct Aim {
    anchor: Option<[f32; 2]>,
    direction: [f32; 2],
}

impl Aim {
    fn new() -> Self {
        Self {
            anchor: None,
            direction: [1.0, 0.0],
        }
    }

    fn follow(&mut self, cursor: [f32; 2]) -> [f32; 2] {
        let anchor = *self.anchor.get_or_insert(cursor);
        let step = [cursor[0] - anchor[0], cursor[1] - anchor[1]];
        let length = step[0].hypot(step[1]);
        if length >= AIM_STEP {
            self.direction = [step[0] / length, step[1] / length];
            self.anchor = Some(cursor);
        }
        self.direction
    }

    fn release(&mut self) {
        self.anchor = None;
    }
}

#[derive(Default)]
struct Finger {
    down: Option<u64>,
    lifted: Option<DeviceId>,
}

impl Finger {
    fn as_mouse(&mut self, event: &WindowEvent) -> Vec<WindowEvent> {
        let WindowEvent::Touch(touch) = event else {
            return vec![event.clone()];
        };
        if touch.phase == TouchPhase::Started && self.down.is_none() {
            self.down = Some(touch.id);
        }
        if self.down != Some(touch.id) {
            return Vec::new();
        }
        let device_id = touch.device_id;
        let moved = WindowEvent::CursorMoved {
            device_id,
            position: touch.location,
        };
        let button = |state| WindowEvent::MouseInput {
            device_id,
            state,
            button: MouseButton::Left,
        };
        match touch.phase {
            TouchPhase::Started => {
                self.lifted = None;
                vec![moved, button(ElementState::Pressed)]
            }
            TouchPhase::Moved => vec![moved],
            TouchPhase::Ended | TouchPhase::Cancelled => {
                self.down = None;
                self.lifted = Some(device_id);
                vec![moved, button(ElementState::Released)]
            }
        }
    }

    fn take_lift(&mut self) -> Option<WindowEvent> {
        self.lifted
            .take()
            .map(|device_id| WindowEvent::CursorLeft { device_id })
    }
}

pub struct Controller<'a> {
    gpu_context: GpuContext<'a>,
    model: Model,
    view: View<'a>,
    last_update: Instant,
    surface_configured: bool,
    overlay: Overlay,
    cursor_position: [f32; 2],
    is_left_mouse_button_pressed: bool,
    touch: Finger,
    aim: Aim,
}

impl<'a> Controller<'a> {
    pub async fn new(window: &'a Window) -> Self {
        let gpu_context = GpuContext::new(window).await;
        let model = Model::new(&gpu_context);
        let view = View::new(window, &gpu_context, model.materials_buffer());
        let surface_configured = view.is_configured();
        let overlay = Overlay::new(
            &gpu_context.device,
            view.surface_format,
            model.materials(),
            model.globals(),
        );

        Self {
            gpu_context,
            model,
            view,
            last_update: Instant::now(),
            surface_configured,
            overlay,
            cursor_position: [0.0, 0.0],
            is_left_mouse_button_pressed: false,
            touch: Finger::default(),
            aim: Aim::new(),
        }
    }

    pub fn handle_event(&mut self, event_loop: EventLoop<()>) {
        event_loop
            .run(move |event, control_flow| match event {
                Event::WindowEvent {
                    ref event,
                    window_id,
                } if window_id == self.view.window().id() => {
                    for event in self.touch.as_mouse(event) {
                        if self.dispatch(&event) {
                            control_flow.exit();
                            return;
                        }
                    }
                }
                _ => {}
            })
            .expect("Failed while running event loop");
    }

    fn dispatch(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.cursor_position = [position.x as f32, position.y as f32];
            }
            WindowEvent::MouseInput {
                state: ElementState::Released,
                button: MouseButton::Left,
                ..
            } => self.is_left_mouse_button_pressed = false,
            _ => {}
        }
        let scale = self.ui_scale();
        if self.overlay.on_window_event(event, scale) {
            return false;
        }
        !self.view.input(event) && self.handle_window_event(event)
    }

    fn handle_window_event(&mut self, event: &WindowEvent) -> bool {
        match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::F1),
                        ..
                    },
                ..
            } => {
                self.overlay.show_developer = !self.overlay.show_developer;
            }
            WindowEvent::MouseInput {
                state,
                button: MouseButton::Left,
                ..
            } => self.is_left_mouse_button_pressed = state.is_pressed(),
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        state: ElementState::Pressed,
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        ..
                    },
                ..
            } => {
                if self.view.window().fullscreen().is_some() {
                    self.view.window().set_fullscreen(None);
                } else {
                    return true;
                }
            }
            WindowEvent::CloseRequested => return true,
            WindowEvent::Resized(physical_size) => {
                log::info!("physical_size: {physical_size:?}");
                self.surface_configured = true;
                self.view.resize(*physical_size, &self.gpu_context);
            }
            WindowEvent::RedrawRequested => {
                let now = Instant::now();
                self.view.window().request_redraw();

                if !self.surface_configured {
                    return false;
                }

                let dt = now
                    .duration_since(self.last_update)
                    .as_secs_f32()
                    .min(config::MAX_FRAME_TIME);
                self.last_update = now;

                if self.is_left_mouse_button_pressed && !self.overlay.captures_pointer() {
                    if let Some(world) = self.view.screen_to_world(self.cursor_position) {
                        let radius = self.overlay.brush_radius();
                        let speed = self.overlay.brush_wind_speed();
                        match self.overlay.brush_mode() {
                            BrushMode::Paint => {
                                self.model
                                    .paint(world, self.overlay.spawn_material(), radius)
                            }
                            BrushMode::Heat | BrushMode::Cool => {
                                self.model
                                    .heat(world, radius, self.overlay.brush_heat_rate() * dt)
                            }
                            BrushMode::Blow => {
                                let [x, y] = self.aim.follow(world);
                                self.model.blow(world, radius, [x * speed, y * speed], 0.0);
                            }
                            BrushMode::Burst => self.model.blow(world, radius, [0.0, 0.0], speed),
                        }
                    }
                } else {
                    self.aim.release();
                }
                self.model.update(dt, &self.gpu_context);

                let scale = self.ui_scale();
                self.overlay.stats.record(
                    dt,
                    self.model.last_substeps(),
                    self.model.live_count(),
                    self.model.pass_timings(),
                );
                self.overlay.run(
                    self.view.size.width,
                    self.view.size.height,
                    scale,
                    self.model.live_count(),
                    self.view.window().fullscreen().is_some(),
                    self.view.world_rect(),
                    self.aim.direction,
                );
                if let Some(leave) = self.touch.take_lift() {
                    self.overlay.on_window_event(&leave, scale);
                }
                self.view
                    .set_menu_height(self.overlay.menu_height() * scale, &self.gpu_context);
                if self.overlay.take_fullscreen_request() {
                    self.toggle_fullscreen();
                }
                let (materials, globals) = (self.overlay.materials, self.overlay.globals);
                self.model
                    .set_parameters(&self.gpu_context, &materials, &globals);
                self.model
                    .set_timing(&self.gpu_context, self.overlay.times_passes());

                self.view.set_air(
                    &self.gpu_context,
                    self.model.globals().wind,
                    &self.model.gust(),
                    dt,
                );
                match self.view.render(
                    &self.gpu_context,
                    self.model.particle_buffer(),
                    self.model.slot_bound(),
                    &mut self.overlay,
                    scale,
                ) {
                    Ok(_) => {}
                    Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                        self.view.resize(self.view.size, &self.gpu_context)
                    }
                    Err(wgpu::SurfaceError::OutOfMemory | wgpu::SurfaceError::Other) => {
                        log::error!("OutOfMemory");
                        return true;
                    }
                    Err(wgpu::SurfaceError::Timeout) => {
                        log::warn!("Surface timeout")
                    }
                }
            }
            _ => {}
        }
        return false;
    }

    fn ui_scale(&self) -> f32 {
        let factor = self.view.window().scale_factor() as f32;
        factor * overlay::zoom_for(self.view.size.width as f32 / factor)
    }

    fn toggle_fullscreen(&self) {
        let window = self.view.window();
        window.set_fullscreen(match window.fullscreen() {
            Some(_) => None,
            None => Some(Fullscreen::Borderless(None)),
        });
    }
}

pub async fn run() {
    let event_loop = EventLoop::new().expect("Failed to create event loop");
    let window = winit::window::WindowBuilder::new()
        .build(&event_loop)
        .expect("Failed to create window");

    #[cfg(target_arch = "wasm32")]
    {
        use winit::platform::web::WindowExtWebSys;
        web_sys::window()
            .and_then(|win| win.document())
            .and_then(|doc| {
                let dst = doc.get_element_by_id("wasm-example")?;
                let canvas = web_sys::Element::from(window.canvas()?);
                dst.append_child(&canvas).ok()?;
                Some(())
            })
            .expect("Couldn't append canvas to document body.");
    }

    let mut controller = Controller::new(&window).await;

    controller.handle_event(event_loop);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blow_aims_along_a_slow_drag_and_keeps_it_after_release() {
        let mut aim = Aim::new();
        assert_eq!(
            aim.follow([100.0, 50.0]),
            [1.0, 0.0],
            "a first press blows right"
        );
        let mut direction = [0.0; 2];
        for frame in 1..=8 {
            direction = aim.follow([100.0 - 0.5 * frame as f32, 50.0]);
        }
        assert_eq!(direction, [-1.0, 0.0], "a slow leftward drag turns it left");
        for wobble in [[0.4, 0.3], [-0.5, 0.2], [0.3, -0.6]] {
            assert_eq!(
                aim.follow([96.0 + wobble[0], 50.0 + wobble[1]]),
                [-1.0, 0.0],
                "trembling in place must not swing the aim",
            );
        }
        aim.release();
        assert_eq!(
            aim.follow([300.0, 20.0]),
            [-1.0, 0.0],
            "the next press keeps it"
        );
    }

    #[test]
    fn the_first_finger_is_the_mouse_and_the_rest_are_ignored() {
        let device_id = unsafe { DeviceId::dummy() };
        let touch = |id, phase, x| {
            WindowEvent::Touch(Touch {
                device_id,
                phase,
                location: winit::dpi::PhysicalPosition::new(x, 0.0),
                force: None,
                id,
            })
        };
        let kinds = |events: Vec<WindowEvent>| -> Vec<&'static str> {
            events
                .iter()
                .map(|event| match event {
                    WindowEvent::CursorMoved { .. } => "move",
                    WindowEvent::MouseInput {
                        state: ElementState::Pressed,
                        ..
                    } => "press",
                    WindowEvent::MouseInput {
                        state: ElementState::Released,
                        ..
                    } => "release",
                    WindowEvent::CursorLeft { .. } => "leave",
                    _ => "other",
                })
                .collect()
        };
        let mut finger = Finger::default();
        assert_eq!(
            kinds(finger.as_mouse(&touch(1, TouchPhase::Started, 5.0))),
            ["move", "press"]
        );
        assert!(finger
            .as_mouse(&touch(2, TouchPhase::Started, 9.0))
            .is_empty());
        assert_eq!(
            kinds(finger.as_mouse(&touch(1, TouchPhase::Moved, 6.0))),
            ["move"]
        );
        assert!(finger
            .as_mouse(&touch(2, TouchPhase::Ended, 9.0))
            .is_empty());
        assert_eq!(
            kinds(finger.as_mouse(&touch(1, TouchPhase::Ended, 6.0))),
            ["move", "release"]
        );
        assert_eq!(kinds(finger.take_lift().into_iter().collect()), ["leave"]);
        assert!(finger.take_lift().is_none());
        finger.as_mouse(&touch(2, TouchPhase::Started, 9.0));
        finger.as_mouse(&touch(2, TouchPhase::Ended, 9.0));
        assert_eq!(
            kinds(finger.as_mouse(&touch(3, TouchPhase::Started, 4.0))),
            ["move", "press"]
        );
        assert!(finger.take_lift().is_none());
    }
}
