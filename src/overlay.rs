use crate::{
    config,
    gpu_timing::Span,
    materials::{Globals, MaterialParams, MATERIALS, MATERIAL_COUNT, NO_TRANSITION},
    pixel_font,
};
use std::collections::VecDeque;
use winit::event::{ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::keyboard::{Key, NamedKey};

const SAMPLE_WINDOW: usize = 120;

const LOG_INTERVAL: f32 = 1.0;

pub struct Stats {
    frame_times: VecDeque<f32>,
    substeps: u32,
    simulated: f32,
    wanted: f32,
    elapsed: f32,
    since_log: f32,
}

impl Stats {
    fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(SAMPLE_WINDOW),
            substeps: 0,
            simulated: 0.0,
            wanted: 0.0,
            elapsed: 0.0,
            since_log: 0.0,
        }
    }

    pub fn record(
        &mut self,
        frame_dt: f32,
        time_scale: f32,
        substeps: u32,
        particles: u32,
        passes: Option<[f32; Span::ALL.len()]>,
    ) {
        if self.frame_times.len() == SAMPLE_WINDOW {
            self.frame_times.pop_front();
        }
        self.frame_times.push_back(frame_dt);
        self.substeps = substeps;

        self.simulated += substeps as f32 * config::SUBSTEP;
        self.wanted += frame_dt * time_scale;
        self.elapsed += frame_dt;
        if self.elapsed > 2.0 {
            self.simulated *= 0.5;
            self.wanted *= 0.5;
            self.elapsed *= 0.5;
        }

        self.since_log += frame_dt;
        if self.since_log >= LOG_INTERVAL {
            self.since_log = 0.0;
            self.log_performance(substeps, particles, passes);
        }
    }

    fn log_performance(
        &self,
        substeps: u32,
        particles: u32,
        passes: Option<[f32; Span::ALL.len()]>,
    ) {
        let gpu = passes.map_or(String::new(), |ms| {
            let per_pass = Span::ALL
                .iter()
                .zip(ms)
                .map(|(span, value)| format!(" {}_ms={value:.3}", span.label()))
                .collect::<String>();
            format!("{per_pass} gpu_total_ms={:.3}", ms.iter().sum::<f32>())
        });
        log::info!(
            "[perf] particles={particles} fps={:.0} mean_ms={:.2} worst_ms={:.2} \
             substeps={substeps} of {} sim_speed={:.3}{gpu}",
            self.frames_per_second(),
            self.mean_frame_time() * 1000.0,
            self.worst_frame_time() * 1000.0,
            config::MAX_SUBSTEPS,
            self.sim_speed(),
        );
    }

    fn mean_frame_time(&self) -> f32 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
    }

    fn readout(&self, particles: u32) -> Readout {
        Readout {
            fps: self.frames_per_second(),
            particles,
            lagging_speed: self.falling_behind().then(|| self.sim_speed()),
        }
    }

    fn falling_behind(&self) -> bool {
        self.simulated < KEEPING_UP * self.wanted
    }

    fn frames_per_second(&self) -> f32 {
        let mean = self.mean_frame_time();
        if mean > 0.0 {
            1.0 / mean
        } else {
            0.0
        }
    }

    fn worst_frame_time(&self) -> f32 {
        self.frame_times.iter().copied().fold(0.0, f32::max)
    }

    fn sim_speed(&self) -> f32 {
        if self.elapsed > 0.0 {
            self.simulated / self.elapsed
        } else {
            0.0
        }
    }
}

#[derive(Copy, Clone, PartialEq, Eq)]
pub enum BrushMode {
    Paint,
    Heat,
    Cool,
    Blow,
    Burst,
    Erase,
}

struct Brush {
    mode: BrushMode,

    material: usize,

    heat_rate: f32,

    wind_speed: f32,

    radius: f32,
}

impl Default for Brush {
    fn default() -> Self {
        Self {
            mode: BrushMode::Paint,
            material: 0,
            heat_rate: 720.0,
            wind_speed: 2000.0,
            radius: 6.0,
        }
    }
}

pub struct Overlay {
    context: egui::Context,
    renderer: egui_wgpu::Renderer,
    events: Vec<egui::Event>,
    modifiers: egui::Modifiers,
    pointer: egui::Pos2,
    captured: bool,
    pub show_developer: bool,
    pub stats: Stats,
    pub materials: [MaterialParams; MATERIAL_COUNT as usize],
    pub globals: Globals,
    selected: usize,
    brush: Brush,
    time_passes: bool,
    time_scale: f32,
    fullscreen_requested: bool,
    reset_requested: bool,
    changelog_open: bool,
    menu_height: f32,
    paint_jobs: Vec<egui::ClippedPrimitive>,
    textures_delta: egui::TexturesDelta,
}

impl Overlay {
    pub fn new(
        device: &wgpu::Device,
        surface_format: wgpu::TextureFormat,
        materials: [MaterialParams; MATERIAL_COUNT as usize],
        globals: Globals,
    ) -> Self {
        Self {
            context: egui::Context::default(),
            renderer: egui_wgpu::Renderer::new(device, surface_format, None, 1, false),
            events: Vec::new(),
            modifiers: egui::Modifiers::default(),
            pointer: egui::Pos2::ZERO,
            captured: false,

            show_developer: false,
            stats: Stats::new(),
            materials,
            globals,
            selected: 0,
            brush: Brush::default(),
            time_passes: false,
            time_scale: REAL_TIME,
            fullscreen_requested: false,
            reset_requested: false,
            changelog_open: false,
            menu_height: 0.0,
            paint_jobs: Vec::new(),
            textures_delta: egui::TexturesDelta::default(),
        }
    }

    pub fn on_window_event(&mut self, event: &WindowEvent, scale: f32) -> bool {
        match event {
            WindowEvent::CursorMoved { position, .. } => {
                self.pointer = egui::pos2(position.x as f32 / scale, position.y as f32 / scale);
                self.events.push(egui::Event::PointerMoved(self.pointer));
                self.captured
            }
            WindowEvent::CursorLeft { .. } => {
                self.events.push(egui::Event::PointerGone);
                false
            }
            WindowEvent::MouseInput { state, button, .. } => {
                let button = match button {
                    MouseButton::Left => Some(egui::PointerButton::Primary),
                    MouseButton::Right => Some(egui::PointerButton::Secondary),
                    MouseButton::Middle => Some(egui::PointerButton::Middle),
                    _ => None,
                };
                if let Some(button) = button {
                    self.events.push(egui::Event::PointerButton {
                        pos: self.pointer,
                        button,
                        pressed: *state == ElementState::Pressed,
                        modifiers: self.modifiers,
                    });
                }
                self.captured
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let d = match delta {
                    MouseScrollDelta::LineDelta(x, y) => egui::vec2(*x * 24.0, *y * 24.0),
                    MouseScrollDelta::PixelDelta(p) => {
                        egui::vec2(p.x as f32 / scale, p.y as f32 / scale)
                    }
                };
                self.events.push(egui::Event::MouseWheel {
                    unit: egui::MouseWheelUnit::Point,
                    delta: d,
                    modifiers: self.modifiers,
                });
                self.captured
            }
            WindowEvent::ModifiersChanged(state) => {
                let s = state.state();
                self.modifiers = egui::Modifiers {
                    alt: s.alt_key(),
                    ctrl: s.control_key(),
                    shift: s.shift_key(),
                    mac_cmd: cfg!(target_os = "macos") && s.super_key(),
                    command: if cfg!(target_os = "macos") {
                        s.super_key()
                    } else {
                        s.control_key()
                    },
                };
                false
            }
            WindowEvent::KeyboardInput { event, .. } => {
                let pressed = event.state == ElementState::Pressed;
                if let Some(key) = translate_key(&event.logical_key) {
                    self.events.push(egui::Event::Key {
                        key,
                        physical_key: None,
                        pressed,
                        repeat: event.repeat,
                        modifiers: self.modifiers,
                    });
                }
                if pressed && !self.modifiers.command && !self.modifiers.ctrl {
                    if let Some(text) = &event.text {
                        if !text.chars().any(char::is_control) {
                            self.events.push(egui::Event::Text(text.to_string()));
                        }
                    }
                }

                self.context.wants_keyboard_input()
            }
            _ => false,
        }
    }

    pub fn spawn_material(&self) -> u32 {
        self.brush.material as u32
    }

    pub fn brush_mode(&self) -> BrushMode {
        self.brush.mode
    }

    pub fn brush_heat_rate(&self) -> f32 {
        match self.brush.mode {
            BrushMode::Paint | BrushMode::Blow | BrushMode::Burst | BrushMode::Erase => 0.0,
            BrushMode::Heat => self.brush.heat_rate,
            BrushMode::Cool => -self.brush.heat_rate,
        }
    }

    pub fn brush_wind_speed(&self) -> f32 {
        self.brush.wind_speed
    }

    pub fn brush_radius(&self) -> f32 {
        self.brush.radius
    }

    pub fn times_passes(&self) -> bool {
        self.time_passes
    }

    pub fn time_scale(&self) -> f32 {
        self.time_scale
    }

    pub fn captures_pointer(&self) -> bool {
        self.captured
    }

    pub fn menu_height(&self) -> f32 {
        self.menu_height
    }

    pub fn take_fullscreen_request(&mut self) -> bool {
        std::mem::take(&mut self.fullscreen_requested)
    }

    pub fn take_reset_request(&mut self) -> bool {
        std::mem::take(&mut self.reset_requested)
    }

    pub fn run(
        &mut self,
        display: &egui_wgpu::ScreenDescriptor,
        particle_count: u32,
        fullscreen: Option<bool>,
        world: [f32; 4],
        blow_direction: [f32; 2],
    ) {
        let scale = display.pixels_per_point;
        let screen = size_in_points(display);
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(egui::pos2(0.0, 0.0), screen)),
            modifiers: self.modifiers,
            events: std::mem::take(&mut self.events),
            ..Default::default()
        };

        let stats = (
            self.stats.mean_frame_time(),
            self.stats.worst_frame_time(),
            self.stats.substeps,
            self.stats.sim_speed(),
        );

        let panel = egui::Rect::from_min_max(
            egui::pos2(0.0, (world[1] + world[3]) / scale),
            screen.to_pos2(),
        );
        self.menu_height = menu_height_for(screen.x, pixel_font::pixel_size(MENU_PIXEL, scale));
        let colours = self.materials.map(|m| m.colour);
        let materials = &mut self.materials;
        let globals = &mut self.globals;
        let selected = &mut self.selected;
        let brush = &mut self.brush;
        let time_passes = &mut self.time_passes;
        let fullscreen_requested = &mut self.fullscreen_requested;
        let show_developer = self.show_developer;
        let time_scale = &mut self.time_scale;
        let readout = self.stats.readout(particle_count);
        let changelog_open = &mut self.changelog_open;
        let reset_requested = &mut self.reset_requested;

        let output = self.context.run(raw, |ctx| {
            let state = MenuState {
                brush: &mut *brush,
                colours: &colours,
                fullscreen,
                world: &mut *globals,
                time_scale: &mut *time_scale,
                readout,
                changelog_open: &mut *changelog_open,
                fullscreen_requested: &mut *fullscreen_requested,
                reset_requested: &mut *reset_requested,
            };
            menu(ctx, panel, scale, state);
            changelog_window(ctx, changelog_open, panel.min.y);

            let (colour, arrow) = match brush.mode {
                BrushMode::Blow => (AIR_ACCENT, Some(blow_direction)),
                _ => (egui::Color32::RED, None),
            };
            brush_outline(ctx, brush.radius, world, scale, colour, arrow);
            if !show_developer {
                return;
            }
            egui::Window::new("Developer")
                .default_pos([12.0, 12.0])
                .default_width(300.0)
                .show(ctx, |ui| {
                    let (mean, worst, substeps, speed) = stats;
                    ui.heading("Performance");
                    egui::Grid::new("stats").num_columns(2).show(ui, |ui| {
                        ui.label("FPS");
                        ui.label(if mean > 0.0 {
                            format!("{:.0}", 1.0 / mean)
                        } else {
                            "—".into()
                        });
                        ui.end_row();
                        ui.label("Frame time");
                        ui.label(format!("{:.2} ms (worst {:.2})", mean * 1000.0, worst * 1000.0));
                        ui.end_row();
                        ui.label("Substeps/frame");
                        ui.label(format!("{substeps} of {}", config::MAX_SUBSTEPS));
                        ui.end_row();
                        ui.label("Sim speed");
                        ui.colored_label(
                            if speed > 0.95 {
                                egui::Color32::LIGHT_GREEN
                            } else {
                                egui::Color32::from_rgb(230, 170, 60)
                            },
                            format!("{speed:.2}x real time"),
                        );
                        ui.end_row();
                        ui.label("Particles");
                        ui.label(format!("{particle_count}"));
                        ui.end_row();
                    });
                    ui.checkbox(time_passes, "Time GPU passes");
                    ui.small(
                        "Logs each dispatch's GPU time with the [perf] line. While on, the \
                         solver runs one pass per dispatch instead of one in total.",
                    );

                    ui.separator();
                    ui.heading("World");
                    ui.add(
                        egui::Slider::new(&mut globals.gravity, 0.0..=800.0)
                            .text("gravity")
                            .suffix(" u/s²"),
                    );
                    ui.add(
                        egui::Slider::new(&mut globals.ambient_density, 0.0..=0.4)
                            .text("air density"),
                    );
                    ui.small(
                        "Buoyancy from the air the world does not simulate. Anything \
                         lighter than this floats — it is what lifts steam, since \
                         gravity alone cannot: acceleration is force over mass, so \
                         density cancels. At zero, gases sink like everything else.",
                    );
                    ui.add(
                        egui::Slider::new(&mut globals.wind, -1000.0..=1000.0)
                            .text("wind")
                            .suffix(" u/s"),
                    );
                    ui.add(
                        egui::Slider::new(&mut globals.air_drag, 0.0..=0.05)
                            .text("air drag"),
                    );
                    ui.add(
                        egui::Slider::new(
                            &mut globals.rest_temperature,
                            config::MIN_TEMPERATURE..=config::MAX_TEMPERATURE,
                        )
                        .text("resting temperature")
                        .suffix(" °"),
                    );
                    ui.small(
                        "The air's speed (positive blows right) and how hard it drags on \
                         what it can reach. Light, small grains are carried first — steam \
                         with any breeze, powder before sand, gravel last — and grains \
                         packed in a heap or a pool are sheltered by their neighbours, so \
                         only the surface is blown. Drag at zero turns off air resistance, \
                         and with it the wind and the Blow and Burst tools.",
                    );
                    ui.add(
                        egui::Slider::new(&mut globals.render_hysteresis, 0.0..=1.0)
                            .text("render hysteresis"),
                    );
                    ui.small("Hysteresis is display-only: it hides sub-pixel jitter without touching physics.");

                    ui.separator();
                    ui.heading("Material");
                    ui.horizontal(|ui| {
                        for (i, material) in MATERIALS.iter().enumerate() {
                            ui.selectable_value(selected, i, material.name);
                        }
                    });
                    ui.small(
                        "Which material the sliders below edit; what the brush paints is \
                         chosen in the menu.",
                    );

                    let m = &mut materials[*selected];
                    ui.horizontal(|ui| {


                        let mut rgb = [m.colour[0], m.colour[1], m.colour[2]];
                        if ui.color_edit_button_rgb(&mut rgb).changed() {
                            m.colour = [rgb[0], rgb[1], rgb[2], 1.0];
                        }
                        ui.label("colour");
                    });


                    ui.add(
                        egui::Slider::new(&mut m.freq_n, 10.0..=config::MAX_FREQ_N)
                            .text("normal stiffness")
                            .suffix(" Hz"),
                    );
                    ui.add(
                        egui::Slider::new(&mut m.freq_t, 2.5..=config::MAX_FREQ_T)
                            .text("static friction stiffness")
                            .suffix(" Hz"),
                    );
                    ui.add(egui::Slider::new(&mut m.zeta_n, 0.0..=1.0).text("normal damping ratio"));
                    ui.add(egui::Slider::new(&mut m.zeta_t, 0.0..=1.0).text("friction damping ratio"));
                    ui.add(egui::Slider::new(&mut m.mu, 0.0..=1.5).text("friction (μ)"));
                    ui.add(egui::Slider::new(&mut m.density, 0.1..=5.0).text("density"));
                    ui.small("Density changes how hard grains shove other materials, not stability.");
                    ui.add(
                        egui::Slider::new(
                            &mut m.radius,
                            config::RADIUS_FLOOR..=config::RADIUS_LIMIT,
                        )
                        .text("radius"),
                    );
                    ui.small("Radius applies to newly spawned grains; existing ones keep theirs.");
                    ui.small(
                        "The floor is a memory budget: the particle buffer holds one \
                         screenful of the smallest grain allowed here.",
                    );

                    ui.separator();
                    ui.heading("Fluid");
                    ui.small(
                        "Pressure stiffness at zero makes this a grain: every fluid term \
                         multiplies out and only the contact springs above apply.",
                    );
                    ui.add(
                        egui::Slider::new(&mut m.pressure_k, 0.0..=1_500_000.0)
                            .text("pressure stiffness"),
                    );
                    ui.add(
                        egui::Slider::new(&mut m.rest_packing, 0.0..=0.5).text("rest packing"),
                    );
                    ui.small(
                        "Rest packing must match what the arrangement actually measures at \
                         this radius: too low and the pool collapses, too high and it boils.",
                    );
                    ui.add(egui::Slider::new(&mut m.viscosity, 0.0..=20.0).text("viscosity"));

                    ui.separator();
                    ui.heading("Stability");
                    stability_readout(ui, m);
                });
        });

        self.captured = self.context.wants_pointer_input();
        self.textures_delta.append(output.textures_delta);
        self.paint_jobs = self
            .context
            .tessellate(output.shapes, output.pixels_per_point);
    }

    pub fn render(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        view: &wgpu::TextureView,
        display: &egui_wgpu::ScreenDescriptor,
    ) {
        if self.paint_jobs.is_empty() {
            return;
        }
        let delta = std::mem::take(&mut self.textures_delta);
        for (id, image) in &delta.set {
            self.renderer.update_texture(device, queue, *id, image);
        }
        self.renderer
            .update_buffers(device, queue, encoder, &self.paint_jobs, display);

        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Overlay pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            })
            .forget_lifetime();
        self.renderer.render(&mut pass, &self.paint_jobs, display);
        drop(pass);

        for id in &delta.free {
            self.renderer.free_texture(id);
        }
    }
}

const MENU_PIXEL: f32 = 1.75;
const MENU_PAD: i32 = 6;
const ROW_HEIGHT: i32 = 18;
const ROW_GAP: i32 = 4;
const ITEM_GAP: i32 = 10;
const GROUP_GAP: i32 = 18;

const TEXT_TOP: i32 = (ROW_HEIGHT - pixel_font::HEIGHT) / 2;
const UNDERLINE_TOP: i32 = TEXT_TOP + pixel_font::HEIGHT + 2;

const NAME_GAP: i32 = 4;

const CHIP: i32 = pixel_font::HEIGHT;
const SLIDER_WIDTH: i32 = 54;
const SNAP_PIXELS: i32 = 2;
const HANDLE_WIDTH: i32 = 3;

const fn hex(rgb: u32) -> egui::Color32 {
    egui::Color32::from_rgb((rgb >> 16) as u8, (rgb >> 8) as u8, rgb as u8)
}
const MENU_FILL: egui::Color32 = hex(0x1d2021);
const MENU_EDGE: egui::Color32 = hex(0x3c3836);
const RAIL: egui::Color32 = hex(0x504945);
const TEXT_DIM: egui::Color32 = hex(0x665c54);
const TEXT: egui::Color32 = hex(0xa89984);
const TEXT_HOVER: egui::Color32 = hex(0xebdbb2);
const TEXT_CHOSEN: egui::Color32 = hex(0xfbf1c7);

const PAINT_ACCENT: egui::Color32 = hex(0xfabd2f);
const HEAT_ACCENT: egui::Color32 = hex(0xfe8019);
const COOL_ACCENT: egui::Color32 = hex(0x83a598);
const AIR_ACCENT: egui::Color32 = hex(0x8ec07c);
const ERASE_ACCENT: egui::Color32 = hex(0xfb4934);

struct MenuSlider {
    label: &'static str,
    low: f32,
    high: f32,
    show: fn(f32) -> String,
    tip: &'static str,
}

impl MenuSlider {
    fn width(&self) -> i32 {
        pixel_font::width(self.label) + NAME_GAP + SLIDER_WIDTH + NAME_GAP + self.value_width()
    }

    fn value_width(&self) -> i32 {
        let widest = |value| pixel_font::width(&(self.show)(value));
        widest(self.low).max(widest(self.high))
    }

    fn snapped(&self, value: f32, default: f32) -> f32 {
        if (self.offset_of(value) - self.offset_of(default)).abs() <= SNAP_PIXELS {
            default
        } else {
            value
        }
    }

    fn offset_of(&self, value: f32) -> i32 {
        let travel = SLIDER_WIDTH - HANDLE_WIDTH;
        (((value - self.low) / (self.high - self.low)).clamp(0.0, 1.0) * travel as f32).round()
            as i32
    }
}

fn without_negative_zero(value: f32) -> f32 {
    if value == 0.0 {
        0.0
    } else {
        value
    }
}

fn whole_number(value: f32) -> String {
    format!("{:.0}", without_negative_zero(value.round()))
}

fn times(value: f32) -> String {
    format!("{value:.2}X")
}

const BRUSH_SIZE: MenuSlider = MenuSlider {
    label: "Size",
    low: 0.0,
    high: config::BRUSH_RADIUS_LIMIT,
    show: whole_number,
    tip: "How far the brush reaches: the red circle around the cursor.",
};

const BRUSH_RATE: MenuSlider = MenuSlider {
    label: "Rate",
    low: 60.0,
    high: 3600.0,
    show: whole_number,
    tip: "How fast Heat and Cool change the temperature, in degrees per second.",
};

const BRUSH_FORCE: MenuSlider = MenuSlider {
    label: "Force",
    low: 100.0,
    high: 3000.0,
    show: whole_number,
    tip: "How hard Blow and Burst push: the speed of the air they make. Steam goes \
          with the gentlest breeze, sand with about half, and gravel wants most of it.",
};

const BRUSH_SLIDERS: [MenuSlider; 3] = [BRUSH_SIZE, BRUSH_RATE, BRUSH_FORCE];

const TIME_ACCENT: egui::Color32 = hex(0xd3869b);

const SIM_SPEED: MenuSlider = MenuSlider {
    label: "Speed",
    low: 0.0,
    high: 2.0,
    show: times,
    tip: "How fast time runs: 1 is real time and 0 pauses. Above 1 a slow computer can \
          fall behind, and the readout beside it then shows the speed actually reached.",
};

const REAL_TIME: f32 = 1.0;
const KEEPING_UP: f32 = 0.95;

#[derive(Copy, Clone)]
struct Readout {
    fps: f32,
    particles: u32,
    lagging_speed: Option<f32>,
}

impl Readout {
    fn text(&self) -> String {
        let counts = format!("{:.0} FPS  {} PARTICLES", self.fps, self.particles);
        match self.lagging_speed {
            Some(speed) => format!("{counts}  RUNNING {speed:.2}X"),
            None => counts,
        }
    }
}

const WORLD_SLIDERS: [(MenuSlider, egui::Color32); 3] = [
    (
        MenuSlider {
            label: "Gravity",
            low: 0.0,
            high: 450.0,
            show: whole_number,
            tip: "How hard everything falls. At zero, grains drift where they are left.",
        },
        TEXT_HOVER,
    ),
    (
        MenuSlider {
            label: "Temp",
            low: -40.0,
            high: 300.0,
            show: whole_number,
            tip: "The temperature everything settles to: the walls hold it and the air \
                  slowly brings every exposed grain to it. Water freezes below 0 and boils \
                  above 100, and plants catch fire at 250.",
        },
        HEAT_ACCENT,
    ),
    (
        MenuSlider {
            label: "Wind",
            low: -1000.0,
            high: 1000.0,
            show: whole_number,
            tip: "A breeze across the whole world, blowing whichever way the slider leans \
                  from centre. Light grains go first, and only surfaces feel it.",
        },
        AIR_ACCENT,
    ),
];

const TOOLS: [(BrushMode, &str, egui::Color32, &str); 6] = [
    (
        BrushMode::Paint,
        "Paint",
        PAINT_ACCENT,
        "Hold the left mouse button in the world to place grains of the chosen material.",
    ),
    (
        BrushMode::Heat,
        "Heat",
        HEAT_ACCENT,
        "Warm what is already there. Grains that cross a threshold change phase on \
         their own: water boils to steam, rock melts.",
    ),
    (
        BrushMode::Cool,
        "Cool",
        COOL_ACCENT,
        "Chill what is already there: water freezes to ice, lava sets to obsidian.",
    ),
    (
        BrushMode::Blow,
        "Blow",
        AIR_ACCENT,
        "Blow air the way the cursor's arrow points; drag to turn it. Light grains \
         go first — steam, then powder, then sand, gravel last — and only the \
         surface of a heap or a pool feels it.",
    ),
    (
        BrushMode::Burst,
        "Burst",
        AIR_ACCENT,
        "Air rushes out of the brush in every direction for as long as you hold it: \
         a click is a puff, holding digs a crater.",
    ),
    (
        BrushMode::Erase,
        "Erase",
        ERASE_ACCENT,
        "Hold the left mouse button in the world to remove whatever the brush touches.",
    ),
];

#[derive(Copy, Clone)]
enum Piece {
    Tools,

    Swatch(usize),

    Controls,

    World,
}

impl Piece {
    fn width(self) -> i32 {
        match self {
            Piece::Tools => {
                TOOLS
                    .iter()
                    .map(|tool| pixel_font::width(tool.1))
                    .sum::<i32>()
                    + (TOOLS.len() as i32 - 1) * ITEM_GAP
            }
            Piece::Swatch(id) => CHIP + NAME_GAP + pixel_font::width(MATERIALS[id].name),
            Piece::Controls => {
                slider_label_width() + NAME_GAP + SLIDER_WIDTH + NAME_GAP + slider_value_width()
            }
            Piece::World => {
                WORLD_SLIDERS
                    .iter()
                    .map(|(slider, _)| slider.width())
                    .sum::<i32>()
                    + (WORLD_SLIDERS.len() as i32 - 1) * ITEM_GAP
            }
        }
    }

    fn gap_after(self, previous: Piece) -> i32 {
        let group = |piece| match piece {
            Piece::Tools => 0,
            Piece::Swatch(_) => 1,
            Piece::Controls => 2,
            Piece::World => 3,
        };
        if group(self) == group(previous) {
            ITEM_GAP
        } else {
            GROUP_GAP
        }
    }
}

fn menu_rows(width: i32) -> Vec<Vec<Piece>> {
    let pieces = std::iter::once(Piece::Tools)
        .chain(
            (0..MATERIALS.len())
                .filter(|&id| MATERIALS[id].palette)
                .map(Piece::Swatch),
        )
        .chain([Piece::Controls, Piece::World]);
    let mut rows: Vec<Vec<Piece>> = Vec::new();
    for piece in pieces {
        let limit = width - reserved_for_fullscreen(rows.len().saturating_sub(1));
        match rows.last_mut() {
            Some(row) if fits_in(row, piece, limit) => row.push(piece),
            _ => rows.push(vec![piece]),
        }
    }
    leave_room_for_last_row_icons(&mut rows, width);
    rows
}

fn leave_room_for_last_row_icons(rows: &mut Vec<Vec<Piece>>, width: i32) {
    let last = rows.len() - 1;
    let limit = width - reserved_for_corners(last, rows.len());
    if row_width(&rows[last]) > limit && rows[last].len() > 1 {
        let piece = rows[last].pop().expect("the last row has a piece to spare");
        rows.push(vec![piece]);
    }
}

fn fits_in(row: &[Piece], piece: Piece, limit: i32) -> bool {
    let previous = row[row.len() - 1];
    row_width(row) + piece.gap_after(previous) + piece.width() <= limit
}

fn reserved_for_fullscreen(row_index: usize) -> i32 {
    if row_index == 0 {
        FULLSCREEN_RESERVE
    } else {
        0
    }
}

fn corner_reserves(row_index: usize, row_count: usize) -> [i32; 2] {
    if row_index + 1 < row_count {
        return [0, reserved_for_fullscreen(row_index)];
    }
    [
        RESET_RESERVE,
        reserved_for_fullscreen(row_index) + INFO_RESERVE,
    ]
}

fn reserved_for_corners(row_index: usize, row_count: usize) -> i32 {
    let [left, right] = corner_reserves(row_index, row_count);
    left + right
}

fn row_width(row: &[Piece]) -> i32 {
    row.iter().map(|piece| piece.width()).sum::<i32>()
        + row
            .windows(2)
            .map(|pair| pair[1].gap_after(pair[0]))
            .sum::<i32>()
}

fn slider_value_width() -> i32 {
    BRUSH_SLIDERS
        .iter()
        .map(MenuSlider::value_width)
        .max()
        .unwrap_or(0)
}

fn slider_label_width() -> i32 {
    BRUSH_SLIDERS
        .iter()
        .map(|slider| pixel_font::width(slider.label))
        .max()
        .unwrap_or(0)
}

const MENU_FULL_SIZE_WIDTH: f32 = 832.0;

const MENU_MIN_ZOOM: f32 = 0.75;

pub fn zoom_for(width: f32) -> f32 {
    (width / MENU_FULL_SIZE_WIDTH).clamp(MENU_MIN_ZOOM, 1.0)
}

fn size_in_points(display: &egui_wgpu::ScreenDescriptor) -> egui::Vec2 {
    let [width, height] = display.size_in_pixels;
    egui::vec2(width as f32, height as f32) / display.pixels_per_point
}

fn menu_height_for(width: f32, pixel: f32) -> f32 {
    let rows = menu_rows((width / pixel).floor() as i32 - 2 * MENU_PAD).len() as i32;
    (2 * MENU_PAD + rows * ROW_HEIGHT + (rows - 1) * ROW_GAP) as f32 * pixel
}

struct MenuState<'a> {
    brush: &'a mut Brush,
    colours: &'a [[f32; 4]],
    fullscreen: Option<bool>,
    world: &'a mut Globals,
    time_scale: &'a mut f32,
    readout: Readout,
    changelog_open: &'a mut bool,
    fullscreen_requested: &'a mut bool,
    reset_requested: &'a mut bool,
}

fn menu(ctx: &egui::Context, panel: egui::Rect, scale: f32, mut state: MenuState<'_>) {
    egui::Area::new(egui::Id::new("menu"))
        .fixed_pos(panel.min)
        .constrain(false)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| menu_contents(ui, panel, scale, &mut state));
}

fn menu_contents(ui: &mut egui::Ui, panel: egui::Rect, scale: f32, state: &mut MenuState<'_>) {
    ui.allocate_rect(panel, egui::Sense::hover());
    let grid = pixel_font::Grid::new(panel.min, MENU_PIXEL, scale);
    let [columns, height] = grid.cell(panel.max);
    ui.painter().rect_filled(panel, 0.0, MENU_FILL);
    grid.fill(ui.painter(), 0, 0, columns + 1, 1, MENU_EDGE);

    let rows = menu_rows(columns - 2 * MENU_PAD);
    let rows_height = rows.len() as i32 * (ROW_HEIGHT + ROW_GAP) - ROW_GAP;
    let first_top = ((height - rows_height) / 2).max(MENU_PAD);
    for (i, row) in rows.iter().enumerate() {
        let [left, right] = corner_reserves(i, rows.len());
        let room = columns - left - right;
        let top = first_top + i as i32 * (ROW_HEIGHT + ROW_GAP);
        menu_row(
            ui,
            grid,
            row,
            [left + (room - row_width(row)) / 2, top],
            state,
        );
    }
    let last_top = first_top + (rows.len() as i32 - 1) * (ROW_HEIGHT + ROW_GAP);
    let corners = Corners {
        right: columns - MENU_PAD - FULLSCREEN.width,
        first_top,
        last_top,
        single_row: rows.len() == 1,
    };
    corner_buttons(ui, grid, corners, state);
}

struct Corners {
    right: i32,
    first_top: i32,
    last_top: i32,
    single_row: bool,
}

fn corner_buttons(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    corners: Corners,
    state: &mut MenuState<'_>,
) {
    let beside_fullscreen = if corners.single_row {
        FULLSCREEN_RESERVE
    } else {
        0
    };
    reset_button(
        ui,
        grid,
        [MENU_PAD, corners.last_top],
        state.reset_requested,
    );
    let info_at = [corners.right - beside_fullscreen, corners.last_top];
    info_button(ui, grid, info_at, state.changelog_open);
    let Some(fullscreen) = state.fullscreen else {
        return;
    };
    if fullscreen_toggle(ui, grid, [corners.right, corners.first_top], fullscreen) {
        *state.fullscreen_requested = true;
    }
}

fn menu_row(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    row: &[Piece],
    at: [i32; 2],
    state: &mut MenuState<'_>,
) {
    let [mut left, top] = at;
    for (i, &piece) in row.iter().enumerate() {
        if i > 0 {
            left += piece.gap_after(row[i - 1]);
        }
        menu_piece(ui, grid, piece, [left, top], state);
        left += piece.width();
    }
}

fn menu_piece(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    piece: Piece,
    at: [i32; 2],
    state: &mut MenuState<'_>,
) {
    let [x, y] = at;
    match piece {
        Piece::Tools => tools(ui, grid, x, y, state.brush),
        Piece::Swatch(id) => swatch(ui, grid, x, y, state.brush, id, state.colours[id]),
        Piece::Controls => controls(ui, grid, x, y, state.brush),
        Piece::World => {
            world_sliders(ui, grid, [x, y], state.world);
            let second = y + ROW_HEIGHT - pixel_font::HEIGHT;
            clock(ui, grid, [x, second], state.time_scale, &state.readout);
        }
    }
}

fn menu_item(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    x: i32,
    y: i32,
    width: i32,
    label: &str,
    selected: bool,
) -> egui::Response {
    let rect = grid.rect(x - ITEM_GAP / 2, y, width + ITEM_GAP, ROW_HEIGHT);
    let response = ui.allocate_rect(rect, egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::selected(
            egui::WidgetType::SelectableLabel,
            ui.is_enabled(),
            selected,
            label,
        )
    });
    response
}

fn label_colour(response: &egui::Response, chosen: Option<egui::Color32>) -> egui::Color32 {
    match chosen {
        Some(colour) => colour,
        None if response.hovered() => TEXT_HOVER,
        None => TEXT,
    }
}

fn accent(mode: BrushMode) -> egui::Color32 {
    TOOLS
        .iter()
        .find(|tool| tool.0 == mode)
        .map_or(TEXT_CHOSEN, |tool| tool.2)
}

fn tools(ui: &mut egui::Ui, grid: pixel_font::Grid, x: i32, y: i32, brush: &mut Brush) {
    let mut left = x;
    for (mode, label, accent, tip) in TOOLS {
        let width = pixel_font::width(label);
        let chosen = brush.mode == mode;
        let response = menu_item(ui, grid, left, y, width, label, chosen);
        let painter = ui.painter();
        grid.text(
            painter,
            left,
            y + TEXT_TOP,
            label,
            label_colour(&response, chosen.then_some(accent)),
        );
        if chosen {
            grid.fill(painter, left, y + UNDERLINE_TOP, width, 1, accent);
        }
        if response.on_hover_text(tip).clicked() {
            brush.mode = mode;
        }
        left += width + ITEM_GAP;
    }
}

fn swatch(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    x: i32,
    y: i32,
    brush: &mut Brush,
    id: usize,
    colour: [f32; 4],
) {
    let [r, g, b, _] = colour;
    let channel = |value: f32| (value * 255.0).round() as u8;
    let fill = egui::Color32::from_rgb(channel(r), channel(g), channel(b));
    let material = &MATERIALS[id];
    let width = Piece::Swatch(id).width();
    let chosen = brush.mode == BrushMode::Paint && brush.material == id;
    let mut response = menu_item(ui, grid, x, y, width, material.name, chosen);
    let painter = ui.painter();
    grid.fill(painter, x, y + TEXT_TOP, CHIP, CHIP, fill);
    let text = label_colour(&response, chosen.then_some(TEXT_CHOSEN));
    grid.text(
        painter,
        x + CHIP + NAME_GAP,
        y + TEXT_TOP,
        material.name,
        text,
    );
    if chosen {
        grid.fill(painter, x, y + UNDERLINE_TOP, width, 1, TEXT_CHOSEN);
    }
    if material.is_static() {
        response = response.on_hover_text("Never moves: painting it lays down terrain.");
    }
    if material.params.sprouts != NO_TRANSITION {
        response = response
            .on_hover_text("Sprouts once it lands and grows into a swaying plant that burns.");
    }
    if response.clicked() {
        brush.material = id;
        brush.mode = BrushMode::Paint;
    }
}

fn controls(ui: &mut egui::Ui, grid: pixel_font::Grid, x: i32, y: i32, brush: &mut Brush) {
    let accent = accent(brush.mode);
    let defaults = Brush::default();
    let label_width = slider_label_width();
    let place = |y| SliderPlace { x, y, label_width };
    let size = Setting {
        value: &mut brush.radius,
        default: defaults.radius,
    };
    menu_slider(ui, grid, place(y), &BRUSH_SIZE, size, Some(accent));

    let enabled = !matches!(brush.mode, BrushMode::Paint | BrushMode::Erase);
    let (slider, value, default) = match brush.mode {
        BrushMode::Blow | BrushMode::Burst => {
            (&BRUSH_FORCE, &mut brush.wind_speed, defaults.wind_speed)
        }
        BrushMode::Paint | BrushMode::Erase | BrushMode::Heat | BrushMode::Cool => {
            (&BRUSH_RATE, &mut brush.heat_rate, defaults.heat_rate)
        }
    };
    let second = place(y + ROW_HEIGHT - pixel_font::HEIGHT);
    let setting = Setting { value, default };
    menu_slider(ui, grid, second, slider, setting, enabled.then_some(accent));
}

fn world_sliders(ui: &mut egui::Ui, grid: pixel_font::Grid, at: [i32; 2], world: &mut Globals) {
    let [x, y] = at;
    let defaults = Globals::default();
    let settings = [
        Setting {
            value: &mut world.gravity,
            default: defaults.gravity,
        },
        Setting {
            value: &mut world.rest_temperature,
            default: defaults.rest_temperature,
        },
        Setting {
            value: &mut world.wind,
            default: defaults.wind,
        },
    ];
    let mut left = x;
    for ((slider, accent), setting) in WORLD_SLIDERS.iter().zip(settings) {
        let label_width = pixel_font::width(slider.label);
        let place = SliderPlace {
            x: left,
            y,
            label_width,
        };
        menu_slider(ui, grid, place, slider, setting, Some(*accent));
        left += slider.width() + ITEM_GAP;
    }
}

fn clock(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    at: [i32; 2],
    time_scale: &mut f32,
    readout: &Readout,
) {
    let [x, y] = at;
    let place = SliderPlace {
        x,
        y,
        label_width: pixel_font::width(WORLD_SLIDERS[0].0.label),
    };
    let speed = Setting {
        value: time_scale,
        default: REAL_TIME,
    };
    menu_slider(ui, grid, place, &SIM_SPEED, speed, Some(TIME_ACCENT));
    grid.text(ui.painter(), x + readout_offset(), y, &readout.text(), TEXT);
}

fn readout_offset() -> i32 {
    pixel_font::width(WORLD_SLIDERS[0].0.label)
        + NAME_GAP
        + SLIDER_WIDTH
        + NAME_GAP
        + SIM_SPEED.value_width()
        + ITEM_GAP
}

#[derive(Copy, Clone)]
struct SliderPlace {
    x: i32,
    y: i32,
    label_width: i32,
}

impl SliderPlace {
    fn rail(self) -> i32 {
        self.x + self.label_width + NAME_GAP
    }

    fn value_x(self) -> i32 {
        self.rail() + SLIDER_WIDTH + NAME_GAP
    }
}

struct Setting<'a> {
    value: &'a mut f32,
    default: f32,
}

struct SliderLook {
    text: egui::Color32,
    fill: egui::Color32,
    knob: egui::Color32,
}

fn menu_slider(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    place: SliderPlace,
    slider: &MenuSlider,
    mut setting: Setting<'_>,
    accent: Option<egui::Color32>,
) {
    let sense = if accent.is_some() {
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::hover()
    };
    let response = ui.allocate_rect(slider_hit_area(grid, place), sense);
    if accent.is_some() {
        drag_slider(&response, grid, place, slider, &mut setting);
        nudge_slider(ui, &response, slider, setting.value);
        reset_on_double_click(&response, &mut setting);
    }
    let value = *setting.value;
    response.widget_info(|| egui::WidgetInfo::slider(accent.is_some(), value as f64, slider.label));
    let look = slider_look(accent, &response);
    paint_slider(
        ui.painter(),
        grid,
        place,
        slider,
        [value, setting.default],
        look,
    );
    response.on_hover_ui(|ui| {
        ui.label(slider.tip);
        ui.label("Double-click to reset.");
    });
}

fn slider_hit_area(grid: pixel_font::Grid, place: SliderPlace) -> egui::Rect {
    grid.rect(
        place.rail() - NAME_GAP / 2,
        place.y - 1,
        SLIDER_WIDTH + NAME_GAP,
        pixel_font::HEIGHT + 2,
    )
}

fn reset_on_double_click(response: &egui::Response, setting: &mut Setting<'_>) {
    if response.double_clicked() {
        *setting.value = setting.default;
    }
}

fn drag_slider(
    response: &egui::Response,
    grid: pixel_font::Grid,
    place: SliderPlace,
    slider: &MenuSlider,
    setting: &mut Setting<'_>,
) {
    let Some(pointer) = response.interact_pointer_pos() else {
        return;
    };
    let travel = SLIDER_WIDTH - HANDLE_WIDTH;
    let start = grid.rect(place.rail(), place.y, HANDLE_WIDTH, 1).center().x;
    let along = (pointer.x - start) / (travel as f32 * grid.rect(0, 0, 1, 1).width());
    let dragged = slider.low + (slider.high - slider.low) * along.clamp(0.0, 1.0);
    *setting.value = slider.snapped(dragged, setting.default);
}

fn nudge_slider(ui: &egui::Ui, response: &egui::Response, slider: &MenuSlider, value: &mut f32) {
    if !response.has_focus() {
        return;
    }
    let step = (slider.high - slider.low) / (SLIDER_WIDTH - HANDLE_WIDTH) as f32;
    let (right, left) = ui.input(|input| {
        (
            input.key_pressed(egui::Key::ArrowRight),
            input.key_pressed(egui::Key::ArrowLeft),
        )
    });
    if right {
        *value = (*value + step).min(slider.high);
    }
    if left {
        *value = (*value - step).max(slider.low);
    }
}

fn slider_look(accent: Option<egui::Color32>, response: &egui::Response) -> SliderLook {
    let Some(accent) = accent else {
        return SliderLook {
            text: TEXT_DIM,
            fill: TEXT_DIM,
            knob: TEXT_DIM,
        };
    };
    let active = response.hovered() || response.dragged();
    SliderLook {
        text: TEXT,
        fill: accent,
        knob: if active { TEXT_CHOSEN } else { TEXT_HOVER },
    }
}

fn paint_slider(
    painter: &egui::Painter,
    grid: pixel_font::Grid,
    place: SliderPlace,
    slider: &MenuSlider,
    [value, default]: [f32; 2],
    look: SliderLook,
) {
    let rail = place.rail();
    let handle = slider.offset_of(value);
    let zero = slider.offset_of(0.0);
    grid.text(painter, place.x, place.y, slider.label, look.text);
    grid.fill(painter, rail, place.y + 2, SLIDER_WIDTH, 3, RAIL);
    let filled = (handle - zero).abs();
    grid.fill(
        painter,
        rail + zero.min(handle),
        place.y + 2,
        filled,
        3,
        look.fill,
    );
    paint_default_mark(painter, grid, place, slider.offset_of(default));
    grid.fill(
        painter,
        rail + handle,
        place.y,
        HANDLE_WIDTH,
        pixel_font::HEIGHT,
        look.knob,
    );
    grid.text(
        painter,
        place.value_x(),
        place.y,
        &(slider.show)(value),
        look.text,
    );
}

fn paint_default_mark(
    painter: &egui::Painter,
    grid: pixel_font::Grid,
    place: SliderPlace,
    offset: i32,
) {
    let x = place.rail() + offset + HANDLE_WIDTH / 2;
    grid.fill(painter, x, place.y, 1, 1, TEXT_DIM);
    grid.fill(painter, x, place.y + pixel_font::HEIGHT - 1, 1, 1, TEXT_DIM);
}

fn brush_outline(
    ctx: &egui::Context,
    radius: f32,
    world: [f32; 4],
    pixels_per_point: f32,
    colour: egui::Color32,
    arrow: Option<[f32; 2]>,
) {
    let Some(pointer) = ctx.pointer_hover_pos() else {
        return;
    };
    if ctx.is_pointer_over_area() {
        return;
    }
    let [left, top, width, _] = world;
    let pixels = [config::PIXEL_WIDTH as i32, config::PIXEL_HEIGHT as i32];

    let scale = width / pixels[0] as f32;
    let centre = [
        (pointer.x * pixels_per_point - left) / scale,
        (pointer.y * pixels_per_point - top) / scale,
    ];
    if !(0.0..pixels[0] as f32).contains(&centre[0])
        || !(0.0..pixels[1] as f32).contains(&centre[1])
    {
        return;
    }
    let cursor = [centre[0].floor() as i32, centre[1].floor() as i32];
    let reach = radius * config::PIXEL_WIDTH as f32 / config::WORLD_WIDTH;
    let inside = |x: i32, y: i32| {
        (0..pixels[0]).contains(&x)
            && (0..pixels[1]).contains(&y)
            && ([x, y] == cursor
                || (x as f32 + 0.5 - centre[0]).hypot(y as f32 + 0.5 - centre[1]) <= reach)
    };

    let edge = |g: i32, origin: f32| (origin + g as f32 * scale - 0.5).ceil() / pixels_per_point;

    let painter = ctx.layer_painter(egui::LayerId::background());
    let block = |x: i32, y: i32| {
        painter.rect_filled(
            egui::Rect::from_min_max(
                egui::pos2(edge(x, left), edge(y, top)),
                egui::pos2(edge(x + 1, left), edge(y + 1, top)),
            ),
            0.0,
            colour,
        );
    };
    let span = reach.ceil() as i32 + 1;
    for y in cursor[1] - span..=cursor[1] + span {
        for x in cursor[0] - span..=cursor[0] + span {
            let enclosed =
                inside(x - 1, y) && inside(x + 1, y) && inside(x, y - 1) && inside(x, y + 1);
            if inside(x, y) && !enclosed {
                block(x, y);
            }
        }
    }

    let Some([dx, dy]) = arrow else {
        return;
    };

    let tip = [
        centre[0] + dx * (reach + 4.0),
        centre[1] + dy * (reach + 4.0),
    ];
    let (sin, cos) = 0.7f32.sin_cos();
    let barbs = [
        [-dx * cos + dy * sin, -dx * sin - dy * cos],
        [-dx * cos - dy * sin, dx * sin - dy * cos],
    ];
    let mut lit = Vec::new();
    let mut trace = |from: [f32; 2], to: [f32; 2]| {
        let steps = ((to[0] - from[0]).hypot(to[1] - from[1]) * 2.0).ceil() as i32;
        for i in 0..=steps {
            let t = i as f32 / steps.max(1) as f32;
            let pixel = [
                (from[0] + (to[0] - from[0]) * t).floor() as i32,
                (from[1] + (to[1] - from[1]) * t).floor() as i32,
            ];
            if !lit.contains(&pixel) {
                lit.push(pixel);
            }
        }
    };
    trace(centre, tip);
    for [bx, by] in barbs {
        trace(tip, [tip[0] + bx * 3.0, tip[1] + by * 3.0]);
    }
    for [x, y] in lit {
        if (0..pixels[0]).contains(&x) && (0..pixels[1]).contains(&y) {
            block(x, y);
        }
    }
}

const FULLSCREEN_RESERVE: i32 = GROUP_GAP + FULLSCREEN.width;
const INFO_RESERVE: i32 = GROUP_GAP + INFO.width;
const RESET_RESERVE: i32 = GROUP_GAP + RESET.width;

const RESET: pixel_font::Glyph = pixel_font::Glyph {
    width: 7,
    rows: [
        0b0011101, 0b0100011, 0b1000111, 0b1000000, 0b1000001, 0b0100010, 0b0011100,
    ],
};

const INFO: pixel_font::Glyph = pixel_font::Glyph {
    width: 7,
    rows: [
        0b0011100, 0b0100010, 0b1001001, 0b1000001, 0b1001001, 0b0101010, 0b0011100,
    ],
};

const FULLSCREEN: pixel_font::Glyph = pixel_font::Glyph {
    width: 7,
    rows: [0b1100011, 0b1000001, 0, 0, 0, 0b1000001, 0b1100011],
};
const EXIT_FULLSCREEN: pixel_font::Glyph = pixel_font::Glyph {
    width: 7,
    rows: [0b0100010, 0b1100011, 0, 0, 0, 0b1100011, 0b0100010],
};

fn fullscreen_toggle(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    at: [i32; 2],
    fullscreen: bool,
) -> bool {
    let (label, icon) = if fullscreen {
        ("Exit fullscreen (Esc)", &EXIT_FULLSCREEN)
    } else {
        ("Fullscreen", &FULLSCREEN)
    };
    icon_button(ui, grid, at, icon, label, false)
}

fn reset_button(ui: &mut egui::Ui, grid: pixel_font::Grid, at: [i32; 2], requested: &mut bool) {
    if icon_button(ui, grid, at, &RESET, "Reset the world", false) {
        *requested = true;
    }
}

fn info_button(ui: &mut egui::Ui, grid: pixel_font::Grid, at: [i32; 2], open: &mut bool) {
    if icon_button(ui, grid, at, &INFO, "Changelog", *open) {
        *open = !*open;
    }
}

fn icon_button(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    at: [i32; 2],
    icon: &pixel_font::Glyph,
    label: &str,
    chosen: bool,
) -> bool {
    let [x, y] = at;
    let rect = grid.rect(x - ITEM_GAP / 2, y, icon.width + ITEM_GAP, ROW_HEIGHT);
    let response = ui.allocate_rect(rect, egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
    });
    let colour = label_colour(&response, chosen.then_some(TEXT_CHOSEN));
    grid.bitmap(ui.painter(), x, y + TEXT_TOP, icon, colour);
    response.on_hover_text(label).clicked()
}

const CHANGELOG: &str = include_str!("../CHANGELOG.md");
const CHANGELOG_MARGIN: f32 = 8.0;

fn changelog_window(ctx: &egui::Context, open: &mut bool, menu_top: f32) {
    let above_menu = ctx.screen_rect().bottom() - menu_top + CHANGELOG_MARGIN;
    let frame = egui::Frame::window(&ctx.style())
        .fill(MENU_FILL)
        .stroke(egui::Stroke::new(1.0, MENU_EDGE));
    egui::Window::new("Changelog")
        .open(open)
        .anchor(egui::Align2::RIGHT_BOTTOM, [-CHANGELOG_MARGIN, -above_menu])
        .default_width(300.0)
        .collapsible(false)
        .resizable(false)
        .frame(frame)
        .show(ctx, |ui| {
            egui::ScrollArea::vertical()
                .max_height(260.0)
                .show(ui, show_changelog);
        });
}

fn show_changelog(ui: &mut egui::Ui) {
    for line in CHANGELOG.lines() {
        if let Some(date) = line.strip_prefix("## ") {
            ui.add_space(6.0);
            ui.label(egui::RichText::new(date).strong());
        } else if let Some(change) = line.strip_prefix("- ") {
            ui.label(format!("• {change}"));
        }
    }
}

fn stability_readout(ui: &mut egui::Ui, m: &MaterialParams) {
    let normal = m.freq_n / config::MAX_FREQ_N;
    let tangential = m.freq_t / config::MAX_FREQ_T;
    let worst = normal.max(tangential);

    egui::Grid::new("stability").num_columns(2).show(ui, |ui| {
        ui.label("normal stiffness");
        ui.label(format!("{:.0}% of tested ceiling", normal * 100.0));
        ui.end_row();
        ui.label("friction stiffness");
        ui.label(format!("{:.0}% of tested ceiling", tangential * 100.0));
        ui.end_row();
    });
    if worst > 0.95 {
        ui.colored_label(
            egui::Color32::from_rgb(230, 170, 60),
            "At the ceiling: stable, but slower to settle and the odd grain may fly.",
        );
    } else {
        ui.colored_label(egui::Color32::LIGHT_GREEN, "Stable.");
    }
    ui.small("Any combination of these sliders is stable, at any density.");

    if m.pressure_k > 0.0 {
        let sound_speed = (m.pressure_k * m.rest_packing / m.density.max(1e-6)).sqrt();
        let dt_max = 0.25 * config::SMOOTHING_RADIUS / sound_speed;
        let margin = dt_max / config::SUBSTEP;
        ui.separator();
        egui::Grid::new("fluid stability")
            .num_columns(2)
            .show(ui, |ui| {
                ui.label("speed of sound");
                ui.label(format!("{sound_speed:.0} u/s"));
                ui.end_row();
                ui.label("CFL margin");
                ui.label(format!("{margin:.2}x the substep"));
                ui.end_row();
            });
        if margin < 1.0 {
            ui.colored_label(
                egui::Color32::from_rgb(230, 120, 90),
                "Pressure waves outrun the substep: the fluid will ring itself apart.",
            );
        } else if margin < 1.5 {
            ui.colored_label(
                egui::Color32::from_rgb(230, 170, 60),
                "Thin margin — lower the pressure stiffness or expect jitter.",
            );
        } else {
            ui.colored_label(egui::Color32::LIGHT_GREEN, "Fluid stable.");
        }
    }
}

fn translate_key(key: &Key) -> Option<egui::Key> {
    Some(match key {
        Key::Named(NamedKey::ArrowDown) => egui::Key::ArrowDown,
        Key::Named(NamedKey::ArrowLeft) => egui::Key::ArrowLeft,
        Key::Named(NamedKey::ArrowRight) => egui::Key::ArrowRight,
        Key::Named(NamedKey::ArrowUp) => egui::Key::ArrowUp,
        Key::Named(NamedKey::Backspace) => egui::Key::Backspace,
        Key::Named(NamedKey::Delete) => egui::Key::Delete,
        Key::Named(NamedKey::Enter) => egui::Key::Enter,
        Key::Named(NamedKey::Escape) => egui::Key::Escape,
        Key::Named(NamedKey::Home) => egui::Key::Home,
        Key::Named(NamedKey::End) => egui::Key::End,
        Key::Named(NamedKey::Tab) => egui::Key::Tab,
        Key::Named(NamedKey::Space) => egui::Key::Space,
        Key::Character(c) => {
            let c = c.chars().next()?;
            egui::Key::from_name(c.to_ascii_uppercase().to_string().as_str())?
        }
        _ => return None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_menu_fits_in_four_rows_from_full_size() {
        let largest = [1.0, 1.25, 1.5, 2.0, 3.0]
            .map(|scale| pixel_font::pixel_size(MENU_PIXEL, scale))
            .into_iter()
            .fold(0.0, f32::max);
        let height = menu_height_for(MENU_FULL_SIZE_WIDTH, largest);
        let four_rows = (2 * MENU_PAD + 4 * ROW_HEIGHT + 3 * ROW_GAP) as f32 * largest;
        assert!(height <= four_rows, "{height} > {four_rows}");

        let narrowest = MENU_FULL_SIZE_WIDTH * MENU_MIN_ZOOM;
        assert_eq!(narrowest / zoom_for(narrowest), MENU_FULL_SIZE_WIDTH);
    }

    #[test]
    fn world_sliders_reach_the_default_world() {
        let world = Globals::default();
        let defaults = [world.gravity, world.rest_temperature, world.wind];
        for ((slider, _), value) in WORLD_SLIDERS.iter().zip(defaults) {
            assert!(
                (slider.low..=slider.high).contains(&value),
                "{} cannot show its default of {value}",
                slider.label,
            );
        }
    }

    fn longest_readout() -> Readout {
        Readout {
            fps: 999.0,
            particles: config::MAX_PARTICLES,
            lagging_speed: Some(1.99),
        }
    }

    #[test]
    fn the_longest_readout_fits_under_the_world_sliders() {
        let line = readout_offset() + pixel_font::width(&longest_readout().text());
        assert!(
            line <= Piece::World.width(),
            "{line} > {}",
            Piece::World.width()
        );
    }

    #[test]
    fn dragging_near_a_default_snaps_to_it() {
        let temperature = &WORLD_SLIDERS[1].0;
        assert_eq!(temperature.snapped(23.0, 20.0), 20.0);
        assert_eq!(
            temperature.snapped(0.0, 20.0),
            0.0,
            "freezing must stay reachable"
        );
        assert_eq!(SIM_SPEED.snapped(0.95, REAL_TIME), REAL_TIME);
        assert_eq!(SIM_SPEED.snapped(0.5, REAL_TIME), 0.5);
    }

    #[test]
    fn values_never_read_negative_zero() {
        assert_eq!(whole_number(-0.3), "0");
        assert_eq!(whole_number(-40.0), "-40");
    }

    #[test]
    fn every_slider_can_show_its_default() {
        let brush = Brush::default();
        let world = Globals::default();
        let defaults = [
            (&BRUSH_SIZE, brush.radius),
            (&BRUSH_RATE, brush.heat_rate),
            (&BRUSH_FORCE, brush.wind_speed),
            (&WORLD_SLIDERS[0].0, world.gravity),
            (&WORLD_SLIDERS[1].0, world.rest_temperature),
            (&WORLD_SLIDERS[2].0, world.wind),
            (&SIM_SPEED, REAL_TIME),
        ];
        for (slider, value) in defaults {
            assert!(
                (slider.low..=slider.high).contains(&value),
                "{} cannot show its default of {value}",
                slider.label,
            );
        }
    }

    #[test]
    fn every_row_leaves_room_for_its_corner_icons() {
        for width in (404..=1500).step_by(7) {
            let rows = menu_rows(width);
            for (i, row) in rows.iter().enumerate() {
                let limit = width - reserved_for_corners(i, rows.len());
                assert!(
                    row_width(row) <= limit,
                    "at width {width}, row {i} is {} wide, over its {limit}",
                    row_width(row),
                );
            }
        }
    }

    #[test]
    fn the_changelog_is_dates_and_changes() {
        let known =
            |line: &str| line.is_empty() || ["# ", "## ", "- "].iter().any(|p| line.starts_with(p));
        for line in CHANGELOG.lines() {
            assert!(known(line), "the changelog window would skip {line:?}");
        }
        assert!(CHANGELOG.lines().any(|line| line.starts_with("## ")));
        assert!(CHANGELOG.lines().any(|line| line.starts_with("- ")));
    }

    #[test]
    fn every_menu_label_has_its_glyphs() {
        let readout = longest_readout().text();
        let values: Vec<String> = BRUSH_SLIDERS
            .iter()
            .chain(WORLD_SLIDERS.iter().map(|(slider, _)| slider))
            .chain([&SIM_SPEED])
            .flat_map(|slider| [(slider.show)(slider.low), (slider.show)(slider.high)])
            .collect();
        let labels = TOOLS
            .iter()
            .map(|tool| tool.1)
            .chain(MATERIALS.iter().map(|material| material.name))
            .chain(BRUSH_SLIDERS.iter().map(|slider| slider.label))
            .chain(WORLD_SLIDERS.iter().map(|(slider, _)| slider.label))
            .chain([SIM_SPEED.label, readout.as_str()])
            .chain(values.iter().map(String::as_str));
        for label in labels {
            let mut drawn = label.chars().filter(|&c| c != ' ');
            assert!(drawn.all(pixel_font::has_glyph), "{label}");
        }
    }
}
