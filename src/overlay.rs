use crate::{
    config,
    gpu_timing::Span,
    materials::{Globals, MaterialParams, MATERIALS, MATERIAL_COUNT},
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
    elapsed: f32,
    since_log: f32,
}

impl Stats {
    fn new() -> Self {
        Self {
            frame_times: VecDeque::with_capacity(SAMPLE_WINDOW),
            substeps: 0,
            simulated: 0.0,
            elapsed: 0.0,
            since_log: 0.0,
        }
    }

    pub fn record(
        &mut self,
        frame_dt: f32,
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
        self.elapsed += frame_dt;
        if self.elapsed > 2.0 {
            self.simulated *= 0.5;
            self.elapsed *= 0.5;
        }

        self.since_log += frame_dt;
        if self.since_log >= LOG_INTERVAL {
            self.since_log = 0.0;
            let mean = self.mean_frame_time();

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
                if mean > 0.0 { 1.0 / mean } else { 0.0 },
                mean * 1000.0,
                self.worst_frame_time() * 1000.0,
                config::MAX_SUBSTEPS,
                self.sim_speed(),
            );
        }
    }

    fn mean_frame_time(&self) -> f32 {
        if self.frame_times.is_empty() {
            return 0.0;
        }
        self.frame_times.iter().sum::<f32>() / self.frame_times.len() as f32
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
}

struct Brush {
    mode: BrushMode,

    material: usize,

    heat_rate: f32,

    wind_speed: f32,

    radius: f32,
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

    fullscreen_requested: bool,

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
            brush: Brush {
                mode: BrushMode::Paint,
                material: 0,
                heat_rate: 720.0,

                wind_speed: 2000.0,
                radius: 6.0,
            },
            time_passes: false,
            fullscreen_requested: false,
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
            BrushMode::Paint | BrushMode::Blow | BrushMode::Burst => 0.0,
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

    pub fn captures_pointer(&self) -> bool {
        self.captured
    }

    pub fn menu_height(&self) -> f32 {
        self.menu_height
    }

    pub fn take_fullscreen_request(&mut self) -> bool {
        std::mem::take(&mut self.fullscreen_requested)
    }

    pub fn run(
        &mut self,
        width: u32,
        height: u32,
        scale: f32,
        particle_count: u32,
        fullscreen: bool,
        world: [f32; 4],
        blow_direction: [f32; 2],
    ) {
        let raw = egui::RawInput {
            screen_rect: Some(egui::Rect::from_min_size(
                egui::pos2(0.0, 0.0),
                egui::vec2(width as f32 / scale, height as f32 / scale),
            )),
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
        let screen = egui::vec2(width as f32 / scale, height as f32 / scale);

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

        let output = self.context.run(raw, |ctx| {
            if menu(ctx, panel, scale, brush, &colours, fullscreen) {
                *fullscreen_requested = true;
            }


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
        width: u32,
        height: u32,
        scale: f32,
    ) {
        if self.paint_jobs.is_empty() {
            return;
        }
        let delta = std::mem::take(&mut self.textures_delta);
        for (id, image) in &delta.set {
            self.renderer.update_texture(device, queue, *id, image);
        }
        let desc = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: scale,
        };
        self.renderer
            .update_buffers(device, queue, encoder, &self.paint_jobs, &desc);

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
        self.renderer.render(&mut pass, &self.paint_jobs, &desc);
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
const SLIDER_WIDTH: i32 = 64;
const SLIDER_LABELS: [&str; 3] = ["Size", "Rate", "Force"];
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

const TOOLS: [(BrushMode, &str, egui::Color32, &str); 5] = [
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
];

#[derive(Copy, Clone)]
enum Piece {
    Tools,

    Swatch(usize),

    Controls,
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
                slider_label_width() + NAME_GAP + SLIDER_WIDTH + ITEM_GAP + FULLSCREEN.width
            }
        }
    }

    fn gap_after(self, previous: Piece) -> i32 {
        let group = |piece| match piece {
            Piece::Tools => 0,
            Piece::Swatch(_) => 1,
            Piece::Controls => 2,
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
        .chain((0..MATERIALS.len()).map(Piece::Swatch))
        .chain(std::iter::once(Piece::Controls));
    let mut rows: Vec<Vec<Piece>> = Vec::new();
    for piece in pieces {
        match rows.last_mut() {
            Some(row)
                if row_width(row) + piece.gap_after(row[row.len() - 1]) + piece.width()
                    <= width =>
            {
                row.push(piece)
            }
            _ => rows.push(vec![piece]),
        }
    }
    rows
}

fn row_width(row: &[Piece]) -> i32 {
    row.iter().map(|piece| piece.width()).sum::<i32>()
        + row
            .windows(2)
            .map(|pair| pair[1].gap_after(pair[0]))
            .sum::<i32>()
}

fn slider_label_width() -> i32 {
    SLIDER_LABELS
        .iter()
        .map(|label| pixel_font::width(label))
        .max()
        .unwrap_or(0)
}

const MENU_FULL_SIZE_WIDTH: f32 = 832.0;

const MENU_MIN_ZOOM: f32 = 0.75;

pub fn zoom_for(width: f32) -> f32 {
    (width / MENU_FULL_SIZE_WIDTH).clamp(MENU_MIN_ZOOM, 1.0)
}

fn menu_height_for(width: f32, pixel: f32) -> f32 {
    let rows = menu_rows((width / pixel).floor() as i32 - 2 * MENU_PAD).len() as i32;
    (2 * MENU_PAD + rows * ROW_HEIGHT + (rows - 1) * ROW_GAP) as f32 * pixel
}

fn menu(
    ctx: &egui::Context,
    panel: egui::Rect,
    scale: f32,
    brush: &mut Brush,
    colours: &[[f32; 4]],
    fullscreen: bool,
) -> bool {
    let mut toggled = false;
    egui::Area::new(egui::Id::new("menu"))
        .fixed_pos(panel.min)
        .constrain(false)
        .order(egui::Order::Foreground)
        .show(ctx, |ui| {
            ui.allocate_rect(panel, egui::Sense::hover());
            let grid = pixel_font::Grid::new(panel.min, MENU_PIXEL, scale);
            let [columns, height] = grid.cell(panel.max);
            ui.painter().rect_filled(panel, 0.0, MENU_FILL);
            grid.fill(ui.painter(), 0, 0, columns + 1, 1, MENU_EDGE);

            let rows = menu_rows(columns - 2 * MENU_PAD);
            let rows_height = rows.len() as i32 * (ROW_HEIGHT + ROW_GAP) - ROW_GAP;

            let mut top = ((height - rows_height) / 2).max(MENU_PAD);
            for row in &rows {
                let mut left = (columns - row_width(row)) / 2;
                let mut previous = None;
                for &piece in row {
                    if let Some(previous) = previous {
                        left += piece.gap_after(previous);
                    }
                    match piece {
                        Piece::Tools => tools(ui, grid, left, top, brush),
                        Piece::Swatch(id) => swatch(ui, grid, left, top, brush, id, colours[id]),
                        Piece::Controls => {
                            toggled |= controls(ui, grid, left, top, brush, fullscreen)
                        }
                    }
                    left += piece.width();
                    previous = Some(piece);
                }
                top += ROW_HEIGHT + ROW_GAP;
            }
        });
    toggled
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
    if response.clicked() {
        brush.material = id;
        brush.mode = BrushMode::Paint;
    }
}

fn controls(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    x: i32,
    y: i32,
    brush: &mut Brush,
    fullscreen: bool,
) -> bool {
    let accent = accent(brush.mode);

    let (first, second) = (y, y + ROW_HEIGHT - pixel_font::HEIGHT);
    menu_slider(
        ui,
        grid,
        x,
        first,
        "Size",
        &mut brush.radius,
        0.0..=config::BRUSH_RADIUS_LIMIT,
        Some(accent),
        "How far the brush reaches: the red circle around the cursor.",
    );
    let enabled = brush.mode != BrushMode::Paint;
    let (label, value, range, tip) = match brush.mode {
        BrushMode::Blow | BrushMode::Burst => (
            "Force",
            &mut brush.wind_speed,
            100.0..=3000.0,
            "How hard Blow and Burst push: the speed of the air they make. Steam goes \
             with the gentlest breeze, sand with about half, and gravel wants most of it.",
        ),
        BrushMode::Paint | BrushMode::Heat | BrushMode::Cool => (
            "Rate",
            &mut brush.heat_rate,
            60.0..=3600.0,
            "How fast Heat and Cool change the temperature, in degrees per second.",
        ),
    };
    menu_slider(
        ui,
        grid,
        x,
        second,
        label,
        value,
        range,
        enabled.then_some(accent),
        tip,
    );
    let icon = x + slider_label_width() + NAME_GAP + SLIDER_WIDTH + ITEM_GAP;
    fullscreen_toggle(ui, grid, icon, y, fullscreen)
}

#[allow(clippy::too_many_arguments)]
fn menu_slider(
    ui: &mut egui::Ui,
    grid: pixel_font::Grid,
    x: i32,
    y: i32,
    label: &str,
    value: &mut f32,
    range: std::ops::RangeInclusive<f32>,
    accent: Option<egui::Color32>,
    tip: &str,
) {
    let (low, high) = (*range.start(), *range.end());
    let rail = x + slider_label_width() + NAME_GAP;
    let travel = SLIDER_WIDTH - HANDLE_WIDTH;
    let hit = grid.rect(
        rail - NAME_GAP / 2,
        y - 1,
        SLIDER_WIDTH + NAME_GAP,
        pixel_font::HEIGHT + 2,
    );
    let sense = if accent.is_some() {
        egui::Sense::click_and_drag()
    } else {
        egui::Sense::hover()
    };
    let response = ui.allocate_rect(hit, sense);
    if accent.is_some() {
        if let Some(pointer) = response.interact_pointer_pos() {
            let start = grid.rect(rail, y, HANDLE_WIDTH, 1).center().x;
            let along = (pointer.x - start) / (travel as f32 * grid.rect(0, 0, 1, 1).width());
            *value = low + (high - low) * along.clamp(0.0, 1.0);
        }
        if response.has_focus() {
            let step = (high - low) / travel as f32;
            ui.input(|input| {
                if input.key_pressed(egui::Key::ArrowRight) {
                    *value = (*value + step).min(high);
                }
                if input.key_pressed(egui::Key::ArrowLeft) {
                    *value = (*value - step).max(low);
                }
            });
        }
    }
    response.widget_info(|| egui::WidgetInfo::slider(accent.is_some(), *value as f64, label));

    let painter = ui.painter();
    let handle = (((*value - low) / (high - low)).clamp(0.0, 1.0) * travel as f32).round() as i32;
    let (text, fill, knob) = match accent {
        Some(accent) => {
            let active = response.hovered() || response.dragged();
            (TEXT, accent, if active { TEXT_CHOSEN } else { TEXT_HOVER })
        }
        None => (TEXT_DIM, TEXT_DIM, TEXT_DIM),
    };
    grid.text(painter, x, y, label, text);
    grid.fill(painter, rail, y + 2, SLIDER_WIDTH, 3, RAIL);
    grid.fill(painter, rail, y + 2, handle, 3, fill);
    grid.fill(
        painter,
        rail + handle,
        y,
        HANDLE_WIDTH,
        pixel_font::HEIGHT,
        knob,
    );
    response.on_hover_text(tip);
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
    x: i32,
    y: i32,
    fullscreen: bool,
) -> bool {
    let (label, icon) = if fullscreen {
        ("Exit fullscreen (Esc)", &EXIT_FULLSCREEN)
    } else {
        ("Fullscreen", &FULLSCREEN)
    };
    let rect = grid.rect(x - ITEM_GAP / 2, y, icon.width + ITEM_GAP, ROW_HEIGHT);
    let response = ui.allocate_rect(rect, egui::Sense::click());
    response.widget_info(|| {
        egui::WidgetInfo::labeled(egui::WidgetType::Button, ui.is_enabled(), label)
    });
    grid.bitmap(
        ui.painter(),
        x,
        y + TEXT_TOP,
        icon,
        label_colour(&response, None),
    );
    response.on_hover_text(label).clicked()
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
    fn the_menu_fits_in_two_rows_from_full_size() {
        let largest = [1.0, 1.25, 1.5, 2.0, 3.0]
            .map(|scale| pixel_font::pixel_size(MENU_PIXEL, scale))
            .into_iter()
            .fold(0.0, f32::max);
        let height = menu_height_for(MENU_FULL_SIZE_WIDTH, largest);
        let two_rows = (2 * MENU_PAD + 2 * ROW_HEIGHT + ROW_GAP) as f32 * largest;
        assert!(height <= two_rows, "{height} > {two_rows}");

        let narrowest = MENU_FULL_SIZE_WIDTH * MENU_MIN_ZOOM;
        assert_eq!(narrowest / zoom_for(narrowest), MENU_FULL_SIZE_WIDTH);
    }

    #[test]
    fn every_menu_label_has_its_glyphs() {
        let labels = TOOLS
            .iter()
            .map(|tool| tool.1)
            .chain(MATERIALS.iter().map(|material| material.name))
            .chain(SLIDER_LABELS);
        for label in labels {
            assert!(label.chars().all(pixel_font::has_glyph), "{label}");
        }
    }
}
