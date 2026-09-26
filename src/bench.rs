use crate::{
    compute_pipeline::Solver, config, ecs, ecs::Particle, gpu_context::GpuContext,
    gpu_timing::Span, materials::MATERIALS,
};
use instant::Instant;
use winit::{event_loop::EventLoop, window::WindowBuilder};

const WARMUP: u32 = 120;
const BATCH_FRAMES: u32 = 60;
const BATCHES: u32 = 8;
const FRAMES: u32 = 240;
const FRAME_DT: f32 = config::SUBSTEP * config::MAX_SUBSTEPS as f32;

pub async fn run() {
    let event_loop = EventLoop::new().expect("event loop");
    let window = WindowBuilder::new()
        .with_visible(false)
        .build(&event_loop)
        .expect("window");
    let gpu_context = GpuContext::new(&window).await;

    let fluid = MATERIALS
        .iter()
        .position(|m| m.is_fluid())
        .expect("a fluid to fill the screen with") as u32;

    println!(
        "{} substeps/frame, {} hash cells",
        config::MAX_SUBSTEPS,
        cell_count(),
    );
    measure(
        &gpu_context,
        "default world",
        &ecs::seed_world(config::PARTICLE_COUNT),
    );
    measure(
        &gpu_context,
        "full screen of water",
        &ecs::seed_full_screen(fluid),
    );
}

fn cell_count() -> u32 {
    let span =
        |extent: f32| ((extent + 2.0 * config::HASH_MARGIN) / config::CELL_SIZE).ceil() as u32;
    span(config::WORLD_WIDTH) * span(config::WORLD_HEIGHT)
}

fn measure(gpu_context: &GpuContext<'_>, label: &str, seed: &[Particle]) {
    let mut solver = Solver::new(gpu_context, seed);
    println!("\n{label}: {} particles", seed.len());

    let on_gpu = solver.read_materials(gpu_context);
    for (i, (want, got)) in MATERIALS.iter().zip(on_gpu.iter()).enumerate() {
        if want.params != *got {
            println!("  MISMATCH at id {i} ({})", want.name);
            println!(
                "    cpu conductivity={} gpu conductivity={}",
                want.params.conductivity, got.conductivity
            );
            println!(
                "    cpu density={} gpu density={}",
                want.params.density, got.density
            );
            println!(
                "    cpu above_point={} gpu above_point={}",
                want.params.above_point, got.above_point
            );
        }
    }
    println!(
        "  materials on GPU: lava conductivity={} density={} below_point={} becomes_below={}",
        on_gpu[7].conductivity, on_gpu[7].density, on_gpu[7].below_point, on_gpu[7].becomes_below,
    );

    for frame in 0..=8 {
        if frame > 0 {
            for _ in 0..25 {
                step(gpu_context, &mut solver);
            }
        }
        let grains = solver.read_particles(gpu_context, seed.len() as u32);
        let mut report = String::new();
        for id in [3u32, 5, 6, 7, 8] {
            let t: Vec<f32> = grains
                .iter()
                .filter(|g| g.material == id)
                .map(|g| g.temperature)
                .collect();
            if t.is_empty() {
                continue;
            }
            let mean = t.iter().sum::<f32>() / t.len() as f32;
            let lo = t.iter().copied().fold(f32::INFINITY, f32::min);
            report += &format!(
                " {}={}@{:.0}({:.0}..)",
                MATERIALS[id as usize].name,
                t.len(),
                mean,
                lo,
            );
        }
        let moving: Vec<&Particle> = grains
            .iter()
            .filter(|g| g.material != ecs::DEAD && !MATERIALS[g.material as usize].is_static())
            .collect();
        let fastest = moving
            .iter()
            .map(|g| g.velocity[0].hypot(g.velocity[1]))
            .fold(0.0f32, f32::max);
        let escaped = moving
            .iter()
            .filter(|g| {
                !(0.0..=config::WORLD_WIDTH).contains(&g.position[0])
                    || !(0.0..=config::WORLD_HEIGHT).contains(&g.position[1])
            })
            .count();
        let mean_y = moving.iter().map(|g| g.position[1]).sum::<f32>() / moving.len().max(1) as f32;
        report += &format!(" | vmax={fastest:.0} out={escaped} mean_y={mean_y:.1}");
        println!("  t={:>4} frames:{report}", frame * 25);
    }

    solver.set_timing(gpu_context, true);
    for _ in 0..WARMUP + FRAMES {
        step(gpu_context, &mut solver);
    }
    match solver.pass_timings() {
        Some(spans) => {
            let total: f32 = spans.iter().sum();
            print!("  attribution (split passes, {total:6.3} ms/frame):");
            for (span, ms) in Span::ALL.iter().zip(spans) {
                print!(" {}={ms:.3}", span.label());
            }
            println!();
        }
        None => println!("  attribution: unavailable (adapter cannot timestamp)"),
    }

    solver.set_timing(gpu_context, false);
    for _ in 0..WARMUP {
        step(gpu_context, &mut solver);
    }
    let mut fastest = f64::INFINITY;
    for _ in 0..BATCHES {
        let start = Instant::now();
        for _ in 0..BATCH_FRAMES {
            solver.step(gpu_context, FRAME_DT);
        }
        gpu_context.device.poll(wgpu::Maintain::Wait);
        let per_frame = start.elapsed().as_secs_f64() * 1000.0 / BATCH_FRAMES as f64;
        fastest = fastest.min(per_frame);
    }
    println!(
        "  frame: {fastest:6.3} ms  ({:.3} ms/substep, {:.1}x realtime)",
        fastest / config::MAX_SUBSTEPS as f64,
        FRAME_DT as f64 * 1000.0 / fastest,
    );
}

fn step(gpu_context: &GpuContext<'_>, solver: &mut Solver) {
    solver.step(gpu_context, FRAME_DT);
    gpu_context.device.poll(wgpu::Maintain::Wait);
}
