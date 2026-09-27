use crate::{config, materials};

#[repr(C)]
#[derive(Copy, Clone, Debug)]
pub struct Particle {
    pub position: [f32; 2],
    pub velocity: [f32; 2],
    pub radius: f32,
    pub material: u32,
    pub drawn: [f32; 2],
    pub packing: f32,
    pub temperature: f32,
}

pub const SIZE_SPREAD: f32 = 0.1;
pub const DEAD: u32 = u32::MAX;

unsafe impl bytemuck::Zeroable for Particle {}
unsafe impl bytemuck::Pod for Particle {}

impl Particle {
    pub fn new(position: [f32; 2], velocity: [f32; 2], radius: f32, material: u32) -> Self {
        Self {
            position,
            velocity,
            radius,
            material,
            drawn: position,
            packing: 0.0,
            temperature: materials::MATERIALS[material as usize]
                .params
                .default_temperature,
        }
    }

    pub fn dead() -> Self {
        Self {
            position: [0.0, 0.0],
            velocity: [0.0, 0.0],
            radius: 0.0,
            material: DEAD,
            drawn: [0.0, 0.0],
            packing: 0.0,
            temperature: config::AMBIENT_TEMPERATURE,
        }
    }
}

pub fn seed_world(grains: u32) -> Vec<Particle> {
    let particles = seed_walls();
    particles
}

pub fn seed_walls() -> Vec<Particle> {
    let (id, wall) = materials::MATERIALS
        .iter()
        .enumerate()
        .find(|(_, m)| m.is_static())
        .expect("a static material to build the world's boundary from");
    let spacing = materials::MATERIALS
        .iter()
        .find(|m| m.is_fluid())
        .map_or(wall.params.rest_spacing(), |f| f.params.rest_spacing());
    let depth = config::WALL_LAYERS as f32 * spacing;

    let mut particles = Vec::new();
    let mut place = |x: f32, y: f32| {
        particles.push(Particle::new(
            [x, y],
            [0.0, 0.0],
            wall.params.radius,
            id as u32,
        ));
    };

    let rows = ((config::WORLD_HEIGHT + 2.0 * depth) / spacing).ceil() as u32;
    for layer in 0..config::WALL_LAYERS {
        let inset = (layer as f32 + 0.5) * spacing;
        for row in 0..rows {
            let y = -depth + (row as f32 + 0.5) * spacing;
            place(-inset, y);
            place(config::WORLD_WIDTH + inset, y);
        }
    }
    let columns = (config::WORLD_WIDTH / spacing).ceil() as u32;
    for layer in 0..config::WALL_LAYERS {
        let inset = (layer as f32 + 0.5) * spacing;
        for column in 0..columns {
            let x = (column as f32 + 0.5) * spacing;
            place(x, -inset);
            place(x, config::WORLD_HEIGHT + inset);
        }
    }
    particles
}

#[cfg(not(target_arch = "wasm32"))]
pub fn seed_full_screen(material: u32) -> Vec<Particle> {
    let mut particles = seed_walls();
    let params = materials::MATERIALS[material as usize].params;
    let spacing = params.rest_spacing();
    let columns = (config::WORLD_WIDTH / spacing) as u32;
    let rows = (config::WORLD_HEIGHT / spacing) as u32;
    for row in 0..rows {
        for column in 0..columns {
            particles.push(Particle::new(
                [
                    (column as f32 + 0.5) * spacing,
                    (row as f32 + 0.5) * spacing,
                ],
                [0.0, 0.0],
                params.radius,
                material,
            ));
        }
    }
    particles
}

pub fn seed_block(count: u32) -> Vec<Particle> {
    let defaults = materials::MaterialParams::defaults();
    let widest = defaults
        .iter()
        .fold(0.0f32, |acc, m| acc.max(m.radius * (1.0 + SIZE_SPREAD)));
    let spacing = widest * 2.55;
    let falling: Vec<u32> = materials::MATERIALS
        .iter()
        .enumerate()
        .filter(|(_, m)| !m.is_static() && !m.is_gas())
        .map(|(id, _)| id as u32)
        .collect();
    let columns = ((config::WORLD_WIDTH / spacing).floor() as u32)
        .saturating_sub(2)
        .max(1);
    let rows = count.div_ceil(columns);

    let block_width = columns as f32 * spacing;
    let block_height = rows as f32 * spacing;
    let origin_x = (config::WORLD_WIDTH - block_width) * 0.5;
    let origin_y = (config::WORLD_HEIGHT - block_height) * 0.25;

    (0..count)
        .map(|i| {
            let (column, row) = (i % columns, i / columns);
            let jitter = spacing * 0.10;
            let bands = falling.len() as u32;
            let material = falling[((row * bands / rows.max(1)).min(bands - 1)) as usize];
            let base = defaults[material as usize].radius;
            Particle::new(
                [
                    origin_x + column as f32 * spacing + random_range(-jitter, jitter),
                    origin_y + row as f32 * spacing + random_range(-jitter, jitter),
                ],
                [random_range(-20.0, 20.0), 0.0],
                base * random_range(1.0 - SIZE_SPREAD, 1.0 + SIZE_SPREAD),
                material,
            )
        })
        .collect()
}

pub fn brush_points(centre: [f32; 2], radius: f32, spacing: f32) -> Vec<[f32; 2]> {
    let row_height = spacing * 0.866_025_4;
    let margin = 0.5 * spacing;
    let mut points = Vec::new();
    let first_row = ((centre[1] - radius) / row_height).floor() as i32;
    let last_row = ((centre[1] + radius) / row_height).ceil() as i32;
    for row in first_row..=last_row {
        let y = row as f32 * row_height;
        let shift = if row.rem_euclid(2) == 1 {
            0.5 * spacing
        } else {
            0.0
        };
        let first_column = ((centre[0] - radius - shift) / spacing).floor() as i32;
        let last_column = ((centre[0] + radius - shift) / spacing).ceil() as i32;
        for column in first_column..=last_column {
            let site = [column as f32 * spacing + shift, y];
            let in_brush = (site[0] - centre[0]).hypot(site[1] - centre[1]) <= radius;
            let in_world = (margin..=config::WORLD_WIDTH - margin).contains(&site[0])
                && (margin..=config::WORLD_HEIGHT - margin).contains(&site[1]);
            if in_brush && in_world {
                points.push(site);
            }
        }
    }
    if points.is_empty() {
        points.push(centre);
    }
    if points.len() > config::MAX_SPAWNS_PER_FRAME as usize {
        let distance = |p: &[f32; 2]| (p[0] - centre[0]).hypot(p[1] - centre[1]);
        points.sort_by(|a, b| distance(a).total_cmp(&distance(b)));
        points.truncate(config::MAX_SPAWNS_PER_FRAME as usize);
    }
    points
}

const RNG_SEED: u32 = 0x5EED_5EED;

thread_local! {
    static RNG: std::cell::Cell<u32> = const { std::cell::Cell::new(RNG_SEED) };
}

pub(crate) fn random_range(min: f32, max: f32) -> f32 {
    let bits = RNG.with(|state| {
        let mut x = state.get();
        x ^= x << 13;
        x ^= x >> 17;
        x ^= x << 5;
        state.set(x);
        x
    });
    let unit = bits as f32 / u32::MAX as f32;
    min + unit * (max - min)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn seeds_every_material_that_can_be_seeded() {
        let particles = seed_world(config::PARTICLE_COUNT);
        for (id, material) in materials::MATERIALS.iter().enumerate() {
            let seeded = particles.iter().filter(|p| p.material == id as u32).count();
            if material.is_gas() {
                assert_eq!(
                    seeded, 0,
                    "{} is a gas and must not be seeded as a falling band",
                    material.name,
                );
            } else {
                assert!(seeded > 0, "{} (id {id}) was never seeded", material.name);
            }
        }
    }

    #[test]
    fn wall_collar_fits_inside_the_padded_hash() {
        let m = config::HASH_MARGIN;
        for p in seed_walls() {
            let (x, y) = (p.position[0], p.position[1]);
            assert!(
                x - p.radius >= -m
                    && x + p.radius <= config::WORLD_WIDTH + m
                    && y - p.radius >= -m
                    && y + p.radius <= config::WORLD_HEIGHT + m,
                "wall particle at ({x}, {y}) r{} escapes the {m}-unit hash margin",
                p.radius,
            );
        }
    }

    #[test]
    fn sweep_covers_the_interaction_range() {
        let reach = config::SWEEP_RADIUS as f32 * config::CELL_SIZE;
        assert!(
            reach >= config::INTERACTION_RANGE,
            "a {}-cell sweep of {}-unit cells reaches {reach}, short of the \
             {}-unit interaction range",
            config::SWEEP_RADIUS,
            config::CELL_SIZE,
            config::INTERACTION_RANGE,
        );
        assert!(
            config::SMOOTHING_RADIUS <= reach,
            "the fluid kernel reaches {} but the sweep only covers {reach}",
            config::SMOOTHING_RADIUS,
        );
    }

    #[test]
    fn smoothing_radius_bounds_contact() {
        let widest_contact = 2.0 * config::RADIUS_LIMIT;
        assert!(
            widest_contact <= config::SMOOTHING_RADIUS,
            "two {}-radius grains touch at {widest_contact}, outside the {} \
             smoothing radius that solve rejects on",
            config::RADIUS_LIMIT,
            config::SMOOTHING_RADIUS,
        );
    }

    #[test]
    fn wall_collar_encloses_the_play_area() {
        let walls = seed_walls();
        let left = walls.iter().any(|p| p.position[0] < 0.0);
        let right = walls.iter().any(|p| p.position[0] > config::WORLD_WIDTH);
        let floor = walls.iter().any(|p| p.position[1] < 0.0);
        let ceiling = walls.iter().any(|p| p.position[1] > config::WORLD_HEIGHT);
        assert!(
            left && right && floor && ceiling,
            "collar is missing a side: left {left}, right {right}, floor {floor}, ceiling {ceiling}",
        );
    }

    #[test]
    fn max_particles_covers_a_full_screen_of_the_smallest_grain() {
        let spacing = materials::MaterialParams {
            radius: config::RADIUS_FLOOR,
            ..materials::MATERIALS[0].params
        }
        .rest_spacing();
        let per_grain = spacing * spacing * 0.866_025_4;
        let screenful = (config::WORLD_WIDTH * config::WORLD_HEIGHT / per_grain).ceil() as u32;
        let collar = seed_walls().len() as u32;
        assert!(
            screenful + collar <= config::MAX_PARTICLES,
            "a full screen of radius-{} grains is {screenful}, plus a {collar}-particle \
             collar = {}, over the {} slots preallocated",
            config::RADIUS_FLOOR,
            screenful + collar,
            config::MAX_PARTICLES,
        );
    }

    #[test]
    fn a_brush_stamp_never_outruns_one_frame_of_spawns() {
        for radius in [0.0, 0.5, 6.0, 24.0, 1000.0] {
            let n = brush_points([100.0, 100.0], radius, 1.54).len() as u32;
            assert!(
                (1..=config::MAX_SPAWNS_PER_FRAME).contains(&n),
                "brush radius {radius} asked for {n} grains in one frame",
            );
        }
    }

    #[test]
    fn brush_points_stay_inside_the_brush() {
        let centre = [100.0, 80.0];
        for radius in [0.5, 6.0, 24.0] {
            for p in brush_points(centre, radius, 1.54) {
                let d = (p[0] - centre[0]).hypot(p[1] - centre[1]);
                assert!(d <= radius + 1e-3, "point {d} out from a {radius} brush");
            }
        }
    }

    #[test]
    fn the_largest_brush_fills_in_one_frame() {
        let tightest = materials::MATERIALS
            .iter()
            .map(|m| m.params.spacing)
            .fold(f32::INFINITY, f32::min);
        let spacing = tightest * config::RADIUS_FLOOR;
        let centre = [config::WORLD_WIDTH * 0.5, config::WORLD_HEIGHT * 0.5];
        let sites = brush_points(centre, config::BRUSH_RADIUS_LIMIT, spacing).len() as u32;
        assert!(
            sites < config::MAX_SPAWNS_PER_FRAME,
            "a {}-unit brush at spacing {spacing} needs at least {sites} sites, but a frame \
             places only {}",
            config::BRUSH_RADIUS_LIMIT,
            config::MAX_SPAWNS_PER_FRAME,
        );
    }

    #[test]
    fn brush_sites_leave_room_for_every_grain() {
        for m in materials::MATERIALS.iter() {
            let largest = m.params.radius * (1.0 + SIZE_SPREAD);
            assert!(
                m.params.rest_spacing() >= 2.0 * largest * 0.9999,
                "{} paints {} apart, but its grains can reach radius {largest}",
                m.name,
                m.params.rest_spacing(),
            );
        }
    }

    #[test]
    fn a_stamp_never_overlaps_itself() {
        let spacing = 1.54;
        for centre in [[256.0, 128.0], [2.0, 3.0], [511.0, 255.0]] {
            let stamp = brush_points(centre, 12.0, spacing);
            for (i, a) in stamp.iter().enumerate() {
                for b in &stamp[i + 1..] {
                    let d = (a[0] - b[0]).hypot(a[1] - b[1]);
                    assert!(
                        d >= spacing * 0.9999,
                        "sites {a:?} and {b:?} are {d} apart, under the {spacing} spacing",
                    );
                }
            }
        }
    }

    #[test]
    fn seeding_repeats_run_to_run() {
        fn draw() -> Vec<f32> {
            (0..16).map(|_| random_range(-1.0, 1.0)).collect()
        }
        let first = std::thread::spawn(draw).join().expect("draw");
        let second = std::thread::spawn(draw).join().expect("draw");
        assert_eq!(first, second, "the world seed must not vary between runs");
        assert!(
            first.windows(2).any(|w| w[0] != w[1]),
            "a generator that returns a constant would also pass the above",
        );
    }

    #[test]
    fn seed_block_fits_inside_the_play_area() {
        for p in seed_block(config::PARTICLE_COUNT) {
            let (x, y, r) = (p.position[0], p.position[1], p.radius);
            assert!(
                x - r > 0.0
                    && x + r < config::WORLD_WIDTH
                    && y - r > 0.0
                    && y + r < config::WORLD_HEIGHT,
                "seeded grain at ({x}, {y}) r{r} is outside the {}x{} play area",
                config::WORLD_WIDTH,
                config::WORLD_HEIGHT,
            );
        }
    }

    #[test]
    fn seeds_only_known_materials() {
        for p in seed_world(config::PARTICLE_COUNT) {
            assert!(
                p.material < materials::MATERIAL_COUNT,
                "bad id {}",
                p.material
            );
        }
    }
}
