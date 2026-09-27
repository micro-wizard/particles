struct RenderParams {
    world: vec2<f32>,
    pixels: vec2<f32>,
};

@group(0) @binding(0) var<uniform> params: RenderParams;
const MATERIAL_NONE: u32 = 0u;
const DEAD: u32 = 0xFFFFFFFFu;

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) @interpolate(flat) material: u32,
};

@vertex
fn vs_main(
    @location(0) corner: vec2<f32>,
    @location(1) centre: vec2<f32>,
    @location(2) radius: f32,
    @location(3) material: u32,
) -> VertexOutput {
    var out: VertexOutput;
    out.material = material;
    if material == DEAD {

        out.clip_position = vec4<f32>(0.0, 0.0, 0.0, 1.0);
        return out;
    }

    let per_unit = params.pixels / params.world;
    let centre_px = centre * per_unit;
    let side = max(1.0, round(radius * 2.0 * per_unit.x));
    let top_left = floor(centre_px - vec2<f32>(side * 0.5));
    let quad_px = top_left + (corner * 0.5 + 0.5) * side;

    out.clip_position = vec4<f32>(
        quad_px.x / params.pixels.x * 2.0 - 1.0,
        1.0 - quad_px.y / params.pixels.y * 2.0,
        0.0,
        1.0,
    );
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return vec4<f32>(f32(in.material + 1u) / 255.0, 0.0, 0.0, 1.0);
}

struct BlitOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) uv: vec2<f32>,
};

@group(0) @binding(0) var pixel_texture: texture_2d<f32>;
@group(0) @binding(1) var pixel_sampler: sampler;

struct Placement {
    offset: vec2<f32>,
    scale: f32,
    _padding: f32,
};
@group(0) @binding(2) var<uniform> placement: Placement;

struct MaterialParams {
    colour: vec4<f32>,
    freq_n: f32,
    freq_t: f32,
    zeta_n: f32,
    zeta_t: f32,
    mu: f32,
    density: f32,
    radius: f32,
    rest_packing: f32,
    pressure_k: f32,
    viscosity: f32,
    is_static: f32,
    spacing: f32,
    conductivity: f32,
    heat_capacity: f32,
    default_temperature: f32,
    thermal_expansion: f32,
    above_point: f32,
    becomes_above: u32,
    below_point: f32,
    becomes_below: u32,
    bond_freq: f32,
    heat_release: f32,
    growth_period: f32,
    sprouts: u32,
    emits: u32,
    emit_period: f32,
    burn_time: f32,
    _padding: u32,
};
@group(0) @binding(3) var<storage, read> materials: array<MaterialParams>;

const LETTERBOX: vec3<f32> = vec3<f32>(0.0, 0.0, 0.0);
const BACKGROUND: vec3<f32> = vec3<f32>(0.07, 0.06, 0.08);
const FILL_NEIGHBOURS: u32 = 4u;
const TONES: u32 = 4u;
const TEXTURE_BLOCK: i32 = 2;

fn hash_pixel(coord: vec2<i32>) -> u32 {
    var h = u32(coord.x) * 374761393u + u32(coord.y) * 668265263u;
    h = (h ^ (h >> 13u)) * 1274126177u;
    return h ^ (h >> 16u);
}

fn material_colour(material: u32, tone: u32) -> vec3<f32> {
    let base = materials[min(material, arrayLength(&materials) - 1u)].colour.rgb;

    let step = (f32(tone % TONES) - 1.5) * 0.026;
    return clamp(base * (1.0 + step), vec3<f32>(0.0), vec3<f32>(1.0));
}

@vertex
fn blit_vs(@builtin(vertex_index) index: u32) -> BlitOutput {
    let corner = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: BlitOutput;
    out.clip_position = vec4<f32>(corner * 2.0 - 1.0, 0.0, 1.0);
    out.uv = vec2<f32>(corner.x, 1.0 - corner.y);
    return out;
}

struct Air {
    gust_centre: vec2<f32>,
    gust_velocity: vec2<f32>,
    gust_radius: f32,
    gust_outflow: f32,
    wind: f32,
    time: f32,
};
@group(0) @binding(4) var<uniform> air: Air;

const TAU: f32 = 6.2831853;
const AIR_TINT: vec3<f32> = vec3<f32>(0.62, 0.72, 0.84);
const AIR_OPACITY: f32 = 0.5;
const LANE: f32 = 4.0;
const DASH_PERIOD: f32 = 24.0;
const STILL: f32 = 5.0;
const FULL_SPEED: f32 = 1500.0;
const DRIFT: f32 = 300.0;

fn drift(speed: f32) -> f32 {
    return DRIFT * speed / (speed + DRIFT);
}

fn streak(h: u32, travelled: f32, speed: f32, period: f32) -> f32 {
    let strength = min(speed / FULL_SPEED, 1.0);
    let dash = min(3.0 + 5.0 * strength, 0.5 * period);
    let phase = f32(h >> 16u) / 65536.0 * period;
    let along = fract((travelled + phase) / period) * period;
    let tail = period - dash;
    if along < tail {
        return 0.0;
    }
    return (0.35 + 0.65 * strength) * (along - tail) / dash;
}

fn lanes(p: vec2<f32>, flow: vec2<f32>, spacing: f32, period: f32) -> f32 {
    let speed = length(flow);
    if speed < STILL {
        return 0.0;
    }
    let ahead = flow / speed;
    let across = dot(p, vec2<f32>(-ahead.y, ahead.x)) - 0.5;
    let lane = round(across / spacing);
    if abs(across - lane * spacing) >= 0.5 {
        return 0.0;
    }

    let h = hash_pixel(vec2<i32>(i32(lane), 0));
    if (h & 1u) == 0u {
        return 0.0;
    }
    return streak(h, dot(p, ahead) - drift(speed) * air.time, speed, period);
}

fn spokes(rel: vec2<f32>, radial: f32, speed: f32, spacing: f32, period: f32) -> f32 {
    if speed < STILL || radial < max(2.5, 0.2 * air.gust_radius) {
        return 0.0;
    }
    let count = max(6.0, floor(TAU * air.gust_radius / spacing));
    let angle = atan2(rel.y, rel.x);
    let spoke = round(angle / TAU * count);

    if abs(radial * sin(angle - spoke / count * TAU)) >= 0.5 {
        return 0.0;
    }

    let seed = (i32(spoke) + i32(count)) % i32(count);
    let h = hash_pixel(vec2<i32>(seed, 1));
    return streak(h, radial - drift(speed) * air.time, speed, period);
}

fn sky(coord: vec2<i32>) -> vec3<f32> {
    let p = vec2<f32>(coord) + 0.5;
    let rel = p - air.gust_centre;
    let radial = length(rel);
    let wind = vec2<f32>(air.wind, 0.0);
    var shine = 0.0;
    if radial < air.gust_radius {
        let fine = clamp(air.gust_radius / 12.0, 0.5, 1.0);
        let spacing = 0.5 * LANE * fine;
        let period = 0.5 * DASH_PERIOD * fine;
        shine = max(
            lanes(p, wind + air.gust_velocity, spacing, period),
            spokes(rel, radial, air.gust_outflow, 2.0 * spacing, period),
        );
    } else {
        shine = lanes(p, wind, LANE, DASH_PERIOD);
    }
    return mix(BACKGROUND, AIR_TINT, AIR_OPACITY * shine);
}

struct LitNeighbours {
    count: u32,
    material_sum: f32,
};

fn lit_neighbours(coord: vec2<i32>, dims: vec2<i32>) -> LitNeighbours {
    var lit = LitNeighbours(0u, 0.0);
    for (var i = 0; i < 9; i++) {
        if i == 4 {
            continue;
        }
        let probe = clamp(coord + vec2<i32>(i % 3 - 1, i / 3 - 1), vec2<i32>(0), dims - vec2<i32>(1));
        let neighbour = textureLoad(pixel_texture, probe, 0);
        if neighbour.a > 0.5 {
            lit.count += 1u;
            lit.material_sum += neighbour.r;
        }
    }
    return lit;
}

fn encoded_material_at(coord: vec2<i32>, dims: vec2<i32>) -> u32 {
    let centre = textureLoad(pixel_texture, coord, 0);
    if centre.a > 0.5 {
        return u32(round(centre.r * 255.0));
    }
    let lit = lit_neighbours(coord, dims);
    if lit.count < FILL_NEIGHBOURS {
        return MATERIAL_NONE;
    }
    return u32(round(lit.material_sum / f32(lit.count) * 255.0));
}

@fragment
fn blit_fs(in: BlitOutput) -> @location(0) vec4<f32> {
    let dims = vec2<i32>(textureDimensions(pixel_texture));
    let local = (in.clip_position.xy - placement.offset) / placement.scale;
    if any(local < vec2<f32>(0.0)) || any(local >= vec2<f32>(dims)) {
        return vec4<f32>(LETTERBOX, 1.0);
    }
    let coord = vec2<i32>(floor(local));
    let encoded = encoded_material_at(coord, dims);
    if encoded == MATERIAL_NONE {
        return vec4<f32>(sky(coord), 1.0);
    }
    let tone = hash_pixel(coord / TEXTURE_BLOCK) % TONES;
    return vec4<f32>(material_colour(encoded - 1u, tone), 1.0);
}
