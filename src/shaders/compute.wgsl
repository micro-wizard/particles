struct Params {
    world: vec2<f32>,
    grid: vec2<u32>,
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
    gust_centre: vec2<f32>,
    gust_velocity: vec2<f32>,
    gust_radius: f32,
    gust_outflow: f32,
    contact_parity: u32,
    _padding: u32,
};

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
};

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    radius: f32,
    material: u32,
    drawn: vec2<f32>,
    packing: f32,
    temperature: f32,
    parent: u32,
    grandparent: u32,
    sprout: u32,
    rest_angle: f32,
    stem_force: vec2<f32>,
};

struct ContactRecord {
    partner: u32,
    _padding: u32,
    tangent: vec2<f32>,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> cell_counts: array<atomic<u32>>;
@group(0) @binding(2) var<storage, read_write> cell_start: array<u32>;
@group(0) @binding(3) var<storage, read_write> cell_cursor: array<atomic<u32>>;

struct CellEntry {
    position: vec2<f32>,
    velocity: vec2<f32>,
    radius: f32,
    packing: f32,
    temperature: f32,
    packed: u32,
};

const PACKED_MATERIAL_SHIFT: u32 = 20u;
const PACKED_INDEX_MASK: u32 = (1u << PACKED_MATERIAL_SHIFT) - 1u;
@group(0) @binding(4) var<storage, read_write> cell_entries: array<CellEntry>;
@group(0) @binding(5) var<uniform> params: Params;
@group(0) @binding(6) var<storage, read_write> contacts: array<ContactRecord>;
@group(0) @binding(7) var<storage, read> materials: array<MaterialParams>;

// The spawner's slot allocator, shared so `grow` can place new plant nodes.
struct SlotAllocator {
    free_count: atomic<u32>,
    high_water: atomic<u32>,
    free_stack: array<u32>,
};

@group(0) @binding(8) var<storage, read_write> allocator: SlotAllocator;
const FRICTIONLESS: f32 = 0.01;
const NO_CONTACT: u32 = 0xFFFFFFFFu;
const DEAD: u32 = 0xFFFFFFFFu;
const NO_TRANSITION: u32 = 0xFFFFFFFFu;
const NO_PARENT: u32 = 0xFFFFFFFFu;
const BEND_SCALE: f32 = 0.15;
const UP: vec2<f32> = vec2<f32>(0.0, -1.0);
const MAX_SHOOTS: u32 = 2u;

struct Link {
    position: vec2<f32>,
    velocity: vec2<f32>,
    radius: f32,
    mass: f32,
    bond_freq: f32,
    zeta: f32,
    rest_angle: f32,
    index: u32,
    parent: u32,
};

fn rotate(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

fn bend_spring(parent: Link, child: Link) -> Mixed {
    var m = spring(
        BEND_SCALE * sqrt(parent.bond_freq * child.bond_freq),
        0.0,
        sqrt(parent.zeta * child.zeta),
        0.0,
        0.5 * min(parent.mass, child.mass),
        0.0,
    );
    m.gamma_n = m.gamma_n / (1.0 + m.gamma_n * params.dt / max(m.mass, 1e-9));
    return m;
}

fn joint_error(grandparent: Link, parent: Link, child: Link) -> vec2<f32> {
    let s = (child.radius + parent.radius) / (parent.radius + grandparent.radius);
    let d = (child.position - parent.position)
        - s * rotate(parent.position - grandparent.position, child.rest_angle);
    let dv = (child.velocity - parent.velocity)
        - s * rotate(parent.velocity - grandparent.velocity, child.rest_angle);
    let m = bend_spring(parent, child);
    return m.k_n * d + m.gamma_n * dv;
}

fn joint_reaction(grandparent: Link, parent: Link, child: Link, error: vec2<f32>) -> vec2<f32> {
    let s = (child.radius + parent.radius) / (parent.radius + grandparent.radius);
    return s * rotate(error, -child.rest_angle);
}

fn in_reach(a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = a - b;
    return dot(d, d) < params.smoothing_radius * params.smoothing_radius;
}

fn link_at(entry: u32) -> Link {
    let e = cell_entries[entry];
    let index = e.packed & PACKED_INDEX_MASK;
    let m = materials[min(e.packed >> PACKED_MATERIAL_SHIFT, params.material_count - 1u)];
    let node = particles[index];
    return Link(
        e.position, e.velocity, e.radius, m.density * e.radius * e.radius,
        m.bond_freq, m.zeta_n, node.rest_angle, index, node.parent,
    );
}

fn root_error(seed: Link, child: Link) -> vec2<f32> {
    let d = (child.position - seed.position)
        - (child.radius + seed.radius) * rotate(UP, child.rest_angle);
    let m = bend_spring(seed, child);
    return m.k_n * d + m.gamma_n * (child.velocity - seed.velocity);
}

fn cell_of(position: vec2<f32>) -> vec2<i32> {
    return vec2<i32>(floor((position + params.hash_margin) * params.inv_cell_size));
}

fn cell_index_of(position: vec2<f32>) -> u32 {
    let cell = cell_of(position);
    let cx = u32(clamp(cell.x, 0, i32(params.grid.x) - 1));
    let cy = u32(clamp(cell.y, 0, i32(params.grid.y) - 1));
    return cx + cy * params.grid.x;
}

@compute @workgroup_size(64)
fn count_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.particle_count {
        return;
    }
    let me = particles[index];
    if me.material == DEAD {
        return;
    }
    atomicAdd(&cell_counts[cell_index_of(me.position)], 1u);
}

@compute @workgroup_size(64)
fn scatter_particles(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.particle_count {
        return;
    }
    let me = particles[index];
    if me.material == DEAD {
        return;
    }
    let position = atomicAdd(&cell_cursor[cell_index_of(me.position)], 1u);

    cell_entries[position] = CellEntry(
        me.position,
        me.velocity,
        me.radius,
        me.packing,
        me.temperature,
        index | (me.material << PACKED_MATERIAL_SHIFT),
    );
}

const PI: f32 = 3.14159265;

fn w_poly6(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    let h4 = h2 * h2;
    let d = h2 - r * r;
    return 4.0 / (PI * h4 * h4) * d * d * d;
}

fn dw_spiky(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    let d = h - r;
    return -30.0 / (PI * h2 * h2 * h) * d * d;
}

fn lap_viscosity(r: f32, h: f32) -> f32 {
    if r >= h {
        return 0.0;
    }
    let h2 = h * h;
    return 40.0 / (PI * h2 * h2 * h) * (h - r);
}

fn pressure_of(m: MaterialParams, packed: f32, temperature: f32) -> f32 {
    let crowding = max(m.pressure_k * (packed - m.rest_packing), 0.0);
    let heat = max(
        1.0 + m.thermal_expansion
            * (temperature - params.temperature_reference) / params.temperature_reference,
        0.0,
    );
    return crowding * heat;
}

fn remembered_tangent(carried: u32, partner: u32) -> vec2<f32> {
    for (var slot = 0u; slot < params.max_contacts; slot++) {
        let record = contacts[carried + slot];
        if record.partner == NO_CONTACT {
            break;
        }
        if record.partner == partner {
            return record.tangent;
        }
    }
    return vec2<f32>(0.0);
}

struct Resolved {
    force: vec2<f32>,
    tangent: vec2<f32>,
};

struct Mixed {
    k_n: f32,
    k_t: f32,
    gamma_n: f32,
    gamma_t: f32,
    mu: f32,
    mass: f32,
};

const TAU: f32 = 6.2831853;

fn spring(freq_n: f32, freq_t: f32, zeta_n: f32, zeta_t: f32, mass: f32, mu: f32) -> Mixed {
    let w_n = TAU * freq_n;
    let w_t = TAU * freq_t;
    var m: Mixed;
    m.k_n = mass * w_n * w_n;
    m.k_t = mass * w_t * w_t;
    m.gamma_n = 2.0 * zeta_n * mass * w_n;
    m.gamma_t = 2.0 * zeta_t * mass * w_t;
    m.mu = mu;
    m.mass = mass;
    return m;
}

fn mix_materials(a: MaterialParams, b: MaterialParams, mass_a: f32, mass_b: f32) -> Mixed {
    let reference = 0.5 * min(mass_a, mass_b);
    return spring(
        sqrt(a.freq_n * b.freq_n),
        sqrt(a.freq_t * b.freq_t),
        sqrt(a.zeta_n * b.zeta_n),
        sqrt(a.zeta_t * b.zeta_t),
        reference,
        min(a.mu, b.mu),
    );
}

fn resolve_contact(
    overlap: f32,
    normal: vec2<f32>,
    relative_velocity: vec2<f32>,
    m: Mixed,
    carried: vec2<f32>,
) -> Resolved {
    let approach = dot(relative_velocity, normal);
    let gamma_n = m.gamma_n / (1.0 + m.gamma_n * params.dt / max(m.mass, 1e-9));
    let gamma_t = m.gamma_t / (1.0 + m.gamma_t * params.dt / max(m.mass, 1e-9));
    let f_n = max(m.k_n * overlap - gamma_n * approach, 0.0);
    let slip = relative_velocity - approach * normal;
    var tangent = carried - dot(carried, normal) * normal;
    tangent += slip * params.dt;
    var f_t = m.k_t * tangent + gamma_t * slip;
    let limit = m.mu * f_n;
    let magnitude = length(f_t);
    if magnitude > limit {
        if limit > 1e-6 && magnitude > 1e-9 {
            f_t = f_t * (limit / magnitude);
            tangent = f_t / max(m.k_t, 1e-6);
        } else {
            f_t = vec2<f32>(0.0);
            tangent = vec2<f32>(0.0);
        }
    }

    var out: Resolved;
    out.force = f_t - f_n * normal;
    out.tangent = tangent;
    return out;
}

fn bond_force(stretch: f32, normal: vec2<f32>, relative_velocity: vec2<f32>, m: Mixed) -> vec2<f32> {
    let gamma = m.gamma_n / (1.0 + m.gamma_n * params.dt / max(m.mass, 1e-9));
    return (m.k_n * stretch + gamma * dot(relative_velocity, normal)) * normal;
}

const NO_ENTRY: u32 = 0xFFFFFFFFu;

fn family_force(me: CellEntry, index: u32, mine: MaterialParams, mass: f32) -> vec2<f32> {
    let own = particles[index];
    let myself = Link(
        me.position, me.velocity, me.radius, mass,
        mine.bond_freq, mine.zeta_n, own.rest_angle, index, own.parent,
    );
    let h = params.smoothing_radius;
    var parent_entry = NO_ENTRY;
    var grandparent_entry = NO_ENTRY;
    var shoot_entries: array<u32, MAX_SHOOTS>;
    var shoot_count = 0u;
    var grandshoot_entries: array<u32, MAX_SHOOTS * MAX_SHOOTS>;
    var grandshoot_count = 0u;
    var force = vec2<f32>(0.0);

    let base = cell_of(me.position);
    let x_lo = max(base.x - params.sweep, 0);
    let x_hi = min(base.x + params.sweep, i32(params.grid.x) - 1);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        let cy = base.y + dy;
        if cy < 0 || cy >= i32(params.grid.y) || x_lo > x_hi {
            continue;
        }
        let row = u32(cy) * params.grid.x;
        let run_end = cell_start[row + u32(x_hi) + 1u];
        for (var entry = cell_start[row + u32(x_lo)]; entry < run_end; entry++) {
            let other = cell_entries[entry];
            let other_index = other.packed & PACKED_INDEX_MASK;
            if other_index == index {
                continue;
            }
            let offset = other.position - me.position;
            let dist_sq = dot(offset, offset);
            if dist_sq >= h * h || dist_sq <= 1e-12 {
                continue;
            }
            let theirs = materials[min(other.packed >> PACKED_MATERIAL_SHIFT, params.material_count - 1u)];
            if theirs.bond_freq <= 0.0 {
                continue;
            }
            let node = particles[other_index];
            if own.grandparent == other_index {
                grandparent_entry = entry;
            }
            if node.grandparent == index && grandshoot_count < MAX_SHOOTS * MAX_SHOOTS {
                grandshoot_entries[grandshoot_count] = entry;
                grandshoot_count += 1u;
            }
            let is_parent = own.parent == other_index;
            let is_child = node.parent == index;
            if is_parent {
                parent_entry = entry;
            } else if is_child && shoot_count < MAX_SHOOTS {
                shoot_entries[shoot_count] = entry;
                shoot_count += 1u;
            }
            if is_parent || is_child {
                let distance = sqrt(dist_sq);
                let touching = me.radius + other.radius;
                let their_mass = theirs.density * other.radius * other.radius;
                let m = spring(
                    sqrt(mine.bond_freq * theirs.bond_freq), 0.0,
                    sqrt(mine.zeta_n * theirs.zeta_n), 0.0,
                    0.5 * min(mass, their_mass), 0.0,
                );
                force += bond_force(distance - touching, offset / distance, other.velocity - me.velocity, m);

                if distance < touching {
                    let mixed = mix_materials(mine, theirs, mass, their_mass);
                    var tangent = vec2<f32>(0.0);
                    if mixed.mu > FRICTIONLESS {
                        let carried = (index * 2u + params.contact_parity) * params.max_contacts;
                        tangent = remembered_tangent(carried, other_index);
                    }
                    force -= resolve_contact(
                        touching - distance,
                        offset / distance,
                        other.velocity - me.velocity,
                        mixed,
                        tangent,
                    ).force;
                }
            }
        }
    }

    let has_parent = parent_entry != NO_ENTRY;
    if has_parent {
        force.y -= params.gravity * (mass - params.ambient_density * me.radius * me.radius);
    }

    var parent: Link;
    if has_parent {
        parent = link_at(parent_entry);
        if own.grandparent == NO_PARENT {
            force -= root_error(parent, myself);
        } else if grandparent_entry != NO_ENTRY {
            let grandparent = link_at(grandparent_entry);
            if in_reach(parent.position, grandparent.position) {
                force -= joint_error(grandparent, parent, myself);
            }
        }
    }
    for (var i = 0u; i < shoot_count; i++) {
        let shoot = link_at(shoot_entries[i]);
        if own.parent == NO_PARENT {
            force += root_error(myself, shoot);
        } else if has_parent && in_reach(shoot.position, parent.position) {
            let error = joint_error(parent, myself, shoot);
            force += error + joint_reaction(parent, myself, shoot, error);
        }
    }
    for (var i = 0u; i < grandshoot_count; i++) {
        let grandshoot = link_at(grandshoot_entries[i]);
        for (var j = 0u; j < shoot_count; j++) {
            let shoot = link_at(shoot_entries[j]);
            if shoot.index == grandshoot.parent && in_reach(shoot.position, grandshoot.position) {
                let error = joint_error(myself, shoot, grandshoot);
                force -= joint_reaction(myself, shoot, grandshoot, error);
            }
        }
    }
    return force;
}

@compute @workgroup_size(64)
fn plant_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry_index = global_id.x;
    if entry_index >= cell_start[params.grid.x * params.grid.y] {
        return;
    }
    let me = cell_entries[entry_index];
    let index = me.packed & PACKED_INDEX_MASK;
    let mine = materials[min(me.packed >> PACKED_MATERIAL_SHIFT, params.material_count - 1u)];
    if mine.bond_freq <= 0.0 || mine.is_static > 0.5 {
        return;
    }
    let mass = mine.density * me.radius * me.radius;
    particles[index].stem_force = family_force(me, index, mine, mass);
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let entry_index = global_id.x;
    if entry_index >= cell_start[params.grid.x * params.grid.y] {
        return;
    }
    let me = cell_entries[entry_index];
    let index = me.packed & PACKED_INDEX_MASK;
    let my_material = me.packed >> PACKED_MATERIAL_SHIFT;
    let mine = materials[min(my_material, params.material_count - 1u)];
    if mine.is_static > 0.5 {
        particles[index].temperature = mine.default_temperature;
        return;
    }
    let my_volume = me.radius * me.radius;
    let mass = mine.density * my_volume;
    var force = vec2<f32>(0.0, params.gravity * (mass - params.ambient_density * my_volume));
    let h = params.smoothing_radius;
    let my_pressure = pressure_of(mine, me.packing, me.temperature);
    var heat = 0.0;
    var shelter = 0.0;
    var packed_next = my_volume * w_poly6(0.0, h);
    let carried = (index * 2u + params.contact_parity) * params.max_contacts;
    let keeping = (index * 2u + 1u - params.contact_parity) * params.max_contacts;
    var kept = 0u;
    let base = cell_of(me.position);
    let x_lo = max(base.x - params.sweep, 0);
    let x_hi = min(base.x + params.sweep, i32(params.grid.x) - 1);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        let cy = base.y + dy;
        if cy < 0 || cy >= i32(params.grid.y) || x_lo > x_hi {
            continue;
        }
        let row = u32(cy) * params.grid.x;
        let run_end = cell_start[row + u32(x_hi) + 1u];

        for (var entry = cell_start[row + u32(x_lo)]; entry < run_end; entry++) {

            let other = cell_entries[entry];
            let other_index = other.packed & PACKED_INDEX_MASK;
            if other_index == index {
                continue;
            }

            let offset = other.position - me.position;
            let dist_sq = dot(offset, offset);
            if dist_sq >= h * h || dist_sq <= 1e-12 {
                continue;
            }
            let distance = sqrt(dist_sq);

            let other_material = other.packed >> PACKED_MATERIAL_SHIFT;
            let theirs = materials[min(other_material, params.material_count - 1u)];
            let their_mass = theirs.density * other.radius * other.radius;
            {
                packed_next += other.radius * other.radius * w_poly6(distance, h);
                let conductivity = 0.5 * (mine.conductivity + theirs.conductivity);
                if conductivity > 0.0 {
                    heat += conductivity * my_volume * (other.radius * other.radius)
                        * (other.temperature - me.temperature)
                        * lap_viscosity(distance, h);
                }

                let their_pressure = pressure_of(theirs, other.packing, other.temperature);

                if my_pressure > 0.0 || their_pressure > 0.0 {
                    let direction = offset / distance;
                    let their_volume = other.radius * other.radius;
                    var pressure_me = my_pressure;
                    var pressure_them = their_pressure;
                    if mine.pressure_k == 0.0 {
                        pressure_me = pressure_of(theirs, me.packing, me.temperature);
                    }
                    if theirs.pressure_k == 0.0 {
                        pressure_them = pressure_of(mine, other.packing, other.temperature);
                    }

                    force += my_volume * their_volume
                        * (pressure_me + pressure_them)
                        * dw_spiky(distance, h) * direction;
                    let viscosity = 0.5 * (mine.viscosity + theirs.viscosity);
                    if viscosity > 0.0 {
                        force += viscosity * my_volume * their_volume
                            * (other.velocity - me.velocity)
                            * lap_viscosity(distance, h);
                    }
                }
            }

            let touching = me.radius + other.radius;
            if distance < touching * params.shelter_reach
                && theirs.thermal_expansion <= 0.0 {
                shelter += other.radius / touching;
            }
            if distance >= touching {
                continue;
            }
            let mixed = mix_materials(mine, theirs, mass, their_mass);
            let gripping = mixed.mu > FRICTIONLESS;
            var spring = vec2<f32>(0.0);
            if gripping {
                spring = remembered_tangent(carried, other_index);
            }
            let resolved = resolve_contact(
                touching - distance,
                offset / distance,
                other.velocity - me.velocity,
                mixed,
                spring,
            );
            force += resolved.force;

            if gripping && kept < params.max_contacts {
                contacts[keeping + kept] =
                    ContactRecord(other_index, 0u, resolved.tangent);
                kept += 1u;
            }
        }
    }

    if kept < params.max_contacts {
        contacts[keeping + kept] = ContactRecord(NO_CONTACT, 0u, vec2<f32>(0.0));
    }

    if mine.bond_freq > 0.0 {
        force += particles[index].stem_force;
    }

    var air = vec2<f32>(params.wind, 0.0);
    let from_gust = me.position - params.gust_centre;
    let gust_distance = length(from_gust);
    if gust_distance < params.gust_radius {
        air += params.gust_velocity
            + params.gust_outflow * from_gust / max(gust_distance, 1e-6);
    }
    let exposure = 1.0 - min(shelter / 3.0, 1.0);
    let relative_air = air - me.velocity;
    let drag = params.air_drag * params.ambient_density * me.radius * exposure
        * length(relative_air);
    force += relative_air * drag / (1.0 + drag * params.dt / mass);

    var velocity = me.velocity + (force / mass) * params.dt;
    let speed = length(velocity);
    if speed > params.max_speed {
        velocity *= params.max_speed / speed;
    }

    let unclamped = me.position + velocity * params.dt;
    let position = clamp(unclamped, vec2<f32>(0.0), params.world);
    velocity = select(velocity, vec2<f32>(0.0), position != unclamped);

    let old_cell = cell_index_of(me.position);
    let new_cell = cell_index_of(position);
    if new_cell != old_cell {
        atomicSub(&cell_counts[old_cell], 1u);
        atomicAdd(&cell_counts[new_cell], 1u);
    }

    var drawn = particles[index].drawn;
    if distance(position, drawn) > params.render_hysteresis {
        drawn = position;
    }
    var temperature = me.temperature
        + heat * params.dt / max(mass * mine.heat_capacity, 1e-9)
        + mine.heat_release * params.dt;
    temperature = clamp(temperature, params.min_temperature, params.max_temperature);

    var material = my_material;
    if mine.becomes_above != NO_TRANSITION && temperature > mine.above_point {
        material = mine.becomes_above;
    } else if mine.becomes_below != NO_TRANSITION && temperature < mine.below_point {
        material = mine.becomes_below;
    }

    if mine.sprouts != NO_TRANSITION {
        let sprout = particles[index].sprout;
        if (sprout & SHOOT_MASK) != 0u && !grown(sprout, mine) {
            particles[index].sprout = sprout + TICK;
        }
    }

    particles[index].position = position;
    particles[index].velocity = velocity;
    particles[index].material = material;
    particles[index].drawn = drawn;
    particles[index].packing = packed_next;
    particles[index].temperature = temperature;
}

const SHOOT_MASK: u32 = 0xFFFFu;
const SIDE_SHOOT: u32 = 0x8000u;
const TICK: u32 = 0x10000u;

fn grown(sprout: u32, m: MaterialParams) -> bool {
    return f32(sprout >> 16u) * params.dt >= m.growth_period;
}

const BRANCH_CHANCE: f32 = 0.06;
const BRANCH_ANGLE: f32 = 1.2;
const MIN_BRANCH: u32 = 6u;
const WANDER: f32 = 0.3;
const UPRIGHT: f32 = 0.35;
const SETTLED_SPEED: f32 = 20.0;
const FLUID_ROOM: f32 = 0.25;

fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn unit_random(seed: u32) -> f32 {
    return f32(hash(seed) >> 8u) / 16777216.0;
}

fn site_blocked(site: vec2<f32>, radius: f32, grower: u32) -> bool {
    let base = cell_of(site);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        for (var dx = -params.sweep; dx <= params.sweep; dx++) {
            let cell = base + vec2<i32>(dx, dy);
            if any(cell < vec2<i32>(0)) || any(cell >= vec2<i32>(params.grid)) {
                continue;
            }
            let c = u32(cell.x) + u32(cell.y) * params.grid.x;
            for (var entry = cell_start[c]; entry < cell_start[c + 1u]; entry++) {
                let other = cell_entries[entry];
                if (other.packed & PACKED_INDEX_MASK) == grower {
                    continue;
                }
                let theirs = materials[min(other.packed >> PACKED_MATERIAL_SHIFT, params.material_count - 1u)];
                let room = select(1.0, FLUID_ROOM, theirs.pressure_k > 0.0);
                if distance(site, other.position) < room * (radius + other.radius) {
                    return true;
                }
            }
        }
    }
    return false;
}

@compute @workgroup_size(64)
fn grow(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.particle_count {
        return;
    }
    let me = particles[index];
    let shoot = me.sprout & SHOOT_MASK;
    if me.material == DEAD || shoot == 0u {
        return;
    }
    let mine = materials[min(me.material, params.material_count - 1u)];
    if mine.sprouts == NO_TRANSITION || !grown(me.sprout, mine) {
        return;
    }

    var stem = UP;
    var rooted = false;
    if me.parent != NO_PARENT {
        let parent = particles[me.parent];
        let theirs = materials[min(parent.material, params.material_count - 1u)];
        let along = me.position - parent.position;
        if parent.material != DEAD && theirs.bond_freq > 0.0 && length(along) > 1e-6 {
            stem = normalize(along);
            rooted = true;
        }
    }
    if !rooted && length(me.velocity) > SETTLED_SPEED {
        return;
    }
    particles[index].sprout = shoot;

    let seed = index ^ bitcast<u32>(me.position.x) ^ (bitcast<u32>(me.position.y) << 1u);
    let side = (shoot & SIDE_SHOOT) != 0u;
    let length_left = shoot & ~SIDE_SHOOT;
    var direction: vec2<f32>;
    if side {
        let turn = select(-BRANCH_ANGLE, BRANCH_ANGLE, unit_random(seed) < 0.5);
        direction = rotate(stem, turn);
    } else {
        let wander = (unit_random(seed) * 2.0 - 1.0) * WANDER;
        direction = normalize(rotate(stem, wander) + UPRIGHT * UP);
    }

    let offshoot = materials[min(mine.sprouts, params.material_count - 1u)];
    let radius = offshoot.radius * (0.9 + 0.2 * unit_random(seed + 1u));
    let site = me.position + direction * (me.radius + radius);
    if any(site < vec2<f32>(radius)) || any(site > params.world - radius)
        || site_blocked(site, radius, index) {
        return;
    }

    let top = atomicSub(&allocator.free_count, 1u);
    if top == 0u || top > arrayLength(&allocator.free_stack) {
        atomicAdd(&allocator.free_count, 1u);
        return;
    }
    let slot = allocator.free_stack[top - 1u];
    atomicMax(&allocator.high_water, slot + 1u);

    let rest_angle = atan2(
        stem.x * direction.y - stem.y * direction.x,
        dot(stem, direction),
    );
    particles[slot] = Particle(
        site,
        me.velocity,
        radius,
        mine.sprouts,
        site,
        0.0,
        me.temperature,
        index,
        me.parent,
        length_left - 1u,
        rest_angle,
        vec2<f32>(0.0),
    );
    let empty = ContactRecord(NO_CONTACT, 0u, vec2<f32>(0.0));
    contacts[(slot * 2u) * params.max_contacts] = empty;
    contacts[(slot * 2u + 1u) * params.max_contacts] = empty;

    var keeps = 0u;
    if !side && length_left > MIN_BRANCH && unit_random(seed + 2u) < BRANCH_CHANCE {
        keeps = SIDE_SHOOT | (length_left / 2u);
    }
    particles[index].sprout = keeps;
}
