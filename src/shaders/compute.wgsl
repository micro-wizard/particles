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
    rest_temperature: f32,
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
    emits: u32,
    emit_period: f32,
    burn_time: f32,
    _padding: u32,
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
const VANISH: u32 = 0xFFFFFFFEu;
const NO_PARENT: u32 = 0xFFFFFFFFu;
const NO_ENTRY: u32 = 0xFFFFFFFFu;
const NO_SLOT: u32 = 0xFFFFFFFFu;

fn material_params(id: u32) -> MaterialParams {
    return materials[min(id, params.material_count - 1u)];
}

fn mass_of(material: MaterialParams, radius: f32) -> f32 {
    return material.density * radius * radius;
}

fn sorted_entry_count() -> u32 {
    return cell_start[params.grid.x * params.grid.y];
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

struct EntryRange {
    first: u32,
    end: u32,
};

fn entries_in_row(centre_cell: vec2<i32>, dy: i32) -> EntryRange {
    let cy = centre_cell.y + dy;
    let x_lo = max(centre_cell.x - params.sweep, 0);
    let x_hi = min(centre_cell.x + params.sweep, i32(params.grid.x) - 1);
    if cy < 0 || cy >= i32(params.grid.y) || x_lo > x_hi {
        return EntryRange(0u, 0u);
    }
    let row = u32(cy) * params.grid.x;
    return EntryRange(cell_start[row + u32(x_lo)], cell_start[row + u32(x_hi) + 1u]);
}

fn within_smoothing_radius(a: vec2<f32>, b: vec2<f32>) -> bool {
    let d = a - b;
    return dot(d, d) < params.smoothing_radius * params.smoothing_radius;
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

fn carried_contacts_of(index: u32) -> u32 {
    return (index * 2u + params.contact_parity) * params.max_contacts;
}

fn kept_contacts_of(index: u32) -> u32 {
    return (index * 2u + 1u - params.contact_parity) * params.max_contacts;
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

fn end_contact_history(index: u32, kept: u32) {
    if kept < params.max_contacts {
        contacts[kept_contacts_of(index) + kept] = ContactRecord(NO_CONTACT, 0u, vec2<f32>(0.0));
    }
}

fn clear_contact_history(slot: u32) {
    let empty = ContactRecord(NO_CONTACT, 0u, vec2<f32>(0.0));
    contacts[(slot * 2u) * params.max_contacts] = empty;
    contacts[(slot * 2u + 1u) * params.max_contacts] = empty;
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

fn grips(a: MaterialParams, b: MaterialParams) -> bool {
    return min(a.mu, b.mu) > FRICTIONLESS;
}

fn implicit_damping(gamma: f32, mass: f32) -> f32 {
    return gamma / (1.0 + gamma * params.dt / max(mass, 1e-9));
}

fn resolve_contact(
    overlap: f32,
    normal: vec2<f32>,
    relative_velocity: vec2<f32>,
    m: Mixed,
    carried: vec2<f32>,
) -> Resolved {
    let approach = dot(relative_velocity, normal);
    let gamma_n = implicit_damping(m.gamma_n, m.mass);
    let gamma_t = implicit_damping(m.gamma_t, m.mass);
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

struct Grain {
    entry: CellEntry,
    index: u32,
    material_id: u32,
    material: MaterialParams,
    volume: f32,
    mass: f32,
    pressure: f32,
};

fn grain_at(entry_index: u32) -> Grain {
    let entry = cell_entries[entry_index];
    let material_id = entry.packed >> PACKED_MATERIAL_SHIFT;
    let material = material_params(material_id);
    return Grain(
        entry,
        entry.packed & PACKED_INDEX_MASK,
        material_id,
        material,
        entry.radius * entry.radius,
        mass_of(material, entry.radius),
        pressure_of(material, entry.packing, entry.temperature),
    );
}

struct Neighbour {
    entry: CellEntry,
    index: u32,
    material: MaterialParams,
    offset: vec2<f32>,
    distance: f32,
    volume: f32,
    mass: f32,
};

fn is_neighbour(me: Grain, other: CellEntry) -> bool {
    let offset = other.position - me.entry.position;
    let dist_sq = dot(offset, offset);
    let h = params.smoothing_radius;
    return (other.packed & PACKED_INDEX_MASK) != me.index && dist_sq > 1e-12 && dist_sq < h * h;
}

fn neighbour_from(me: Grain, other: CellEntry) -> Neighbour {
    let offset = other.position - me.entry.position;
    let material = material_params(other.packed >> PACKED_MATERIAL_SHIFT);
    return Neighbour(
        other,
        other.packed & PACKED_INDEX_MASK,
        material,
        offset,
        sqrt(dot(offset, offset)),
        other.radius * other.radius,
        mass_of(material, other.radius),
    );
}

fn touching(me: Grain, them: Neighbour) -> bool {
    return them.distance < me.entry.radius + them.entry.radius;
}

fn heat_conducted_from(me: Grain, them: Neighbour) -> f32 {
    let conductivity = 0.5 * (me.material.conductivity + them.material.conductivity);
    if conductivity <= 0.0 {
        return 0.0;
    }
    return conductivity * me.volume * them.volume
        * (them.entry.temperature - me.entry.temperature)
        * lap_viscosity(them.distance, params.smoothing_radius);
}

fn pressure_force_from(me: Grain, them: Neighbour, their_pressure: f32) -> vec2<f32> {
    var pressure_me = me.pressure;
    var pressure_them = their_pressure;
    if me.material.pressure_k == 0.0 {
        pressure_me = pressure_of(them.material, me.entry.packing, me.entry.temperature);
    }
    if them.material.pressure_k == 0.0 {
        pressure_them = pressure_of(me.material, them.entry.packing, them.entry.temperature);
    }
    return me.volume * them.volume
        * (pressure_me + pressure_them)
        * dw_spiky(them.distance, params.smoothing_radius) * (them.offset / them.distance);
}

fn viscous_force_from(me: Grain, them: Neighbour) -> vec2<f32> {
    let viscosity = 0.5 * (me.material.viscosity + them.material.viscosity);
    if viscosity <= 0.0 {
        return vec2<f32>(0.0);
    }
    return viscosity * me.volume * them.volume
        * (them.entry.velocity - me.entry.velocity)
        * lap_viscosity(them.distance, params.smoothing_radius);
}

fn fluid_force_from(me: Grain, them: Neighbour) -> vec2<f32> {
    let their_pressure = pressure_of(them.material, them.entry.packing, them.entry.temperature);
    if !(me.pressure > 0.0 || their_pressure > 0.0) {
        return vec2<f32>(0.0);
    }
    return pressure_force_from(me, them, their_pressure) + viscous_force_from(me, them);
}

fn shelter_from(me: Grain, them: Neighbour) -> f32 {
    let contact_distance = me.entry.radius + them.entry.radius;
    if them.distance >= contact_distance * params.shelter_reach
        || them.material.thermal_expansion > 0.0 {
        return 0.0;
    }
    return them.entry.radius / contact_distance;
}

fn contact_between(me: Grain, them: Neighbour) -> Resolved {
    let mixed = mix_materials(me.material, them.material, me.mass, them.mass);
    var carried = vec2<f32>(0.0);
    if grips(me.material, them.material) {
        carried = remembered_tangent(carried_contacts_of(me.index), them.index);
    }
    return resolve_contact(
        me.entry.radius + them.entry.radius - them.distance,
        them.offset / them.distance,
        them.entry.velocity - me.entry.velocity,
        mixed,
        carried,
    );
}

struct Surroundings {
    force: vec2<f32>,
    heat: f32,
    shelter: f32,
    packing: f32,
    contacts_kept: u32,
};

fn remember_contact(me: Grain, them: Neighbour, tangent: vec2<f32>, around: ptr<function, Surroundings>) {
    if !grips(me.material, them.material) || (*around).contacts_kept >= params.max_contacts {
        return;
    }
    contacts[kept_contacts_of(me.index) + (*around).contacts_kept] =
        ContactRecord(them.index, 0u, tangent);
    (*around).contacts_kept += 1u;
}

fn press_against(me: Grain, them: Neighbour, around: ptr<function, Surroundings>) {
    if !touching(me, them) {
        return;
    }
    let contact = contact_between(me, them);
    (*around).force += contact.force;
    remember_contact(me, them, contact.tangent, around);
}

fn feel_neighbour(me: Grain, other: CellEntry, around: ptr<function, Surroundings>) {
    if !is_neighbour(me, other) {
        return;
    }
    let them = neighbour_from(me, other);
    (*around).packing += them.volume * w_poly6(them.distance, params.smoothing_radius);
    (*around).heat += heat_conducted_from(me, them);
    (*around).force += fluid_force_from(me, them);
    (*around).shelter += shelter_from(me, them);
    press_against(me, them, around);
}

fn sense_surroundings(me: Grain) -> Surroundings {
    var around = Surroundings(
        vec2<f32>(0.0), 0.0, 0.0, me.volume * w_poly6(0.0, params.smoothing_radius), 0u,
    );
    let centre_cell = cell_of(me.entry.position);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        let row = entries_in_row(centre_cell, dy);
        for (var entry = row.first; entry < row.end; entry++) {
            feel_neighbour(me, cell_entries[entry], &around);
        }
    }
    return around;
}

fn weight_of(me: Grain) -> vec2<f32> {
    return vec2<f32>(0.0, params.gravity * (me.mass - params.ambient_density * me.volume));
}

fn stem_force_on(me: Grain) -> vec2<f32> {
    if me.material.bond_freq <= 0.0 {
        return vec2<f32>(0.0);
    }
    return particles[me.index].stem_force;
}

fn air_velocity_at(position: vec2<f32>) -> vec2<f32> {
    var air = vec2<f32>(params.wind, 0.0);
    let from_gust = position - params.gust_centre;
    let gust_distance = length(from_gust);
    if gust_distance < params.gust_radius {
        air += params.gust_velocity
            + params.gust_outflow * from_gust / max(gust_distance, 1e-6);
    }
    return air;
}

fn exposure_to_air(shelter: f32) -> f32 {
    return 1.0 - min(shelter / 3.0, 1.0);
}

fn air_drag_on(me: Grain, shelter: f32) -> vec2<f32> {
    let relative_air = air_velocity_at(me.entry.position) - me.entry.velocity;
    let drag = params.air_drag * params.ambient_density * me.entry.radius
        * exposure_to_air(shelter) * length(relative_air);
    return relative_air * drag / (1.0 + drag * params.dt / me.mass);
}

fn speed_limited(velocity: vec2<f32>) -> vec2<f32> {
    let speed = length(velocity);
    if speed > params.max_speed {
        return velocity * (params.max_speed / speed);
    }
    return velocity;
}

fn move_cell_count(old_position: vec2<f32>, new_position: vec2<f32>) {
    let old_cell = cell_index_of(old_position);
    let new_cell = cell_index_of(new_position);
    if new_cell != old_cell {
        atomicSub(&cell_counts[old_cell], 1u);
        atomicAdd(&cell_counts[new_cell], 1u);
    }
}

fn drawn_after_moving_to(index: u32, position: vec2<f32>) -> vec2<f32> {
    let drawn = particles[index].drawn;
    return select(drawn, position, distance(position, drawn) > params.render_hysteresis);
}

fn move_grain(me: Grain, force: vec2<f32>) -> vec2<f32> {
    var velocity = speed_limited(me.entry.velocity + (force / me.mass) * params.dt);
    let unclamped = me.entry.position + velocity * params.dt;
    let position = clamp(unclamped, vec2<f32>(0.0), params.world);
    velocity = select(velocity, vec2<f32>(0.0), position != unclamped);
    move_cell_count(me.entry.position, position);
    particles[me.index].drawn = drawn_after_moving_to(me.index, position);
    particles[me.index].position = position;
    particles[me.index].velocity = velocity;
    return position;
}

fn material_at_temperature(me: Grain, temperature: f32) -> u32 {
    let m = me.material;
    if m.becomes_above != NO_TRANSITION && temperature > m.above_point {
        return m.becomes_above;
    }
    if m.becomes_below != NO_TRANSITION && temperature < m.below_point {
        return m.becomes_below;
    }
    return me.material_id;
}

const AIR_HEAT_EXCHANGE: f32 = 0.1;

fn heat_exchanged_with_air(me: Grain, shelter: f32) -> f32 {
    return AIR_HEAT_EXCHANGE * exposure_to_air(shelter)
        * (params.rest_temperature - me.entry.temperature)
        * params.dt / max(me.material.heat_capacity, 1e-9);
}

fn release_slot(index: u32) {
    let top = atomicAdd(&allocator.free_count, 1u);
    if top >= arrayLength(&allocator.free_stack) {
        atomicSub(&allocator.free_count, 1u);
        return;
    }
    allocator.free_stack[top] = index;
}

fn retire(index: u32, position: vec2<f32>) {
    particles[index].material = DEAD;
    atomicSub(&cell_counts[cell_index_of(position)], 1u);
    release_slot(index);
}

fn heat_grain(me: Grain, around: Surroundings, position: vec2<f32>) {
    let temperature = clamp(
        me.entry.temperature
            + around.heat * params.dt / max(me.mass * me.material.heat_capacity, 1e-9)
            + heat_exchanged_with_air(me, around.shelter)
            + me.material.heat_release * params.dt,
        params.min_temperature,
        params.max_temperature,
    );
    particles[me.index].temperature = temperature;
    let material = material_at_temperature(me, temperature);
    if material == VANISH {
        retire(me.index, position);
        return;
    }
    particles[me.index].material = material;
    if material != me.material_id && starts_or_stops_burning(me.material, material) {
        particles[me.index].sprout = 0u;
    }
}

fn starts_or_stops_burning(was: MaterialParams, becomes: u32) -> bool {
    return was.emits != NO_TRANSITION || material_params(becomes).emits != NO_TRANSITION;
}

fn offshoot_period(me: Grain, sprout: u32) -> f32 {
    if me.material.emits != NO_TRANSITION {
        return me.material.emit_period;
    }
    if me.material.sprouts != NO_TRANSITION && (sprout & SHOOT_MASK) != 0u {
        return me.material.growth_period;
    }
    return 0.0;
}

fn tick_offshoot_clock(me: Grain) {
    if me.material.sprouts == NO_TRANSITION && me.material.emits == NO_TRANSITION {
        return;
    }
    let sprout = particles[me.index].sprout;
    let period = offshoot_period(me, sprout);
    if period > 0.0 && !period_elapsed(sprout, period) {
        particles[me.index].sprout = sprout + TICK;
    }
}

@compute @workgroup_size(64)
fn solve(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= sorted_entry_count() {
        return;
    }
    let me = grain_at(global_id.x);
    if me.material.is_static > 0.5 {
        particles[me.index].temperature = params.rest_temperature;
        return;
    }
    let around = sense_surroundings(me);
    end_contact_history(me.index, around.contacts_kept);
    let force = around.force + weight_of(me) + stem_force_on(me) + air_drag_on(me, around.shelter);
    let position = move_grain(me, force);
    particles[me.index].packing = around.packing;
    heat_grain(me, around, position);
    tick_offshoot_clock(me);
}

const BEND_SCALE: f32 = 0.15;
const UP: vec2<f32> = vec2<f32>(0.0, -1.0);
const MAX_SHOOTS: u32 = 2u;
const MAX_GRANDSHOOTS: u32 = MAX_SHOOTS * MAX_SHOOTS;

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
    grandparent: u32,
};

fn link_at(entry_index: u32) -> Link {
    let e = cell_entries[entry_index];
    let index = e.packed & PACKED_INDEX_MASK;
    let m = material_params(e.packed >> PACKED_MATERIAL_SHIFT);
    let node = particles[index];
    return Link(
        e.position, e.velocity, e.radius, mass_of(m, e.radius),
        m.bond_freq, m.zeta_n, node.rest_angle, index, node.parent, node.grandparent,
    );
}

fn rotate(v: vec2<f32>, angle: f32) -> vec2<f32> {
    let c = cos(angle);
    let s = sin(angle);
    return vec2<f32>(c * v.x - s * v.y, s * v.x + c * v.y);
}

fn signed_angle(from_direction: vec2<f32>, to_direction: vec2<f32>) -> f32 {
    return atan2(
        from_direction.x * to_direction.y - from_direction.y * to_direction.x,
        dot(from_direction, to_direction),
    );
}

fn bond_between(me: Grain, them: Neighbour) -> vec2<f32> {
    let m = spring(
        sqrt(me.material.bond_freq * them.material.bond_freq), 0.0,
        sqrt(me.material.zeta_n * them.material.zeta_n), 0.0,
        0.5 * min(me.mass, them.mass), 0.0,
    );
    let normal = them.offset / them.distance;
    let stretch = them.distance - (me.entry.radius + them.entry.radius);
    let closing = dot(them.entry.velocity - me.entry.velocity, normal);
    return (m.k_n * stretch + implicit_damping(m.gamma_n, m.mass) * closing) * normal;
}

fn contact_solve_applies(me: Grain, them: Neighbour) -> vec2<f32> {
    if !touching(me, them) {
        return vec2<f32>(0.0);
    }
    return contact_between(me, them).force;
}

struct Family {
    parent: u32,
    grandparent: u32,
    shoots: array<u32, MAX_SHOOTS>,
    shoot_count: u32,
    grandshoots: array<u32, MAX_GRANDSHOOTS>,
    grandshoot_count: u32,
    bond_force: vec2<f32>,
};

fn add_shoot(family: ptr<function, Family>, entry: u32) {
    if (*family).shoot_count < MAX_SHOOTS {
        (*family).shoots[(*family).shoot_count] = entry;
        (*family).shoot_count += 1u;
    }
}

fn add_grandshoot(family: ptr<function, Family>, entry: u32) {
    if (*family).grandshoot_count < MAX_GRANDSHOOTS {
        (*family).grandshoots[(*family).grandshoot_count] = entry;
        (*family).grandshoot_count += 1u;
    }
}

fn meet_relative(me: Grain, myself: Link, entry: u32, family: ptr<function, Family>) {
    let other = cell_entries[entry];
    if !is_neighbour(me, other) {
        return;
    }
    let them = neighbour_from(me, other);
    if them.material.bond_freq <= 0.0 {
        return;
    }
    let their_links = particles[them.index];
    if myself.grandparent == them.index {
        (*family).grandparent = entry;
    }
    if their_links.grandparent == me.index {
        add_grandshoot(family, entry);
    }
    let is_parent = myself.parent == them.index;
    let is_child = their_links.parent == me.index;
    if is_parent {
        (*family).parent = entry;
    } else if is_child {
        add_shoot(family, entry);
    }
    if is_parent || is_child {
        (*family).bond_force += bond_between(me, them) - contact_solve_applies(me, them);
    }
}

fn find_family(me: Grain, myself: Link) -> Family {
    var family: Family;
    family.parent = NO_ENTRY;
    family.grandparent = NO_ENTRY;
    let centre_cell = cell_of(me.entry.position);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        let row = entries_in_row(centre_cell, dy);
        for (var entry = row.first; entry < row.end; entry++) {
            meet_relative(me, myself, entry, &family);
        }
    }
    return family;
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
    m.gamma_n = implicit_damping(m.gamma_n, m.mass);
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

fn root_error(seed: Link, child: Link) -> vec2<f32> {
    let d = (child.position - seed.position)
        - (child.radius + seed.radius) * rotate(UP, child.rest_angle);
    let m = bend_spring(seed, child);
    return m.k_n * d + m.gamma_n * (child.velocity - seed.velocity);
}

fn bend_as_child(myself: Link, family: ptr<function, Family>) -> vec2<f32> {
    if (*family).parent == NO_ENTRY {
        return vec2<f32>(0.0);
    }
    let parent = link_at((*family).parent);
    if myself.grandparent == NO_PARENT {
        return -root_error(parent, myself);
    }
    if (*family).grandparent == NO_ENTRY {
        return vec2<f32>(0.0);
    }
    let grandparent = link_at((*family).grandparent);
    if !within_smoothing_radius(parent.position, grandparent.position) {
        return vec2<f32>(0.0);
    }
    return -joint_error(grandparent, parent, myself);
}

fn bend_shoot(myself: Link, family: ptr<function, Family>, shoot: Link) -> vec2<f32> {
    if myself.parent == NO_PARENT {
        return root_error(myself, shoot);
    }
    if (*family).parent == NO_ENTRY {
        return vec2<f32>(0.0);
    }
    let parent = link_at((*family).parent);
    if !within_smoothing_radius(shoot.position, parent.position) {
        return vec2<f32>(0.0);
    }
    let error = joint_error(parent, myself, shoot);
    return error + joint_reaction(parent, myself, shoot, error);
}

fn bend_as_parent(myself: Link, family: ptr<function, Family>) -> vec2<f32> {
    var force = vec2<f32>(0.0);
    for (var i = 0u; i < (*family).shoot_count; i++) {
        force += bend_shoot(myself, family, link_at((*family).shoots[i]));
    }
    return force;
}

fn shoot_entry_with_index(family: ptr<function, Family>, index: u32) -> u32 {
    for (var i = 0u; i < (*family).shoot_count; i++) {
        let entry = (*family).shoots[i];
        if (cell_entries[entry].packed & PACKED_INDEX_MASK) == index {
            return entry;
        }
    }
    return NO_ENTRY;
}

fn bend_grandshoot(myself: Link, family: ptr<function, Family>, grandshoot: Link) -> vec2<f32> {
    let shoot_entry = shoot_entry_with_index(family, grandshoot.parent);
    if shoot_entry == NO_ENTRY {
        return vec2<f32>(0.0);
    }
    let shoot = link_at(shoot_entry);
    if !within_smoothing_radius(shoot.position, grandshoot.position) {
        return vec2<f32>(0.0);
    }
    let error = joint_error(myself, shoot, grandshoot);
    return -joint_reaction(myself, shoot, grandshoot, error);
}

fn bend_as_grandparent(myself: Link, family: ptr<function, Family>) -> vec2<f32> {
    var force = vec2<f32>(0.0);
    for (var i = 0u; i < (*family).grandshoot_count; i++) {
        force += bend_grandshoot(myself, family, link_at((*family).grandshoots[i]));
    }
    return force;
}

fn weight_carried_by_stem(me: Grain, family: ptr<function, Family>) -> vec2<f32> {
    if (*family).parent == NO_ENTRY {
        return vec2<f32>(0.0);
    }
    return -weight_of(me);
}

@compute @workgroup_size(64)
fn plant_forces(@builtin(global_invocation_id) global_id: vec3<u32>) {
    if global_id.x >= sorted_entry_count() {
        return;
    }
    let me = grain_at(global_id.x);
    if me.material.bond_freq <= 0.0 || me.material.is_static > 0.5 {
        return;
    }
    let myself = link_at(global_id.x);
    var family = find_family(me, myself);
    particles[me.index].stem_force = family.bond_force
        + weight_carried_by_stem(me, &family)
        + bend_as_child(myself, &family)
        + bend_as_parent(myself, &family)
        + bend_as_grandparent(myself, &family);
}

const SHOOT_MASK: u32 = 0xFFFFu;
const SIDE_SHOOT: u32 = 0x8000u;
const TICK: u32 = 0x10000u;
const BRANCH_CHANCE: f32 = 0.06;
const BRANCH_ANGLE: f32 = 1.2;
const MIN_BRANCH: u32 = 6u;
const WANDER: f32 = 0.3;
const UPRIGHT: f32 = 0.35;
const SETTLED_SPEED: f32 = 20.0;
const FLUID_ROOM: f32 = 0.25;

fn period_elapsed(sprout: u32, period: f32) -> bool {
    return f32(sprout >> 16u) * params.dt >= period;
}

fn hash(x: u32) -> u32 {
    var h = x * 747796405u + 2891336453u;
    h = ((h >> ((h >> 28u) + 4u)) ^ h) * 277803737u;
    return (h >> 22u) ^ h;
}

fn unit_random(seed: u32) -> f32 {
    return f32(hash(seed) >> 8u) / 16777216.0;
}

fn entry_crowds_site(other: CellEntry, site: vec2<f32>, radius: f32, grower: u32) -> bool {
    if (other.packed & PACKED_INDEX_MASK) == grower {
        return false;
    }
    let fluid = material_params(other.packed >> PACKED_MATERIAL_SHIFT).pressure_k > 0.0;
    let room = select(1.0, FLUID_ROOM, fluid);
    return distance(site, other.position) < room * (radius + other.radius);
}

fn row_crowds_site(row: EntryRange, site: vec2<f32>, radius: f32, grower: u32) -> bool {
    for (var entry = row.first; entry < row.end; entry++) {
        if entry_crowds_site(cell_entries[entry], site, radius, grower) {
            return true;
        }
    }
    return false;
}

fn site_blocked(site: vec2<f32>, radius: f32, grower: u32) -> bool {
    let centre_cell = cell_of(site);
    for (var dy = -params.sweep; dy <= params.sweep; dy++) {
        if row_crowds_site(entries_in_row(centre_cell, dy), site, radius, grower) {
            return true;
        }
    }
    return false;
}

fn site_outside_world(site: vec2<f32>, radius: f32) -> bool {
    return any(site < vec2<f32>(radius)) || any(site > params.world - radius);
}

fn ready_to_grow(me: Particle) -> bool {
    if me.material == DEAD || (me.sprout & SHOOT_MASK) == 0u {
        return false;
    }
    let mine = material_params(me.material);
    return mine.sprouts != NO_TRANSITION && period_elapsed(me.sprout, mine.growth_period);
}

struct Stem {
    direction: vec2<f32>,
    rooted: bool,
};

fn stem_of(me: Particle) -> Stem {
    if me.parent == NO_PARENT {
        return Stem(UP, false);
    }
    let parent = particles[me.parent];
    let along = me.position - parent.position;
    if parent.material == DEAD || material_params(parent.material).bond_freq <= 0.0
        || length(along) <= 1e-6 {
        return Stem(UP, false);
    }
    return Stem(normalize(along), true);
}

fn growth_direction(stem: vec2<f32>, shoot: u32, seed: u32) -> vec2<f32> {
    if (shoot & SIDE_SHOOT) != 0u {
        return rotate(stem, select(-BRANCH_ANGLE, BRANCH_ANGLE, unit_random(seed) < 0.5));
    }
    let wander = (unit_random(seed) * 2.0 - 1.0) * WANDER;
    return normalize(rotate(stem, wander) + UPRIGHT * UP);
}

fn sprout_left_after_growing(shoot: u32, seed: u32) -> u32 {
    let length_left = shoot & ~SIDE_SHOOT;
    if (shoot & SIDE_SHOOT) != 0u || length_left <= MIN_BRANCH
        || unit_random(seed + 2u) >= BRANCH_CHANCE {
        return 0u;
    }
    return SIDE_SHOOT | (length_left / 2u);
}

fn claim_free_slot() -> u32 {
    let top = atomicSub(&allocator.free_count, 1u);
    if top == 0u || top > arrayLength(&allocator.free_stack) {
        atomicAdd(&allocator.free_count, 1u);
        return NO_SLOT;
    }
    let slot = allocator.free_stack[top - 1u];
    atomicMax(&allocator.high_water, slot + 1u);
    return slot;
}

struct Offshoot {
    site: vec2<f32>,
    radius: f32,
    rest_angle: f32,
};

fn offshoot_node(me: Particle, index: u32, offshoot: Offshoot) -> Particle {
    return Particle(
        offshoot.site,
        me.velocity,
        offshoot.radius,
        material_params(me.material).sprouts,
        offshoot.site,
        0.0,
        me.temperature,
        index,
        me.parent,
        (me.sprout & SHOOT_MASK & ~SIDE_SHOOT) - 1u,
        offshoot.rest_angle,
        vec2<f32>(0.0),
    );
}

fn plan_offshoot(me: Particle, stem: vec2<f32>, seed: u32) -> Offshoot {
    let direction = growth_direction(stem, me.sprout & SHOOT_MASK, seed);
    let offshoot_material = material_params(material_params(me.material).sprouts);
    let radius = offshoot_material.radius * (0.9 + 0.2 * unit_random(seed + 1u));
    return Offshoot(
        me.position + direction * (me.radius + radius),
        radius,
        signed_angle(stem, direction),
    );
}

fn random_seed(index: u32, position: vec2<f32>) -> u32 {
    return index ^ bitcast<u32>(position.x) ^ (bitcast<u32>(position.y) << 1u);
}

fn grow_shoot(index: u32, me: Particle) {
    let stem = stem_of(me);
    if !stem.rooted && length(me.velocity) > SETTLED_SPEED {
        return;
    }
    let shoot = me.sprout & SHOOT_MASK;
    particles[index].sprout = shoot;
    let seed = random_seed(index, me.position);
    let offshoot = plan_offshoot(me, stem.direction, seed);
    if site_outside_world(offshoot.site, offshoot.radius)
        || site_blocked(offshoot.site, offshoot.radius, index) {
        return;
    }
    let slot = claim_free_slot();
    if slot == NO_SLOT {
        return;
    }
    particles[slot] = offshoot_node(me, index, offshoot);
    clear_contact_history(slot);
    particles[index].sprout = sprout_left_after_growing(shoot, seed);
}

const FLAME_ATTEMPTS: u32 = 5u;
const FLAME_SPREAD: f32 = 0.8;
const FLAME_LIFT: f32 = 30.0;
const FLAME_FLICKER: f32 = 15.0;

fn ready_to_emit(me: Particle) -> bool {
    if me.material == DEAD {
        return false;
    }
    let mine = material_params(me.material);
    return mine.emits != NO_TRANSITION && period_elapsed(me.sprout, mine.emit_period);
}

fn flame_site(me: Particle, radius: f32, attempt: u32, side: f32) -> vec2<f32> {
    let lean = f32((attempt + 1u) / 2u) * FLAME_SPREAD * select(side, -side, attempt % 2u == 0u);
    return me.position + rotate(UP, lean) * (me.radius + radius);
}

fn place_flame(me: Particle, site: vec2<f32>, radius: f32, seed: u32) {
    let slot = claim_free_slot();
    if slot == NO_SLOT {
        return;
    }
    let fire = material_params(me.material).emits;
    let flicker = (unit_random(seed + 2u) * 2.0 - 1.0) * FLAME_FLICKER;
    particles[slot] = Particle(
        site,
        me.velocity + vec2<f32>(flicker, -FLAME_LIFT),
        radius,
        fire,
        site,
        0.0,
        material_params(fire).default_temperature,
        NO_PARENT,
        NO_PARENT,
        0u,
        0.0,
        vec2<f32>(0.0),
    );
    clear_contact_history(slot);
}

fn fuel_spent(me: Particle, mine: MaterialParams) -> bool {
    return f32(me.sprout & SHOOT_MASK) * mine.emit_period >= mine.burn_time;
}

fn emit_flame(index: u32, me: Particle) {
    let mine = material_params(me.material);
    if fuel_spent(me, mine) && mine.becomes_above != NO_TRANSITION {
        particles[index].material = mine.becomes_above;
        return;
    }
    particles[index].sprout = (me.sprout & SHOOT_MASK) + 1u;
    let seed = random_seed(index, me.position);
    let fire = material_params(material_params(me.material).emits);
    let radius = fire.radius * (0.9 + 0.2 * unit_random(seed));
    let side = select(-1.0, 1.0, unit_random(seed + 1u) < 0.5);
    for (var attempt = 0u; attempt < FLAME_ATTEMPTS; attempt++) {
        let site = flame_site(me, radius, attempt, side);
        if !site_outside_world(site, radius) && !site_blocked(site, radius, index) {
            place_flame(me, site, radius, seed);
            return;
        }
    }
}

@compute @workgroup_size(64)
fn grow_and_emit(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= params.particle_count {
        return;
    }
    let me = particles[index];
    if ready_to_grow(me) {
        grow_shoot(index, me);
    } else if ready_to_emit(me) {
        emit_flame(index, me);
    }
}
