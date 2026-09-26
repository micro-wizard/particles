















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
};

struct Particle {
    position: vec2<f32>,
    velocity: vec2<f32>,
    radius: f32,
    material: u32,
    drawn: vec2<f32>,





    packing: f32,


    temperature: f32,
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





















const FRICTIONLESS: f32 = 0.01;

const NO_CONTACT: u32 = 0xFFFFFFFFu;

const DEAD: u32 = 0xFFFFFFFFu;


const NO_TRANSITION: u32 = 0xFFFFFFFFu;













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
        + heat * params.dt / max(mass * mine.heat_capacity, 1e-9);
    temperature = clamp(temperature, params.min_temperature, params.max_temperature);










    var material = my_material;
    if mine.becomes_above != NO_TRANSITION && temperature > mine.above_point {
        material = mine.becomes_above;
    } else if mine.becomes_below != NO_TRANSITION && temperature < mine.below_point {
        material = mine.becomes_below;
    }

    particles[index] = Particle(
        position, velocity, me.radius, material, drawn, packed_next, temperature,
    );
}
