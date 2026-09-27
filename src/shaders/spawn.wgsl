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

struct SpawnRequest {
    position: vec2<f32>,
    radius: f32,
    material: u32,


    temperature: f32,
    budget: u32,
};

struct SpawnBatch {
    count: u32,
    slot_bound: u32,
    max_contacts: u32,
    _padding: u32,
    lo: vec2<f32>,
    hi: vec2<f32>,
    requests: array<SpawnRequest>,
};

struct SlotAllocator {
    free_count: atomic<u32>,
    high_water: atomic<u32>,
    free_stack: array<u32>,
};

struct BrushStroke {
    centre: vec2<f32>,
    radius: f32,
    delta: f32,
    slot_bound: u32,
    protected_below: u32,
};

@group(0) @binding(0) var<storage, read_write> particles: array<Particle>;
@group(0) @binding(1) var<storage, read_write> contacts: array<ContactRecord>;
@group(0) @binding(2) var<storage, read> batch: SpawnBatch;
@group(0) @binding(3) var<storage, read_write> allocator: SlotAllocator;
@group(0) @binding(5) var<storage, read> stroke: BrushStroke;
@group(0) @binding(6) var<storage, read_write> blocked: array<atomic<u32>>;

const DEAD: u32 = 0xFFFFFFFFu;
const NO_CONTACT: u32 = 0xFFFFFFFFu;
const NO_PARENT: u32 = 0xFFFFFFFFu;

fn overlaps(a: vec2<f32>, a_radius: f32, b: vec2<f32>, b_radius: f32) -> bool {
    return distance(a, b) < a_radius + b_radius;
}

@compute @workgroup_size(64)
fn check_spawns(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= batch.slot_bound {
        return;
    }
    let grain = particles[index];
    if grain.material == DEAD {
        return;
    }
    if any(grain.position + grain.radius < batch.lo) || any(grain.position - grain.radius > batch.hi) {
        return;
    }
    for (var i = 0u; i < batch.count; i++) {
        let request = batch.requests[i];
        if overlaps(grain.position, grain.radius, request.position, request.radius) {
            atomicStore(&blocked[i], 1u);
        }
    }
}

@compute @workgroup_size(64)
fn commit_spawns(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let i = global_id.x;
    if i >= batch.count || atomicLoad(&blocked[i]) != 0u {
        return;
    }

    let top = atomicSub(&allocator.free_count, 1u);
    if top == 0u || top > arrayLength(&allocator.free_stack) {
        atomicAdd(&allocator.free_count, 1u);
        return;
    }
    let slot = allocator.free_stack[top - 1u];
    atomicMax(&allocator.high_water, slot + 1u);

    let request = batch.requests[i];
    particles[slot] = Particle(
        request.position,
        vec2<f32>(0.0),
        request.radius,
        request.material,
        request.position,
        0.0,
        request.temperature,
        NO_PARENT,
        NO_PARENT,
        request.budget,
        0.0,
        vec2<f32>(0.0),
    );
    let empty = ContactRecord(NO_CONTACT, 0u, vec2<f32>(0.0));
    contacts[(slot * 2u) * batch.max_contacts] = empty;
    contacts[(slot * 2u + 1u) * batch.max_contacts] = empty;
}

fn under_stroke(index: u32) -> bool {
    if index < stroke.protected_below || index >= stroke.slot_bound {
        return false;
    }
    let grain = particles[index];
    return grain.material != DEAD && distance(grain.position, stroke.centre) <= stroke.radius;
}

@compute @workgroup_size(64)
fn heat_brush(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if under_stroke(index) {
        particles[index].temperature += stroke.delta;
    }
}

fn release_slot(index: u32) {
    let top = atomicAdd(&allocator.free_count, 1u);
    if top >= arrayLength(&allocator.free_stack) {
        atomicSub(&allocator.free_count, 1u);
        return;
    }
    allocator.free_stack[top] = index;
}

@compute @workgroup_size(64)
fn erase_brush(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if under_stroke(index) {
        particles[index].material = DEAD;
        release_slot(index);
    }
}

fn erased(relative: u32) -> bool {
    return relative != NO_PARENT && particles[relative].material == DEAD;
}

@compute @workgroup_size(64)
fn forget_erased_relatives(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let index = global_id.x;
    if index >= stroke.slot_bound || particles[index].material == DEAD {
        return;
    }
    if erased(particles[index].parent) {
        particles[index].parent = NO_PARENT;
    }
    if erased(particles[index].grandparent) {
        particles[index].grandparent = NO_PARENT;
    }
}
