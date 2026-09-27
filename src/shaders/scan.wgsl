struct ScanParams {
    world: vec2<f32>,
    grid: vec2<u32>,
};

@group(0) @binding(1) var<storage, read_write> cell_counts: array<vec4<u32>>;
@group(0) @binding(2) var<storage, read_write> cell_start: array<vec4<u32>>;
@group(0) @binding(3) var<storage, read_write> cell_cursor: array<vec4<u32>>;
@group(0) @binding(5) var<uniform> params: ScanParams;
const SCAN_WORKGROUP: u32 = 256u;
const SCAN_GROUP: u32 = 16u;

var<workgroup> partials: array<u32, SCAN_WORKGROUP>;
var<workgroup> group_totals: array<u32, SCAN_GROUP>;

struct CellRun {
    first: u32,
    end: u32,
};

fn cells_of_thread(t: u32, vec4_count: u32) -> CellRun {
    let per_thread = (vec4_count + SCAN_WORKGROUP - 1u) / SCAN_WORKGROUP;
    let first = t * per_thread;
    return CellRun(first, min(first + per_thread, vec4_count));
}

fn particles_in(cells: CellRun) -> u32 {
    var sum = 0u;
    for (var v = cells.first; v < cells.end; v++) {
        let n = cell_counts[v];
        sum += n.x + n.y + n.z + n.w;
    }
    return sum;
}

fn sum_of_group_partials(leader: u32) -> u32 {
    var total = 0u;
    for (var k = 0u; k < SCAN_GROUP; k++) {
        total += partials[leader + k];
    }
    return total;
}

fn particles_before_thread(t: u32) -> u32 {
    let group = t / SCAN_GROUP;
    var running = 0u;
    for (var g = 0u; g < group; g++) {
        running += group_totals[g];
    }
    for (var k = group * SCAN_GROUP; k < t; k++) {
        running += partials[k];
    }
    return running;
}

fn write_cell_starts(cells: CellRun, first_start: u32) -> u32 {
    var running = first_start;
    for (var v = cells.first; v < cells.end; v++) {
        let n = cell_counts[v];
        let starts = vec4<u32>(
            running,
            running + n.x,
            running + n.x + n.y,
            running + n.x + n.y + n.z,
        );
        cell_start[v] = starts;
        cell_cursor[v] = starts;
        running += n.x + n.y + n.z + n.w;
    }
    return running;
}

@compute @workgroup_size(256)
fn scan_cells(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let t = local_id.x;
    let vec4_count = (params.grid.x * params.grid.y + 3u) / 4u;
    let cells = cells_of_thread(t, vec4_count);

    partials[t] = particles_in(cells);
    workgroupBarrier();

    if t % SCAN_GROUP == 0u {
        group_totals[t / SCAN_GROUP] = sum_of_group_partials(t);
    }
    workgroupBarrier();

    let total = write_cell_starts(cells, particles_before_thread(t));
    if t == SCAN_WORKGROUP - 1u {
        cell_start[vec4_count] = vec4<u32>(total);
    }
}
