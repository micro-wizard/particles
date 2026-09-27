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

@compute @workgroup_size(256)
fn scan_cells(@builtin(local_invocation_id) local_id: vec3<u32>) {
    let t = local_id.x;
    let cell_count = params.grid.x * params.grid.y;
    let vec4_count = (cell_count + 3u) / 4u;
    let per_thread = (vec4_count + SCAN_WORKGROUP - 1u) / SCAN_WORKGROUP;
    let first = t * per_thread;

    var sum = 0u;
    for (var i = 0u; i < per_thread; i++) {
        let v = first + i;
        if v < vec4_count {
            let n = cell_counts[v];
            sum += n.x + n.y + n.z + n.w;
        }
    }
    partials[t] = sum;
    workgroupBarrier();

    let group = t / SCAN_GROUP;
    if t % SCAN_GROUP == 0u {
        var total = 0u;
        for (var k = 0u; k < SCAN_GROUP; k++) {
            total += partials[t + k];
        }
        group_totals[group] = total;
    }
    workgroupBarrier();

    var running = 0u;
    for (var g = 0u; g < group; g++) {
        running += group_totals[g];
    }
    for (var k = group * SCAN_GROUP; k < t; k++) {
        running += partials[k];
    }

    for (var i = 0u; i < per_thread; i++) {
        let v = first + i;
        if v < vec4_count {
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
    }
    if t == SCAN_WORKGROUP - 1u {
        cell_start[vec4_count] = vec4<u32>(running);
    }
}
