// BitLinear WebGPU WGSL Shader
// High-performance GPU kernel for BitNet ternary quantized matrix multiplication
// 
// Features:
// - Vectorized ternary weight unpacking
// - Optimized matrix multiplication with packed weights  
// - Support for different matrix sizes
// - Efficient workgroup memory usage

// Workgroup size optimized for most GPUs
const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WEIGHTS_PER_BYTE = 4u;

// Decode ternary weights from packed format
// Each byte contains 4 ternary weights (2 bits each)
// 00 -> -1, 01 -> 0, 10 -> +1, 11 -> +1 (fallback)
fn unpack_ternary_weight(packed_byte: u32, weight_idx: u32) -> f32 {
    let shift = (weight_idx % WEIGHTS_PER_BYTE) * 2u;
    let bits = (packed_byte >> shift) & 3u;
    
    switch bits {
        case 0u: { return -1.0; }  // 00 -> -1
        case 1u: { return 0.0; }   // 01 -> 0  
        case 2u: { return 1.0; }   // 10 -> +1
        case 3u: { return 1.0; }   // 11 -> +1 (fallback)
        default: { return 0.0; }
    }
}

// BitLinear matrix multiplication kernel
// Computes: output = input @ ternary_weights.T * scales
@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

// Uniform parameters
struct Uniforms {
    batch_size: u32,
    in_dim: u32,
    out_dim: u32,
    input_offset: u32,
}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

// Shared memory for workgroup optimization
var<workgroup> shared_input: array<f32, 256>;
var<workgroup> shared_weights: array<f32, 256>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn bitlinear_main(@builtin(global_invocation_id) global_id: vec3<u32>,
                  @builtin(local_invocation_id) local_id: vec3<u32>,
                  @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    
    let batch_idx = global_id.x;
    let out_idx = global_id.y;
    
    // Bounds check
    if (batch_idx >= uniforms.batch_size || out_idx >= uniforms.out_dim) {
        return;
    }
    
    // Compute matrix multiplication for this output element
    var sum = 0.0;
    
    // Process input features in chunks for better memory access
    let chunk_size = 32u;
    let num_chunks = (uniforms.in_dim + chunk_size - 1u) / chunk_size;
    
    for (var chunk = 0u; chunk < num_chunks; chunk++) {
        let chunk_start = chunk * chunk_size;
        let chunk_end = min(chunk_start + chunk_size, uniforms.in_dim);
        
        // Load input chunk into shared memory (if within workgroup bounds)
        let local_thread_id = local_id.x * WORKGROUP_SIZE_Y + local_id.y;
        for (var i = local_thread_id; i < chunk_size && chunk_start + i < uniforms.in_dim; i += WORKGROUP_SIZE_X * WORKGROUP_SIZE_Y) {
            let global_in_idx = chunk_start + i;
            let input_base = batch_idx * uniforms.in_dim + global_in_idx;
            shared_input[i] = input_data[input_base];
        }
        
        workgroupBarrier();
        
        // Process this chunk
        for (var in_idx = chunk_start; in_idx < chunk_end; in_idx++) {
            // Get input value from shared memory
            let local_in_idx = in_idx - chunk_start;
            let input_val = shared_input[local_in_idx];
            
            // Calculate packed weight index
            // Weights are stored in row-major order: [out_dim, in_dim]
            let weight_linear_idx = out_idx * uniforms.in_dim + in_idx;
            let packed_idx = weight_linear_idx / WEIGHTS_PER_BYTE;
            let weight_sub_idx = weight_linear_idx % WEIGHTS_PER_BYTE;
            
            // Unpack ternary weight
            let packed_value = packed_weights[packed_idx];
            let weight_val = unpack_ternary_weight(packed_value, weight_sub_idx);
            
            // Accumulate: input * weight
            sum += input_val * weight_val;
        }
        
        workgroupBarrier();
    }
    
    // Apply per-output scaling factor
    let scale = scales[out_idx];
    let final_output = sum * scale;
    
    // Write result
    let output_idx = batch_idx * uniforms.out_dim + out_idx;
    output_data[output_idx] = final_output;
}

// Alternative kernel optimized for smaller matrices
@compute @workgroup_size(32, 1, 1)
fn bitlinear_small(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let linear_id = global_id.x;
    let total_outputs = uniforms.batch_size * uniforms.out_dim;
    
    if (linear_id >= total_outputs) {
        return;
    }
    
    let batch_idx = linear_id / uniforms.out_dim;
    let out_idx = linear_id % uniforms.out_dim;
    
    var sum = 0.0;
    
    // Simple linear processing - good for small matrices
    for (var in_idx = 0u; in_idx < uniforms.in_dim; in_idx++) {
        // Get input
        let input_base = batch_idx * uniforms.in_dim + in_idx;
        let input_val = input_data[input_base];
        
        // Get packed weight
        let weight_linear_idx = out_idx * uniforms.in_dim + in_idx;
        let packed_idx = weight_linear_idx / WEIGHTS_PER_BYTE;
        let weight_sub_idx = weight_linear_idx % WEIGHTS_PER_BYTE;
        
        let packed_value = packed_weights[packed_idx];
        let weight_val = unpack_ternary_weight(packed_value, weight_sub_idx);
        
        sum += input_val * weight_val;
    }
    
    // Apply scaling and store
    let scale = scales[out_idx];
    output_data[linear_id] = sum * scale;
}

// Specialized kernel for very large matrices with dp4a support
// Uses packed 4x INT8 operations for maximum throughput
@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)  
fn bitlinear_dp4a(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_idx = global_id.y;
    
    if (batch_idx >= uniforms.batch_size || out_idx >= uniforms.out_dim) {
        return;
    }
    
    var sum = 0.0;
    
    // Process 4 inputs at a time using vector operations
    let vec4_dim = uniforms.in_dim / 4u;
    let remainder = uniforms.in_dim % 4u;
    
    // Vectorized processing
    for (var vec_idx = 0u; vec_idx < vec4_dim; vec_idx++) {
        let base_in_idx = vec_idx * 4u;
        
        // Load 4 input values
        let input_base = batch_idx * uniforms.in_dim + base_in_idx;
        let input_vec = vec4<f32>(
            input_data[input_base],
            input_data[input_base + 1u],
            input_data[input_base + 2u], 
            input_data[input_base + 3u]
        );
        
        // Load and unpack 4 weight values (from 1 packed byte)
        let weight_linear_idx = out_idx * uniforms.in_dim + base_in_idx;
        let packed_idx = weight_linear_idx / WEIGHTS_PER_BYTE;
        let packed_byte = packed_weights[packed_idx];
        
        let weight_vec = vec4<f32>(
            unpack_ternary_weight(packed_byte, 0u),
            unpack_ternary_weight(packed_byte, 1u),
            unpack_ternary_weight(packed_byte, 2u),
            unpack_ternary_weight(packed_byte, 3u)
        );
        
        // Vectorized multiply-accumulate
        sum += dot(input_vec, weight_vec);
    }
    
    // Handle remainder elements
    for (var i = 0u; i < remainder; i++) {
        let in_idx = vec4_dim * 4u + i;
        let input_base = batch_idx * uniforms.in_dim + in_idx;
        let input_val = input_data[input_base];
        
        let weight_linear_idx = out_idx * uniforms.in_dim + in_idx;
        let packed_idx = weight_linear_idx / WEIGHTS_PER_BYTE;
        let weight_sub_idx = weight_linear_idx % WEIGHTS_PER_BYTE;
        
        let packed_value = packed_weights[packed_idx];
        let weight_val = unpack_ternary_weight(packed_value, weight_sub_idx);
        
        sum += input_val * weight_val;
    }
    
    // Apply scaling and store
    let scale = scales[out_idx];
    let output_idx = batch_idx * uniforms.out_dim + out_idx;
    output_data[output_idx] = sum * scale;
}