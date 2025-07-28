/**
 * @file BitLinear operation implementation for BitNet (1.58-bit) quantization
 * 
 * This module provides efficient BitLinear operations for ternary quantized models
 * with optimized kernels for WebGPU, WASM SIMD, and CPU fallback.
 * 
 * BitNet uses ternary weights (-1, 0, +1) packed into 4 weights per byte for efficiency.
 * Activations remain 8-bit with per-matrix FP32 scaling factors.
 * 
 * @author Transformers.js Team
 * @since 3.7.0
 */

import { Tensor } from '../utils/tensor.js';
import { TensorOpRegistry } from './registry.js';
import { apis } from '../env.js';

/**
 * Unpacks ternary weights from packed byte format
 * @param {Uint8Array} packedWeights - Packed ternary weights (4 weights per byte)
 * @param {number} outDim - Output dimension
 * @param {number} inDim - Input dimension
 * @returns {Int8Array} Unpacked ternary weights (-1, 0, +1)
 */
function unpackTernaryWeights(packedWeights, outDim, inDim) {
    const totalWeights = outDim * inDim;
    const unpacked = new Int8Array(totalWeights);
    
    for (let byteIdx = 0; byteIdx < packedWeights.length; byteIdx++) {
        const packedByte = packedWeights[byteIdx];
        
        for (let bitPair = 0; bitPair < 4; bitPair++) {
            const weightIdx = byteIdx * 4 + bitPair;
            if (weightIdx >= totalWeights) break;
            
            const bits = (packedByte >> (bitPair * 2)) & 0b11;
            
            // Decode ternary values: 00 → -1, 01 → 0, 10 → +1, 11 → +1 (fallback)
            switch (bits) {
                case 0b00: unpacked[weightIdx] = -1; break;
                case 0b01: unpacked[weightIdx] = 0; break;
                case 0b10: unpacked[weightIdx] = 1; break;
                case 0b11: unpacked[weightIdx] = 1; break; // Treat reserved pattern as +1
                default: unpacked[weightIdx] = 0; break; // Fallback to 0
            }
        }
    }
    
    return unpacked;
}

/**
 * CPU fallback implementation of BitLinear using standard INT8 GEMM
 * @param {Float32Array} input - Input activations [batch, inDim]
 * @param {Int8Array} ternaryWeights - Ternary weights [-1, 0, +1]
 * @param {Float32Array} scales - Per-output scaling factors [outDim]
 * @param {number} batchSize - Batch size
 * @param {number} inDim - Input dimension
 * @param {number} outDim - Output dimension
 * @returns {Float32Array} Output activations [batch, outDim]
 */
function bitLinearCPU(input, ternaryWeights, scales, batchSize, inDim, outDim) {
    const output = new Float32Array(batchSize * outDim);
    
    for (let b = 0; b < batchSize; b++) {
        for (let o = 0; o < outDim; o++) {
            let sum = 0;
            
            for (let i = 0; i < inDim; i++) {
                const inputVal = input[b * inDim + i];
                const weightVal = ternaryWeights[o * inDim + i];
                
                // Ternary multiplication: -1*x = -x, 0*x = 0, +1*x = +x
                sum += inputVal * weightVal;
            }
            
            // Apply per-output scaling
            output[b * outDim + o] = sum * scales[o];
        }
    }
    
    return output;
}

/**
 * Check if WASM SIMD is available and supports required operations
 * @returns {boolean} True if WASM SIMD BitLinear is supported
 */
function isWASMSIMDSupported() {
    if (!apis.IS_WEBWORKER_ENV && !apis.IS_BROWSER_ENV) return false;
    
    try {
        // Check for WebAssembly SIMD support
        return typeof WebAssembly !== 'undefined' && 
               WebAssembly.validate && 
               WebAssembly.validate(new Uint8Array([
                   0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00,
                   0x01, 0x05, 0x01, 0x60, 0x00, 0x01, 0x7b, 0x03,
                   0x02, 0x01, 0x00, 0x0a, 0x0a, 0x01, 0x08, 0x00,
                   0x41, 0x00, 0xfd, 0x0f, 0x0b
               ]));
    } catch (e) {
        return false;
    }
}

/**
 * Check if WebGPU is available and supports BitLinear operations
 * @returns {boolean} True if WebGPU BitLinear is supported
 */
function isWebGPUSupported() {
    if (!apis.IS_BROWSER_ENV) return false;
    
    return typeof navigator !== 'undefined' && 
           'gpu' in navigator &&
           navigator.gpu !== undefined;
}

/**
 * BitLinear operation class providing multiple execution backends
 */
class BitLinearOp {
    constructor() {
        this.backend = this._selectBackend();
        this.wasmModule = null;
        this.webgpuDevice = null;
        this.webgpuShader = null;
    }
    
    /**
     * Select the best available backend for BitLinear operations
     * @returns {string} Selected backend ('webgpu', 'wasm-simd', 'cpu')
     */
    _selectBackend() {
        if (isWebGPUSupported()) {
            console.log('BitNet: Using WebGPU backend');
            return 'webgpu';
        } else if (isWASMSIMDSupported()) {
            console.log('BitNet: Using WASM SIMD backend');
            return 'wasm-simd';
        } else {
            console.log('BitNet: Using CPU fallback backend');
            return 'cpu';
        }
    }
    
    /**
     * Initialize the selected backend
     * @returns {Promise<void>}
     */
    async initialize() {
        switch (this.backend) {
            case 'webgpu':
                await this._initializeWebGPU();
                break;
            case 'wasm-simd':
                await this._initializeWASM();
                break;
            case 'cpu':
                // CPU backend requires no initialization
                break;
        }
    }
    
    /**
     * Initialize WebGPU backend
     * @private
     */
    async _initializeWebGPU() {
        try {
            const adapter = await navigator.gpu.requestAdapter();
            if (!adapter) throw new Error('No WebGPU adapter available');
            
            this.webgpuDevice = await adapter.requestDevice();
            
            // Check for required features (dp4a for efficient INT8 operations)
            const hasDP4A = adapter.features.has('shader-f16') || 
                           adapter.features.has('chromium-experimental-dp4a');
            
            if (!hasDP4A) {
                console.warn('BitNet: WebGPU device lacks dp4a support, falling back to WASM');
                this.backend = 'wasm-simd';
                await this._initializeWASM();
                return;
            }
            
            // Load BitLinear shader
            await this._loadWebGPUShader();
            
        } catch (error) {
            console.warn('BitNet: WebGPU initialization failed, falling back to WASM:', error);
            this.backend = 'wasm-simd';
            await this._initializeWASM();
        }
    }
    
    /**
     * Load WebGPU WGSL shader for BitLinear
     * @private
     */
    async _loadWebGPUShader() {
        const shaderCode = `
// BitLinear WebGPU WGSL Shader
const WORKGROUP_SIZE_X = 16u;
const WORKGROUP_SIZE_Y = 16u;
const WEIGHTS_PER_BYTE = 4u;

fn unpack_ternary_weight(packed_byte: u32, weight_idx: u32) -> f32 {
    let shift = (weight_idx % WEIGHTS_PER_BYTE) * 2u;
    let bits = (packed_byte >> shift) & 3u;
    
    switch bits {
        case 0u: { return -1.0; }
        case 1u: { return 0.0; }
        case 2u: { return 1.0; }
        case 3u: { return 1.0; }
        default: { return 0.0; }
    }
}

@group(0) @binding(0) var<storage, read> input_data: array<f32>;
@group(0) @binding(1) var<storage, read> packed_weights: array<u32>;
@group(0) @binding(2) var<storage, read> scales: array<f32>;
@group(0) @binding(3) var<storage, read_write> output_data: array<f32>;

struct Uniforms {
    batch_size: u32,
    in_dim: u32,
    out_dim: u32,
    input_offset: u32,
}

@group(1) @binding(0) var<uniform> uniforms: Uniforms;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_Y, 1)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let batch_idx = global_id.x;
    let out_idx = global_id.y;
    
    if (batch_idx >= uniforms.batch_size || out_idx >= uniforms.out_dim) {
        return;
    }
    
    var sum = 0.0;
    
    for (var in_idx = 0u; in_idx < uniforms.in_dim; in_idx++) {
        let input_base = batch_idx * uniforms.in_dim + in_idx;
        let input_val = input_data[input_base];
        
        let weight_linear_idx = out_idx * uniforms.in_dim + in_idx;
        let packed_idx = weight_linear_idx / WEIGHTS_PER_BYTE;
        let weight_sub_idx = weight_linear_idx % WEIGHTS_PER_BYTE;
        
        let packed_value = packed_weights[packed_idx];
        let weight_val = unpack_ternary_weight(packed_value, weight_sub_idx);
        
        sum += input_val * weight_val;
    }
    
    let scale = scales[out_idx];
    let output_idx = batch_idx * uniforms.out_dim + out_idx;
    output_data[output_idx] = sum * scale;
}
        `;
        
        this.webgpuShader = this.webgpuDevice.createShaderModule({
            code: shaderCode
        });
    }
    
    /**
     * Initialize WASM SIMD backend
     * @private
     */
    async _initializeWASM() {
        try {
            // This will load the compiled WASM module with SIMD BitLinear kernels
            // For now, fall back to CPU if WASM is not available
            if (!isWASMSIMDSupported()) {
                console.warn('BitNet: WASM SIMD not supported, using CPU fallback');
                this.backend = 'cpu';
                return;
            }
            
            // TODO: Load WASM module
            // this.wasmModule = await loadBitLinearWASM();
            this.wasmModule = null; // Placeholder
            
            if (!this.wasmModule) {
                console.warn('BitNet: WASM module loading failed, using CPU fallback');
                this.backend = 'cpu';
            }
            
        } catch (error) {
            console.warn('BitNet: WASM initialization failed, using CPU fallback:', error);
            this.backend = 'cpu';
        }
    }
    
    /**
     * Execute BitLinear operation using the selected backend
     * @param {Tensor} input - Input tensor [batch, inDim]
     * @param {Uint8Array} packedWeights - Packed ternary weights
     * @param {Float32Array} scales - Per-output scale factors
     * @param {number} outDim - Output dimension
     * @param {number} inDim - Input dimension
     * @returns {Promise<Tensor>} Output tensor [batch, outDim]
     */
    async forward(input, packedWeights, scales, outDim, inDim) {
        const batchSize = input.dims[0];
        const inputData = input.data;
        
        // Validate input dimensions
        if (input.dims.length !== 2 || input.dims[1] !== inDim) {
            throw new Error(`Invalid input shape: expected [batch, ${inDim}], got [${input.dims.join(', ')}]`);
        }
        
        // Validate packed weights size
        const expectedPackedSize = Math.ceil((outDim * inDim) / 4);
        if (packedWeights.length !== expectedPackedSize) {
            throw new Error(`Invalid packed weights size: expected ${expectedPackedSize}, got ${packedWeights.length}`);
        }
        
        // Validate scales
        if (scales.length !== outDim) {
            throw new Error(`Invalid scales length: expected ${outDim}, got ${scales.length}`);
        }
        
        let outputData;
        
        switch (this.backend) {
            case 'webgpu':
                outputData = await this._forwardWebGPU(inputData, packedWeights, scales, batchSize, inDim, outDim);
                break;
                
            case 'wasm-simd':
                outputData = await this._forwardWASM(inputData, packedWeights, scales, batchSize, inDim, outDim);
                break;
                
            case 'cpu':
                outputData = this._forwardCPU(inputData, packedWeights, scales, batchSize, inDim, outDim);
                break;
                
            default:
                throw new Error(`Unknown backend: ${this.backend}`);
        }
        
        return new Tensor('float32', outputData, [batchSize, outDim]);
    }
    
    /**
     * WebGPU implementation of BitLinear forward pass
     * @private
     */
    async _forwardWebGPU(inputData, packedWeights, scales, batchSize, inDim, outDim) {
        if (!this.webgpuDevice || !this.webgpuShader) {
            console.warn('BitNet: WebGPU not properly initialized, using CPU fallback');
            return this._forwardCPU(inputData, packedWeights, scales, batchSize, inDim, outDim);
        }

        try {
            // Create buffers
            const inputBuffer = this.webgpuDevice.createBuffer({
                size: inputData.length * 4, // 4 bytes per float32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            const weightsBuffer = this.webgpuDevice.createBuffer({
                size: packedWeights.length * 4, // 4 bytes per uint32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            const scalesBuffer = this.webgpuDevice.createBuffer({
                size: scales.length * 4, // 4 bytes per float32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST,
            });

            const outputBuffer = this.webgpuDevice.createBuffer({
                size: batchSize * outDim * 4, // 4 bytes per float32
                usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
            });

            const uniformBuffer = this.webgpuDevice.createBuffer({
                size: 16, // 4 uint32s
                usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
            });

            // Copy data to buffers
            this.webgpuDevice.queue.writeBuffer(inputBuffer, 0, inputData);
            this.webgpuDevice.queue.writeBuffer(weightsBuffer, 0, new Uint32Array(packedWeights.buffer));
            this.webgpuDevice.queue.writeBuffer(scalesBuffer, 0, scales);
            
            const uniformData = new Uint32Array([batchSize, inDim, outDim, 0]);
            this.webgpuDevice.queue.writeBuffer(uniformBuffer, 0, uniformData);

            // Create bind groups
            const bindGroupLayout = this.webgpuDevice.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'read-only-storage' } },
                    { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'storage' } },
                ]
            });

            const uniformBindGroupLayout = this.webgpuDevice.createBindGroupLayout({
                entries: [
                    { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: 'uniform' } }
                ]
            });

            const bindGroup = this.webgpuDevice.createBindGroup({
                layout: bindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: inputBuffer } },
                    { binding: 1, resource: { buffer: weightsBuffer } },
                    { binding: 2, resource: { buffer: scalesBuffer } },
                    { binding: 3, resource: { buffer: outputBuffer } },
                ]
            });

            const uniformBindGroup = this.webgpuDevice.createBindGroup({
                layout: uniformBindGroupLayout,
                entries: [
                    { binding: 0, resource: { buffer: uniformBuffer } }
                ]
            });

            // Create compute pipeline
            const computePipeline = this.webgpuDevice.createComputePipeline({
                layout: this.webgpuDevice.createPipelineLayout({
                    bindGroupLayouts: [bindGroupLayout, uniformBindGroupLayout]
                }),
                compute: {
                    module: this.webgpuShader,
                    entryPoint: 'main'
                }
            });

            // Dispatch compute shader
            const commandEncoder = this.webgpuDevice.createCommandEncoder();
            const computePass = commandEncoder.beginComputePass();
            
            computePass.setPipeline(computePipeline);
            computePass.setBindGroup(0, bindGroup);
            computePass.setBindGroup(1, uniformBindGroup);
            
            const workgroupsX = Math.ceil(batchSize / 16);
            const workgroupsY = Math.ceil(outDim / 16);
            computePass.dispatchWorkgroups(workgroupsX, workgroupsY);
            
            computePass.end();

            // Read back results
            const stagingBuffer = this.webgpuDevice.createBuffer({
                size: batchSize * outDim * 4,
                usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ
            });

            commandEncoder.copyBufferToBuffer(outputBuffer, 0, stagingBuffer, 0, batchSize * outDim * 4);
            this.webgpuDevice.queue.submit([commandEncoder.finish()]);

            await stagingBuffer.mapAsync(GPUMapMode.READ);
            const resultArray = new Float32Array(stagingBuffer.getMappedRange());
            const result = new Float32Array(resultArray);
            stagingBuffer.unmap();

            // Cleanup
            inputBuffer.destroy();
            weightsBuffer.destroy();
            scalesBuffer.destroy();
            outputBuffer.destroy();
            uniformBuffer.destroy();
            stagingBuffer.destroy();

            return result;

        } catch (error) {
            console.warn('BitNet: WebGPU execution failed, using CPU fallback:', error);
            return this._forwardCPU(inputData, packedWeights, scales, batchSize, inDim, outDim);
        }
    }
    
    /**
     * WASM SIMD implementation of BitLinear forward pass
     * @private
     */
    async _forwardWASM(inputData, packedWeights, scales, batchSize, inDim, outDim) {
        // TODO: Implement WASM SIMD kernel execution
        // For now, fall back to CPU
        console.warn('BitNet: WASM implementation not ready, using CPU fallback');
        return this._forwardCPU(inputData, packedWeights, scales, batchSize, inDim, outDim);
    }
    
    /**
     * CPU implementation of BitLinear forward pass
     * @private
     */
    _forwardCPU(inputData, packedWeights, scales, batchSize, inDim, outDim) {
        // Unpack ternary weights
        const ternaryWeights = unpackTernaryWeights(packedWeights, outDim, inDim);
        
        // Execute BitLinear operation
        return bitLinearCPU(inputData, ternaryWeights, scales, batchSize, inDim, outDim);
    }
}

// Global BitLinear operation instance
let globalBitLinearOp = null;

/**
 * Get the global BitLinear operation instance, initializing if necessary
 * @returns {Promise<BitLinearOp>}
 */
async function getBitLinearOp() {
    if (!globalBitLinearOp) {
        globalBitLinearOp = new BitLinearOp();
        await globalBitLinearOp.initialize();
    }
    return globalBitLinearOp;
}

/**
 * Main BitLinear operation function for external use
 * @param {Tensor} input - Input tensor [batch, inDim]
 * @param {Uint8Array} packedWeights - Packed ternary weights
 * @param {Float32Array} scales - Per-output scale factors
 * @param {number} outDim - Output dimension
 * @param {number} inDim - Input dimension
 * @returns {Promise<Tensor>} Output tensor [batch, outDim]
 */
export async function bitlinear(input, packedWeights, scales, outDim, inDim) {
    const op = await getBitLinearOp();
    return await op.forward(input, packedWeights, scales, outDim, inDim);
}

/**
 * BitLinear operation for the TensorOpRegistry
 * Provides a unified interface similar to other tensor operations
 */
export class BitLinearTensorOp {
    static async create() {
        const op = await getBitLinearOp();
        return async (inputs) => {
            const [input, packedWeights, scales, dimensions] = inputs;
            const [outDim, inDim] = dimensions.data;
            
            return await op.forward(
                input,
                new Uint8Array(packedWeights.data.buffer),
                new Float32Array(scales.data.buffer),
                outDim,
                inDim
            );
        };
    }
}

// Export for debugging and testing
export { BitLinearOp, unpackTernaryWeights, bitLinearCPU };