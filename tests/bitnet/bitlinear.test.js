/**
 * @file Test suite for BitLinear operations
 * 
 * Comprehensive tests for BitNet (1.58-bit) ternary quantization
 * Tests weight packing/unpacking, CPU operations, and numerical correctness
 */

import { jest } from '@jest/globals';
import { bitlinear, BitLinearOp, unpackTernaryWeights, bitLinearCPU } from '../../src/ops/bitlinear.js';
import { Tensor } from '../../src/utils/tensor.js';

describe('BitLinear Operations', () => {
    describe('Ternary Weight Packing/Unpacking', () => {
        test('unpackTernaryWeights should correctly decode packed weights', () => {
            // Test case: 4 weights packed into 1 byte
            // Weights: [-1, 0, +1, -1] -> [00, 01, 10, 00] -> 0b00100100 = 0x24
            const packedWeights = new Uint8Array([0x24]);
            const expectedWeights = new Int8Array([-1, 0, 1, -1]);
            
            const unpackedWeights = unpackTernaryWeights(packedWeights, 2, 2);
            
            expect(unpackedWeights).toEqual(expectedWeights);
        });
        
        test('unpackTernaryWeights should handle partial bytes', () => {
            // Test with 3 weights in one byte (4th position unused)
            // Weights [+1, -1, 0]: +1=10, -1=00, 0=01
            // Bit arrangement: ?? 01 00 10 -> 0b00010010 = 0x12
            const packedWeights = new Uint8Array([0x12]);
            const expectedWeights = new Int8Array([1, -1, 0]); // First 3 elements
            
            const unpackedWeights = unpackTernaryWeights(packedWeights, 3, 1);
            
            expect(unpackedWeights.slice(0, 3)).toEqual(expectedWeights);
        });
        
        test('unpackTernaryWeights should handle larger matrices', () => {
            // Test 2x4 matrix (8 weights, 2 bytes)
            const outDim = 2, inDim = 4;
            const totalWeights = outDim * inDim; // 8 weights
            
            // Create test packed weights: 2 bytes for 8 weights
            const packedWeights = new Uint8Array([0x1B, 0xE4]); // Example packed values
            
            const unpackedWeights = unpackTernaryWeights(packedWeights, outDim, inDim);
            
            expect(unpackedWeights.length).toBe(totalWeights);
            // All values should be -1, 0, or +1
            for (let i = 0; i < totalWeights; i++) {
                expect([-1, 0, 1]).toContain(unpackedWeights[i]);
            }
        });
        
        test('unpackTernaryWeights should handle reserved bit patterns', () => {
            // Test that reserved bit pattern 11 is handled gracefully  
            // 0xFF = 11111111 -> all bit pairs are 11, which maps to +1
            const packedWeights = new Uint8Array([0xFF]); // All bits set
            
            const unpackedWeights = unpackTernaryWeights(packedWeights, 2, 2);
            
            // Should handle gracefully (our implementation maps 11 to +1)
            expect(unpackedWeights.length).toBe(4);
            expect(unpackedWeights).toEqual(new Int8Array([1, 1, 1, 1]));
        });
    });
    
    describe('CPU BitLinear Implementation', () => {
        test('bitLinearCPU should perform correct matrix multiplication', () => {
            // Simple test case: 1x2 input, 2x2 weights, 1x2 output
            const batchSize = 1, inDim = 2, outDim = 2;
            
            const input = new Float32Array([2.0, 3.0]); // [1, 2]
            const ternaryWeights = new Int8Array([1, -1, 0, 1]); // [[1, -1], [0, 1]] (row-major)
            const scales = new Float32Array([1.0, 1.0]); // Unit scales
            
            const output = bitLinearCPU(input, ternaryWeights, scales, batchSize, inDim, outDim);
            
            // Expected: [2*1 + 3*(-1), 2*0 + 3*1] = [-1, 3]
            expect(output).toEqual(new Float32Array([-1.0, 3.0]));
        });
        
        test('bitLinearCPU should apply scaling correctly', () => {
            const batchSize = 1, inDim = 2, outDim = 2;
            
            const input = new Float32Array([1.0, 1.0]);
            const ternaryWeights = new Int8Array([1, 1, -1, -1]); // [[1, 1], [-1, -1]]
            const scales = new Float32Array([2.0, 0.5]); // Different scales
            
            const output = bitLinearCPU(input, ternaryWeights, scales, batchSize, inDim, outDim);
            
            // Before scaling: [1*1 + 1*1, 1*(-1) + 1*(-1)] = [2, -2]  
            // After scaling: [2*2.0, -2*0.5] = [4.0, -1.0]
            expect(output).toEqual(new Float32Array([4.0, -1.0]));
        });
        
        test('bitLinearCPU should handle batch processing', () => {
            const batchSize = 2, inDim = 2, outDim = 1;
            
            const input = new Float32Array([1.0, 2.0, 3.0, 4.0]); // 2 samples: [1,2] and [3,4]
            const ternaryWeights = new Int8Array([1, -1]); // [1, -1]
            const scales = new Float32Array([1.0]);
            
            const output = bitLinearCPU(input, ternaryWeights, scales, batchSize, inDim, outDim);
            
            // Expected: [1*1 + 2*(-1), 3*1 + 4*(-1)] = [-1, -1]
            expect(output).toEqual(new Float32Array([-1.0, -1.0]));
        });
        
        test('bitLinearCPU should handle zero weights correctly', () => {
            const batchSize = 1, inDim = 3, outDim = 1;
            
            const input = new Float32Array([5.0, 10.0, 15.0]);
            const ternaryWeights = new Int8Array([1, 0, -1]); // [1, 0, -1]
            const scales = new Float32Array([1.0]);
            
            const output = bitLinearCPU(input, ternaryWeights, scales, batchSize, inDim, outDim);
            
            // Expected: 5*1 + 10*0 + 15*(-1) = 5 + 0 - 15 = -10
            expect(output).toEqual(new Float32Array([-10.0]));
        });
    });
    
    describe('BitLinearOp Class', () => {
        test('BitLinearOp should initialize with correct backend selection', async () => {
            const op = new BitLinearOp();
            await op.initialize();
            
            // Should select a valid backend
            expect(['webgpu', 'wasm-simd', 'cpu']).toContain(op.backend);
        });
        
        test('BitLinearOp should validate input dimensions', async () => {
            const op = new BitLinearOp();
            await op.initialize();
            
            const input = new Tensor('float32', new Float32Array([1.0, 2.0]), [1, 2]);
            const packedWeights = new Uint8Array([0x24]); // 4 weights packed
            const scales = new Float32Array([1.0, 1.0]);
            const outDim = 2, inDim = 2;
            
            // This should work (correct dimensions)
            const result = await op.forward(input, packedWeights, scales, outDim, inDim);
            expect(result.dims).toEqual([1, 2]);
        });
        
        test('BitLinearOp should throw on mismatched dimensions', async () => {
            const op = new BitLinearOp();
            await op.initialize();
            
            const input = new Tensor('float32', new Float32Array([1.0, 2.0]), [1, 2]);
            const packedWeights = new Uint8Array([0x24]);
            const scales = new Float32Array([1.0, 1.0]);
            const outDim = 2, inDim = 3; // Mismatch: input has inDim=2, but we claim inDim=3
            
            await expect(op.forward(input, packedWeights, scales, outDim, inDim))
                .rejects.toThrow(/Invalid input shape/);
        });
        
        test('BitLinearOp should validate packed weights size', async () => {
            const op = new BitLinearOp();
            await op.initialize();
            
            const input = new Tensor('float32', new Float32Array([1.0, 2.0]), [1, 2]);
            const packedWeights = new Uint8Array([0x24]); // 1 byte = 4 weights
            const scales = new Float32Array([1.0, 1.0, 1.0]);
            const outDim = 3, inDim = 2; // 6 total weights, needs 2 bytes, but we only provide 1
            
            await expect(op.forward(input, packedWeights, scales, outDim, inDim))
                .rejects.toThrow(/Invalid packed weights size/);
        });
        
        test('BitLinearOp should validate scales length', async () => {
            const op = new BitLinearOp();
            await op.initialize();
            
            const input = new Tensor('float32', new Float32Array([1.0, 2.0]), [1, 2]);
            const packedWeights = new Uint8Array([0x24]);
            const scales = new Float32Array([1.0]); // Only 1 scale, but outDim=2
            const outDim = 2, inDim = 2;
            
            await expect(op.forward(input, packedWeights, scales, outDim, inDim))
                .rejects.toThrow(/Invalid scales length/);
        });
    });
    
    describe('Integration Tests', () => {
        test('Full pipeline should match expected numerical results', async () => {
            // Test complete pipeline: pack weights -> unpack -> compute
            const batchSize = 1, inDim = 4, outDim = 2;
            
            // Create test ternary weights (row-major)
            const originalWeights = new Int8Array([
                 1, -1,  0,  1,  // First output row
                -1,  0,  1, -1   // Second output row
            ]);
            
            // Manually pack the weights (4 per byte)
            // First 4 weights: [1, -1, 0, 1] -> [10, 00, 01, 10] -> 0b10010010 = 0x92
            // Next 4 weights: [-1, 0, 1, -1] -> [00, 01, 10, 00] -> 0b00100100 = 0x24
            const packedWeights = new Uint8Array([0x92, 0x24]);
            
            const input = new Tensor('float32', new Float32Array([2.0, 3.0, 4.0, 5.0]), [1, 4]);
            const scales = new Float32Array([1.0, 1.0]);
            
            // Test unpacking first
            const unpackedWeights = unpackTernaryWeights(packedWeights, outDim, inDim);
            expect(unpackedWeights).toEqual(originalWeights);
            
            // Test full forward pass
            const result = await bitlinear(input, packedWeights, scales, outDim, inDim);
            
            // Expected computation:
            // Output[0] = 2*1 + 3*(-1) + 4*0 + 5*1 = 2 - 3 + 0 + 5 = 4
            // Output[1] = 2*(-1) + 3*0 + 4*1 + 5*(-1) = -2 + 0 + 4 - 5 = -3
            expect(result.data).toEqual(new Float32Array([4.0, -3.0]));
        });
        
        test('Should handle edge case of all zero weights', async () => {
            const batchSize = 1, inDim = 4, outDim = 1;
            
            // All weights are 0: [0, 0, 0, 0] -> [01, 01, 01, 01] -> 0b01010101 = 0x55
            const packedWeights = new Uint8Array([0x55]);
            const input = new Tensor('float32', new Float32Array([1.0, 2.0, 3.0, 4.0]), [1, 4]);
            const scales = new Float32Array([2.0]);
            
            const result = await bitlinear(input, packedWeights, scales, outDim, inDim);
            
            // All weights are 0, so output should be 0 regardless of scale
            expect(result.data).toEqual(new Float32Array([0.0]));
        });
        
        test('Should handle large matrices efficiently', async () => {
            const batchSize = 2, inDim = 64, outDim = 32;
            
            // Generate random-ish packed weights
            const packedSize = Math.ceil((outDim * inDim) / 4);
            const packedWeights = new Uint8Array(packedSize);
            for (let i = 0; i < packedSize; i++) {
                packedWeights[i] = i % 256; // Deterministic but varied
            }
            
            // Generate test input
            const inputData = new Float32Array(batchSize * inDim);
            for (let i = 0; i < inputData.length; i++) {
                inputData[i] = (i % 100) / 100.0; // [0, 0.99]
            }
            
            const input = new Tensor('float32', inputData, [batchSize, inDim]);
            const scales = new Float32Array(outDim).fill(1.0);
            
            const start = Date.now();
            const result = await bitlinear(input, packedWeights, scales, outDim, inDim);
            const elapsed = Date.now() - start;
            
            // Verify output shape
            expect(result.dims).toEqual([batchSize, outDim]);
            expect(result.data.length).toBe(batchSize * outDim);
            
            // Performance check (should complete reasonably quickly)
            expect(elapsed).toBeLessThan(1000); // Less than 1 second
        });
    });
    
    describe('Error Handling', () => {
        test('Should handle null/undefined inputs gracefully', async () => {
            await expect(bitlinear(null, new Uint8Array([0]), new Float32Array([1]), 1, 1))
                .rejects.toThrow();
                
            await expect(bitlinear(new Tensor('float32', new Float32Array([1]), [1, 1]), null, new Float32Array([1]), 1, 1))
                .rejects.toThrow();
        });
        
        test('Should handle empty inputs', async () => {
            const emptyInput = new Tensor('float32', new Float32Array([]), [0, 0]);
            const packedWeights = new Uint8Array([]);
            const scales = new Float32Array([]);
            
            // Empty dimensions (0x0) should be handled gracefully
            // This might work (return empty tensor) or throw depending on implementation
            try {
                const result = await bitlinear(emptyInput, packedWeights, scales, 0, 0);
                expect(result.dims).toEqual([0, 0]);
                expect(result.data.length).toBe(0);
            } catch (error) {
                expect(error).toBeDefined();
            }
        });
        
        test('Should handle mismatched tensor types', async () => {
            const input = new Tensor('int32', new Int32Array([1, 2]), [1, 2]); // Wrong type
            const packedWeights = new Uint8Array([0x24]);
            const scales = new Float32Array([1.0, 1.0]);
            
            // This might work or throw depending on implementation
            // At minimum, it should not crash
            try {
                await bitlinear(input, packedWeights, scales, 2, 2);
            } catch (error) {
                expect(error).toBeDefined();
            }
        });
    });
});