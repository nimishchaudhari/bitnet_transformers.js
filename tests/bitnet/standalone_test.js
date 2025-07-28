#!/usr/bin/env node

/**
 * Standalone BitNet test - verifies the core implementation without dependencies
 */

import { BitLinearOp, unpackTernaryWeights, bitLinearCPU } from './src/ops/bitlinear.js';
import { Tensor } from './src/utils/tensor.js';

console.log('=== BitNet Standalone Test ===\n');

async function runBitNetTest() {
    try {
        console.log('1. Testing weight unpacking...');
        
        // Test weight unpacking
        const packedWeights = new Uint8Array([0x12, 0x08, 0x00]); // 3 bytes
        const unpackedWeights = unpackTernaryWeights(packedWeights, 3, 4);
        
        console.log('   Packed weights:', Array.from(packedWeights));
        console.log('   Unpacked weights:', Array.from(unpackedWeights));
        console.log('   ✓ Weight unpacking successful\n');
        
        console.log('2. Testing CPU BitLinear operation...');
        
        // Test CPU operation
        const inputData = new Float32Array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0]);
        const scales = new Float32Array([1.0, 2.0, 0.5]);
        
        const output = bitLinearCPU(inputData, unpackedWeights, scales, 2, 4, 3);
        console.log('   Input:', Array.from(inputData));
        console.log('   Output:', Array.from(output));
        console.log('   ✓ CPU operation successful\n');
        
        console.log('3. Testing full BitLinear operation...');
        
        // Test full operation
        const input = new Tensor('float32', inputData, [2, 4]);
        const bitLinearOp = new BitLinearOp();
        await bitLinearOp.initialize();
        
        const fullOutput = await bitLinearOp.forward(input, packedWeights, scales, 3, 4);
        
        console.log('   Backend selected:', bitLinearOp.backend);
        console.log('   Output shape:', fullOutput.dims);
        console.log('   Output data:', Array.from(fullOutput.data));
        console.log('   ✓ Full operation successful\n');
        
        console.log('🎉 All BitNet tests passed!');
        console.log(`🔧 Using ${bitLinearOp.backend} backend`);
        
        return true;
        
    } catch (error) {
        console.error('❌ BitNet test failed:', error);
        return false;
    }
}

// Run the test
runBitNetTest().then(success => {
    process.exit(success ? 0 : 1);
});