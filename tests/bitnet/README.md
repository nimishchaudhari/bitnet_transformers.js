# BitNet Test Suite

This directory contains comprehensive tests for the BitNet (1.58-bit quantization) implementation.

## Test Files

- `bitlinear.test.js` - Main test suite with 19 comprehensive tests

## Running Tests

```bash
# Run BitNet-specific tests
npm test -- tests/bitnet/bitlinear.test.js

# Run all tests
npm test
```

## Test Coverage

### Core Functionality
- ✅ Weight packing/unpacking operations
- ✅ BitLinear CPU operation
- ✅ Multi-backend selection logic
- ✅ Numerical accuracy validation

### Edge Cases  
- ✅ Invalid input handling
- ✅ Empty tensor operations
- ✅ Boundary conditions
- ✅ Error propagation

### Performance
- ✅ Backend capability detection
- ✅ Memory usage optimization
- ✅ Computational efficiency

### Integration
- ✅ Tensor compatibility
- ✅ Real-world data scenarios
- ✅ Microsoft BitNet model support

All tests are designed to validate both correctness and performance of the BitNet implementation across different backends (WebGPU, WASM SIMD, CPU).