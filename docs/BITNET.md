# BitNet (1.58-bit Quantization) Implementation

BitNet is an extreme quantization technique that reduces model weights to just 1.58 bits per parameter using ternary values (-1, 0, +1). This implementation provides **4x memory compression** while maintaining competitive performance through optimized multi-backend inference.

## 🚀 Key Features

- **🔥 Extreme Compression**: 4x smaller models compared to int8 quantization
- **⚡ GPU Acceleration**: WebGPU compute shaders for high-performance inference
- **🔄 Multi-Backend Support**: Automatic fallback (WebGPU → WASM SIMD → CPU)
- **🎯 Microsoft BitNet Support**: Compatible with official Microsoft BitNet models
- **🧪 Comprehensive Testing**: 19 passing tests covering all functionality

## 📁 Implementation Structure

```
src/ops/bitlinear.js          # Core BitLinear operation with multi-backend support
webgpu/shaders/bitlinear.wgsl # WebGPU compute shaders for GPU acceleration
wasm/bitlinear.cpp            # WASM SIMD kernel for optimized CPU fallback
tests/bitnet/bitlinear.test.js # Comprehensive test suite (19 tests)
scripts/bitnet_export.py      # Generic BitNet model export script
export_microsoft_bitnet_production.py # Microsoft BitNet model export
```

## 🔧 Core API

### BitLinear Operation

```javascript
import { bitlinear } from '@huggingface/transformers';

// Execute BitLinear operation with ternary weights
const output = await bitlinear(
  input,        // Input tensor [batch_size, input_dim]
  packedWeights, // Packed ternary weights (4 weights per byte)
  scales,       // Per-output channel scaling factors
  outputDim,    // Output dimension
  inputDim      // Input dimension
);
```

### Weight Packing Format

BitNet uses an efficient packing format storing 4 ternary weights per byte:
- `00` → -1
- `01` → 0 
- `10` → +1
- `11` → +1 (fallback)

```javascript
// Example: Pack weights [1, -1, 0, 1] into single byte
const packedByte = 0x92; // Binary: 10 01 00 10
```

## 🏭 Model Export Pipeline

### Generic BitNet Models

```python
python scripts/bitnet_export.py \
  --model_path /path/to/model \
  --output_dir ./bitnet_exported \
  --quantize_weights
```

### Microsoft BitNet Models

```python  
python export_microsoft_bitnet_production.py \
  --model_id microsoft/bitnet-b1.58-2B-4T \
  --output_dir ./microsoft_bitnet \
  --batch_size 1
```

## 🖥️ Backend Architecture

### Automatic Backend Selection

```javascript
// The system automatically selects the best available backend:
// 1. WebGPU (if supported) - Highest performance
// 2. WASM SIMD (if supported) - Good performance 
// 3. CPU JavaScript - Universal fallback
```

### WebGPU Implementation

High-performance GPU compute using WGSL shaders:
- **Adaptive GPU Selection**: Requests high-performance GPU first, falls back gracefully
- **Performance Features**: Uses shader-f16 and dp4a when available for better performance
- **Vectorized Operations**: Optimized ternary weight unpacking and matrix multiplication
- **Workgroup Optimization**: Configurable workgroup sizes for different GPU architectures
- **Chrome Performance Fix**: Mitigates Chrome issue #369219127 (powerPreference ignored on Windows)

#### Performance Optimization Tips

**Windows Users**: Enable `chrome://flags/#force-high-performance-gpu` for discrete GPU usage

**macOS Users**: BitNet automatically selects discrete GPU when on AC power

**All Platforms**: Check console for GPU adapter details and performance hints

### WASM SIMD Implementation

Optimized CPU fallback using SIMD instructions:
- Vectorized operations where possible
- Cache-friendly memory access
- Minimal JavaScript overhead

## 🧪 Testing & Validation

### Run Test Suite

```bash
# Run comprehensive BitNet test suite
npm test -- tests/bitnet/bitlinear.test.js

# Run standalone validation
node test_bitnet_standalone.js
```

### Test Coverage

- ✅ Weight packing/unpacking correctness
- ✅ Numerical accuracy validation
- ✅ Multi-backend operation testing
- ✅ Error handling and edge cases
- ✅ Performance benchmarking
- ✅ Microsoft BitNet compatibility

## 🌐 Browser Integration

### WebGPU Chat Example

The WebGPU chat example demonstrates BitNet integration:

```bash
cd examples/webgpu-chat
npm install
npm run dev
# Visit http://localhost:5174
```

Features:
- BitNet backend capability detection
- Real-time performance monitoring
- Console logging for debugging
- Seamless integration with existing workflows

## 📊 Performance Benefits

### Memory Usage
- **4x compression** vs int8 quantization
- **16x compression** vs float32 weights
- Reduced model download times
- Lower memory footprint during inference

### Compute Performance
- **WebGPU**: Parallel GPU compute for large models
- **WASM SIMD**: Vectorized CPU operations
- **Automatic optimization**: Always uses fastest available backend

## 🔍 Implementation Details

### Weight Unpacking

```javascript
function unpackTernaryWeights(packedWeights, outDim, inDim) {
  const totalWeights = outDim * inDim;
  const unpackedWeights = new Int8Array(totalWeights);
  
  for (let i = 0; i < totalWeights; i++) {
    const byteIndex = Math.floor(i / 4);
    const bitOffset = (i % 4) * 2;
    const packedValue = (packedWeights[byteIndex] >> bitOffset) & 0x3;
    
    // Convert 2-bit value to ternary weight
    unpackedWeights[i] = packedValue === 0 ? -1 : 
                        packedValue === 1 ? 0 : 1;
  }
  
  return unpackedWeights;
}
```

### Matrix Multiplication

```javascript
function bitLinearCPU(input, weights, scales, batchSize, inDim, outDim) {
  const output = new Float32Array(batchSize * outDim);
  
  for (let b = 0; b < batchSize; b++) {
    for (let o = 0; o < outDim; o++) {
      let sum = 0;
      for (let i = 0; i < inDim; i++) {
        sum += input[b * inDim + i] * weights[o * inDim + i];
      }
      output[b * outDim + o] = sum * scales[o];
    }
  }
  
  return output;
}
```

## 🚀 Future Enhancements

- **Model Hub Integration**: Direct BitNet model loading from Hugging Face
- **Training Support**: Gradient computation for fine-tuning
- **Additional Formats**: Support for other extreme quantization schemes
- **Mobile Optimization**: React Native and mobile browser optimizations

## 📚 References

- [BitNet: Scaling 1-bit Transformers for Large Language Models](https://arxiv.org/abs/2310.11453)
- [Microsoft BitNet Models](https://huggingface.co/microsoft/bitnet-b1.58-2B-4T)
- [WebGPU Specification](https://gpuweb.github.io/gpuweb/)
- [ONNX Runtime Web](https://onnxruntime.ai/docs/tutorials/web/)