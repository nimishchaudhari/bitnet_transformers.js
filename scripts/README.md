# BitNet Export Scripts

This directory contains Python scripts for exporting BitNet quantized models to browser-compatible formats.

## Scripts

### `bitnet_export.py` - Generic BitNet Export
Converts standard ONNX models to BitNet format with ternary quantization.

**Usage:**
```bash
python scripts/bitnet_export.py \
  --model_path /path/to/model \
  --output_dir ./bitnet_exported \
  --quantize_weights
```

**Features:**
- Generic ONNX model support
- Ternary weight quantization (-1, 0, +1)
- Efficient weight packing (4 weights per byte)
- Metadata preservation

### `../export_microsoft_bitnet_production.py` - Microsoft BitNet Export
Specialized script for Microsoft's official BitNet models with uint8 weight decoding.

**Usage:**
```bash
python export_microsoft_bitnet_production.py \
  --model_id microsoft/bitnet-b1.58-2B-4T \
  --output_dir ./microsoft_bitnet \
  --batch_size 1
```

**Features:**
- Microsoft BitNet model support
- uint8 → ternary weight conversion
- Production-ready export pipeline
- Safetensors format support

## Requirements

```bash
pip install torch transformers onnx safetensors numpy
```

## Output Format

Both scripts generate:
- `model.onnx` - Quantized ONNX model
- `*.bin` - Packed ternary weight files
- `config.json` - Model configuration
- Weight analysis and validation logs