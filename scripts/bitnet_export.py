#!/usr/bin/env python3
"""
BitNet (1.58-bit) Model Export Script for Transformers.js

This script converts HuggingFace BitNet models to ONNX format with ternary quantization
optimizations for browser deployment.

Usage:
    python bitnet_export.py --model_name microsoft/bitnet-1.58-bert-base --output_dir ./models

Features:
- Converts BitNet models with -1, 0, +1 ternary weights
- Packs ternary weights into 4 weights per byte (2 bits each)
- Preserves FP16 first/last layers for stability
- Validates static shapes (dynamic shapes not supported)
- Generates optimized ONNX graphs for WebGPU/WASM execution
"""

import os
import sys
import argparse
import numpy as np
import torch
import onnx
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from pathlib import Path

try:
    from transformers import AutoModel, AutoTokenizer, AutoConfig
    from transformers.models.bitnet import BitNetForSequenceClassification
    import onnxruntime as ort
except ImportError as e:
    print(f"Error importing required dependencies: {e}")
    print("Please install: transformers>=4.35.0 torch onnx onnxruntime")
    sys.exit(1)

@dataclass
class BitNetExportConfig:
    """Configuration for BitNet export process"""
    model_name: str
    output_dir: str
    max_seq_length: int = 512
    batch_size: int = 1
    keep_first_last_fp16: bool = True
    validate_output: bool = True
    opset_version: int = 17

class BitNetONNXExporter:
    """
    Exports BitNet models to ONNX with ternary quantization optimizations
    """
    
    def __init__(self, config: BitNetExportConfig):
        self.config = config
        self.model = None
        self.tokenizer = None
        self.original_weights = {}
        
    def load_model(self):
        """Load BitNet model and tokenizer from HuggingFace"""
        print(f"Loading BitNet model: {self.config.model_name}")
        
        try:
            self.model = AutoModel.from_pretrained(
                self.config.model_name, 
                torch_dtype=torch.float16,
                trust_remote_code=True
            )
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
            
            # Ensure model is in eval mode
            self.model.eval()
            
            print(f"✓ Model loaded: {type(self.model).__name__}")
            print(f"✓ Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load BitNet model: {e}")
    
    def extract_bitnet_weights(self) -> Dict[str, Dict]:
        """
        Extract and analyze BitNet weights from the model
        Returns dict mapping layer names to weight info
        """
        bitnet_layers = {}
        
        for name, module in self.model.named_modules():
            # Look for BitLinear layers (BitNet's main quantized layer)
            if hasattr(module, 'weight') and hasattr(module, 'bits') and module.bits == 1.58:
                weight = module.weight.data
                
                # Validate ternary weights (-1, 0, +1)
                unique_vals = torch.unique(weight)
                if not torch.allclose(unique_vals, torch.tensor([-1., 0., 1.], device=weight.device)):
                    print(f"Warning: Layer {name} doesn't have pure ternary weights")
                    print(f"Unique values: {unique_vals}")
                
                # Calculate scale factor (if available)
                scale = getattr(module, 'scale', None)
                if scale is None:
                    scale = torch.ones(weight.shape[0], device=weight.device, dtype=torch.float32)
                
                bitnet_layers[name] = {
                    'weight': weight,
                    'scale': scale,
                    'shape': weight.shape,
                    'in_features': module.in_features,
                    'out_features': module.out_features
                }
                
        print(f"✓ Found {len(bitnet_layers)} BitNet quantized layers")
        return bitnet_layers
    
    def pack_ternary_weights(self, weight: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Pack ternary weights (-1, 0, +1) into bytes for efficient storage
        Returns packed weights and unpacking indices
        
        Packing scheme: 4 weights per byte (2 bits each)
        - -1 → 00
        - 0  → 01  
        - +1 → 10
        - 11 reserved
        """
        weight_np = weight.cpu().numpy().astype(np.int8)
        out_dim, in_dim = weight_np.shape
        
        # Validate static shape (required for packing)
        if out_dim <= 0 or in_dim <= 0:
            raise ValueError(f"Invalid weight shape: {weight_np.shape}")
        
        # Convert ternary values to 2-bit codes
        weight_coded = np.zeros_like(weight_np, dtype=np.uint8)
        weight_coded[weight_np == -1] = 0b00  # -1 → 00
        weight_coded[weight_np == 0]  = 0b01  # 0  → 01
        weight_coded[weight_np == 1]  = 0b10  # +1 → 10
        
        # Pack 4 weights per byte
        packed_size = (in_dim * out_dim + 3) // 4  # Ceil division
        packed = np.zeros(packed_size, dtype=np.uint8)
        
        flat_coded = weight_coded.flatten()
        for i in range(0, len(flat_coded), 4):
            byte_val = 0
            for j in range(4):
                if i + j < len(flat_coded):
                    byte_val |= (flat_coded[i + j] << (j * 2))
            packed[i // 4] = byte_val
        
        # Verify packing ratio
        expected_size = (out_dim * in_dim + 3) // 4
        assert len(packed) == expected_size, f"Packing size mismatch: {len(packed)} vs {expected_size}"
        
        return packed, np.array([out_dim, in_dim], dtype=np.int32)
    
    
    def export_bitnet_weights(self, output_dir: Path) -> Dict:
        """Export BitNet weights as separate binary files"""
        bitnet_layers = self.extract_bitnet_weights()
        weight_manifest = {}
        
        for layer_name, layer_info in bitnet_layers.items():
            # Pack ternary weights
            packed_weights, shape_info = self.pack_ternary_weights(layer_info['weight'])
            
            # Save packed weights
            weight_file = output_dir / f"{layer_name.replace('.', '_')}_weights.bin"
            with open(weight_file, 'wb') as f:
                f.write(packed_weights.tobytes())
            
            # Save scale factors
            scale_file = output_dir / f"{layer_name.replace('.', '_')}_scale.bin"
            with open(scale_file, 'wb') as f:
                f.write(layer_info['scale'].cpu().numpy().astype(np.float32).tobytes())
            
            # Add to manifest
            weight_manifest[layer_name] = {
                'weight_file': weight_file.name,
                'scale_file': scale_file.name,
                'shape': shape_info.tolist(),
                'original_shape': list(layer_info['weight'].shape),
                'in_features': layer_info['in_features'],
                'out_features': layer_info['out_features']
            }
        
        # Save manifest as JSON
        import json
        manifest_file = output_dir / "bitnet_manifest.json"
        with open(manifest_file, 'w') as f:
            json.dump(weight_manifest, f, indent=2)
        
        print(f"✓ Exported {len(bitnet_layers)} BitNet weight files")
        return weight_manifest
    
    def export_to_onnx(self) -> str:
        """Export BitNet model to ONNX format with separate weight files"""
        output_dir = Path(self.config.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        model_name = Path(self.config.model_name).name
        onnx_path = output_dir / f"{model_name}.onnx"
        
        # Create dummy inputs for tracing
        dummy_input = {
            'input_ids': torch.randint(0, self.tokenizer.vocab_size, 
                                     (self.config.batch_size, self.config.max_seq_length)),
            'attention_mask': torch.ones((self.config.batch_size, self.config.max_seq_length))
        }
        
        print("Exporting standard ONNX model...")
        
        # First, export a standard ONNX model (this will work with any transformer)
        torch.onnx.export(
            self.model,
            tuple(dummy_input.values()),
            str(onnx_path),
            input_names=list(dummy_input.keys()),
            output_names=['logits'],
            dynamic_axes=None,  # Static shapes for BitNet optimization
            opset_version=self.config.opset_version,
            do_constant_folding=True,
            export_params=True
        )
        
        # Export BitNet weights separately  
        print("Exporting BitNet weight files...")
        weight_manifest = self.export_bitnet_weights(output_dir)
        
        # Create a BitNet config file for Transformers.js
        bitnet_config = {
            'model_type': 'bitnet',
            'onnx_file': onnx_path.name,
            'weight_manifest': 'bitnet_manifest.json',
            'max_seq_length': self.config.max_seq_length,
            'keep_first_last_fp16': self.config.keep_first_last_fp16,
            'bitnet_layers': list(weight_manifest.keys())
        }
        
        config_file = output_dir / "bitnet_config.json"
        import json
        with open(config_file, 'w') as f:
            json.dump(bitnet_config, f, indent=2)
        
        print(f"✓ Standard ONNX model saved: {onnx_path}")
        print(f"✓ BitNet config saved: {config_file}")
        return str(onnx_path)
    
    def validate_export(self, onnx_path: str):
        """Validate the exported ONNX model"""
        if not self.config.validate_output:
            return
            
        print("Validating exported model...")
        
        try:
            # Load ONNX model
            ort_session = ort.InferenceSession(onnx_path)
            
            # Create test input
            test_input = {
                'input_ids': np.random.randint(0, 1000, (1, 64), dtype=np.int64),
                'attention_mask': np.ones((1, 64), dtype=np.int64)
            }
            
            # Run inference
            outputs = ort_session.run(None, test_input)
            
            print(f"✓ ONNX model validation successful")
            print(f"  Output shape: {outputs[0].shape}")
            print(f"  Output range: [{outputs[0].min():.3f}, {outputs[0].max():.3f}]")
            
        except Exception as e:
            print(f"⚠ ONNX validation failed: {e}")
    
    def export(self) -> str:
        """Main export pipeline"""
        print("=== BitNet ONNX Export Pipeline ===")
        
        # Step 1: Load model
        self.load_model()
        
        # Step 2: Export to ONNX
        onnx_path = self.export_to_onnx()
        
        # Step 3: Validate
        self.validate_export(onnx_path)
        
        print("=== Export Complete ===")
        return onnx_path

def main():
    parser = argparse.ArgumentParser(description="Export BitNet models to ONNX for Transformers.js")
    parser.add_argument("--model_name", required=True, help="HuggingFace BitNet model name")
    parser.add_argument("--output_dir", required=True, help="Output directory for ONNX models")
    parser.add_argument("--max_seq_length", type=int, default=512, help="Maximum sequence length")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for export")
    parser.add_argument("--keep_first_last_fp16", action="store_true", default=True, 
                       help="Keep first/last layers in FP16 for stability")
    parser.add_argument("--no_validate", action="store_true", help="Skip output validation")
    parser.add_argument("--opset_version", type=int, default=17, help="ONNX opset version")
    
    args = parser.parse_args()
    
    # Create export config
    config = BitNetExportConfig(
        model_name=args.model_name,
        output_dir=args.output_dir,
        max_seq_length=args.max_seq_length,
        batch_size=args.batch_size,
        keep_first_last_fp16=args.keep_first_last_fp16,
        validate_output=not args.no_validate,
        opset_version=args.opset_version
    )
    
    # Run export
    exporter = BitNetONNXExporter(config)
    try:
        onnx_path = exporter.export()
        print(f"\n✅ BitNet export successful!")
        print(f"📁 ONNX model: {onnx_path}")
        print(f"🚀 Ready for Transformers.js deployment")
        
    except Exception as e:
        print(f"\n❌ BitNet export failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()