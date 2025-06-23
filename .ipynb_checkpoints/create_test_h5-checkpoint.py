#!/usr/bin/env python3
"""
Create synthetic H5 test files for TITAN WSI embedding app
Run this script to generate test data that matches the expected format
FIXED: Coordinate data type issue for TITAN model compatibility
"""

import h5py
import numpy as np
import os
from pathlib import Path

def create_realistic_features(num_patches, feature_dim=768, slide_type='tumor'):
    """Create realistic-looking histopathology features"""
    
    # Base features
    features = np.random.randn(num_patches, feature_dim).astype(np.float32)
    
    # Add slide-type specific patterns
    if slide_type == 'tumor':
        # Tumor slides tend to have higher variance and some specific patterns
        features = features * 1.5
        # Add some correlated features (tumor markers)
        tumor_signal = np.random.randn(num_patches, 1) * 2
        features[:, :80] += tumor_signal  # First 80 features correlated
        
    elif slide_type == 'normal':
        # Normal tissue has more uniform, lower variance features
        features = features * 0.8
        features += np.random.normal(0, 0.1, features.shape)  # More uniform
        
    elif slide_type == 'inflammatory':
        # Inflammatory tissue has different patterns
        features = features * 1.2
        # Add periodic patterns (inflammatory infiltrates)
        for i in range(0, num_patches, 20):
            end_idx = min(i+10, num_patches)
            features[i:end_idx, 150:200] += 1.5
    
    # Normalize to reasonable range
    features = (features - features.mean()) / features.std()
    features = features * 0.5  # Scale to typical embedding range
    
    return features

def create_realistic_coords(num_patches, slide_width=10000, slide_height=8000, patch_size=256):
    """Create realistic patch coordinates with correct data type"""
    
    # Generate coordinates that don't overlap too much
    coords = []
    attempts = 0
    max_attempts = num_patches * 10
    
    while len(coords) < num_patches and attempts < max_attempts:
        x = np.random.randint(0, slide_width - patch_size)
        y = np.random.randint(0, slide_height - patch_size)
        
        # Check for reasonable spacing (avoid too much overlap)
        too_close = False
        for existing_x, existing_y in coords:
            if abs(x - existing_x) < patch_size//2 and abs(y - existing_y) < patch_size//2:
                too_close = True
                break
        
        if not too_close:
            coords.append([x, y])
        
        attempts += 1
    
    # If we couldn't generate enough non-overlapping coords, fill randomly
    while len(coords) < num_patches:
        x = np.random.randint(0, slide_width - patch_size)
        y = np.random.randint(0, slide_height - patch_size)
        coords.append([x, y])
    
    # FIXED: Use int64 instead of int32 for TITAN model compatibility
    return np.array(coords, dtype=np.int64)

def create_test_h5_file(filename, num_patches, slide_type='tumor', feature_dim=768):
    """Create a complete H5 test file with correct data types"""
    
    print(f"Creating {filename}...")
    print(f"  Type: {slide_type}")
    print(f"  Patches: {num_patches}")
    print(f"  Feature dim: {feature_dim}")
    
    # Generate features and coordinates
    features = create_realistic_features(num_patches, feature_dim, slide_type)
    coords = create_realistic_coords(num_patches)
    
    # Verify data types
    print(f"  Features dtype: {features.dtype}")
    print(f"  Coords dtype: {coords.dtype}")
    
    # Create H5 file
    with h5py.File(filename, 'w') as f:
        # Create main datasets with explicit data types
        f.create_dataset('features', data=features, compression='gzip', dtype=np.float32)
        coord_dataset = f.create_dataset('coords', data=coords, compression='gzip', dtype=np.int64)
        
        # Add required attributes
        coord_dataset.attrs['patch_size_level0'] = np.int64(256)  # Ensure consistent type
        
        # Add optional metadata (makes it more realistic)
        f.attrs['slide_type'] = slide_type
        f.attrs['magnification'] = '20x'
        f.attrs['patch_size'] = np.int64(256)
        f.attrs['num_patches'] = np.int64(num_patches)
        f.attrs['feature_extractor'] = 'ResNet50'  # Simulate common extractor
        
    print(f"  ✅ Created {filename}")
    print(f"  Features shape: {features.shape}")
    print(f"  Coords shape: {coords.shape}")
    print(f"  File size: {os.path.getsize(filename) / 1024 / 1024:.1f} MB")
    print()

def detect_model_feature_dim():
    """Try to detect the expected feature dimension from TITAN model config"""
    try:
        import json
        from pathlib import Path
        
        # Check if we can find model config
        model_paths = [
            "/mnt/titan_model/config.json",
            "./titan_model/config.json",
            "config.json"
        ]
        
        for config_path in model_paths:
            if Path(config_path).exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                # Look for feature dimension hints
                possible_keys = [
                    'hidden_size', 'feature_dim', 'input_dim', 
                    'patch_feature_dim', 'vision_config.hidden_size'
                ]
                
                for key in possible_keys:
                    if '.' in key:
                        # Handle nested keys
                        parts = key.split('.')
                        value = config
                        for part in parts:
                            if part in value:
                                value = value[part]
                            else:
                                break
                        else:
                            print(f"Found feature dimension from {key}: {value}")
                            return value
                    elif key in config:
                        print(f"Found feature dimension from {key}: {config[key]}")
                        return config[key]
        
        print("Could not detect feature dimension from model config, using default 768")
        return 768
        
    except Exception as e:
        print(f"Error detecting feature dimension: {e}, using default 768")
        return 768

def create_test_dataset():
    """Create a complete test dataset with various slide types"""
    
    # Try to detect the correct feature dimension
    feature_dim = detect_model_feature_dim()
    
    print("🔬 Creating TITAN Test Dataset (FIXED VERSION - 768D Features)")
    print("=" * 60)
    print(f"Using feature dimension: {feature_dim}")
    print()
    
    # Create output directory
    output_dir = Path("titan_test_data_fixed")
    output_dir.mkdir(exist_ok=True)
    
    # Test configurations
    test_configs = [
        # Small slides for quick testing
        {'name': 'small_tumor_slide.h5', 'patches': 300, 'type': 'tumor', 'feature_dim': feature_dim},
        {'name': 'small_normal_slide.h5', 'patches': 250, 'type': 'normal', 'feature_dim': feature_dim},
        
        # Medium slides for realistic testing
        {'name': 'medium_tumor_slide.h5', 'patches': 800, 'type': 'tumor', 'feature_dim': feature_dim},
        {'name': 'medium_normal_slide.h5', 'patches': 750, 'type': 'normal', 'feature_dim': feature_dim},
        {'name': 'inflammatory_slide.h5', 'patches': 600, 'type': 'inflammatory', 'feature_dim': feature_dim},
        
        # Large slide for performance testing
        {'name': 'large_mixed_slide.h5', 'patches': 1500, 'type': 'tumor', 'feature_dim': feature_dim},
        
        # Edge cases
        {'name': 'tiny_slide.h5', 'patches': 50, 'type': 'normal', 'feature_dim': feature_dim},
        {'name': 'very_large_slide.h5', 'patches': 3000, 'type': 'tumor', 'feature_dim': feature_dim}
    ]
    
    # Create all test files
    for config in test_configs:
        filepath = output_dir / config['name']
        create_test_h5_file(
            str(filepath), 
            config['patches'], 
            config['type'],
            config['feature_dim']
        )
    
    print("📊 Test Dataset Summary")
    print("=" * 30)
    
    total_size = sum(os.path.getsize(output_dir / config['name']) 
                    for config in test_configs)
    
    print(f"Total files: {len(test_configs)}")
    print(f"Total size: {total_size / 1024 / 1024:.1f} MB")
    print(f"Output directory: {output_dir.absolute()}")
    
    print("\n🚀 Ready to test!")
    print(f"Upload files from: {output_dir.absolute()}")
    
    return output_dir

def validate_h5_file(filepath):
    """Validate that H5 file has correct structure and data types"""
    try:
        with h5py.File(filepath, 'r') as f:
            # Check required datasets
            assert 'features' in f, "Missing 'features' dataset"
            assert 'coords' in f, "Missing 'coords' dataset"
            
            features = f['features']
            coords = f['coords']
            
            # Check shapes
            assert len(features.shape) == 2, f"Features should be 2D, got {features.shape}"
            assert len(coords.shape) == 2, f"Coords should be 2D, got {coords.shape}"
            assert coords.shape[1] == 2, f"Coords should have 2 columns, got {coords.shape[1]}"
            assert features.shape[0] == coords.shape[0], "Features and coords should have same number of rows"
            
            # Check data types (CRITICAL for TITAN)
            assert features.dtype == np.float32, f"Features should be float32, got {features.dtype}"
            assert coords.dtype == np.int64, f"Coords should be int64, got {coords.dtype}"
            
            # Check attributes
            assert 'patch_size_level0' in coords.attrs, "Missing patch_size_level0 attribute"
            
            print(f"✅ {filepath} is valid")
            print(f"   Features: {features.shape}, dtype: {features.dtype}")
            print(f"   Coords: {coords.shape}, dtype: {coords.dtype}")
            print(f"   Patch size: {coords.attrs['patch_size_level0']}")
            
            return True
            
    except Exception as e:
        print(f"❌ {filepath} is invalid: {e}")
        return False

def convert_existing_h5_file(input_path, output_path):
    """Convert existing H5 file to have correct data types"""
    try:
        print(f"Converting {input_path} -> {output_path}")
        
        with h5py.File(input_path, 'r') as src:
            # Read data
            features = src['features'][:]
            coords = src['coords'][:]
            
            # Get attributes
            patch_size = src['coords'].attrs.get('patch_size_level0', 256)
            
            print(f"  Original - Features: {features.dtype}, Coords: {coords.dtype}")
            
            # Convert data types
            features = features.astype(np.float32)
            coords = coords.astype(np.int64)
            
            print(f"  Converted - Features: {features.dtype}, Coords: {coords.dtype}")
            
            # Write to new file
            with h5py.File(output_path, 'w') as dst:
                dst.create_dataset('features', data=features, compression='gzip', dtype=np.float32)
                coord_dataset = dst.create_dataset('coords', data=coords, compression='gzip', dtype=np.int64)
                coord_dataset.attrs['patch_size_level0'] = np.int64(patch_size)
                
                # Copy other attributes if they exist
                for key, value in src.attrs.items():
                    dst.attrs[key] = value
        
        print(f"  ✅ Converted successfully")
        return True
        
    except Exception as e:
        print(f"  ❌ Conversion failed: {e}")
        return False

def diagnose_titan_model():
    """Diagnose TITAN model to understand expected input dimensions"""
    try:
        import torch
        from transformers import AutoModel, AutoConfig
        
        print("🔍 Diagnosing TITAN Model...")
        print("=" * 40)
        
        # Try to load config first
        try:
            config = AutoConfig.from_pretrained("/mnt/titan_model", trust_remote_code=True)
            print("✅ Successfully loaded model config")
            
            # Print relevant config attributes
            for attr in dir(config):
                if any(keyword in attr.lower() for keyword in ['hidden', 'dim', 'size', 'feature']):
                    if not attr.startswith('_'):
                        value = getattr(config, attr)
                        if isinstance(value, (int, float)):
                            print(f"  {attr}: {value}")
            
        except Exception as e:
            print(f"❌ Could not load config: {e}")
        
        # Try to analyze the model structure without loading it fully
        try:
            print("\n🔍 Analyzing model files...")
            from pathlib import Path
            model_dir = Path("/mnt/titan_model")
            
            # Look for pytorch_model.bin or model.safetensors
            model_files = []
            for pattern in ["*.bin", "*.safetensors", "*.pt", "*.pth"]:
                model_files.extend(model_dir.glob(pattern))
            
            print(f"Found {len(model_files)} model files:")
            for f in model_files[:5]:  # Show first 5
                print(f"  {f.name}")
                
        except Exception as e:
            print(f"❌ Could not analyze model files: {e}")
            
        print("\n💡 Common feature dimensions for WSI models:")
        print("  - ResNet50 (ImageNet): 2048")
        print("  - ViT-Base: 768") 
        print("  - ViT-Large: 1024")
        print("  - CONCH: 768")
        print("  - CTransPath: 768")
        
    except Exception as e:
        print(f"❌ Diagnosis failed: {e}")

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == 'diagnose':
        # Diagnose TITAN model
        diagnose_titan_model()
        
    elif len(sys.argv) > 1 and sys.argv[1] == 'convert':
        # Convert existing files
        if len(sys.argv) != 4:
            print("Usage: python script.py convert <input.h5> <output.h5>")
            sys.exit(1)
        
        input_file = sys.argv[2]
        output_file = sys.argv[3]
        convert_existing_h5_file(input_file, output_file)
    
    else:
        # Create new test dataset
        output_dir = create_test_dataset()
        
        print("\n🔍 Validating created files...")
        print("=" * 40)
        
        # Validate all created files
        for h5_file in output_dir.glob("*.h5"):
            validate_h5_file(h5_file)
        
        print("\n📋 Usage Instructions:")
        print("1. Run your TITAN Streamlit app")
        print("2. Upload H5 files from the 'titan_test_data_fixed' directory")
        print("3. Test with different slide types and sizes")
        print("4. Compare embeddings between tumor/normal slides")
        
        print("\n💡 Recommended testing order:")
        print("- Start with small files (small_*.h5)")
        print("- Try different slide types") 
        print("- Test visualization with 3+ files")
        print("- Performance test with large files")
        
        print("\n🔧 Available commands:")
        print("python script.py                    # Create new test dataset")
        print("python script.py diagnose           # Diagnose TITAN model dimensions") 
        print("python script.py convert input.h5 output.h5  # Convert existing files")