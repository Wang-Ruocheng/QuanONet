#!/usr/bin/env python3
"""
Simple data file checking tool

Check the format and structure of existing data files
"""

import numpy as np
import os

def check_npz_file(file_path):
    """Check NPZ file structure"""
    print(f"\nChecking file: {file_path}")
    print("=" * 50)
    
    if not os.path.exists(file_path):
        print(f"❌ File does not exist: {file_path}")
        return
    
    file_size = os.path.getsize(file_path)
    print(f"File size: {file_size / (1024**2):.2f} MB")
    
    try:
        # Try different loading methods
        try:
            data = np.load(file_path)
        except ValueError as e:
            if "allow_pickle" in str(e):
                print("⚠️  Need to allow pickle loading, retrying...")
                data = np.load(file_path, allow_pickle=True)
            else:
                raise e
                
        print(f"✅ File loaded successfully")
        print(f"Contained keys:")
        
        for key in data.files:
            try:
                array = data[key]
                print(f"  📊 {key}: {type(array)} {getattr(array, 'shape', 'N/A')} {getattr(array, 'dtype', 'N/A')}")
                
                # If it's a numpy array, show more information
                if isinstance(array, np.ndarray):
                    size_mb = array.nbytes / (1024**2)
                    print(f"      Size: ({size_mb:.2f} MB)")
                    
                    # Show data range (min/max values)
                    if array.dtype in [np.float32, np.float64, np.int32, np.int64] and array.size > 0:
                        min_val = np.min(array)
                        max_val = np.max(array)
                        mean_val = np.mean(array)
                        std_val = np.std(array)
                        print(f"      📈 Data statistics:")
                        print(f"         Min value: {min_val:.8f}")
                        print(f"         Max value: {max_val:.8f}")
                        print(f"         Mean value: {mean_val:.8f}")
                        print(f"         Standard deviation: {std_val:.8f}")
                        print(f"         Data range: [{min_val:.6f}, {max_val:.6f}]")
                    
                    # Show first few values (if array is not too large)
                    if array.size <= 10:
                        print(f"      Values: {array}")
                    elif array.ndim == 1 and array.size <= 100:
                        print(f"      First 5 values: {array[:5]}")
                    elif array.ndim >= 2 and array.size <= 20:
                        print(f"      Partial data:\n{array}")
                        
                else:
                    # For non-numpy arrays (like lists, objects, etc.)
                    if hasattr(array, '__len__'):
                        print(f"      Length: {len(array)}")
                        if len(array) <= 5:
                            print(f"      Content: {array}")
                        else:
                            print(f"      First few elements: {array[:3] if hasattr(array, '__getitem__') else 'N/A'}")
                    else:
                        print(f"      Content: {array}")
                        
            except Exception as e:
                print(f"  ❌ {key}: Failed to read - {str(e)}")
                
        data.close()
        
    except Exception as e:
        print(f"❌ File loading failed: {str(e)}")

def main():
    """Main function"""
    # Check various data files, focusing on Diffusion operator data
    files_to_check = [
        "data/Diffusion_Operator_data/Diffusion_Operator_data_1000_1.npz",
        # "data/Inverse_Operator_dataset_1000_1000_100_10_100.npz",
        # "data/Burgers_Operator_data/Burgers_Operator_data_1000_1.npz"
    ]
    
    for file_path in files_to_check:
        check_npz_file(file_path)

if __name__ == "__main__":
    main()
