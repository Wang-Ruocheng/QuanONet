import mindspore as ms
import numpy as np
import argparse
import os

def main():
    parser = argparse.ArgumentParser(description="Convert MindSpore .ckpt to .npz format for Qiskit hardware inference.")
    parser.add_argument("ckpt_path", type=str, help="Path to the input MindSpore .ckpt file.")
    parser.add_argument("--output", type=str, default=None, help="Path for the output .npz file (optional).")
    args = parser.parse_args()

    ckpt_path = args.ckpt_path

    if not os.path.exists(ckpt_path):
        print(f"❌ Error: Checkpoint file not found at '{ckpt_path}'")
        return

    if args.output:
        save_path = args.output
    else:
        save_path = os.path.splitext(ckpt_path)[0] + ".npz"

    print(f"Loading checkpoint: {ckpt_path} ...")
    try:
        param_dict = ms.load_checkpoint(ckpt_path)
    except Exception as e:
        print(f"❌ Failed to load checkpoint. Error: {e}")
        return

    np_dict = {}
    for key, param in param_dict.items():
        np_dict[key] = param.asnumpy()
        print(f"  -> Extracted: {key}, Shape: {np_dict[key].shape}")

    np.savez(save_path, **np_dict)
    print(f"\n✅ Conversion successful! Saved to: {save_path}")

if __name__ == "__main__":
    main()