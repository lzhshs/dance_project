"""Inspect an AIST++ SMPL motion file to understand its format."""
import pickle
import numpy as np
import sys

path = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/lucy_lzh/dance_project/aist_data/motions/gBR_sBM_cAll_d04_mBR0_ch01.pkl"

with open(path, "rb") as f:
    data = pickle.load(f)

print(f"File: {path}")
print(f"Top-level type: {type(data).__name__}")

if isinstance(data, dict):
    print(f"Keys: {list(data.keys())}")
    for k, v in data.items():
        if isinstance(v, np.ndarray):
            print(f"  {k}: ndarray shape={v.shape} dtype={v.dtype}")
            print(f"    min={v.min():.4f} max={v.max():.4f} mean={v.mean():.4f}")
        else:
            print(f"  {k}: {type(v).__name__} = {v}")
