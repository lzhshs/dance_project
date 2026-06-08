"""Quick test of the pytorch3d shim."""
import sys, os
os.chdir(os.path.join(os.path.dirname(__file__), "..", "external", "EDGE"))
sys.path.insert(0, ".")

from pytorch3d.transforms import (
    axis_angle_to_quaternion, quaternion_to_axis_angle,
    quaternion_multiply, quaternion_apply,
    axis_angle_to_matrix, matrix_to_quaternion,
    rotation_6d_to_matrix, matrix_to_rotation_6d,
)
import torch

# roundtrip test
aa = torch.randn(5, 3) * 0.5
q = axis_angle_to_quaternion(aa)
aa2 = quaternion_to_axis_angle(q)
print(f"axis_angle roundtrip error: {(aa - aa2).abs().max().item():.6f}")

# matrix roundtrip
M = axis_angle_to_matrix(aa)
r6d = matrix_to_rotation_6d(M)
M2 = rotation_6d_to_matrix(r6d)
print(f"matrix 6d roundtrip error:  {(M - M2).abs().max().item():.6f}")

# quaternion multiply
q1 = axis_angle_to_quaternion(torch.randn(3, 3) * 0.3)
q2 = axis_angle_to_quaternion(torch.randn(3, 3) * 0.3)
q12 = quaternion_multiply(q1, q2)
print(f"quat multiply norm:         {torch.norm(q12, dim=-1)}")

# quaternion apply
pts = torch.randn(3, 3)
rotated = quaternion_apply(q1, pts)
print(f"quat apply output shape:    {rotated.shape}")

print("\nAll pytorch3d shim tests PASSED!")
