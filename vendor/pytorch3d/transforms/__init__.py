import math

import torch
import torch.nn.functional as F


def axis_angle_to_quaternion(axis_angle):
    angles = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    small = angles.abs() < 1e-8
    sin_half_over_angle = torch.empty_like(angles)
    sin_half_over_angle[~small] = torch.sin(half_angles[~small]) / angles[~small]
    sin_half_over_angle[small] = 0.5 - (angles[small] * angles[small]) / 48.0
    return torch.cat([torch.cos(half_angles), axis_angle * sin_half_over_angle], dim=-1)


def quaternion_to_axis_angle(quaternions):
    quaternions = F.normalize(quaternions, dim=-1)
    vector = quaternions[..., 1:]
    norms = torch.linalg.norm(vector, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2.0 * half_angles
    small = angles.abs() < 1e-8
    scale = torch.empty_like(angles)
    scale[~small] = angles[~small] / norms[~small]
    scale[small] = 2.0
    return vector * scale


def quaternion_to_matrix(quaternions):
    quaternions = F.normalize(quaternions, dim=-1)
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0
    matrix = torch.stack(
        [
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ],
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_quaternion(matrix):
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError("Invalid rotation matrix shape")
    m00 = matrix[..., 0, 0]
    m11 = matrix[..., 1, 1]
    m22 = matrix[..., 2, 2]
    qw = 0.5 * torch.sqrt(torch.clamp(1 + m00 + m11 + m22, min=0))
    qx = 0.5 * torch.sqrt(torch.clamp(1 + m00 - m11 - m22, min=0))
    qy = 0.5 * torch.sqrt(torch.clamp(1 - m00 + m11 - m22, min=0))
    qz = 0.5 * torch.sqrt(torch.clamp(1 - m00 - m11 + m22, min=0))
    qx = torch.copysign(qx, matrix[..., 2, 1] - matrix[..., 1, 2])
    qy = torch.copysign(qy, matrix[..., 0, 2] - matrix[..., 2, 0])
    qz = torch.copysign(qz, matrix[..., 1, 0] - matrix[..., 0, 1])
    return F.normalize(torch.stack([qw, qx, qy, qz], dim=-1), dim=-1)


def axis_angle_to_matrix(axis_angle):
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def matrix_to_axis_angle(matrix):
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def matrix_to_rotation_6d(matrix):
    return matrix[..., :2, :].clone().reshape(matrix.shape[:-2] + (6,))


def rotation_6d_to_matrix(d6):
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def quaternion_multiply(a, b):
    aw, ax, ay, az = torch.unbind(a, -1)
    bw, bx, by, bz = torch.unbind(b, -1)
    return torch.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        dim=-1,
    )


def quaternion_invert(quaternion):
    scaling = torch.tensor([1, -1, -1, -1], dtype=quaternion.dtype, device=quaternion.device)
    return quaternion * scaling


def quaternion_apply(quaternion, point):
    real_parts = torch.zeros(point.shape[:-1] + (1,), dtype=point.dtype, device=point.device)
    point_as_quaternion = torch.cat([real_parts, point], dim=-1)
    out = quaternion_multiply(
        quaternion_multiply(quaternion, point_as_quaternion), quaternion_invert(quaternion)
    )
    return out[..., 1:]


class RotateAxisAngle:
    def __init__(self, angle, axis="X", degrees=True):
        if degrees:
            angle = angle * math.pi / 180.0
        vector = torch.zeros(3)
        vector[{"X": 0, "Y": 1, "Z": 2}[axis.upper()]] = angle
        self.matrix = axis_angle_to_matrix(vector)

    def transform_points(self, points):
        matrix = self.matrix.to(dtype=points.dtype, device=points.device)
        return torch.matmul(points, matrix.transpose(-1, -2))
