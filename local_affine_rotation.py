#!/usr/bin/env python

import SimpleITK as sitk
import numpy as np
from numpy.linalg import lstsq, svd
import math


def polar_decomposition(J):
    U, _, Vt = svd(J)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


def rotation_matrix_to_axis_angle(R):
    angle = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2.0)))
    if abs(angle) < 1e-6:
        return np.array([0, 0, 1]), 0.0
    rx, ry, rz = R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz])
    axis = axis / np.linalg.norm(axis)
    return axis, math.degrees(angle)


def fit_local_affine(disp_img, point_phys, patch_radius=2):
    spacing = np.array(disp_img.GetSpacing())
    origin = np.array(disp_img.GetOrigin())
    direction = np.array(disp_img.GetDirection()).reshape(3, 3)

    # Convert physical point to index
    point_index = np.linalg.inv(direction) @ ((np.array(point_phys) - origin) / spacing)
    point_index = np.round(point_index).astype(int)

    size = disp_img.GetSize()

    src_pts = []
    dst_pts = []

    for dz in range(-patch_radius, patch_radius + 1):
        for dy in range(-patch_radius, patch_radius + 1):
            for dx in range(-patch_radius, patch_radius + 1):
                idx = point_index + np.array([dx, dy, dz])
                if np.any(idx < 0) or np.any(idx >= size):
                    continue

                idx_tuple = tuple(idx.tolist())
                disp = np.array(disp_img[idx_tuple])
                p = origin + direction @ ((idx + 0.5) * spacing)
                q = p + disp
                src_pts.append(p)
                dst_pts.append(q)

    src = np.array(src_pts).T  # shape (3, N)
    dst = np.array(dst_pts).T

    # Affine fit: dst = A @ src + t
    # Solve dst = [A | t] * [src; 1]
    src_aug = np.vstack([src, np.ones((1, src.shape[1]))])
    A_full, _, _, _ = lstsq(src_aug.T, dst.T, rcond=None)
    A = A_full[:3, :].T  # shape (3,3)

    return A


if __name__ == "__main__":
    disp_fn = "displacement_field.nii.gz"
    print(f"Estimating local rotation from affine fit in displacement field: {disp_fn}")
    warp = sitk.ReadImage(disp_fn, sitk.sitkVectorFloat32)
    point = [0.0, 0.0, 0.0]  # physical coordinate to evaluate

    J = fit_local_affine(warp, point, patch_radius=1)
    R = polar_decomposition(J)
    axis, angle_deg = rotation_matrix_to_axis_angle(R)

    print("Local rotation estimated from affine fit:")
    print("  Rotation Axis : [{:.6f}, {:.6f}, {:.6f}]".format(*axis))
    print("  Rotation Angle: {:.6f} degrees".format(angle_deg))
