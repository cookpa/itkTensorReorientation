#!/usr/bin/env python

import sys
import math
import argparse
import numpy as np
import itk
from pathlib import Path

def polar_decomposition(J):
    u, _, vh = np.linalg.svd(J)
    R = u @ vh
    if np.linalg.det(R) < 0:
        u[:, -1] *= -1
        R = u @ vh
    return R

def rotation_matrix_to_axis_angle(R):
    angle = math.acos(min(1.0, max(-1.0, (np.trace(R) - 1) / 2.0)))
    if abs(angle) < 1e-6:
        return np.array([0, 0, 1]), 0.0
    rx, ry, rz = R[2,1] - R[1,2], R[0,2] - R[2,0], R[1,0] - R[0,1]
    axis = np.array([rx, ry, rz])
    axis = axis / np.linalg.norm(axis)
    return axis, math.degrees(angle)

def get_jacobian_from_displacement(transform, point):
    Jinv = itk.vnl_matrix_fixed[itk.F, 3, 3]()
    transform.ComputeInverseJacobianWithRespectToPosition(point, Jinv)
    return np.array([[Jinv.get(i, j) for j in range(3)] for i in range(3)], dtype=np.float32)

def get_inverse_affine_matrix(transform_file):
    composite = itk.transformread(transform_file)
    # Get the first transform from composite
    mat = composite[0].GetMatrix()
    inv = np.linalg.inv(mat)
    return inv.astype(np.float32)

def print_rotation(label, J):
    R = polar_decomposition(J)
    axis, angle_deg = rotation_matrix_to_axis_angle(R)
    print(f"{label}")
    print("  Rotation Axis : [{:.6f}, {:.6f}, {:.6f}]".format(*axis))
    print("  Rotation Angle: {:.6f} degrees".format(angle_deg))
    print()

def main():
    parser = argparse.ArgumentParser(description="Report local rotation from displacement and/or affine transform.")
    parser.add_argument("x", type=float)
    parser.add_argument("y", type=float)
    parser.add_argument("z", type=float)
    parser.add_argument("--displacement", type=Path, help="Displacement field image")
    parser.add_argument("--affine", type=Path, help="Affine transform file (.tfm)")
    args = parser.parse_args()

    point = itk.Point[itk.F, 3]()
    point[0], point[1], point[2] = args.x, args.y, args.z

    if not args.displacement and not args.affine:
        parser.error("You must specify at least one of --displacement or --affine.")

    if args.displacement:
        VectorType = itk.Vector[itk.F, 3]
        ImageType = itk.Image[VectorType, 3]
        reader = itk.ImageFileReader[ImageType].New()
        reader.SetFileName(str(args.displacement))
        reader.Update()
        disp_img = reader.GetOutput()

        TransformType = itk.DisplacementFieldTransform[itk.F, 3]
        transform = TransformType.New()
        transform.SetDisplacementField(disp_img)

        J_disp = get_jacobian_from_displacement(transform, point)
        print_rotation("Displacement field rotation:", J_disp)

    if args.affine:
        J_affine = get_inverse_affine_matrix(str(args.affine))
        print_rotation("Affine transform rotation:", J_affine)

if __name__ == "__main__":
    main()

