#!/usr/bin/env python

import SimpleITK as sitk
import numpy as np
import math
import argparse

def rotation_matrix_from_axis_angle(axis, angle_rad):
    """Rodrigues' rotation formula."""
    axis = axis / np.linalg.norm(axis)
    K = np.array([[0, -axis[2], axis[1]],
                  [axis[2], 0, -axis[0]],
                  [-axis[1], axis[0], 0]])
    R = np.eye(3) + math.sin(angle_rad) * K + (1 - math.cos(angle_rad)) * (K @ K)
    return R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, help="Random seed")
    parser.add_argument("--angle", type=float, help="Angle in degrees to rotate (default = random)", default=0)
    parser.add_argument("--axis", type=float, nargs=3, help="Axis of rotation (default = random)", default=None)
    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    # Random axis and angle
    axis = args.axis if args.axis is not None else np.random.randn(3)
    axis /= np.linalg.norm(axis)

    angle_deg = args.angle if args.angle != 0 else np.random.uniform(-89, 89)

    angle_rad = math.radians(angle_deg)

    # Get rotation matrix
    R = rotation_matrix_from_axis_angle(axis, angle_rad)

    # Reference image geometry
    size = [64, 64, 64]
    spacing = [1, 1, 1]
    origin = [-2.0, -2.0, -2.0]
    direction = [1.0, 0, 0, 0, 1.0, 0, 0, 0, 1.0]

    # Compute center of image in physical space
    center = [origin[i] + 0.5 * spacing[i] * (size[i] - 1) for i in range(3)]

    # Build transform
    transform = sitk.Euler3DTransform()
    transform.SetCenter(center)
    transform.SetMatrix(R.flatten())

    # Displacement field
    disp = sitk.TransformToDisplacementField(
        transform,
        sitk.sitkVectorFloat32,
        size,
        origin,
        spacing,
        direction,
    )

    # Save files
    affine_output_file = "rigid_affine.tfm"
    disp_output_file = "displacement_field.nii.gz"
    sitk.WriteTransform(transform, affine_output_file)
    sitk.WriteImage(disp, disp_output_file)

    # Print rotation info
    print("Affine transform written to: {}".format(affine_output_file))
    print("Displacement field written to: {}".format(disp_output_file))
    print("Ground truth rotation:")
    print("  Axis : [{:.6f}, {:.6f}, {:.6f}]".format(*axis))
    print("  Angle: {:.6f} degrees".format(angle_deg))

if __name__ == "__main__":
    main()

