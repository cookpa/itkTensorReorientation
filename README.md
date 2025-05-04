# itkTensorReorientation
Exploring tensor reorientation using ITK transforms

## Requirements

Python with ITK and SimpleITK installed. Both packages are needed because some operations
aren't available in SimpleITK, and some things that are easy in SimpleITK are hard in ITK.


Scripts here use a combination of ITK and SimpleITK to generate random rotations as affine
or displacement field transforms, and then estimate the local orientation that would be
applied to tensors in ITK / ANTs.

To make a transform pair: generate_random_rotation.py

To find the local rotation: get_local_rotation.py

To use an alternative (slower) method: local_affine_rotation.py

The local_affine_rotation.py script uses regression in a neighborhood to estimate the
affine parameters. It appears to be more accurate when the ground truth is a pure
rotation, but it remains to be seen if this helps in real data.

## Observations

It seems that displacement fields underestimate large rotations. You can see this by
generating a random rotation with

```
generate_random_rotation.py --angle 5
```

This will print an axis and angle, and then generate the rotation matrix / df.

```
$ generate_random_rotation.py --angle 5
Affine transform written to: rigid_affine.tfm
Displacement field written to: displacement_field.nii.gz
Ground truth rotation:
  Axis : [-0.252716, -0.029336, 0.967096]
  Angle: 5.000000 degrees
```

Test with

```
$ ./get_local_rotation.py --displacement displacement_field.nii.gz --affine rigid_affine.tfm 0 0 0
Displacement field rotation:
  Rotation Axis : [0.252717, 0.029337, -0.967095]
  Rotation Angle: 4.962306 degrees

Affine transform rotation:
  Rotation Axis : [0.252716, 0.029336, -0.967096]
  Rotation Angle: 4.999985 degrees
```

We can use the same axis and make the angle bigger

```
$ ./generate_random_rotation.py --angle 15 --axis -0.252716 -0.029336 0.967096
Affine transform written to: rigid_affine.tfm
Displacement field written to: displacement_field.nii.gz
Ground truth rotation:
  Axis : [-0.252716, -0.029336, 0.967096]
  Angle: 15.000000 degrees

$ ./get_local_rotation.py --displacement displacement_field.nii.gz --affine rigid_affine.tfm 0 0 0
Displacement field rotation:
  Rotation Axis : [0.252716, 0.029336, -0.967096]
  Rotation Angle: 14.051925 degrees

Affine transform rotation:
  Rotation Axis : [0.252716, 0.029336, -0.967096]
  Rotation Angle: 15.000003 degrees

$ ./generate_random_rotation.py --angle 30 --axis -0.252716 -0.029336 0.967096
Affine transform written to: rigid_affine.tfm
Displacement field written to: displacement_field.nii.gz
Ground truth rotation:
  Axis : [-0.252716, -0.029336, 0.967096]
  Angle: 30.000000 degrees

$ ./get_local_rotation.py --displacement displacement_field.nii.gz --affine rigid_affine.tfm 0 0 0
Displacement field rotation:
  Rotation Axis : [0.252716, 0.029335, -0.967096]
  Rotation Angle: 23.793974 degrees

Affine transform rotation:
  Rotation Axis : [0.252716, 0.029336, -0.967096]
  Rotation Angle: 29.999995 degrees
```

The displacement field is reasonably accurate up to about 20 degrees, but then starts to
underestimate the rotation.

## Conclusions

It would be better to use the composite transform for reorentation, rather than a single
warp field. ITK will then use the affine transform for the affine part, which should be
more accurate if there are large global rotations.

