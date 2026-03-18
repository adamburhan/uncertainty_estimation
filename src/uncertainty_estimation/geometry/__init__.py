"""Geometry primitives used by training, evaluation, and inference.

This package is the canonical home for camera/stereo geometry and bearing-space
projection utilities.
"""

from .bearings import linear, to_3d_cov, to_homogeneous
from .stereo import depth_from_disparity, extract_covs, reproject
