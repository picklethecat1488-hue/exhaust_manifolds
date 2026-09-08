"""Unit tests for spatial transforms, surface normals, and port matching."""

import jax.numpy as jnp
import numpy as np
from model import Position3D, BoundaryParam, ShapeCode
from provider.transforms import (
    compute_surface_normal,
    match_intake_drain_ports,
    point_in_surface_hole,
)


def test_position_3d_data_model():
    """Verify Position3D parsing, conversion, and iteration."""
    # From floats
    p1 = Position3D(x=1.0, y=2.0, z=3.0)
    assert p1.x == 1.0 and p1.y == 2.0 and p1.z == 3.0
    assert p1.to_tuple() == (1.0, 2.0, 3.0)
    assert np.allclose(p1.to_array(), np.array([1.0, 2.0, 3.0], dtype=np.float32))

    # Unpacking
    x, y, z = p1
    assert (x, y, z) == (1.0, 2.0, 3.0)

    # Indexing
    assert p1[0] == 1.0 and p1[1] == 2.0 and p1[2] == 3.0

    # From tuple / list
    p2 = Position3D.model_validate((4.0, 5.0, 6.0))
    assert p2 == Position3D(x=4.0, y=5.0, z=6.0)

    # From string
    p3 = Position3D.model_validate("7.0 8.0 9.0")
    assert p3 == Position3D(x=7.0, y=8.0, z=9.0)

    # Equality with tuple
    assert p1 == (1.0, 2.0, 3.0)


def test_compute_surface_normal():
    """Verify surface normal calculation for cylinders and spheres."""
    # Cylinder top cap normal
    pos_top = jnp.array([0.0, 0.010, 0.050])
    norm_top = compute_surface_normal(int(ShapeCode.CYLINDER), pos_top, radius=0.030, height=0.050)
    assert np.allclose(norm_top, np.array([0.0, 0.0, 1.0]))

    # Cylinder bottom floor normal
    pos_bot = jnp.array([0.0, 0.010, 0.0])
    norm_bot = compute_surface_normal(int(ShapeCode.CYLINDER), pos_bot, radius=0.030, height=0.050)
    assert np.allclose(norm_bot, np.array([0.0, 0.0, -1.0]))

    # Cylinder side wall normal
    pos_side = jnp.array([0.030, 0.0, 0.025])
    norm_side = compute_surface_normal(int(ShapeCode.CYLINDER), pos_side, radius=0.030, height=0.050)
    assert np.allclose(norm_side, np.array([1.0, 0.0, 0.0]), atol=1e-3)

    # Canopy normal
    pos_sph = jnp.array([0.0, 0.050, 0.0])
    norm_sph = compute_surface_normal(int(ShapeCode.CANOPY), pos_sph, radius=0.050, height=0.0)
    assert np.allclose(norm_sph, np.array([0.0, 1.0, 0.0]), atol=1e-3)


def test_match_intake_drain_ports():
    """Verify world-space intake and drain port matching and normal opposing logic."""
    # Setup two boundaries: Boundary 0 has a drain at (0, 0.028, 0.050) with normal (0, 0, 1)
    # Boundary 1 has an intake at (0, 0.028, 0.050) with normal (0, 0, -1)
    b_pos_arr = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    b_orn_arr = jnp.array(
        [
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=jnp.float32,
    )

    # Create dummy b_params tensor (2, 60)
    b_params = np.zeros((2, 60), dtype=np.float32)

    # Boundary 0: has drain at (0, 0.028, 0.050) pointing +Z
    b_params[0, BoundaryParam.HAS_DRAIN] = 1.0
    b_params[0, BoundaryParam.DRAIN_POS : BoundaryParam.DRAIN_POS + 3] = [0.0, 0.028, 0.050]
    b_params[0, BoundaryParam.DRAIN_NORMAL : BoundaryParam.DRAIN_NORMAL + 3] = [0.0, 0.0, 1.0]

    # Boundary 1: has intake at (0, 0.028, 0.050) pointing -Z
    b_params[1, BoundaryParam.HAS_INTAKE] = 1.0
    b_params[1, BoundaryParam.INTAKE_POS : BoundaryParam.INTAKE_POS + 3] = [0.0, 0.028, 0.050]
    b_params[1, BoundaryParam.INTAKE_NORMAL : BoundaryParam.INTAKE_NORMAL + 3] = [0.0, 0.0, -1.0]

    matches_mask, dist_matrix = match_intake_drain_ports(b_pos_arr, b_orn_arr, jnp.array(b_params))

    # Intake on Boundary 1 matches Drain on Boundary 0: matches_mask[1, 0] must be True
    assert bool(matches_mask[1, 0])
    assert not bool(matches_mask[0, 1])  # Boundary 0 has no intake, Boundary 1 has no drain
    assert dist_matrix[1, 0] < 1e-3


def test_point_in_surface_hole():
    """Verify analytical 3D point projection and hole cutting using surface normals."""
    # 1. Planar hole with Z normal (e.g. lid top plate at z=0)
    hole_pos = jnp.array([0.0, 0.028, 0.0], dtype=jnp.float32)
    hole_norm = jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32)
    hole_r = 0.006

    # Test point at center
    p_center = jnp.array([[0.0, 0.028, 0.0]], dtype=jnp.float32)
    assert bool(point_in_surface_hole(p_center, hole_pos, hole_norm, hole_r)[0])

    # Test point inside radius (at r = 4mm)
    p_inside = jnp.array([[0.003, 0.028, 0.002]], dtype=jnp.float32)
    assert bool(point_in_surface_hole(p_inside, hole_pos, hole_norm, hole_r, normal_tol=0.005)[0])

    # Test point outside radius (at r = 8mm)
    p_outside = jnp.array([[0.008, 0.028, 0.0]], dtype=jnp.float32)
    assert not bool(point_in_surface_hole(p_outside, hole_pos, hole_norm, hole_r)[0])

    # Test point too far along normal (z = 20mm)
    p_far_z = jnp.array([[0.0, 0.028, 0.020]], dtype=jnp.float32)
    assert not bool(point_in_surface_hole(p_far_z, hole_pos, hole_norm, hole_r, normal_tol=0.005)[0])

    # 2. Side wall hole with Y normal (e.g. casing tangential exhaust along +Y)
    exhaust_pos = jnp.array([0.0, 0.028, 0.005], dtype=jnp.float32)
    exhaust_norm = jnp.array([0.0, 1.0, 0.0], dtype=jnp.float32)
    exhaust_r = 0.005

    p_exhaust_in = jnp.array([[0.002, 0.028, 0.006]], dtype=jnp.float32)
    assert bool(point_in_surface_hole(p_exhaust_in, exhaust_pos, exhaust_norm, exhaust_r)[0])

    p_exhaust_out = jnp.array([[0.008, 0.028, 0.005]], dtype=jnp.float32)
    assert not bool(point_in_surface_hole(p_exhaust_out, exhaust_pos, exhaust_norm, exhaust_r)[0])
