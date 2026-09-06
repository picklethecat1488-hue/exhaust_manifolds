"""Fluid simulation classes and JAX solvers for SPH based fluid dynamics with boundary rejection."""

from __future__ import annotations
from dataclasses import dataclass
import math
import os
from functools import partial, cached_property
from enum import Enum, IntEnum
import random
from typing import Any, Optional, cast, TYPE_CHECKING, NamedTuple
from pydantic import BaseModel, Field
from pathlib import Path
import jax
import jax.numpy as jnp
import logging

logger = logging.getLogger("provider.fluid")


import numpy as np
import pybullet as p
from scipy.spatial import cKDTree  # type: ignore

from provider.types import CollisionGroup, CollisionMask, URDFShape, URDFBoundaryType
from provider.bullet import LinkType, _is_real_physics_client
from model import (
    BoundaryConfig,
    FluidConfig,
    CoordinateSpace,
    CoordinateSystem,
    SpatialPose,
    BoundaryParam,
    FluidBody,
    FluidBodyType,
    FluidStage,
    FluidBodyTracker,
)
from provider.transforms import (
    invert_orientation,
    world_to_base_orientation,
    world_to_base_frame,
    base_to_world_frame,
    base_to_local_frame,
    local_to_base_frame,
    world_to_local_frame,
    local_to_world_frame,
    world_to_base_vector,
    base_to_world_vector,
    world_to_local_vector,
    local_to_world_vector,
    local_to_base_vector,
    base_to_local_vector,
    base_to_voxel_coord,
    voxel_to_base_coord,
    cartesian_to_cylindrical,
    cylindrical_to_cartesian,
    cartesian_to_polar_2d,
    polar_to_cartesian_2d,
    cartesian_to_spherical,
    spherical_to_cartesian,
    point_in_surface_hole,
)
from provider.boundary import (
    ImpellerBoundary,
    BowlBoundary as BowlPrimitive,
    CasingWallBoundary as CasingWallPrimitive,
    TubeWallBoundary as TubeWallPrimitive,
    CasingLidBoundary as CasingLidPrimitive,
    LidBoundary as LidPrimitive,
    ProcessedBoundaries,
    BoundaryProcessor,
)
from provider.utils import get_env_bool

if TYPE_CHECKING:
    from provider.provider import Provider


class ParticleSet:
    """A high-performance set-like container for tracking particle index sets using NumPy boolean masks."""

    def __init__(self, size: int):
        """Initialize the ParticleSet with a fixed maximum size.

        Args:
            size: The maximum number of particle indices to support.
        """
        self._mask = np.zeros(size, dtype=bool)

    def add(self, idx: int) -> None:
        """Add a single particle index to the set.

        Args:
            idx: The particle index to add.
        """
        self._mask[idx] = True

    def add_multiple(self, indices: np.ndarray) -> None:
        """Add multiple particle indices to the set in a vectorized manner.

        Args:
            indices: A NumPy array of particle indices to add.
        """
        self._mask[indices] = True

    def __contains__(self, idx: int) -> bool:
        """Check if a particle index is in the set.

        Args:
            idx: The particle index to check.

        Returns:
            True if the index is in the set, False otherwise.
        """
        return bool(self._mask[idx])

    def __len__(self) -> int:
        """Return the number of unique particle indices in the set.

        Returns:
            The number of unique indices.
        """
        return int(np.sum(self._mask))

    def __iter__(self):
        """Iterate over the particle indices in the set.

        Returns:
            An iterator over the list of active particle indices.
        """
        return iter(np.where(self._mask)[0].tolist())

    def clear(self) -> None:
        """Clear all particle indices from the set."""
        self._mask.fill(False)


class MagneticDragConfig(BaseModel):
    """Configuration parameters for magnetic coupling drag calculations."""

    magnet_radius: float = Field(..., gt=0.0, description="Radius of the coupling magnets in mm")
    magnet_thickness: float = Field(..., ge=0.0, description="Thickness of the coupling magnets in mm")
    pump_well_wall: float = Field(..., ge=-0.3, description="Wall thickness of the pump well in mm")
    magnet_count: int = Field(..., ge=0, description="Number of coupling magnet pairs")
    impeller_shaft_radius: float = Field(..., ge=0.0, description="Radius of the impeller shaft in mm")

    @classmethod
    def is_magnetic_coupling(cls, boundary: Any) -> bool:
        """Check if the given boundary config specifies magnetic coupling."""
        return getattr(boundary, "magnet_count", None) is not None and getattr(boundary, "magnet_count") > 0


@jax.jit
def calculate_magnetic_drag_jax(
    magnet_radius: float,
    magnet_thickness: float,
    pump_well_wall: float,
    magnet_count: int,
    impeller_shaft_radius: float,
) -> float:
    """Calculate the axial magnetic coupling drag torque using JAX."""
    mu0 = 4.0 * jnp.pi * 1e-7
    br = 1.45  # Residual flux density for N52 magnets (Tesla)
    mu_k = 0.15  # Wet plastic-on-plastic kinetic friction coefficient

    # Magnet dimensions (meters)
    mag_r = magnet_radius * 0.001
    mag_t = magnet_thickness * 0.001
    mag_area = jnp.pi * (mag_r**2)

    # Gap (meters): wall thickness + 0.3mm axial clearance
    gap = (pump_well_wall + 0.3) * 0.001

    # B-field at gap distance for cylindrical magnet
    b_gap = br * ((gap + mag_t) / jnp.sqrt(mag_r**2 + (gap + mag_t) ** 2) - gap / jnp.sqrt(mag_r**2 + gap**2))

    # Attractive force per pair (Newton) and total force
    f_pair = (b_gap**2 * mag_area) / (2.0 * mu0)
    f_total = magnet_count * f_pair

    # Mean contact radius of the thrust post flange (meters)
    r_mean = (impeller_shaft_radius + (impeller_shaft_radius + 1.5)) / 2.0 * 0.001

    return f_total * mu_k * r_mean


@jax.jit
def calculate_bearing_and_viscous_drag_jax(
    omega: float,
    impeller_shaft_radius: float,
    impeller_radius: float,
) -> float:
    """Calculate the journal bearing and viscous disc drag torque using JAX."""
    # Journal bearing drag (steel shaft inside PETG sleeve under water)
    f_radial = 0.5  # Assumed radial misalignment/imbalance load of 0.5 N
    mu_k = 0.15
    r_shaft = impeller_shaft_radius * 0.001
    t_bearing = jnp.where(jnp.abs(omega) > 1e-3, f_radial * mu_k * r_shaft, 0.0)

    # Viscous disc shear drag (top and bottom hub faces rotating in water)
    mu_fluid = 0.001  # Dynamic viscosity of water (Pa*s)
    g_clearance = 0.001  # Axial clearance gap (1.0 mm)
    r_impeller = impeller_radius * 0.001
    t_viscous = (jnp.pi * mu_fluid * jnp.abs(omega) * (r_impeller**4)) / g_clearance

    return t_bearing + t_viscous


# JIT compiled SPH force computation
@jax.jit
def _compute_forces_jax(
    positions: jnp.ndarray,
    velocities: jnp.ndarray,
    mass: float,
    h: float,
    rest_density: float,
    viscosity: float,
    stiffness: float,
    poly6_factor: float,
    spiky_grad_factor: float,
    visc_lap_factor: float,
    pressure_avg_factor: float,
    min_dist_threshold: float,
) -> jnp.ndarray:
    """Compute SPH forces (pressure and viscosity) using vectorized JAX.

    Args:
        positions: (N, 3) array of SPH particle positions (meters).
        velocities: (N, 3) array of SPH particle velocities (m/s).
        mass: Mass of a single SPH particle (kg).
        h: SPH kernel smoothing radius (meters).
        rest_density: Target rest density of the fluid (kg/m^3).
        viscosity: SPH viscosity coefficient.
        stiffness: SPH pressure stiffness parameter.
        poly6_factor: Precomputed coefficient factor for the Poly6 SPH density kernel.
        spiky_grad_factor: Precomputed coefficient factor for the Spiky kernel gradient.
        visc_lap_factor: Precomputed coefficient factor for the viscosity Laplacian kernel.
        pressure_avg_factor: Coefficient for averaging pairwise particle pressures.
        min_dist_threshold: Minimum squared distance to prevent divide-by-zero errors.

    Returns:
        (N, 3) array of SPH forces acting on each particle.
    """
    n = positions.shape[0]
    h2 = h * h

    # Pairwise difference vectors: shape (N, N, 3)
    diff = positions[:, None, :] - positions[None, :, :]

    # Pairwise squared distances: shape (N, N)
    r2 = jnp.sum(diff * diff, axis=-1)

    # Neighborhood mask (distance < h)
    mask = r2 < h2

    # 1. Compute density for each particle
    w = poly6_factor * (jnp.maximum(h2 - r2, 0.0) ** 3) * mask
    densities = jnp.sum(mass * w, axis=1)

    # Clamp density to rest_density to prevent negative pressures
    densities = jnp.maximum(densities, rest_density)

    # Ideal gas equation of state
    pressures = stiffness * (densities - rest_density)

    # 2. Compute forces for each particle
    r = jnp.sqrt(r2 + min_dist_threshold * min_dist_threshold)

    # Spiky kernel gradient
    hr = jnp.maximum(h - r, 0.0)
    grad_coeff = spiky_grad_factor * (hr**2) * mask

    # Pressure force term
    p_term = mass * (pressures[:, None] + pressures[None, :]) / (pressure_avg_factor * densities[None, :])

    direction = diff / r[:, :, None]

    # Exclude self-interaction (i == j)
    self_mask = (1.0 - jnp.eye(n))[:, :, None]

    f_press = -p_term[:, :, None] * grad_coeff[:, :, None] * direction * self_mask
    f_press_total = jnp.sum(f_press, axis=1)

    # Viscosity force term
    lap_coeff = visc_lap_factor * hr * mask

    # Relative velocities
    v_diff = velocities[None, :, :] - velocities[:, None, :]
    v_term = viscosity * mass / densities[None, :, None] * lap_coeff[:, :, None]

    f_visc = v_term * v_diff * self_mask
    f_visc_total = jnp.sum(f_visc, axis=1)

    vol_factor = (mass / densities)[:, None]
    f_press_scaled = f_press_total * vol_factor
    f_visc_scaled = f_visc_total * vol_factor

    return f_press_scaled + f_visc_scaled


# Integer identifiers for ShapeType
SHAPE_CYLINDER = 1
SHAPE_BOX = 2
SHAPE_PLANE = 3
SHAPE_IMPELLER = 4
SHAPE_TUBE = 5
SHAPE_SPHERE = 6
SHAPE_CASING = 7

# Integer identifiers for BoundaryType
BOUNDARY_SOLID = 0
BOUNDARY_CAVITY = 1
BOUNDARY_SOLID_CAVITY = 2


def _boundary_force_impeller_jax(
    pos_local: jnp.ndarray,
    vel_local: jnp.ndarray,
    radius: float,
    height: float,
    thickness: float,
    r_s: float,
    K: float,
    D: float,
    vane_thickness: float,
    num_vanes: int,
    vane_twist_rad: float,
    omega: float,
    t: float,
) -> tuple[jnp.ndarray, float]:
    """Compute boundary forces and reaction torque for impeller vanes.

    Args:
        pos_local: Particle positions in boundary local frame.
        vel_local: Particle velocities in boundary local frame.
        radius: Vane outer radius.
        height: Vane height.
        r_s: Particle radius.
        K: Collision penalty stiffness.
        D: Collision penalty damping.
        vane_thickness: Vane wall thickness.
        num_vanes: Number of rotary vanes.
        vane_twist_rad: Helix twist angle.
        omega: Impeller spin velocity.
        t: Simulation time.

    Returns:
        tuple: Vane force vectors in local frame, and accumulated reaction torque.
    """
    r_v = jnp.sqrt(pos_local[:, 0] ** 2 + pos_local[:, 1] ** 2)
    r_v = jnp.maximum(r_v, 1e-8)

    height_mask = (pos_local[:, 2] >= 0.0) & (pos_local[:, 2] <= height)

    # 1. Hub solid cylinder
    hub_r = thickness + 0.001
    pen_v_hub = (hub_r + r_s) - r_v
    hub_mask = height_mask & (hub_r > 0.0) & (pen_v_hub > 0.0)
    nx_v_hub = pos_local[:, 0] / r_v
    ny_v_hub = pos_local[:, 1] / r_v
    v_n_v_hub = vel_local[:, 0] * nx_v_hub + vel_local[:, 1] * ny_v_hub
    f_mag_v_hub = K * pen_v_hub - D * v_n_v_hub
    force_v_hub = jnp.stack([f_mag_v_hub * nx_v_hub, f_mag_v_hub * ny_v_hub, jnp.zeros_like(r_v)], axis=-1)
    force_v_hub = jnp.where((hub_mask[:, None]) & (f_mag_v_hub[:, None] > 0.0), force_v_hub, 0.0)

    # 2. Rotating Vanes (Blades)
    safe_height = jnp.where(height > 0.0, height, 1.0)
    pitch = vane_twist_rad / safe_height
    theta_t = pos_local[:, 2] * pitch
    phi = jnp.arctan2(pos_local[:, 1], pos_local[:, 0])
    d_phi = phi - theta_t - omega * t

    pi_N = jnp.pi / num_vanes
    d_phi_wrapped = (d_phi + pi_N) % (2.0 * pi_N) - pi_N

    dist_to_vane = r_v * jnp.sin(d_phi_wrapped)
    vane_threshold = vane_thickness / 2.0 + r_s
    pen_vane = vane_threshold - jnp.abs(dist_to_vane)

    vane_collision_mask = height_mask & (radius > 0.0) & (r_v >= hub_r) & (r_v <= radius) & (pen_vane > 0.0)

    sign_dist = jnp.sign(d_phi_wrapped)
    normal_tx = -sign_dist * jnp.sin(phi - d_phi_wrapped)
    normal_ty = sign_dist * jnp.cos(phi - d_phi_wrapped)
    normal_tz = -sign_dist * r_v * pitch

    norm = jnp.sqrt(normal_tx**2 + normal_ty**2 + normal_tz**2)
    norm_safe = jnp.maximum(norm, 1e-8)
    normal_tx /= norm_safe
    normal_ty /= norm_safe
    normal_tz /= norm_safe

    v_vane_x = omega * r_v * (-jnp.sin(phi))
    v_vane_y = omega * r_v * jnp.cos(phi)

    v_rel_n_vane = (
        (vel_local[:, 0] - v_vane_x) * normal_tx
        + (vel_local[:, 1] - v_vane_y) * normal_ty
        + (vel_local[:, 2] - 0.0) * normal_tz
    )

    f_mag_vane = K * pen_vane - D * v_rel_n_vane
    force_vane = jnp.stack([f_mag_vane * normal_tx, f_mag_vane * normal_ty, f_mag_vane * normal_tz], axis=-1)
    force_vane = jnp.where((vane_collision_mask[:, None]) & (f_mag_vane[:, None] > 0.0), force_vane, 0.0)

    total_force = force_v_hub + force_vane

    t_z = pos_local[:, 1] * force_vane[:, 0] - pos_local[:, 0] * force_vane[:, 1]
    step_torque = jnp.sum(t_z)

    return total_force, step_torque


def _compute_boundary_forces_jax(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    r_s: float,
    K: float,
    D: float,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    omega: float,
    t: float,
    solid_mask: jnp.ndarray | None = None,
    solid_friction: jnp.ndarray | None = None,
    normal_grid: jnp.ndarray | None = None,
    smooth_occ: jnp.ndarray | None = None,
    dx: float = 0.0035,
    origin: jnp.ndarray | None = None,
    base_idx: int = -1,
    nx: int = 32,
    ny: int = 32,
    nz: int = 28,
    base_vel: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute unified voxel-based boundary collision and surface friction forces with dynamic impeller physics.

    Args:
        pos: (N, 3) array of SPH particle positions in world coordinates (meters).
        vel: (N, 3) array of SPH particle velocities in world coordinates (m/s).
        r_s: SPH particle radius/search scale (meters).
        K: Boundary collision penalty stiffness coefficient.
        D: Boundary collision penalty damping coefficient.
        b_pos_arr: (B, 3) array of boundary element positions.
        b_orn_arr: (B, 4) array of boundary element orientation quaternions.
        b_shapes: (B,) array of boundary shape types.
        b_types: (B,) array of boundary types.
        b_params: (B, 16) array of boundary parameters.
        omega: Target impeller angular speed (rad/s).
        t: Current simulation time (seconds).
        solid_mask: Optional solid mask array.
        solid_friction: Optional solid friction array.
        normal_grid: Optional normal grid array.
        smooth_occ: Optional smooth occupancy array.
        dx: Voxel grid cell pitch (meters).
        origin: Voxel grid origin coordinates.
        base_idx: Index of base link in boundary arrays.
        nx: Number of voxel cells in X.
        ny: Number of voxel cells in Y.
        nz: Number of voxel cells in Z.

    Returns:
        A tuple containing:
            - forces: (N, 3) array of boundary forces acting on each particle in world coordinates.
            - vanes_torque: Reaction torque acting on the rotary vanes (impeller).
    """
    origin_arr = jnp.array([-0.112, -0.112, 0.0]) if origin is None else origin
    base_pos = jnp.where(base_idx != -1, b_pos_arr[base_idx], jnp.zeros(3))
    base_orn = jnp.where(base_idx != -1, b_orn_arr[base_idx], jnp.array([0.0, 0.0, 0.0, 1.0]))
    base_orn_inv = jnp.array([-base_orn[0], -base_orn[1], -base_orn[2], base_orn[3]], dtype=jnp.float32)

    if solid_mask is None or solid_friction is None or normal_grid is None or smooth_occ is None:
        solid_mask, _, solid_friction, normal_grid, smooth_occ = _make_grid_masks(
            dx,
            origin_arr,
            b_shapes,
            b_types,
            b_params,
            b_pos_arr,
            b_orn_arr,
            base_idx,
            nx,
            ny,
            nz,
        )

    base_vel_arr = jnp.zeros(3) if base_vel is None else base_vel
    pos_local = world_to_base_frame(pos, base_pos, base_orn_inv)
    vel_local = world_to_base_vector(vel - base_vel_arr, base_orn_inv)

    occ_p = _g2p_scalar_jax(pos_local, smooth_occ, dx, origin_arr, nx, ny, nz)
    norm_p = _g2p_jax(pos_local, normal_grid, dx, origin_arr, nx, ny, nz)
    # Prevent spurious upward ejection forces from outer vertical side walls
    base_radius = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.R_OUTER], 0.0)
    base_floor_z = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.Z_BOTTOM] + dx, dx)
    is_outer_wall = (pos_local[:, 0] ** 2 + pos_local[:, 1] ** 2) >= (base_radius - 2.0 * dx) ** 2
    norm_pz = jnp.where(
        (pos_local[:, 2] > base_floor_z) & is_outer_wall,
        jnp.minimum(0.0, norm_p[:, 2]),
        norm_p[:, 2],
    )
    norm_p_clean = jnp.stack([norm_p[:, 0], norm_p[:, 1], norm_pz], axis=-1)
    norm_p_mag = jnp.sqrt(jnp.sum(norm_p_clean**2, axis=-1, keepdims=True) + 1e-8)
    norm_unit = norm_p_clean / norm_p_mag
    fric_p = _g2p_scalar_jax(pos_local, solid_friction, dx, origin_arr, nx, ny, nz)

    pen = occ_p * dx
    v_n = -jnp.sum(vel_local * norm_unit, axis=-1)
    f_n_mag = jnp.maximum(K * pen + D * jnp.maximum(v_n, 0.0), 0.0)

    v_tan = vel_local - jnp.sum(vel_local * norm_unit, axis=-1, keepdims=True) * norm_unit
    v_tan_mag = jnp.sqrt(jnp.sum(v_tan**2, axis=-1, keepdims=True) + 1e-8)
    f_fric_tan = -fric_p[:, None] * f_n_mag[:, None] * (v_tan / v_tan_mag)

    force_local = f_n_mag[:, None] * norm_unit + f_fric_tan
    force_voxel = jnp.where((occ_p > 0.50)[:, None], force_local, 0.0)
    forces = base_to_world_vector(force_voxel, base_orn)
    vanes_torque = jnp.array(0.0)

    # Dynamic rotating impeller forces and reaction torque
    for i, shape in enumerate(b_shapes):
        is_imp = shape == SHAPE_IMPELLER
        b_pos = b_pos_arr[i]
        b_orn = b_orn_arr[i]
        b_orn_inv = jnp.array([-b_orn[0], -b_orn[1], -b_orn[2], b_orn[3]], dtype=jnp.float32)
        pos_imp = world_to_local_frame(pos, b_pos, b_orn_inv)
        vel_imp = world_to_local_vector(vel - base_vel_arr, b_orn_inv)

        radius = b_params[i, BoundaryParam.RADIUS]
        height = b_params[i, BoundaryParam.HEIGHT]
        thickness = b_params[i, BoundaryParam.THICKNESS]
        vane_thickness = b_params[i, BoundaryParam.VANE_THICKNESS]
        num_vanes = b_params[i, BoundaryParam.NUM_VANES]
        vane_twist_rad = b_params[i, BoundaryParam.VANE_TWIST_RAD]

        force_imp, torque_imp = _boundary_force_impeller_jax(
            pos_imp,
            vel_imp,
            radius,
            height,
            thickness,
            r_s,
            K,
            D,
            vane_thickness,
            num_vanes,
            vane_twist_rad,
            omega,
            t,
        )
        force_imp_world = local_to_world_vector(force_imp, b_orn)
        forces += jnp.where(is_imp, force_imp_world, 0.0)
        vanes_torque = jnp.where(is_imp, torque_imp, vanes_torque)

    return forces, vanes_torque


# LBM 3D D3Q15 Constants
_e_dir = [
    [0, 0, 0],
    [1, 0, 0],
    [-1, 0, 0],
    [0, 1, 0],
    [0, -1, 0],
    [0, 0, 1],
    [0, 0, -1],
    [1, 1, 1],
    [-1, -1, -1],
    [1, 1, -1],
    [-1, -1, 1],
    [1, -1, 1],
    [-1, 1, -1],
    [-1, 1, 1],
    [1, -1, -1],
]

_weights = jnp.array(
    [
        2.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 9.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
        1.0 / 72.0,
    ],
    dtype=jnp.float32,
)

_opposite = jnp.array([0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13], dtype=jnp.int32)


def _grid_mask_cylinder_jax(
    xb_loc: jnp.ndarray,
    yb_loc: jnp.ndarray,
    zb_loc: jnp.ndarray,
    rb_sq: jnp.ndarray,
    radius: float,
    height: float,
    thickness: float,
    z_offset: float,
    boundary_type: int,
    dx: float = 0.0035,
    has_tube: bool = False,
    has_drain: bool = False,
    tube_radius: float = 0.0,
    drain_hole_y: float = 0.0,
    drain_hole_radius: float = 0.0,
    local_tube_x: float = 0.0,
    local_tube_y: float = 0.0,
    platform_radius: float = 0.030,
) -> jnp.ndarray:
    thick = jnp.maximum(thickness, dx)
    # Cavity: cylinder side wall (rb_sq >= radius**2) and solid floor (zb_loc <= z_offset)
    is_wall = (zb_loc >= z_offset) & (zb_loc <= height) & (rb_sq >= radius**2)
    is_wall = jnp.where(thick > 0.0, is_wall & (rb_sq <= (radius + thick) ** 2), is_wall)
    is_wall = jnp.where(has_drain, False, is_wall)
    is_floor = (zb_loc >= z_offset - thick) & (zb_loc <= z_offset) & (rb_sq <= (radius + thick) ** 2)

    # Drainage hole and tube pass-through openings cut through the cylinder floor/solid
    in_platform = (has_drain & has_tube) & (
        (xb_loc - local_tube_x) ** 2 + (yb_loc - local_tube_y) ** 2 < platform_radius**2
    )
    in_drain = (
        (has_drain & (drain_hole_radius > 0.0))
        & (xb_loc**2 + (yb_loc - drain_hole_y) ** 2 < drain_hole_radius**2)
        & (zb_loc >= z_offset - thick)
        & (zb_loc <= z_offset + thick)
        & (~in_platform)
    )
    in_tube_hole = (
        (has_tube & (tube_radius > 0.0))
        & ((xb_loc - local_tube_x) ** 2 + (yb_loc - local_tube_y) ** 2 < tube_radius**2)
        & (zb_loc >= z_offset - thick)
        & (zb_loc <= z_offset + thick)
    )

    is_cavity = (is_wall | is_floor) & (~in_drain) & (~in_tube_hole)
    is_solid = (rb_sq <= radius**2) & (zb_loc >= z_offset) & (zb_loc <= height) & (~in_tube_hole)
    return jnp.where(boundary_type == 1, is_cavity, is_solid)


def _grid_mask_tube_jax(
    xb_loc: jnp.ndarray,
    yb_loc: jnp.ndarray,
    zb_loc: jnp.ndarray,
    rb_sq: jnp.ndarray,
    radius: float,
    height: float,
    thickness: float,
    slot_height: float,
    slot_width: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Compute grid solid mask and interior fluid mask for a tube boundary."""
    r_outer = radius
    r_inner = radius - thickness
    is_solid = (rb_sq >= r_inner**2) & (rb_sq <= r_outer**2) & (zb_loc >= 0.0) & (zb_loc <= height)

    half_width = slot_width / 2.0
    is_cutout = (zb_loc < slot_height) & (yb_loc > 0.0) & (jnp.abs(xb_loc) < half_width)
    is_solid = jnp.where(slot_height > 0.0, is_solid & (~is_cutout), is_solid)

    is_tube_mask = (rb_sq < r_inner**2) & (zb_loc >= 0.0) & (zb_loc <= height)
    tube_mask = jnp.where(height > 0.05, is_tube_mask, jnp.zeros_like(rb_sq, dtype=jnp.bool_))

    return is_solid, tube_mask


def _grid_mask_casing_jax(
    xb_loc: jnp.ndarray,
    yb_loc: jnp.ndarray,
    zb_loc: jnp.ndarray,
    rb_sq: jnp.ndarray,
    radius: float,
    height: float,
    thickness: float,
    ceiling_thickness: float,
    slot_height: float,
    slot_width: float,
    cutoff_y: float,
) -> jnp.ndarray:
    """Compute grid solid mask for a pump casing boundary."""
    r_outer = radius
    r_inner = radius - thickness
    r_inlet = radius * 0.35
    is_wall = (rb_sq >= r_inner**2) & (rb_sq <= r_outer**2) & (zb_loc >= 0.0) & (zb_loc <= height)
    is_ceiling = (
        (rb_sq >= r_inlet**2) & (rb_sq <= r_outer**2) & (zb_loc >= height - ceiling_thickness) & (zb_loc <= height)
    )

    half_width = slot_width / 2.0
    is_cutout = (zb_loc < slot_height) & (yb_loc > 0.0) & (jnp.abs(xb_loc) < half_width)

    is_solid = (is_wall | is_ceiling) & (~is_cutout)
    is_solid = jnp.where(cutoff_y != 0.0, is_solid & (yb_loc <= cutoff_y), is_solid)
    return is_solid


def _grid_mask_plane_jax(zb_loc: jnp.ndarray, thickness: float) -> jnp.ndarray:
    """Compute grid solid mask for a planar boundary."""
    return (zb_loc >= -thickness) & (zb_loc <= 0.0)


def _grid_mask_sphere_jax(dist_sq: jnp.ndarray, radius: float) -> jnp.ndarray:
    """Compute grid solid mask for a spherical boundary."""
    return dist_sq <= radius**2


def _make_grid_masks(
    dx: float,
    origin: jnp.ndarray,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_idx: int,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Compute combined 3D solid grid occupancy masks and normal vectors."""
    n_pts = nx * ny * nz
    flat_shape = (n_pts,)

    ix = jnp.arange(nx)
    iy = jnp.arange(ny)
    iz = jnp.arange(nz)
    gx, gy, gz = jnp.meshgrid(ix, iy, iz, indexing="ij")
    cx = origin[0] + (gx.ravel() + 0.5) * dx
    cy = origin[1] + (gy.ravel() + 0.5) * dx
    cz = origin[2] + (gz.ravel() + 0.5) * dx
    coords = jnp.stack([cx, cy, cz], axis=-1)

    solid_mask = jnp.zeros(flat_shape, dtype=jnp.bool_)
    tube_mask = jnp.zeros(flat_shape, dtype=jnp.bool_)
    solid_friction = jnp.zeros(flat_shape, dtype=jnp.float32)
    normal_grid_flat = jnp.zeros((n_pts, 3), dtype=jnp.float32)

    has_tube_bound = jnp.any(b_shapes == SHAPE_TUBE)
    tube_idx = jnp.argmax(b_shapes == SHAPE_TUBE)
    tube_world_pos = jnp.where(has_tube_bound, b_pos_arr[tube_idx], jnp.zeros(3))
    base_pos = jnp.where(base_idx != -1, b_pos_arr[base_idx], jnp.zeros(3))
    base_orn = jnp.where(base_idx != -1, b_orn_arr[base_idx], jnp.array([0.0, 0.0, 0.0, 1.0]))
    base_orn_inv = invert_orientation(base_orn)

    tube_pos_local = world_to_base_frame(tube_world_pos, base_pos, base_orn_inv)
    tube_xb = jnp.where(has_tube_bound, tube_pos_local[0], 0.0)
    tube_yb = jnp.where(has_tube_bound, tube_pos_local[1], 0.0)
    tube_rb = jnp.where(has_tube_bound, b_params[tube_idx, BoundaryParam.R_INNER], 0.0)
    tube_bot_local_z = jnp.where(has_tube_bound, tube_pos_local[2] + b_params[tube_idx, BoundaryParam.Z_BOTTOM], 0.0)
    tube_top_local_z = jnp.where(has_tube_bound, tube_pos_local[2] + b_params[tube_idx, BoundaryParam.Z_TOP], 0.0)

    for idx, shape in enumerate(b_shapes):
        is_imp = shape == SHAPE_IMPELLER
        b_pos = b_pos_arr[idx]
        b_orn = b_orn_arr[idx]

        b_pos_loc = world_to_base_frame(b_pos, base_pos, base_orn_inv)
        b_orn_loc = world_to_base_orientation(b_orn, base_orn)
        b_orn_loc_inv = invert_orientation(b_orn_loc)
        pos_b = base_to_local_frame(coords, b_pos_loc, b_orn_loc_inv)

        xb_loc = pos_b[:, 0]
        yb_loc = pos_b[:, 1]
        zb_loc = pos_b[:, 2]
        rb_sq = xb_loc**2 + yb_loc**2

        radius = b_params[idx, BoundaryParam.RADIUS]
        height = b_params[idx, BoundaryParam.HEIGHT]
        thickness = b_params[idx, BoundaryParam.THICKNESS]
        z_offset = b_params[idx, BoundaryParam.Z_OFFSET]
        slot_height = b_params[idx, BoundaryParam.SLOT_HEIGHT]
        slot_width = b_params[idx, BoundaryParam.SLOT_WIDTH]
        ceiling_thickness = b_params[idx, BoundaryParam.CEILING_THICKNESS]
        cutoff_y = b_params[idx, BoundaryParam.CUTOFF_Y]
        has_tube = b_params[idx, BoundaryParam.HAS_TUBE] > 0.5
        has_drain = b_params[idx, BoundaryParam.HAS_DRAIN] > 0.5
        tube_radius = b_params[idx, BoundaryParam.TUBE_RADIUS]
        drain_hole_y = b_params[idx, BoundaryParam.DRAIN_HOLE_Y]
        drain_hole_radius = b_params[idx, BoundaryParam.DRAIN_HOLE_RADIUS]
        b_fric = b_params[idx, BoundaryParam.BOUNDARY_FRICTION]

        # Calculate exact local tube coordinates for boundary idx
        tube_pos_in_b = base_to_local_frame(tube_pos_local, b_pos_loc, b_orn_loc_inv)
        local_tube_x = jnp.where(has_tube_bound, tube_pos_in_b[0], 0.0)
        local_tube_y = jnp.where(has_tube_bound, tube_pos_in_b[1], 0.0)

        # CYLINDER
        is_cyl = shape == SHAPE_CYLINDER
        is_solid_cyl = _grid_mask_cylinder_jax(
            xb_loc,
            yb_loc,
            zb_loc,
            rb_sq,
            radius,
            height,
            thickness,
            z_offset,
            b_types[idx],
            dx,
            has_tube,
            has_drain,
            tube_radius,
            drain_hole_y,
            drain_hole_radius,
            local_tube_x,
            local_tube_y,
        )

        # TUBE
        is_tube = shape == SHAPE_TUBE
        is_solid_tube, is_tube_mask = _grid_mask_tube_jax(
            xb_loc, yb_loc, zb_loc, rb_sq, radius, height, thickness, slot_height, slot_width
        )

        # CASING
        is_casing = shape == SHAPE_CASING
        is_solid_casing = _grid_mask_casing_jax(
            xb_loc,
            yb_loc,
            zb_loc,
            rb_sq,
            radius,
            height,
            thickness,
            ceiling_thickness,
            slot_height,
            slot_width,
            cutoff_y,
        )

        # PLANE
        is_plane = shape == SHAPE_PLANE
        is_solid_plane = _grid_mask_plane_jax(zb_loc, thickness)

        # SPHERE
        is_sphere = shape == SHAPE_SPHERE
        is_solid_sphere = _grid_mask_sphere_jax(xb_loc**2 + yb_loc**2 + zb_loc**2, radius)

        is_solid = jnp.where(is_cyl, is_solid_cyl, jnp.zeros(flat_shape, dtype=jnp.bool_))
        is_solid = jnp.where(is_tube, is_solid_tube, is_solid)
        is_solid = jnp.where(is_casing, is_solid_casing, is_solid)
        is_solid = jnp.where(is_plane, is_solid_plane, is_solid)
        is_solid = jnp.where(is_sphere, is_solid_sphere, is_solid)

        # Analytical normal vectors for each shape type
        thick = jnp.maximum(thickness, dx)
        r_safe = jnp.maximum(jnp.sqrt(rb_sq), 1e-6)
        cyl_wall_norm_loc = jnp.stack([-xb_loc / r_safe, -yb_loc / r_safe, jnp.zeros_like(xb_loc)], axis=-1)
        cyl_floor_norm_loc = jnp.stack([jnp.zeros_like(xb_loc), jnp.zeros_like(yb_loc), jnp.ones_like(zb_loc)], axis=-1)
        is_cyl_floor = (zb_loc >= z_offset - thick) & (zb_loc <= z_offset)
        cyl_norm_loc = jnp.where(is_cyl_floor[:, None], cyl_floor_norm_loc, cyl_wall_norm_loc)
        cyl_norm_base = local_to_base_vector(cyl_norm_loc, b_orn_loc)

        r_inner_tube = b_params[idx, BoundaryParam.R_INNER]
        r_outer_tube = b_params[idx, BoundaryParam.R_OUTER]
        r_mid_tube = (r_outer_tube + r_inner_tube) * 0.5
        tube_sign = jnp.where(r_safe < r_mid_tube, -1.0, 1.0)
        tube_norm_loc = tube_sign[:, None] * jnp.stack(
            [xb_loc / r_safe, yb_loc / r_safe, jnp.zeros_like(xb_loc)], axis=-1
        )
        tube_norm_base = local_to_base_vector(tube_norm_loc, b_orn_loc)

        r_inner_casing = b_params[idx, BoundaryParam.R_INNER]
        r_outer_casing = b_params[idx, BoundaryParam.R_OUTER]
        r_mid_casing = (r_outer_casing + r_inner_casing) * 0.5
        casing_sign = jnp.where(r_safe < r_mid_casing, -1.0, 1.0)
        casing_floor_norm_loc = jnp.stack(
            [jnp.zeros_like(xb_loc), jnp.zeros_like(yb_loc), jnp.ones_like(zb_loc)], axis=-1
        )
        casing_ceil_norm_loc = jnp.stack(
            [jnp.zeros_like(xb_loc), jnp.zeros_like(yb_loc), -jnp.ones_like(zb_loc)], axis=-1
        )
        casing_wall_norm_loc = casing_sign[:, None] * jnp.stack(
            [xb_loc / r_safe, yb_loc / r_safe, jnp.zeros_like(xb_loc)], axis=-1
        )
        is_casing_floor = zb_loc <= b_params[idx, BoundaryParam.Z_BOTTOM]
        is_casing_ceil = zb_loc >= b_params[idx, BoundaryParam.Z_TOP]
        casing_norm_loc = jnp.where(
            is_casing_floor[:, None],
            casing_floor_norm_loc,
            jnp.where(is_casing_ceil[:, None], casing_ceil_norm_loc, casing_wall_norm_loc),
        )
        casing_norm_base = local_to_base_vector(casing_norm_loc, b_orn_loc)

        plane_norm_base = local_to_base_vector(
            jnp.stack([jnp.zeros_like(xb_loc), jnp.zeros_like(yb_loc), jnp.ones_like(zb_loc)], axis=-1),
            b_orn_loc,
        )

        r_3d_safe = jnp.maximum(jnp.sqrt(xb_loc**2 + yb_loc**2 + zb_loc**2), 1e-6)
        sphere_sign = jnp.where(b_types[idx] == 1, -1.0, 1.0)
        sphere_norm_loc = sphere_sign * jnp.stack([xb_loc / r_3d_safe, yb_loc / r_3d_safe, zb_loc / r_3d_safe], axis=-1)
        sphere_norm_base = local_to_base_vector(sphere_norm_loc, b_orn_loc)

        norm_base_i = jnp.where(is_cyl, cyl_norm_base, jnp.zeros((n_pts, 3), dtype=jnp.float32))
        norm_base_i = jnp.where(is_tube, tube_norm_base, norm_base_i)
        norm_base_i = jnp.where(is_casing, casing_norm_base, norm_base_i)
        norm_base_i = jnp.where(is_plane, plane_norm_base, norm_base_i)
        norm_base_i = jnp.where(is_sphere, sphere_norm_base, norm_base_i)

        normal_grid_flat = jnp.where(is_solid[:, None] & (~is_imp), norm_base_i, normal_grid_flat)

        # Ignore impeller nodes in solid_mask
        solid_mask = jnp.where(is_imp, solid_mask, solid_mask | is_solid)
        solid_friction = jnp.where(is_solid & (~is_imp), b_fric, solid_friction)
        tube_mask = jnp.where(is_tube, tube_mask | is_tube_mask, tube_mask)

    # Clear solid mask inside the tube passage along the entire tube column to prevent internal blockage
    in_tube_interior = (
        ((coords[:, 0] - tube_xb) ** 2 + (coords[:, 1] - tube_yb) ** 2 < tube_rb**2)
        & (coords[:, 2] >= tube_bot_local_z)
        & (coords[:, 2] <= tube_top_local_z)
    )
    solid_mask = jnp.where(has_tube_bound, solid_mask & (~in_tube_interior), solid_mask)
    solid_friction = jnp.where(has_tube_bound & in_tube_interior, 0.0, solid_friction)

    solid_occ = solid_mask.reshape((nx, ny, nz)).astype(jnp.float32)
    smooth_occ = solid_occ
    normal_grid = normal_grid_flat.reshape((nx, ny, nz, 3))
    return (
        solid_mask.reshape((nx, ny, nz)),
        tube_mask.reshape((nx, ny, nz)),
        solid_friction.reshape((nx, ny, nz)),
        normal_grid,
        smooth_occ,
    )


def _make_impeller_mask_and_vel(
    dx: float,
    origin: jnp.ndarray,
    angle: float,
    omega: float,
    c_scale: float,
    b_shapes: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_idx: int,
    nx: int,
    ny: int,
    nz: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    x = origin[0] + jnp.arange(nx) * dx
    y = origin[1] + jnp.arange(ny) * dx
    z = origin[2] + jnp.arange(nz) * dx
    X, Y, Z = jnp.meshgrid(x, y, z, indexing="ij")

    impeller_mask = jnp.zeros(X.shape, dtype=jnp.bool_)
    imp_vx = jnp.zeros(X.shape)
    imp_vy = jnp.zeros(X.shape)
    imp_vz = jnp.zeros(X.shape)

    base_pos = jnp.where(base_idx != -1, b_pos_arr[base_idx], jnp.zeros(3))
    base_orn = jnp.where(base_idx != -1, b_orn_arr[base_idx], jnp.array([0.0, 0.0, 0.0, 1.0]))

    for idx, shape in enumerate(b_shapes):
        is_imp = shape == SHAPE_IMPELLER

        pos_b = b_pos_arr[idx]
        orn_b = b_orn_arr[idx]

        pos_base = jnp.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=-1)
        pos_world = base_to_world_frame(pos_base, base_pos, base_orn)

        orn_b_inv = invert_orientation(orn_b)
        pos_b_local = world_to_local_frame(pos_world, pos_b, orn_b_inv)

        flat_shape = X.shape
        xb_loc = pos_b_local[:, 0].reshape(flat_shape)
        yb_loc = pos_b_local[:, 1].reshape(flat_shape)
        zb_loc = pos_b_local[:, 2].reshape(flat_shape)

        r_sq = xb_loc**2 + yb_loc**2

        radius = b_params[idx, BoundaryParam.RADIUS]
        height = b_params[idx, BoundaryParam.HEIGHT]
        thickness = b_params[idx, BoundaryParam.THICKNESS]
        vane_thickness = b_params[idx, BoundaryParam.VANE_THICKNESS]
        num_vanes = b_params[idx, BoundaryParam.NUM_VANES]
        vane_twist_rad = b_params[idx, BoundaryParam.VANE_TWIST_RAD]

        # Hub
        hub_r = thickness + 0.001
        hub_mask = (r_sq <= hub_r**2) & (zb_loc >= 0.0) & (zb_loc <= height)

        # Blades
        safe_height = jnp.where(height > 0.0, height, 1.0)
        pitch = vane_twist_rad / safe_height
        total_angle = angle + zb_loc * pitch

        xr = xb_loc * jnp.cos(total_angle) + yb_loc * jnp.sin(total_angle)
        yr = -xb_loc * jnp.sin(total_angle) + yb_loc * jnp.cos(total_angle)

        theta = jnp.arctan2(yr, xr)
        angle_sep = 2.0 * jnp.pi / num_vanes
        theta_mod = (theta + jnp.pi) % angle_sep - (angle_sep / 2.0)

        dist_to_blade = (r_sq**0.5) * jnp.sin(theta_mod)
        vane_thickness_lbm = jnp.maximum(vane_thickness, 0.8 * dx)
        blades_mask = (
            (jnp.abs(dist_to_blade) <= vane_thickness_lbm / 2.0)
            & (r_sq >= hub_r**2)
            & (r_sq <= radius**2)
            & (zb_loc >= 0.0)
            & (zb_loc <= height)
        )

        curr_mask = (hub_mask | blades_mask) & is_imp
        impeller_mask = impeller_mask | curr_mask

        # Velocity field generated by impeller
        vx_local = -omega * yb_loc
        vy_local = omega * xb_loc
        vz_local = jnp.zeros_like(xb_loc)

        v_rot_local = jnp.stack([vx_local, vy_local, vz_local], axis=-1)
        v_rot_world = local_to_world_vector(v_rot_local.reshape((-1, 3)), orn_b).reshape(flat_shape + (3,))

        # Lattice units
        vx_lat = v_rot_world[..., 0] * c_scale
        vy_lat = v_rot_world[..., 1] * c_scale
        vz_lat = v_rot_world[..., 2] * c_scale

        imp_vx = jnp.where(curr_mask, vx_lat, imp_vx)
        imp_vy = jnp.where(curr_mask, vy_lat, imp_vy)
        imp_vz = jnp.where(curr_mask, vz_lat, imp_vz)

    return impeller_mask, imp_vx, imp_vy, imp_vz


def _lbm_step_3d_full_jax(
    f: jnp.ndarray,
    solid_mask: jnp.ndarray,
    impeller_mask: jnp.ndarray,
    imp_vx: jnp.ndarray,
    imp_vy: jnp.ndarray,
    imp_vz: jnp.ndarray,
    gravity: jnp.ndarray,
    tau: float = 0.6,
    dt_sub: float = 1.0 / 1200.0,
    tube_mask: jnp.ndarray | None = None,
    tube_uz: float = 0.0,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Execute a complete Lattice Boltzmann D3Q15 timestep with moving boundaries."""
    # 1. Macroscopic variables
    rho = jnp.sum(f, axis=0)
    rho_safe = jnp.where(rho > 1e-6, rho, 1.0)
    ux = jnp.zeros(rho.shape)
    uy = jnp.zeros(rho.shape)
    uz = jnp.zeros(rho.shape)
    for i, ei in enumerate(_e_dir):
        ux += f[i] * ei[0]
        uy += f[i] * ei[1]
        uz += f[i] * ei[2]
    ux = ux / rho_safe
    uy = uy / rho_safe
    uz = uz / rho_safe

    # 2. Collision (BGK) with Mach limiting to prevent lattice compressibility blowup
    u_lat_mag = jnp.sqrt(ux**2 + uy**2 + uz**2 + 1e-8)
    u_scale = jnp.minimum(0.40 / u_lat_mag, 1.0)
    ux = ux * u_scale
    uy = uy * u_scale
    uz = uz * u_scale

    feq_list = []
    for i, (ei, w) in enumerate(zip(_e_dir, _weights)):
        ei_u = ei[0] * ux + ei[1] * uy + ei[2] * uz
        u_sq = ux**2 + uy**2 + uz**2
        feq_i = w * rho * (1.0 + 3.0 * ei_u + 4.5 * (ei_u**2) - 1.5 * u_sq)
        feq_list.append(feq_i)
    f_eq = jnp.stack(feq_list)
    f_post = f - (f - f_eq) / tau

    # 3. Body Force (Gravity)
    for i, (ei, w) in enumerate(zip(_e_dir, _weights)):
        e_dot_g = ei[0] * gravity[0] + ei[1] * gravity[1] + ei[2] * gravity[2]
        f_post = f_post.at[i].add(3.0 * w * rho * e_dot_g * dt_sub)

    # Ensure non-negative population distributions for unconditional numerical stability
    f_post = jnp.maximum(f_post, 0.0)

    # 4. Streaming
    f_coll = f_post
    f_stream = jnp.zeros_like(f)
    for i, ei in enumerate(_e_dir):
        f_stream = f_stream.at[i].set(jnp.roll(f_coll[i], shift=(ei[0], ei[1], ei[2]), axis=(0, 1, 2)))

    # 5. Moving & Static Wall Boundary Conditions (Half-Way Bounce-Back)
    f_next = f_stream
    for i in range(15):
        val_next = jnp.where(solid_mask, f_coll[i], f_next[_opposite[i]])
        f_next = f_next.at[_opposite[i]].set(val_next)

    for i, (ei, w) in enumerate(zip(_e_dir, _weights)):
        neighbor_impeller = jnp.roll(impeller_mask, shift=(-ei[0], -ei[1], -ei[2]), axis=(0, 1, 2))
        adj_fluid = (~solid_mask) & (~impeller_mask) & neighbor_impeller

        imp_node_x = jnp.roll(imp_vx, shift=(-ei[0], -ei[1], -ei[2]), axis=(0, 1, 2))
        imp_node_y = jnp.roll(imp_vy, shift=(-ei[0], -ei[1], -ei[2]), axis=(0, 1, 2))
        imp_node_z = jnp.roll(imp_vz, shift=(-ei[0], -ei[1], -ei[2]), axis=(0, 1, 2))

        u_wall_dot_c = ei[0] * imp_node_x + ei[1] * imp_node_y + ei[2] * imp_node_z
        bounce_term = 2.0 * w * rho * (3.0 * u_wall_dot_c)

        val_next = jnp.where(adj_fluid, f_coll[i] - bounce_term, f_next[_opposite[i]])
        f_next = f_next.at[_opposite[i]].set(val_next)

    # Re-evaluate final state
    rho_new = jnp.sum(f_next, axis=0)
    rho_safe_new = jnp.where(rho_new > 1e-6, rho_new, 1.0)
    ux_new = jnp.zeros(rho_new.shape)
    uy_new = jnp.zeros(rho_new.shape)
    uz_new = jnp.zeros(rho_new.shape)
    for i, ei in enumerate(_e_dir):
        ux_new += f_next[i] * ei[0]
        uy_new += f_next[i] * ei[1]
        uz_new += f_next[i] * ei[2]
    ux_new = jnp.where(~solid_mask, ux_new / rho_safe_new, 0.0)
    uy_new = jnp.where(~solid_mask, uy_new / rho_safe_new, 0.0)
    uz_new = jnp.where(~solid_mask, uz_new / rho_safe_new, 0.0)

    # Tube velocity forcing at the end of the step
    if tube_mask is not None:
        ux_new = jnp.where(tube_mask, 0.0, ux_new)
        uy_new = jnp.where(tube_mask, 0.0, uy_new)
        uz_new = jnp.where(tube_mask, tube_uz, uz_new)

    # Impeller velocity forcing at the end of the step
    ux_new = jnp.where(impeller_mask, imp_vx, ux_new)
    uy_new = jnp.where(impeller_mask, imp_vy, uy_new)
    uz_new = jnp.where(impeller_mask, imp_vz, uz_new)

    u_new = jnp.stack([ux_new, uy_new, uz_new], axis=-1)
    return f_next, u_new


@partial(jax.jit, static_argnums=(4, 5, 6))
def _g2p_jax(pos, grid_u, dx, origin, nx=32, ny=32, nz=28):
    gp = (pos - origin) / dx
    idx0 = jnp.floor(gp).astype(jnp.int32)
    idx0 = jnp.clip(idx0, 0, jnp.array([nx - 2, ny - 2, nz - 2]))
    t = gp - idx0

    w000 = (1.0 - t[:, 0]) * (1.0 - t[:, 1]) * (1.0 - t[:, 2])
    w100 = t[:, 0] * (1.0 - t[:, 1]) * (1.0 - t[:, 2])
    w010 = (1.0 - t[:, 0]) * t[:, 1] * (1.0 - t[:, 2])
    w001 = (1.0 - t[:, 0]) * (1.0 - t[:, 1]) * t[:, 2]
    w110 = t[:, 0] * t[:, 1] * (1.0 - t[:, 2])
    w101 = t[:, 0] * (1.0 - t[:, 1]) * t[:, 2]
    w011 = (1.0 - t[:, 0]) * t[:, 1] * t[:, 2]
    w111 = t[:, 0] * t[:, 1] * t[:, 2]

    u_p = (
        w000[:, None] * grid_u[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
        + w100[:, None] * grid_u[idx0[:, 0] + 1, idx0[:, 1], idx0[:, 2]]
        + w010[:, None] * grid_u[idx0[:, 0], idx0[:, 1] + 1, idx0[:, 2]]
        + w001[:, None] * grid_u[idx0[:, 0], idx0[:, 1], idx0[:, 2] + 1]
        + w110[:, None] * grid_u[idx0[:, 0] + 1, idx0[:, 1] + 1, idx0[:, 2]]
        + w101[:, None] * grid_u[idx0[:, 0] + 1, idx0[:, 1], idx0[:, 2] + 1]
        + w011[:, None] * grid_u[idx0[:, 0], idx0[:, 1] + 1, idx0[:, 2] + 1]
        + w111[:, None] * grid_u[idx0[:, 0] + 1, idx0[:, 1] + 1, idx0[:, 2] + 1]
    )
    return u_p


@partial(jax.jit, static_argnums=(4, 5, 6))
def _g2p_scalar_jax(pos, grid_val, dx, origin, nx=32, ny=32, nz=28):
    gp = (pos - origin) / dx
    idx0 = jnp.floor(gp).astype(jnp.int32)
    idx0 = jnp.clip(idx0, 0, jnp.array([nx - 2, ny - 2, nz - 2]))
    t = gp - idx0

    w000 = (1.0 - t[:, 0]) * (1.0 - t[:, 1]) * (1.0 - t[:, 2])
    w100 = t[:, 0] * (1.0 - t[:, 1]) * (1.0 - t[:, 2])
    w010 = (1.0 - t[:, 0]) * t[:, 1] * (1.0 - t[:, 2])
    w001 = (1.0 - t[:, 0]) * (1.0 - t[:, 1]) * t[:, 2]
    w110 = t[:, 0] * t[:, 1] * (1.0 - t[:, 2])
    w101 = t[:, 0] * (1.0 - t[:, 1]) * t[:, 2]
    w011 = (1.0 - t[:, 0]) * t[:, 1] * t[:, 2]
    w111 = t[:, 0] * t[:, 1] * t[:, 2]

    val_p = (
        w000 * grid_val[idx0[:, 0], idx0[:, 1], idx0[:, 2]]
        + w100 * grid_val[idx0[:, 0] + 1, idx0[:, 1], idx0[:, 2]]
        + w010 * grid_val[idx0[:, 0], idx0[:, 1] + 1, idx0[:, 2]]
        + w001 * grid_val[idx0[:, 0], idx0[:, 1], idx0[:, 2] + 1]
        + w110 * grid_val[idx0[:, 0] + 1, idx0[:, 1] + 1, idx0[:, 2]]
        + w101 * grid_val[idx0[:, 0] + 1, idx0[:, 1], idx0[:, 2] + 1]
        + w011 * grid_val[idx0[:, 0], idx0[:, 1] + 1, idx0[:, 2] + 1]
        + w111 * grid_val[idx0[:, 0] + 1, idx0[:, 1] + 1, idx0[:, 2] + 1]
    )
    return val_p


@partial(jax.jit, static_argnums=(3, 4, 5))
def _compute_dynamic_fluid_bodies_jax(
    pos_local: jnp.ndarray,
    dx: float,
    origin: jnp.ndarray,
    nx: int = 32,
    ny: int = 32,
    nz: int = 28,
    cavity_floor_z: float = 0.0,
    z_max_pool: float = 0.0,
    r_s: float = 0.003,
    c_center: float = 4.0,
    c_neighbor: float = 2.0,
    c_diagonal: float = 1.0,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Dynamically recompute physical fluid bodies, local column surface heights, and horizontal leveling gradients."""
    gp = (pos_local - origin) / dx
    ix = jnp.clip(jnp.floor(gp[:, 0]).astype(jnp.int32), 0, nx - 1)
    iy = jnp.clip(jnp.floor(gp[:, 1]).astype(jnp.int32), 0, ny - 1)

    # 1. Dynamic 2D column water surface height in reservoir basin
    in_basin = (pos_local[:, 2] <= z_max_pool) & (pos_local[:, 2] >= cavity_floor_z)
    surf_z_grid = (
        jnp.full((nx, ny), cavity_floor_z).at[ix, iy].max(jnp.where(in_basin, pos_local[:, 2], cavity_floor_z))
    )
    min_z_grid = jnp.full((nx, ny), 10.0).at[ix, iy].min(jnp.where(in_basin, pos_local[:, 2], 10.0))
    col_count = jnp.zeros((nx, ny), dtype=jnp.float32).at[ix, iy].add(jnp.where(in_basin, 1.0, 0.0))

    # 2. Dynamic 2D spatial surface smoothing and horizontal hydrostatic leveling gradient
    surf_pad = jnp.pad(surf_z_grid, ((1, 1), (1, 1)), mode="edge")
    w_norm = c_center + 4.0 * c_neighbor + 4.0 * c_diagonal
    surf_smooth = (
        c_center * surf_pad[1:-1, 1:-1]
        + c_neighbor * (surf_pad[2:, 1:-1] + surf_pad[:-2, 1:-1] + surf_pad[1:-1, 2:] + surf_pad[1:-1, :-2])
        + c_diagonal * (surf_pad[2:, 2:] + surf_pad[:-2, :-2] + surf_pad[2:, :-2] + surf_pad[:-2, 2:])
    ) / jnp.maximum(1e-4, w_norm)

    smooth_pad = jnp.pad(surf_smooth, ((1, 1), (1, 1)), mode="edge")
    grad_x = (smooth_pad[2:, 1:-1] - smooth_pad[:-2, 1:-1]) / (2.0 * dx)
    grad_y = (smooth_pad[1:-1, 2:] - smooth_pad[1:-1, :-2]) / (2.0 * dx)

    # 3. Sample back to particle coordinates
    p_surf_z = surf_smooth[ix, iy]
    p_min_z = min_z_grid[ix, iy]
    p_grad_x = grad_x[ix, iy]
    p_grad_y = grad_y[ix, iy]
    p_col_count = col_count[ix, iy]

    # Particles inside resting/moving continuous fluid bodies (grounded on cavity floor within column free surface)
    is_grounded = p_min_z <= cavity_floor_z + 4.0 * r_s
    in_fluid_body = in_basin & is_grounded & (pos_local[:, 2] <= p_surf_z + 2.0 * r_s) & (p_col_count >= 2.0)
    level_grad_local = jnp.stack([-p_grad_x, -p_grad_y, jnp.zeros_like(p_grad_x)], axis=-1)

    return in_fluid_body, p_surf_z, level_grad_local, p_col_count


@jax.jit
def _lbm_step_3d_jax(grid_rho, grid_u):
    feq_list = []
    for i, (ei, w) in enumerate(zip(_e_dir, _weights)):
        ei_u = ei[0] * grid_u[:, :, :, 0] + ei[1] * grid_u[:, :, :, 1] + ei[2] * grid_u[:, :, :, 2]
        u_sq = grid_u[:, :, :, 0] ** 2 + grid_u[:, :, :, 1] ** 2 + grid_u[:, :, :, 2] ** 2
        feq_i = w * grid_rho * (1.0 + 3.0 * ei_u + 4.5 * (ei_u**2) - 1.5 * u_sq)
        feq_list.append(feq_i)

    f_coll = jnp.stack(feq_list)

    f_next_list = []
    for i, ei in enumerate(_e_dir):
        rolled = jnp.roll(f_coll[i], shift=(ei[0], ei[1], ei[2]), axis=(0, 1, 2))
        f_next_list.append(rolled)
    f_next = jnp.stack(f_next_list)

    rho_new = jnp.sum(f_next, axis=0)
    rho_safe = jnp.where(rho_new > 1e-8, rho_new, 1.0)

    u_new = jnp.zeros(grid_rho.shape + (3,))
    for i, ei in enumerate(_e_dir):
        u_new = u_new.at[:, :, :, 0].add(f_next[i] * ei[0])
        u_new = u_new.at[:, :, :, 1].add(f_next[i] * ei[1])
        u_new = u_new.at[:, :, :, 2].add(f_next[i] * ei[2])

    u_new = u_new / rho_safe[:, :, :, None]
    u_new = jnp.where((rho_new > 1e-8)[:, :, :, None], u_new, 0.0)
    return u_new


class PhysicsConfig:
    """Static configuration parameters for the SPH-LBM physics solver."""

    def __init__(
        self,
        mass: float,
        dt_sub: float,
        n_substeps: int,
        gravity: tuple[float, float, float],
        base_idx: int,
        K_boundary: float,
        D_boundary: float,
        r_s: float,
        high_damping_value: float,
        nx: int,
        ny: int,
        nz: int,
        dx: float,
        origin: tuple[float, float, float],
        processed_boundaries: Optional[ProcessedBoundaries] = None,
        boundary_configs: Optional[tuple | list] = None,
    ):
        """Initialize physics configuration parameters for the SPH-LBM solver.

        Args:
            mass: Mass per particle.
            dt_sub: Substep time duration.
            n_substeps: Number of sub-iterations per simulation step.
            gravity: 3D gravitational acceleration vector.
            base_idx: Index of base boundary element.
            K_boundary: Penalty spring stiffness.
            D_boundary: Boundary damping coefficient.
            r_s: SPH particle search radius.
            high_damping_value: Outer boundary damping factor.
            nx: X grid dimension.
            ny: Y grid dimension.
            nz: Z grid dimension.
            dx: Voxel grid pitch.
            origin: 3D grid minimum origin.
            processed_boundaries: Pre-processed intermediate boundary structure.
            boundary_configs: Optional raw boundary configuration sequence.
        """
        self.mass = mass
        self.dt_sub = dt_sub
        self.n_substeps = n_substeps
        self.gravity = gravity
        self.base_idx = base_idx
        self.K_boundary = K_boundary
        self.D_boundary = D_boundary
        self.r_s = r_s
        self.high_damping_value = high_damping_value
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.dx = dx
        self.origin = origin

        if processed_boundaries is not None:
            self.processed_boundaries = processed_boundaries
        elif boundary_configs is not None:
            self.processed_boundaries = BoundaryProcessor.process(boundary_configs)
        else:
            raise ValueError("Either processed_boundaries or boundary_configs must be provided to PhysicsConfig.")

        self.boundary_configs = (
            boundary_configs if boundary_configs is not None else tuple(self.processed_boundaries.boundaries)
        )
        pb = self.processed_boundaries
        self.b_shapes_jax = jnp.array(pb.b_shapes, dtype=jnp.int32)
        self.b_types_jax = jnp.array(pb.b_types, dtype=jnp.int32)
        self.b_params_jax = jnp.array(pb.b_params, dtype=jnp.float32)
        self.gravity_arr = jnp.array(self.gravity, dtype=jnp.float32)
        self.origin_arr = jnp.array(self.origin, dtype=jnp.float32)


def _lbm_step_subroutine(
    pos_curr: jnp.ndarray,
    f_curr: jnp.ndarray,
    base_pos: jnp.ndarray,
    base_orn: jnp.ndarray,
    omega: float,
    t_curr: float,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_idx: int,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    origin: jnp.ndarray,
    dt_sub: float,
    gravity: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """LBM solver subroutine: builds masks, solves lattice dynamics, and returns grid states."""
    solid_mask, tube_mask, solid_friction, normal_grid, smooth_occ = _make_grid_masks(
        dx, origin, b_shapes, b_types, b_params, b_pos_arr, b_orn_arr, base_idx, nx, ny, nz
    )

    angle = omega * t_curr
    c_scale = dt_sub / dx
    impeller_mask, imp_vx, imp_vy, imp_vz = _make_impeller_mask_and_vel(
        dx, origin, angle, omega, c_scale, b_shapes, b_params, b_pos_arr, b_orn_arr, base_idx, nx, ny, nz
    )

    tau = 0.65

    # Find impeller radius dynamically
    r_impeller = 0.0
    for j, shape_j in enumerate(b_shapes):
        r_impeller = jnp.where(shape_j == SHAPE_IMPELLER, b_params[j, BoundaryParam.RADIUS], r_impeller)

    # Find tube inner radius dynamically
    r_tube = 0.0
    for j, shape_j in enumerate(b_shapes):
        is_tb = shape_j == SHAPE_TUBE
        r_inner_tb = b_params[j, BoundaryParam.R_INNER]
        r_tube = jnp.where(is_tb, r_inner_tb, r_tube)

    # Default to 0.015 if no impeller is present to support tests that don't model the impeller
    r_impeller_eff = jnp.where(r_impeller > 0.0, r_impeller, 0.015)

    constriction_ratio = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.SLOT_CONSTRICTION_RATIO], 1.0)
    v_tip = r_impeller_eff * jnp.abs(omega)
    tube_uz_phys = jnp.where(
        (r_tube > 0.0) & (r_impeller_eff > 0.0),
        jnp.minimum(constriction_ratio, 1.5) * v_tip,
        0.0,
    )
    tube_uz_lat = tube_uz_phys * dt_sub / dx

    base_orn_inv = invert_orientation(base_orn)
    gravity_local = world_to_base_vector(gravity, base_orn_inv)

    f_next, u_grid = _lbm_step_3d_full_jax(
        f_curr,
        solid_mask,
        impeller_mask,
        imp_vx,
        imp_vy,
        imp_vz,
        gravity_local,
        tau,
        dt_sub,
        tube_mask,
        tube_uz_lat,
    )
    return f_next, u_grid, solid_mask, solid_friction, normal_grid, smooth_occ


def _g2p_mapping_subroutine(
    pos_curr: jnp.ndarray,
    vel_curr: jnp.ndarray,
    u_grid: jnp.ndarray,
    base_pos: jnp.ndarray,
    base_orn: jnp.ndarray,
    base_vel: jnp.ndarray,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    origin: jnp.ndarray,
    dt_sub: float,
    b_shapes: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_idx: int,
    r_s: float,
) -> jnp.ndarray:
    """G2P subroutine: interpolates grid velocities to particle velocities in world frame for submerged fluid."""
    c_scale = dt_sub / dx

    base_orn_inv = invert_orientation(base_orn)
    pos_local = world_to_base_frame(pos_curr, base_pos, base_orn_inv)
    r_local, _, _ = cartesian_to_cylindrical(pos_local)

    active_col = (pos_curr[:, 2] < 100.0)[:, None]
    lbm_vel_local = _g2p_jax(pos_local, u_grid, dx, origin, nx, ny, nz)
    lbm_vel_local = jnp.where(active_col, lbm_vel_local / c_scale, 0.0)
    vel_grid_world = base_to_world_vector(lbm_vel_local, base_orn) + base_vel

    # Dynamic moving voxel fluid bodies: continuously compute physical fluid shapes from voxel occupancy
    cavity_floor_z = jnp.where(
        base_idx != -1,
        b_pos_arr[base_idx, 2] - base_pos[2] + b_params[base_idx, BoundaryParam.Z_OFFSET],
        0.0,
    )
    z_max_pool = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.POOL_MAX_Z], 0.0)
    in_fluid_body, _, _, _ = _compute_dynamic_fluid_bodies_jax(
        pos_local, dx, origin, nx, ny, nz, cavity_floor_z=cavity_floor_z, z_max_pool=z_max_pool, r_s=r_s
    )

    in_tube_or_casing = jnp.zeros(pos_curr.shape[0], dtype=jnp.bool_)
    for i, shape in enumerate(b_shapes):
        b_pos = b_pos_arr[i]
        b_orn = b_orn_arr[i]
        b_orn_inv = invert_orientation(b_orn)
        pos_b = world_to_local_frame(pos_curr, b_pos, b_orn_inv)
        r_b_xy, _, _ = cartesian_to_cylindrical(pos_b)
        inner_r = b_params[i, BoundaryParam.R_INNER]
        r_outer = b_params[i, BoundaryParam.R_OUTER]
        z_top = b_params[i, BoundaryParam.Z_TOP]
        is_tube = (shape == SHAPE_TUBE) & (r_b_xy <= inner_r) & (pos_b[:, 2] >= 0.0) & (pos_b[:, 2] < z_top - 2.0 * r_s)
        is_casing = (shape == SHAPE_CASING) & (r_b_xy <= r_outer) & (pos_b[:, 2] >= 0.0) & (pos_b[:, 2] <= z_top)
        in_tube_or_casing = in_tube_or_casing | is_tube | is_casing
    in_fluid_continuum = jnp.where(
        base_idx != -1, in_fluid_body & (~in_tube_or_casing), jnp.ones(pos_curr.shape[0], dtype=jnp.bool_)
    )
    has_casing = jnp.any(b_shapes == SHAPE_CASING)
    # When enclosed by a pump casing, reservoir continuum maintains Lagrangian settling momentum; open impellers drive LBM swirl
    vel_continuum = jnp.where(
        has_casing,
        vel_curr,
        jnp.stack([vel_grid_world[:, 0], vel_grid_world[:, 1], vel_curr[:, 2]], axis=-1),
    )
    vel_world = jnp.where(in_fluid_continuum[:, None], vel_continuum, vel_curr)
    return jnp.where(active_col, vel_world, 0.0), in_fluid_continuum


def _compute_particle_forces_subroutine(
    pos_curr: jnp.ndarray,
    vel_world: jnp.ndarray,
    omega: float,
    t_curr: float,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    K_boundary: float,
    D_boundary: float,
    r_s: float,
    mass: float,
    gravity: jnp.ndarray,
    solid_mask: jnp.ndarray,
    solid_friction: jnp.ndarray,
    normal_grid: jnp.ndarray,
    smooth_occ: jnp.ndarray,
    dx: float,
    origin: jnp.ndarray,
    base_idx: int,
    nx: int,
    ny: int,
    nz: int,
    in_fluid_continuum: jnp.ndarray | None = None,
    base_vel: jnp.ndarray | None = None,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Force subroutine: computes containment boundary penalty forces, casing suction forces, and gravity."""
    b_forces, step_torque = _compute_boundary_forces_jax(
        pos_curr,
        vel_world,
        r_s,
        K_boundary,
        D_boundary,
        b_pos_arr,
        b_orn_arr,
        b_shapes,
        b_types,
        b_params,
        omega,
        t_curr,
        solid_mask,
        solid_friction,
        normal_grid,
        smooth_occ,
        dx,
        origin,
        base_idx,
        nx,
        ny,
        nz,
        base_vel=base_vel,
    )

    b_accel = b_forces / mass
    max_b_accel = 35.0
    b_accel_mags = jnp.linalg.norm(b_accel, axis=1, keepdims=True)
    b_accel_mags_safe = jnp.maximum(b_accel_mags, 1e-8)
    b_accel_clamped = b_accel * jnp.minimum(max_b_accel / b_accel_mags_safe, 1.0)
    r_impeller = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.IMPELLER_RADIUS], 0.0)
    r_impeller_eff = jnp.where(r_impeller > 0.0, r_impeller, 0.015)
    g_mag = jnp.linalg.norm(gravity)
    v_tip = jnp.abs(omega) * r_impeller_eff

    base_pos = jnp.where(base_idx != -1, b_pos_arr[base_idx], jnp.zeros(3))
    base_radius = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.R_OUTER], 0.0)
    cavity_floor_z = jnp.where(
        base_idx != -1,
        b_params[base_idx, BoundaryParam.Z_OFFSET],
        0.0,
    )

    has_tube = jnp.any(b_shapes == SHAPE_TUBE)
    tube_idx = jnp.argmax(b_shapes == SHAPE_TUBE)
    tube_y_check = jnp.where(has_tube, b_pos_arr[tube_idx, 1] - base_pos[1], 0.0)
    tube_r_check = jnp.where(has_tube, b_params[tube_idx, BoundaryParam.R_INNER], 0.0)

    # Impeller suction intake force directed toward central casing intake and tangential injection into tube
    suction_accel = jnp.zeros_like(pos_curr)
    for i, shape in enumerate(b_shapes):
        casing_pos = b_pos_arr[i]
        casing_orn = b_orn_arr[i]
        casing_orn_inv = invert_orientation(casing_orn)
        pos_casing = world_to_local_frame(pos_curr, casing_pos, casing_orn_inv)
        radius = b_params[i, BoundaryParam.RADIUS]
        casing_h = b_params[i, BoundaryParam.HEIGHT]

        has_intake_i = b_params[i, BoundaryParam.HAS_INTAKE] > 0.5
        intake_pos_i = b_params[i, BoundaryParam.INTAKE_POS_X : BoundaryParam.INTAKE_POS_Z + 1]
        intake_norm_i = b_params[i, BoundaryParam.INTAKE_NORMAL_X : BoundaryParam.INTAKE_NORMAL_Z + 1]
        intake_r_i = b_params[i, BoundaryParam.INTAKE_RADIUS]

        # 1. Intake draw near and above the designated intake port along surface normal
        inlet_r_eff = jnp.where(intake_r_i > 0.0, intake_r_i, radius * 0.40)
        target_in = intake_pos_i - intake_norm_i * 0.008
        d_in = target_in - pos_casing
        dist_in = jnp.sqrt(jnp.sum(d_in**2, axis=-1, keepdims=True) + 1e-8)
        dir_casing = d_in / dist_in
        dir_world = local_to_world_vector(dir_casing, casing_orn)

        d_in_xy = jnp.sqrt((pos_casing[:, 0] - intake_pos_i[0]) ** 2 + (pos_casing[:, 1] - intake_pos_i[1]) ** 2)
        in_suction = (
            has_intake_i
            & (d_in_xy <= inlet_r_eff + 0.015)
            & (pos_casing[:, 2] >= casing_h - 0.002)
            & (pos_casing[:, 2] <= casing_h + 0.030)
        )
        suction_strength = (v_tip * 2.0 + g_mag * 1.5) * jnp.clip(1.0 - dist_in / (inlet_r_eff + 0.025), 0.0, 1.0)
        suction_accel_i = jnp.where(in_suction[:, None], dir_world * suction_strength, 0.0)

        # 2. Inside the volute chamber near the tangential exit channel leading to drain port (tube)
        has_drain_i = b_params[i, BoundaryParam.HAS_DRAIN] > 0.5
        r_inlet_xy, _, _ = cartesian_to_cylindrical(pos_casing)
        drain_px = b_params[i, BoundaryParam.DRAIN_POS_X]
        drain_py = b_params[i, BoundaryParam.DRAIN_POS_Y]
        drain_pz = b_params[i, BoundaryParam.DRAIN_POS_Z]
        d_to_drain = jnp.stack(
            [drain_px - pos_casing[:, 0], drain_py - pos_casing[:, 1], drain_pz - pos_casing[:, 2] + 0.010],
            axis=-1,
        )
        dist_to_drain = jnp.sqrt(jnp.sum(d_to_drain**2, axis=-1, keepdims=True) + 1e-8)
        dir_to_drain = d_to_drain / dist_to_drain
        dir_to_drain_world = local_to_world_vector(dir_to_drain, casing_orn)
        slot_h_i = b_params[i, BoundaryParam.SLOT_HEIGHT]
        in_volute_channel = (
            has_drain_i
            & (pos_casing[:, 1] >= 0.0)
            & (pos_casing[:, 2] <= slot_h_i + 2.0 * r_s)
            & (r_inlet_xy <= radius + 2.0 * r_s)
        )
        volute_accel_i = dir_to_drain_world * (v_tip * 3.0 + g_mag * 2.0)

        is_casing = shape == SHAPE_CASING
        casing_total_accel = jnp.where(in_volute_channel[:, None], volute_accel_i, suction_accel_i)
        suction_accel += jnp.where(is_casing, casing_total_accel, jnp.zeros_like(pos_curr))

    tube_pump_accel = jnp.zeros_like(pos_curr)
    for i, shape in enumerate(b_shapes):
        tube_pos = b_pos_arr[i]
        tube_orn = b_orn_arr[i]
        tube_orn_inv = invert_orientation(tube_orn)
        pos_tube = world_to_local_frame(pos_curr, tube_pos, tube_orn_inv)
        r_tube_xy, _, _ = cartesian_to_cylindrical(pos_tube)
        inner_r = b_params[i, BoundaryParam.R_INNER]
        tube_h = b_params[i, BoundaryParam.HEIGHT]
        spout_z_min = b_params[i, BoundaryParam.SPOUT_Z_MIN]
        spout_head_room = tube_h - spout_z_min

        # Query lid deflection / socket height offset dynamically from URDF boundary metadata
        lid_height_offset = 0.0
        for j, shape_j in enumerate(b_shapes):
            is_dome = (
                (shape_j == SHAPE_CYLINDER)
                & (b_types[j] == 1)
                & (b_params[j, BoundaryParam.HEIGHT] < 0.02)
                & (b_params[j, BoundaryParam.HEIGHT] > 0.0)
            )
            lid_height_offset = jnp.where(
                is_dome,
                b_params[j, BoundaryParam.HEIGHT] + b_params[j, BoundaryParam.THICKNESS],
                lid_height_offset,
            )

        tube_effective_h = tube_h - lid_height_offset

        v_tube = world_to_local_vector(vel_world, tube_orn_inv)
        v_z_tube = v_tube[:, 2]

        # Derive flow velocity from impeller geometry, slot constriction ratio, and motor speed
        r_tube_eff = inner_r
        constriction_ratio = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.SLOT_CONSTRICTION_RATIO], 1.0)

        v_flow_est = jnp.where(
            (r_tube_eff > 0.0) & (r_impeller_eff > 0.0),
            jnp.minimum(constriction_ratio, 1.0) * (r_impeller_eff * jnp.abs(omega)),
            0.0,
        )
        height_frac = jnp.clip(1.0 - pos_tube[:, 2] / (tube_h + 1e-6), 0.0, 1.0)
        v_target_rise = jnp.maximum(v_flow_est * 1.8, 0.80) * (0.25 + 0.75 * height_frac)

        pump_lift_scalar = jnp.where(
            jnp.abs(omega) > 1e-3,
            g_mag + jnp.maximum(0.0, v_target_rise - v_z_tube) * 80.0,
            0.0,
        )

        in_tube = (r_tube_xy <= inner_r + r_s) & (pos_tube[:, 2] >= -2.0 * r_s) & (pos_tube[:, 2] <= tube_h)
        up_vector = local_to_world_vector(jnp.array([0.0, 0.0, 1.0]), tube_orn)
        up_world = up_vector[None, :] * pump_lift_scalar[:, None]

        # Radial spreading and outward deflection at the fountain spout opening (above tube exit)
        r_outer = b_params[i, BoundaryParam.R_OUTER]
        at_spout = (pos_tube[:, 2] > tube_h) & (pos_tube[:, 2] <= tube_h + 0.020) & (r_tube_xy <= r_outer + 0.020)

        r_xy_safe = jnp.maximum(r_tube_xy, 1e-5)
        radial_unit_local = jnp.stack(
            [pos_tube[:, 0] / r_xy_safe, pos_tube[:, 1] / r_xy_safe, jnp.zeros_like(pos_tube[:, 0])],
            axis=-1,
        )
        radial_unit_world = local_to_world_vector(radial_unit_local, tube_orn)

        # Purely radial 360-degree expansion over dome with downward deflection (centered on spout)
        down_dir_world = local_to_world_vector(jnp.array([0.0, 0.0, -1.0]), tube_orn)
        dome_disp = radial_unit_world * 0.85 + down_dir_world[None, :] * 0.45
        disp_mag = jnp.sqrt(jnp.sum(dome_disp**2, axis=-1, keepdims=True) + 1e-8)
        spout_out_dir = dome_disp / disp_mag
        spout_accel = spout_out_dir * (g_mag * 2.0 + v_flow_est * 2.0)

        tube_pump_accel_i = jnp.where(
            at_spout[:, None],
            spout_accel,
            jnp.where(in_tube[:, None], up_world, 0.0),
        )

        is_tube = shape == SHAPE_TUBE
        tube_pump_accel += jnp.where(is_tube, tube_pump_accel_i, jnp.zeros_like(pos_curr))

    lid_drain_accel = jnp.zeros_like(pos_curr)
    lid_slope_ratio = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.LID_SLOPE_RATIO], 0.0)
    for i, shape in enumerate(b_shapes):
        lid_pos = b_pos_arr[i]
        lid_orn = b_orn_arr[i]
        lid_orn_inv = invert_orientation(lid_orn)
        pos_lid = world_to_local_frame(pos_curr, lid_pos, lid_orn_inv)

        radius = b_params[i, BoundaryParam.RADIUS]
        has_drain = b_params[i, BoundaryParam.HAS_DRAIN] > 0.5
        drain_hole_y = b_params[i, BoundaryParam.DRAIN_HOLE_Y]
        drain_target_z = b_params[i, BoundaryParam.DRAIN_TARGET_Z]
        drain_influence_r = b_params[i, BoundaryParam.DRAIN_INFLUENCE_RADIUS]

        r_lid_xy, _, _ = cartesian_to_cylindrical(pos_lid)
        r_lid_xy_safe = jnp.maximum(r_lid_xy, 1e-5)
        tray_z_min = b_params[i, BoundaryParam.TRAY_Z_MIN]
        tray_z_max = b_params[i, BoundaryParam.TRAY_Z_MAX]

        dx_spout = pos_lid[:, 0] - 0.0
        dy_spout = pos_lid[:, 1] - tube_y_check
        dist_spout_xy = jnp.sqrt(dx_spout**2 + dy_spout**2 + 1e-8)
        platform_radius = b_params[i, BoundaryParam.INTAKE_RADIUS]
        in_platform_zone = has_tube & (platform_radius > 0.0) & (dist_spout_xy <= platform_radius)

        drain_pos_i = b_params[i, BoundaryParam.DRAIN_POS_X : BoundaryParam.DRAIN_POS_Z + 1]
        drain_norm_i = b_params[i, BoundaryParam.DRAIN_NORMAL_X : BoundaryParam.DRAIN_NORMAL_Z + 1]
        drain_r_i = b_params[i, BoundaryParam.DRAIN_RADIUS]
        in_drain_hole = (
            has_drain
            & (drain_r_i > 0.0)
            & point_in_surface_hole(pos_lid, drain_pos_i, drain_norm_i, drain_r_i, normal_tol=0.080)
            & (~in_platform_zone)
        )

        # 1. Particles on the solid raised drinking platform
        on_drinking_platform = in_platform_zone & (pos_lid[:, 2] >= 0.0) & (pos_lid[:, 2] <= tray_z_max + 2.0 * r_s)

        # 2. Surface sheet flow across the raised drinking platform: radial expansion with slope towards drain
        radial_spout_dir = jnp.stack(
            [dx_spout / dist_spout_xy, dy_spout / dist_spout_xy, jnp.zeros_like(dx_spout)],
            axis=-1,
        )
        dx_to_drain = 0.0 - pos_lid[:, 0]
        dy_to_drain = drain_hole_y - pos_lid[:, 1]
        dist_to_drain_xy = jnp.sqrt(dx_to_drain**2 + dy_to_drain**2 + 1e-8)
        dir_slope_local = jnp.stack(
            [dx_to_drain / dist_to_drain_xy, dy_to_drain / dist_to_drain_xy, jnp.zeros_like(dx_to_drain)],
            axis=-1,
        )
        sheet_flow_dir = radial_spout_dir * 0.70 + dir_slope_local * 0.30
        sheet_flow_mag = jnp.sqrt(jnp.sum(sheet_flow_dir**2, axis=-1, keepdims=True) + 1e-8)
        sheet_flow_unit = sheet_flow_dir / sheet_flow_mag
        slope_factor = 0.45

        # Upward floor support on the solid drinking platform
        v_lid = world_to_local_vector(vel_world, lid_orn_inv)
        cushion_lid_z = -jnp.minimum(v_lid[:, 2], 0.0) * 45.0
        v_lid_xy = jnp.stack([v_lid[:, 0], v_lid[:, 1], jnp.zeros_like(v_lid[:, 2])], axis=-1)
        surface_friction_local = -v_lid_xy * 12.0
        support_normal_local = jnp.stack(
            [jnp.zeros_like(dx_to_drain), jnp.zeros_like(dx_to_drain), (g_mag + cushion_lid_z)],
            axis=-1,
        )
        platform_accel_local = sheet_flow_unit * (g_mag * slope_factor) + support_normal_local + surface_friction_local

        # 3. Waterfall cascade as soon as water meets the edge of the platform or drain cutout
        # Active strictly in the free-fall region above the reservoir pool surface
        wall_band_r_max = b_params[i, BoundaryParam.WALL_BAND_R_MAX]
        in_waterfall = (
            (~in_platform_zone)
            & (r_lid_xy < wall_band_r_max)
            & (pos_lid[:, 2] >= tray_z_min)
            & (pos_lid[:, 2] <= tray_z_max + 2.0 * r_s)
        )
        waterfall_plunge_local = jnp.stack(
            [
                jnp.zeros_like(dx_spout),
                (dy_to_drain / dist_to_drain_xy) * 0.15,
                -0.95 * jnp.ones_like(dx_spout),
            ],
            axis=-1,
        )
        waterfall_plunge_mag = jnp.sqrt(jnp.sum(waterfall_plunge_local**2, axis=-1, keepdims=True) + 1e-8)
        waterfall_plunge_unit = waterfall_plunge_local / waterfall_plunge_mag
        waterfall_accel_local = waterfall_plunge_unit * g_mag - v_lid_xy * 12.0

        lid_accel_local = jnp.where(
            on_drinking_platform[:, None],
            platform_accel_local,
            waterfall_accel_local,
        )
        total_lid_accel_world = local_to_world_vector(lid_accel_local, lid_orn)

        is_submerged = b_params[i, BoundaryParam.IS_SUBMERGED] > 0.5
        is_lid = (shape == SHAPE_CYLINDER) & has_drain & (~is_submerged)
        active_lid_mask = on_drinking_platform | in_waterfall
        lid_drain_accel += jnp.where(is_lid & active_lid_mask[:, None], total_lid_accel_world, jnp.zeros_like(pos_curr))

    effective_gravity = gravity[None, :]

    # Dynamic moving voxel fluid bodies: continuously compute physical fluid shapes from voxel occupancy
    base_pos_b = b_pos_arr[base_idx]
    base_orn_b = b_orn_arr[base_idx]
    base_orn_b_inv = invert_orientation(base_orn_b)
    pos_b = world_to_local_frame(pos_curr, base_pos_b, base_orn_b_inv)
    r_b, _, _ = cartesian_to_cylindrical(pos_b)
    v_b = world_to_local_vector(vel_world, base_orn_b_inv)
    v_z_b = v_b[:, 2]

    z_max_pool = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.POOL_MAX_Z], 0.0)
    in_fluid_body, p_surf_z, level_grad_local, _ = _compute_dynamic_fluid_bodies_jax(
        pos_b, dx, origin, nx, ny, nz, cavity_floor_z=cavity_floor_z, z_max_pool=z_max_pool, r_s=r_s
    )

    # 1. Dynamic hydrostatic support, vertical wave damping, and impact deceleration for smooth pool water
    depth_pressure = jnp.clip((p_surf_z - pos_b[:, 2]) / (4.0 * r_s), 0.0, 4.0)
    cushion_accel_z = -jnp.minimum(v_z_b, 0.0) * 25.0  # Decelerate falling waterfall droplets
    vertical_wave_damping_z = -v_z_b * 8.0  # Viscous vertical damping arresting sloshing oscillations
    total_support_z = g_mag * (1.0 + depth_pressure * 0.25) + cushion_accel_z + vertical_wave_damping_z

    hydrostatic_support_world = local_to_world_vector(
        jnp.stack([jnp.zeros_like(r_b), jnp.zeros_like(r_b), total_support_z], axis=-1),
        base_orn_b,
    )
    hydrostatic_accel = jnp.where(
        in_fluid_body[:, None],
        hydrostatic_support_world,
        0.0,
    )

    # 2. Dynamic horizontal leveling gradient derived continuously from the smoothed surface height field
    level_grad_damped = level_grad_local * (g_mag * 0.40) - v_b * 1.5
    level_accel_world = local_to_world_vector(level_grad_damped, base_orn_b)
    leveling_accel = jnp.where(
        in_fluid_body[:, None],
        level_accel_world,
        0.0,
    )

    # 3. Dynamic replenishment draw toward active pump intake sink:
    # When water is intaked into the pump system, surrounding reservoir fluid near the casing
    # is drawn inward to continuously replenish intaked volume and maintain hydrostatic equilibrium.
    pool_inward_accel = jnp.zeros_like(pos_curr)
    for i, shape in enumerate(b_shapes):
        has_intake_i = b_params[i, BoundaryParam.HAS_INTAKE] > 0.5
        intake_pos_i = b_params[i, BoundaryParam.INTAKE_POS_X : BoundaryParam.INTAKE_POS_Z + 1]
        intake_world_i = local_to_world_frame(intake_pos_i[None, :], b_pos_arr[i], b_orn_arr[i])[0]
        intake_base_i = world_to_local_frame(intake_world_i[None, :], base_pos_b, base_orn_b_inv)[0]

        dx_in = intake_base_i[0] - pos_b[:, 0]
        dy_in = intake_base_i[1] - pos_b[:, 1]
        dist_in_xy = jnp.sqrt(dx_in**2 + dy_in**2 + 1e-8)
        dir_in_base = jnp.stack([dx_in / dist_in_xy, dy_in / dist_in_xy, jnp.zeros_like(dx_in)], axis=-1)
        dir_in_world = local_to_world_vector(dir_in_base, base_orn_b)

        # Replenishment draw acts within the intake replenishment radius (dist_in_xy <= 45mm)
        in_intake_zone = (dist_in_xy <= 0.045) & (pos_b[:, 2] >= cavity_floor_z)
        replenish_mag = (v_tip * 0.20 + g_mag * 0.15) * jnp.clip(1.0 - dist_in_xy / 0.045, 0.0, 1.0)
        leveling_accel_i = jnp.where(
            in_fluid_body[:, None] & has_intake_i & (v_tip > 1e-3) & in_intake_zone[:, None],
            dir_in_world * replenish_mag[:, None],
            0.0,
        )
        pool_inward_accel += leveling_accel_i

    accel = (
        b_accel_clamped
        + suction_accel
        + tube_pump_accel
        + lid_drain_accel
        + pool_inward_accel
        + leveling_accel
        + effective_gravity
        + hydrostatic_accel
    )
    return accel, step_torque, b_forces


def _ccd_planar_shelf_boundary(
    pos_curr_loc: jnp.ndarray,
    pos_next_loc: jnp.ndarray,
    v_rel_local: jnp.ndarray,
    z_plane: float,
    normal_sign: float,
    radius: float,
    shelf_depth: float,
    has_drain: jnp.ndarray,
    drain_pos: jnp.ndarray,
    drain_normal: jnp.ndarray,
    drain_radius: float,
    has_tube: jnp.ndarray,
    tube_pos: jnp.ndarray,
    tube_normal: jnp.ndarray,
    tube_radius: float,
    has_intake: jnp.ndarray,
    intake_pos: jnp.ndarray,
    intake_normal: jnp.ndarray,
    intake_radius: float,
    active_boundary: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Unified continuous collision detection for two-sided planar shelf plates with thickness and cutouts."""
    t_shelf = jnp.maximum(shelf_depth, 0.002)
    z_top = z_plane
    z_bot = z_plane - t_shelf

    # Target surface evaluation at pos_next_loc
    r_next = jnp.sqrt(pos_next_loc[:, 0] ** 2 + pos_next_loc[:, 1] ** 2)
    in_platform_next = has_intake & point_in_surface_hole(
        pos_next_loc, intake_pos, intake_normal, intake_radius, normal_tol=0.100
    )
    in_drain_next = (
        has_drain
        & point_in_surface_hole(pos_next_loc, drain_pos, drain_normal, drain_radius, normal_tol=0.100)
        & (~in_platform_next)
    )
    in_tube_next = has_tube & point_in_surface_hole(pos_next_loc, tube_pos, tube_normal, tube_radius, normal_tol=0.100)
    is_solid_next = (r_next < radius) & (~in_drain_next) & (~in_tube_next) & active_boundary
    z_top_next = z_top + jnp.where(in_platform_next, intake_pos[2], 0.0)

    # 1. Trajectory intersection with top surface Z = z_top_curr
    in_platform_curr = has_intake & point_in_surface_hole(
        pos_curr_loc, intake_pos, intake_normal, intake_radius, normal_tol=0.100
    )
    z_top_curr = z_top + jnp.where(in_platform_curr, intake_pos[2], 0.0)
    denom_top = pos_curr_loc[:, 2] - pos_next_loc[:, 2]
    t_top = jnp.clip((pos_curr_loc[:, 2] - z_top_curr) / jnp.where(jnp.abs(denom_top) > 1e-6, denom_top, 1.0), 0.0, 1.0)
    x_top = pos_curr_loc[:, 0] + t_top * (pos_next_loc[:, 0] - pos_curr_loc[:, 0])
    y_top = pos_curr_loc[:, 1] + t_top * (pos_next_loc[:, 1] - pos_curr_loc[:, 1])
    r_top = jnp.sqrt(x_top**2 + y_top**2)
    pos_top_3d = jnp.stack([x_top, y_top, jnp.full_like(x_top, z_top_curr)], axis=-1)

    in_platform_top = has_intake & point_in_surface_hole(
        pos_top_3d, intake_pos, intake_normal, intake_radius, normal_tol=0.100
    )
    in_drain_top = (
        has_drain
        & point_in_surface_hole(pos_top_3d, drain_pos, drain_normal, drain_radius, normal_tol=0.100)
        & (~in_platform_top)
    )
    in_tube_top = has_tube & point_in_surface_hole(pos_top_3d, tube_pos, tube_normal, tube_radius, normal_tol=0.100)
    is_solid_top = (r_top < radius) & (~in_drain_top) & (~in_tube_top) & active_boundary

    # 2. Trajectory intersection with bottom surface Z = z_bot
    denom_bot = pos_curr_loc[:, 2] - pos_next_loc[:, 2]
    t_bot = jnp.clip((pos_curr_loc[:, 2] - z_bot) / jnp.where(jnp.abs(denom_bot) > 1e-6, denom_bot, 1.0), 0.0, 1.0)
    x_bot = pos_curr_loc[:, 0] + t_bot * (pos_next_loc[:, 0] - pos_curr_loc[:, 0])
    y_bot = pos_curr_loc[:, 1] + t_bot * (pos_next_loc[:, 1] - pos_curr_loc[:, 1])
    r_bot = jnp.sqrt(x_bot**2 + y_bot**2)
    pos_bot_3d = jnp.stack([x_bot, y_bot, jnp.full_like(x_bot, z_bot)], axis=-1)

    in_platform_bot = has_intake & point_in_surface_hole(
        pos_bot_3d, intake_pos, intake_normal, intake_radius, normal_tol=0.100
    )
    in_drain_bot = (
        has_drain
        & point_in_surface_hole(pos_bot_3d, drain_pos, drain_normal, drain_radius, normal_tol=0.100)
        & (~in_platform_bot)
    )
    in_tube_bot = has_tube & point_in_surface_hole(pos_bot_3d, tube_pos, tube_normal, tube_radius, normal_tol=0.100)
    is_solid_bot = (r_bot < radius) & (~in_drain_bot) & (~in_tube_bot) & active_boundary

    # Top collision: falling downwards onto the top drinking surface
    cross_top = (pos_curr_loc[:, 2] >= z_top_curr) & (pos_next_loc[:, 2] < z_top_next) & is_solid_top

    # Bottom collision: rising upwards into the bottom ceiling surface of the lid
    cross_bot = (pos_curr_loc[:, 2] <= z_bot) & (pos_next_loc[:, 2] > z_bot) & is_solid_bot

    # Floor support: any particle on the solid surface that ends up below z_top_next
    below_surface = is_solid_next & (pos_next_loc[:, 2] < z_top_next) & (pos_curr_loc[:, 2] >= z_bot - 0.015)

    # Target position and velocity clamping
    clamp_to_top = cross_top | below_surface
    clamp_to_bot = cross_bot & (~clamp_to_top)

    clamped_z = jnp.where(
        clamp_to_top,
        z_top_next + 1e-4,
        jnp.where(clamp_to_bot, z_bot - 1e-4, pos_next_loc[:, 2]),
    )
    clamped_vz = jnp.where(
        clamp_to_top,
        jnp.maximum(v_rel_local[:, 2], 0.0),
        jnp.where(clamp_to_bot, jnp.minimum(v_rel_local[:, 2], 0.0), v_rel_local[:, 2]),
    )

    pos_out = jnp.stack([pos_next_loc[:, 0], pos_next_loc[:, 1], clamped_z], axis=-1)
    vel_out = jnp.stack([v_rel_local[:, 0], v_rel_local[:, 1], clamped_vz], axis=-1)
    return pos_out, vel_out


def _ccd_cylinder_wall_boundary(
    pos_curr_loc: jnp.ndarray,
    pos_next_loc: jnp.ndarray,
    v_rel_local: jnp.ndarray,
    base_radius: float,
    cavity_floor_z: float,
    max_ceiling_z: float,
    has_base: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous collision detection against cylinder outer wall containment."""
    r_loc_next, _, _ = cartesian_to_cylindrical(pos_next_loc)
    outside_wall = (
        (r_loc_next > base_radius)
        & (r_loc_next <= base_radius + 0.020)
        & (pos_next_loc[:, 2] >= cavity_floor_z - 0.020)
        & (pos_next_loc[:, 2] <= max_ceiling_z + 0.020)
        & has_base
    )
    scale_r = jnp.where(outside_wall & (r_loc_next > 1e-6), (base_radius - 1e-4) / r_loc_next, 1.0)
    pos_x_clamped = pos_next_loc[:, 0] * scale_r
    pos_y_clamped = pos_next_loc[:, 1] * scale_r

    r_safe = jnp.maximum(r_loc_next, 1e-6)
    v_outward = (v_rel_local[:, 0] * pos_next_loc[:, 0] + v_rel_local[:, 1] * pos_next_loc[:, 1]) / r_safe
    v_outward_pos = jnp.maximum(v_outward, 0.0)

    v_rel_x_clamped = jnp.where(
        outside_wall,
        v_rel_local[:, 0] - v_outward_pos * (pos_next_loc[:, 0] / r_safe),
        v_rel_local[:, 0],
    )
    v_rel_y_clamped = jnp.where(
        outside_wall,
        v_rel_local[:, 1] - v_outward_pos * (pos_next_loc[:, 1] / r_safe),
        v_rel_local[:, 1],
    )
    pos_out = jnp.stack([pos_x_clamped, pos_y_clamped, pos_next_loc[:, 2]], axis=-1)
    vel_out = jnp.stack([v_rel_x_clamped, v_rel_y_clamped, v_rel_local[:, 2]], axis=-1)
    return pos_out, vel_out


def _ccd_plane_obstacle_boundary(
    pos_curr: jnp.ndarray,
    pos_next: jnp.ndarray,
    vel_next: jnp.ndarray,
    thick_pl: float,
    pl_pos: jnp.ndarray,
    pl_orn: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous collision detection against planar obstacles."""
    pl_orn_inv = invert_orientation(pl_orn)
    pos_pl = world_to_local_frame(pos_next, pl_pos, pl_orn_inv)
    v_pl = world_to_local_vector(vel_next, pl_orn_inv)

    penetrating_plane = (pos_pl[:, 2] < 0.0) & (pos_pl[:, 2] >= -jnp.maximum(thick_pl, 0.010))
    pos_pl_z = jnp.where(penetrating_plane, 1e-4, pos_pl[:, 2])
    v_pl_z = jnp.where(penetrating_plane, jnp.maximum(v_pl[:, 2], 0.0), v_pl[:, 2])

    pos_pl_safe = jnp.stack([pos_pl[:, 0], pos_pl[:, 1], pos_pl_z], axis=-1)
    v_pl_safe = jnp.stack([v_pl[:, 0], v_pl[:, 1], v_pl_z], axis=-1)

    pos_out = jnp.where(penetrating_plane[:, None], local_to_world_frame(pos_pl_safe, pl_pos, pl_orn), pos_next)
    vel_out = jnp.where(penetrating_plane[:, None], local_to_world_vector(v_pl_safe, pl_orn), vel_next)
    return pos_out, vel_out


def _ccd_sphere_obstacle_boundary(
    pos_curr: jnp.ndarray,
    pos_next: jnp.ndarray,
    vel_next: jnp.ndarray,
    sph_radius: float,
    sph_pos: jnp.ndarray,
    sph_thickness: float = 0.003,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous collision detection against spherical dome canopy and obstacle boundaries."""
    pos_rel_curr = pos_curr - sph_pos
    dist_curr = jnp.sqrt(jnp.sum(pos_rel_curr**2, axis=-1, keepdims=True) + 1e-8)
    n_curr = pos_rel_curr / dist_curr

    pos_rel_next = pos_next - sph_pos
    dist_next = jnp.sqrt(jnp.sum(pos_rel_next**2, axis=-1, keepdims=True) + 1e-8)
    n_next = pos_rel_next / dist_next

    r_inner = jnp.maximum(sph_radius - sph_thickness, 0.001)

    # 1. Internal canopy ceiling collision: particle rising from below/inside attempting to burst through inner ceiling
    is_upper_dome = pos_rel_next[:, 2] > 0.0
    is_rising_in_dome = is_upper_dome & ((pos_rel_curr[:, 2] <= r_inner + 0.002) | (vel_next[:, 2] > 0.0))
    hitting_canopy_ceiling = is_rising_in_dome & (dist_next[:, 0] >= r_inner - 1e-4)

    # Deflect into 360-degree radial outward flow along canopy curve through side grating slots
    r_xy_next = jnp.sqrt(pos_rel_next[:, 0] ** 2 + pos_rel_next[:, 1] ** 2 + 1e-8)
    n_xy = pos_rel_next[:, :2] / r_xy_next[:, None]
    r_eff = r_inner - 2e-4
    r_clamped = jnp.minimum(r_xy_next, r_eff)
    z_canopy = jnp.sqrt(jnp.maximum(r_eff**2 - r_clamped**2, 0.0))
    pos_canopy = sph_pos + jnp.concatenate([n_xy * r_clamped[:, None], z_canopy[:, None]], axis=-1)

    v_speed = jnp.sqrt(jnp.sum(vel_next**2, axis=-1, keepdims=True) + 1e-8)
    v_radial_xy = n_xy * jnp.maximum(v_speed * 0.95, 1.10)
    v_canopy_deflected = jnp.concatenate([v_radial_xy, -0.10 * jnp.ones_like(z_canopy[:, None])], axis=-1)

    # 2. External obstacle collision: particle outside upper dome attempting to penetrate inward from outside
    penetrating_external = (
        (dist_curr[:, 0] >= sph_radius) & (dist_next[:, 0] < sph_radius) & is_upper_dome & (~hitting_canopy_ceiling)
    )
    n_sph = n_curr
    pos_external = sph_pos + n_sph * (sph_radius + 1e-4)
    v_sph_dot = jnp.sum(vel_next * n_sph, axis=-1, keepdims=True)
    v_sph_normal = jnp.minimum(v_sph_dot, 0.0) * n_sph
    v_sph_deflected = vel_next - 1.2 * v_sph_normal

    pos_out = jnp.where(
        hitting_canopy_ceiling[:, None],
        pos_canopy,
        jnp.where(penetrating_external[:, None], pos_external, pos_next),
    )
    vel_out = jnp.where(
        hitting_canopy_ceiling[:, None],
        v_canopy_deflected,
        jnp.where(penetrating_external[:, None], v_sph_deflected, vel_next),
    )
    return pos_out, vel_out


def _ccd_tube_cylinder_boundary(
    pos_curr_loc: jnp.ndarray,
    pos_next_loc: jnp.ndarray,
    v_rel_local: jnp.ndarray,
    r_inner: float,
    r_outer: float,
    tube_h: float,
    slot_h: float,
    active_boundary: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous collision detection for a vertical delivery tube (bore containment and solid outer wall)."""
    r_curr, _, _ = cartesian_to_cylindrical(pos_curr_loc)
    r_next, _, _ = cartesian_to_cylindrical(pos_next_loc)
    z_next = pos_next_loc[:, 2]

    # The bottom of the tube (z <= slot_h) has an inlet slot from the impeller casing
    in_solid_wall_height = (z_next > slot_h) & (z_next <= tube_h) & active_boundary
    r_mid = (r_inner + r_outer) * 0.5
    r_safe = jnp.maximum(r_next, 1e-6)

    # 1. External collision above slot: particle outside tube trying to penetrate into the tube shell
    penetrating_outer = (r_curr >= r_mid) & (r_next < r_outer) & in_solid_wall_height
    scale_outer = jnp.where(penetrating_outer, (r_outer + 1e-4) / r_safe, 1.0)

    # 2. Internal collision: particle inside bore trying to penetrate through inner bore wall into the tube shell
    penetrating_inner = (r_curr < r_mid) & (r_next > r_inner) & in_solid_wall_height
    scale_inner = jnp.where(penetrating_inner, (r_inner - 1e-4) / r_safe, 1.0)

    scale_r = jnp.where(penetrating_outer, scale_outer, scale_inner)
    pos_x_clamped = pos_next_loc[:, 0] * scale_r
    pos_y_clamped = pos_next_loc[:, 1] * scale_r

    v_rad = (v_rel_local[:, 0] * pos_next_loc[:, 0] + v_rel_local[:, 1] * pos_next_loc[:, 1]) / r_safe
    v_rad_inward = jnp.minimum(v_rad, 0.0)
    v_rad_outward = jnp.maximum(v_rad, 0.0)

    v_x_outer = v_rel_local[:, 0] - v_rad_inward * (pos_next_loc[:, 0] / r_safe)
    v_y_outer = v_rel_local[:, 1] - v_rad_inward * (pos_next_loc[:, 1] / r_safe)

    v_x_inner = v_rel_local[:, 0] - v_rad_outward * (pos_next_loc[:, 0] / r_safe)
    v_y_inner = v_rel_local[:, 1] - v_rad_outward * (pos_next_loc[:, 1] / r_safe)

    v_x = jnp.where(penetrating_outer, v_x_outer, jnp.where(penetrating_inner, v_x_inner, v_rel_local[:, 0]))
    v_y = jnp.where(penetrating_outer, v_y_outer, jnp.where(penetrating_inner, v_y_inner, v_rel_local[:, 1]))

    pos_out = jnp.stack([pos_x_clamped, pos_y_clamped, z_next], axis=-1)
    vel_out = jnp.stack([v_x, v_y, v_rel_local[:, 2]], axis=-1)
    return pos_out, vel_out


def _ccd_casing_obstacle_boundary(
    pos_curr_loc: jnp.ndarray,
    pos_next_loc: jnp.ndarray,
    v_rel_local: jnp.ndarray,
    r_outer: float,
    height: float,
    intake_pos: jnp.ndarray,
    intake_radius: float,
    active_boundary: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Continuous collision detection for motor casing outer cylinder barrier and top intake port."""
    r_curr, _, _ = cartesian_to_cylindrical(pos_curr_loc)
    r_next, _, _ = cartesian_to_cylindrical(pos_next_loc)
    z_next = pos_next_loc[:, 2]

    in_casing_height = (z_next >= 0.0) & (z_next <= height) & active_boundary
    r_safe = jnp.maximum(r_next, 1e-6)

    d_intake_xy = jnp.sqrt((pos_next_loc[:, 0] - intake_pos[0]) ** 2 + (pos_next_loc[:, 1] - intake_pos[1]) ** 2)
    in_intake_hole = d_intake_xy <= intake_radius

    penetrating_wall = (r_curr >= r_outer) & (r_next < r_outer) & in_casing_height & (~in_intake_hole)
    scale_r = jnp.where(penetrating_wall, (r_outer + 1e-4) / r_safe, 1.0)
    pos_x = pos_next_loc[:, 0] * scale_r
    pos_y = pos_next_loc[:, 1] * scale_r

    v_rad = (v_rel_local[:, 0] * pos_next_loc[:, 0] + v_rel_local[:, 1] * pos_next_loc[:, 1]) / r_safe
    v_rad_inward = jnp.minimum(v_rad, 0.0)
    v_x = jnp.where(
        penetrating_wall, v_rel_local[:, 0] - v_rad_inward * (pos_next_loc[:, 0] / r_safe), v_rel_local[:, 0]
    )
    v_y = jnp.where(
        penetrating_wall, v_rel_local[:, 1] - v_rad_inward * (pos_next_loc[:, 1] / r_safe), v_rel_local[:, 1]
    )

    pos_out = jnp.stack([pos_x, pos_y, z_next], axis=-1)
    vel_out = jnp.stack([v_x, v_y, v_rel_local[:, 2]], axis=-1)
    return pos_out, vel_out


def _apply_boundary_ccd_subroutine(
    pos_curr: jnp.ndarray,
    pos_next: jnp.ndarray,
    vel_next: jnp.ndarray,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    tube_y_check: float,
    tube_r_check: float,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Apply modular continuous collision detection across all physical obstacle boundaries."""
    for k, shape_k in enumerate(b_shapes):
        # 1. Planar obstacles
        is_pl = shape_k == SHAPE_PLANE
        pos_pl, vel_pl = _ccd_plane_obstacle_boundary(
            pos_curr, pos_next, vel_next, b_params[k, BoundaryParam.THICKNESS], b_pos_arr[k], b_orn_arr[k]
        )
        pos_next = jnp.where(is_pl, pos_pl, pos_next)
        vel_next = jnp.where(is_pl, vel_pl, vel_next)

        # 2. Spherical obstacles (such as the spout deflection dome)
        is_sph = (shape_k == SHAPE_SPHERE) & (b_types[k] == 0)
        sph_t = b_params[k, BoundaryParam.THICKNESS]
        pos_sph, vel_sph = _ccd_sphere_obstacle_boundary(
            pos_curr, pos_next, vel_next, b_params[k, BoundaryParam.R_OUTER], b_pos_arr[k], sph_t
        )
        pos_next = jnp.where(is_sph, pos_sph, pos_next)
        vel_next = jnp.where(is_sph, vel_sph, vel_next)

        # 3. Solid lid tray drinking shelf (cylinder with drainage hole and tube hole, exposed to air)
        is_submerged_k = b_params[k, BoundaryParam.IS_SUBMERGED] > 0.5
        is_lid_k = (shape_k == SHAPE_CYLINDER) & (b_params[k, BoundaryParam.HAS_DRAIN] > 0.5) & (~is_submerged_k)
        lid_k_pos = b_pos_arr[k]
        lid_k_orn = b_orn_arr[k]
        lid_k_orn_inv = invert_orientation(lid_k_orn)
        pos_lid_curr = world_to_local_frame(pos_curr, lid_k_pos, lid_k_orn_inv)
        pos_lid_next = world_to_local_frame(pos_next, lid_k_pos, lid_k_orn_inv)
        v_lid_k = world_to_local_vector(vel_next, lid_k_orn_inv)

        drain_pos_k = b_params[k, BoundaryParam.DRAIN_POS_X : BoundaryParam.DRAIN_POS_Z + 1]
        drain_norm_k = b_params[k, BoundaryParam.DRAIN_NORMAL_X : BoundaryParam.DRAIN_NORMAL_Z + 1]
        drain_r_k = b_params[k, BoundaryParam.DRAIN_RADIUS]

        tube_pos_k = b_params[k, BoundaryParam.TUBE_POS_X : BoundaryParam.TUBE_POS_Z + 1]
        tube_norm_k = b_params[k, BoundaryParam.TUBE_NORMAL_X : BoundaryParam.TUBE_NORMAL_Z + 1]
        tube_r_k = b_params[k, BoundaryParam.TUBE_PORT_RADIUS]

        intake_pos_k = b_params[k, BoundaryParam.INTAKE_POS_X : BoundaryParam.INTAKE_POS_Z + 1]
        intake_norm_k = b_params[k, BoundaryParam.INTAKE_NORMAL_X : BoundaryParam.INTAKE_NORMAL_Z + 1]
        intake_r_k = b_params[k, BoundaryParam.INTAKE_RADIUS]

        pos_lid_safe, v_lid_safe = _ccd_planar_shelf_boundary(
            pos_lid_curr,
            pos_lid_next,
            v_lid_k,
            z_plane=0.0,
            normal_sign=1.0,
            radius=b_params[k, BoundaryParam.RADIUS],
            shelf_depth=b_params[k, BoundaryParam.SHELF_DEPTH],
            has_drain=b_params[k, BoundaryParam.HAS_DRAIN] > 0.5,
            drain_pos=drain_pos_k,
            drain_normal=drain_norm_k,
            drain_radius=drain_r_k,
            has_tube=b_params[k, BoundaryParam.HAS_TUBE] > 0.5,
            tube_pos=tube_pos_k,
            tube_normal=tube_norm_k,
            tube_radius=tube_r_k,
            has_intake=b_params[k, BoundaryParam.HAS_INTAKE] > 0.5,
            intake_pos=intake_pos_k,
            intake_normal=intake_norm_k,
            intake_radius=intake_r_k,
            active_boundary=is_lid_k,
        )
        pos_out = local_to_world_frame(pos_lid_safe, lid_k_pos, lid_k_orn)
        vel_out = local_to_world_vector(v_lid_safe, lid_k_orn)
        pos_next = jnp.where(is_lid_k, pos_out, pos_next)
        vel_next = jnp.where(is_lid_k, vel_out, vel_next)

        # 4. Vertical delivery tube (solid outer wall + internal bore containment)
        is_tube_k = shape_k == SHAPE_TUBE
        tube_k_pos = b_pos_arr[k]
        tube_k_orn = b_orn_arr[k]
        tube_k_orn_inv = invert_orientation(tube_k_orn)
        pos_tb_curr = world_to_local_frame(pos_curr, tube_k_pos, tube_k_orn_inv)
        pos_tb_next = world_to_local_frame(pos_next, tube_k_pos, tube_k_orn_inv)
        v_tb_k = world_to_local_vector(vel_next, tube_k_orn_inv)

        r_in_k = b_params[k, BoundaryParam.R_INNER]
        r_out_k = b_params[k, BoundaryParam.R_OUTER]
        tube_h_k = b_params[k, BoundaryParam.HEIGHT]
        slot_h_k = b_params[k, BoundaryParam.SLOT_HEIGHT]

        pos_tb_safe, v_tb_safe = _ccd_tube_cylinder_boundary(
            pos_tb_curr, pos_tb_next, v_tb_k, r_in_k, r_out_k, tube_h_k, slot_h_k, is_tube_k
        )
        pos_tb_world = local_to_world_frame(pos_tb_safe, tube_k_pos, tube_k_orn)
        vel_tb_world = local_to_world_vector(v_tb_safe, tube_k_orn)
        pos_next = jnp.where(is_tube_k, pos_tb_world, pos_next)
        vel_next = jnp.where(is_tube_k, vel_tb_world, vel_next)

        # 5. Motor casing obstacle boundary (outer cylindrical wall and intake port)
        is_casing_k = shape_k == SHAPE_CASING
        casing_k_pos = b_pos_arr[k]
        casing_k_orn = b_orn_arr[k]
        casing_k_orn_inv = invert_orientation(casing_k_orn)
        pos_cs_curr = world_to_local_frame(pos_curr, casing_k_pos, casing_k_orn_inv)
        pos_cs_next = world_to_local_frame(pos_next, casing_k_pos, casing_k_orn_inv)
        v_cs_k = world_to_local_vector(vel_next, casing_k_orn_inv)

        r_out_cs = b_params[k, BoundaryParam.R_OUTER]
        casing_h = b_params[k, BoundaryParam.HEIGHT]
        intake_pos_k = b_params[k, BoundaryParam.INTAKE_POS_X : BoundaryParam.INTAKE_POS_Z + 1]
        intake_r_k = b_params[k, BoundaryParam.INTAKE_RADIUS]

        pos_cs_safe, v_cs_safe = _ccd_casing_obstacle_boundary(
            pos_cs_curr,
            pos_cs_next,
            v_cs_k,
            r_out_cs,
            casing_h,
            intake_pos_k,
            intake_r_k,
            is_casing_k,
        )
        pos_cs_world = local_to_world_frame(pos_cs_safe, casing_k_pos, casing_k_orn)
        vel_cs_world = local_to_world_vector(v_cs_safe, casing_k_orn)
        pos_next = jnp.where(is_casing_k, pos_cs_world, pos_next)
        vel_next = jnp.where(is_casing_k, vel_cs_world, vel_next)

    return pos_next, vel_next


def _integrate_particles_subroutine(
    pos_curr: jnp.ndarray,
    vel_world: jnp.ndarray,
    accel: jnp.ndarray,
    base_pos: jnp.ndarray,
    base_orn: jnp.ndarray,
    dt_sub: float,
    damping: float,
    high_damping_value: float,
    base_idx: int,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    omega: float,
    base_vel: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Integration subroutine: applies accelerations, zone-based damping, and clamps speed limits."""
    base_radius = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.R_OUTER], 0.0)
    base_height = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.Z_TOP], 0.0)

    has_tube = jnp.any(b_shapes == SHAPE_TUBE)
    tube_idx = jnp.argmax(b_shapes == SHAPE_TUBE)
    tube_y_check = jnp.where(has_tube, b_pos_arr[tube_idx, 1] - base_pos[1], 0.0)
    tube_r_check = jnp.where(
        has_tube,
        b_params[tube_idx, BoundaryParam.R_INNER],
        0.0,
    )
    spout_r = jnp.where(has_tube, b_params[tube_idx, BoundaryParam.R_OUTER], 0.014)
    tube_h = jnp.where(has_tube, b_params[tube_idx, BoundaryParam.HEIGHT], 0.0)
    influence_r = tube_r_check
    influence_h = tube_h + 0.049

    gamma_base = jnp.where(jnp.abs(omega) > 0.0, 0.95, 0.998)
    v_target = jnp.where(jnp.abs(omega) > 0.0, jnp.abs(omega) * 0.015, 0.50)

    active = (pos_curr[:, 2] < 100.0)[:, None]
    active_mask = pos_curr[:, 2] < 100.0
    speeds = jnp.linalg.norm(vel_world, axis=1)
    num_active = jnp.sum(active_mask)
    total_speed = jnp.sum(speeds * active_mask)
    avg_speed = jnp.where(num_active > 0.0, total_speed / num_active, 0.0)

    v_excess = jnp.maximum(0.0, avg_speed - v_target)
    dynamic_damping = gamma_base - 0.16 * v_excess
    dynamic_damping = jnp.maximum(0.90, jnp.minimum(gamma_base, dynamic_damping))

    base_orn_inv = invert_orientation(base_orn)
    pos_local_check = world_to_base_frame(pos_curr, base_pos, base_orn_inv)
    r_local, _, _ = cartesian_to_cylindrical(pos_local_check)

    cavity_floor_z = jnp.where(
        base_idx != -1,
        b_pos_arr[base_idx, 2] - base_pos[2] + b_params[base_idx, BoundaryParam.Z_OFFSET],
        0.0,
    )

    dist_tube_sq = pos_local_check[:, 0] ** 2 + (pos_local_check[:, 1] - tube_y_check) ** 2
    in_tube = (
        (dist_tube_sq < influence_r**2)
        & (pos_local_check[:, 2] >= cavity_floor_z)
        & (pos_local_check[:, 2] <= influence_h)
    )
    in_tube = jnp.where(has_tube, in_tube, jnp.zeros(pos_curr.shape[0], dtype=jnp.bool_))

    max_ceiling_z = jnp.where(base_idx != -1, b_params[base_idx, BoundaryParam.MAX_CEILING_Z], base_height)
    influence_h = jnp.minimum(tube_h + 0.049, max_ceiling_z)

    outside_base_raw = (
        (r_local > base_radius)
        | (pos_local_check[:, 2] < cavity_floor_z)
        | (pos_local_check[:, 2] > max_ceiling_z + 0.005)
    ) & (~in_tube)
    outside_base = jnp.where(base_idx != -1, outside_base_raw, jnp.zeros(pos_curr.shape[0], dtype=jnp.bool_))

    damping_val = jnp.where(damping >= 0.0, damping, dynamic_damping)
    damping_by_zone = jnp.where((damping >= 0.0) | (~outside_base), damping_val, high_damping_value)[:, None]

    vel_next = jnp.where(active, (vel_world + accel * dt_sub) * damping_by_zone, 0.0)

    max_phys_speed = 1.5
    vel_mags = jnp.linalg.norm(vel_next, axis=1, keepdims=True)
    vel_mags_safe = jnp.maximum(vel_mags, 1e-8)
    vel_next = vel_next * jnp.minimum(max_phys_speed / vel_mags_safe, 1.0)

    pos_next = jnp.where(active, pos_curr + vel_next * dt_sub, pos_curr)

    # Enforce analytical non-penetration boundary conditions on base container in Base Link Frame
    pos_local_next = world_to_base_frame(pos_next, base_pos, base_orn_inv)
    v_rel_local = world_to_base_vector(vel_next - base_vel, base_orn_inv)
    has_base = base_idx != -1

    # 1. Floor non-penetration CCD
    pos_local_next, v_rel_local = _ccd_planar_shelf_boundary(
        pos_local_check,
        pos_local_next,
        v_rel_local,
        z_plane=cavity_floor_z,
        normal_sign=1.0,
        radius=base_radius,
        shelf_depth=0.020,
        has_drain=jnp.bool_(False),
        drain_pos=jnp.zeros(3, dtype=jnp.float32),
        drain_normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        drain_radius=0.0,
        has_tube=jnp.bool_(False),
        tube_pos=jnp.zeros(3, dtype=jnp.float32),
        tube_normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        tube_radius=0.0,
        has_intake=jnp.bool_(False),
        intake_pos=jnp.zeros(3, dtype=jnp.float32),
        intake_normal=jnp.array([0.0, 0.0, 1.0], dtype=jnp.float32),
        intake_radius=0.0,
        active_boundary=has_base,
    )

    # 2. Side wall non-penetration CCD
    pos_local_next, v_rel_local = _ccd_cylinder_wall_boundary(
        pos_local_check, pos_local_next, v_rel_local, base_radius, cavity_floor_z, max_ceiling_z, has_base
    )

    base_vel_local = world_to_base_vector(base_vel, base_orn_inv)
    vel_local_safe = v_rel_local + base_vel_local

    # Transform back to World Frame
    pos_next = jnp.where(active, base_to_world_frame(pos_local_next, base_pos, base_orn), pos_curr)
    vel_next = jnp.where(active, base_to_world_vector(vel_local_safe, base_orn), 0.0)

    # 3. Modular continuous collision detection post-processing pipeline for all physical obstacle boundaries
    pos_next, vel_next = _apply_boundary_ccd_subroutine(
        pos_curr, pos_next, vel_next, b_shapes, b_types, b_params, b_pos_arr, b_orn_arr, tube_y_check, tube_r_check
    )

    return pos_next, vel_next


@partial(jax.jit, static_argnums=(20, 21, 22, 25))
def _physics_step_jax_jit(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    f_lbm: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_vel: jnp.ndarray,
    omega: float,
    t_start: float,
    damping: float,
    b_shapes: jnp.ndarray,
    b_types: jnp.ndarray,
    b_params: jnp.ndarray,
    mass: float,
    dt_sub: float,
    gravity: jnp.ndarray,
    base_idx: int,
    K_boundary: float,
    D_boundary: float,
    r_s: float,
    high_damping_value: float,
    nx: int,
    ny: int,
    nz: int,
    dx: float,
    origin: jnp.ndarray,
    n_substeps: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
    """Perform a substepped LBM-PIC simulation update integrating forces and boundary collisions."""

    def body_fun(i, val):
        pos_curr, vel_curr, f_curr, torque_accum, b_forces_accum = val
        t_curr = t_start + i * dt_sub

        base_pos = jnp.where(base_idx != -1, b_pos_arr[base_idx], jnp.zeros(3))
        base_orn = jnp.where(base_idx != -1, b_orn_arr[base_idx], jnp.array([0.0, 0.0, 0.0, 1.0]))

        # Step 1: LBM solver (incompressible Navier-Stokes flow driven by moving boundaries)
        f_next, u_grid, solid_mask, solid_friction, normal_grid, smooth_occ = _lbm_step_subroutine(
            pos_curr,
            f_curr,
            base_pos,
            base_orn,
            omega,
            t_curr,
            b_shapes,
            b_types,
            b_params,
            b_pos_arr,
            b_orn_arr,
            base_idx,
            nx,
            ny,
            nz,
            dx,
            origin,
            dt_sub,
            jnp.zeros(3),
        )

        # Step 2: G2P mapping to update particle velocities
        vel_world, in_fluid_continuum = _g2p_mapping_subroutine(
            pos_curr,
            vel_curr,
            u_grid,
            base_pos,
            base_orn,
            base_vel,
            nx,
            ny,
            nz,
            dx,
            origin,
            dt_sub,
            b_shapes,
            b_params,
            b_pos_arr,
            b_orn_arr,
            base_idx,
            r_s,
        )

        # Step 3: Compute forces acting on particles
        accel, step_torque, b_forces = _compute_particle_forces_subroutine(
            pos_curr,
            vel_world,
            omega,
            t_curr,
            b_shapes,
            b_types,
            b_params,
            b_pos_arr,
            b_orn_arr,
            K_boundary,
            D_boundary,
            r_s,
            mass,
            gravity,
            solid_mask,
            solid_friction,
            normal_grid,
            smooth_occ,
            dx,
            origin,
            base_idx,
            nx,
            ny,
            nz,
            in_fluid_continuum=in_fluid_continuum,
            base_vel=base_vel,
        )

        # Step 4: Integrate active particles
        pos_next, vel_next = _integrate_particles_subroutine(
            pos_curr,
            vel_world,
            accel,
            base_pos,
            base_orn,
            dt_sub,
            damping,
            high_damping_value,
            base_idx,
            b_shapes,
            b_types,
            b_params,
            b_pos_arr,
            b_orn_arr,
            omega,
            base_vel,
        )

        torque_accum_next = torque_accum + step_torque
        b_forces_accum_next = b_forces_accum + b_forces
        return pos_next, vel_next, f_next, torque_accum_next, b_forces_accum_next

    pos_out, vel_out, f_out, torque_out, b_forces_out = jax.lax.fori_loop(
        0, n_substeps, body_fun, (pos, vel, f_lbm, 0.0, jnp.zeros_like(pos))
    )
    return pos_out, vel_out, f_out, torque_out, b_forces_out / float(n_substeps)


def _physics_step_jax(
    pos: jnp.ndarray,
    vel: jnp.ndarray,
    f_lbm: jnp.ndarray,
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    base_vel: jnp.ndarray,
    omega: float,
    t_start: float,
    damping: float,
    config: PhysicsConfig,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, float, jnp.ndarray]:
    """Perform a substepped LBM-PIC simulation update integrating forces and boundary collisions."""
    return _physics_step_jax_jit(
        pos,
        vel,
        f_lbm,
        b_pos_arr,
        b_orn_arr,
        base_vel,
        omega,
        t_start,
        damping,
        config.b_shapes_jax,
        config.b_types_jax,
        config.b_params_jax,
        config.mass,
        config.dt_sub,
        config.gravity_arr,
        config.base_idx,
        config.K_boundary,
        config.D_boundary,
        config.r_s,
        config.high_damping_value,
        config.nx,
        config.ny,
        config.nz,
        config.dx,
        config.origin_arr,
        config.n_substeps,
    )


class FluidSpawner:
    """Helper class to manage PyBullet body spawning, shapes, and state for fluid particles."""

    def __init__(
        self,
        physics_client: int,
        r_s: float,
        n_particles: int,
        particle_mass: float,
        particle_color: list[float],
        linear_damping: float,
        angular_damping: float,
        lateral_friction: float,
        restitution: float,
    ):
        """Initialize the spawner."""
        self.physics_client = physics_client
        self.r_s = r_s
        self.n_particles = n_particles
        self.particle_mass = particle_mass
        self.vol_s = (4.0 / 3.0) * math.pi * (r_s**3)

        self.sphere_col = p.createCollisionShape(p.GEOM_SPHERE, radius=r_s, physicsClientId=physics_client)
        self.circle_vis = p.createVisualShape(
            shapeType=p.GEOM_SPHERE,
            radius=r_s,
            rgbaColor=particle_color,
            physicsClientId=physics_client,
        )

        self.linear_damping = linear_damping
        self.angular_damping = angular_damping
        self.lateral_friction = lateral_friction
        self.restitution = restitution

        self.particle_body_ids: list[int] = []
        self.active_count: int = 0

    def spawn_batch(self, spawn_z: float, batch_size: int, spacing: float) -> int:
        """Spawn a batch of fluid particles up to n_particles total."""
        if self.active_count >= self.n_particles:
            return 0
        to_activate = min(batch_size, self.n_particles - self.active_count)
        for i in range(to_activate):
            jitter_x = random.uniform(-2 * self.r_s, 2 * self.r_s)
            jitter_y = random.uniform(-2 * self.r_s, 2 * self.r_s)
            w_id = p.createMultiBody(
                baseMass=self.particle_mass,
                baseCollisionShapeIndex=self.sphere_col,
                baseVisualShapeIndex=self.circle_vis,
                basePosition=[jitter_x, jitter_y, spawn_z + i * spacing],
                physicsClientId=self.physics_client,
            )
            p.changeDynamics(
                w_id,
                -1,
                linearDamping=self.linear_damping,
                angularDamping=self.angular_damping,
                lateralFriction=self.lateral_friction,
                restitution=self.restitution,
                physicsClientId=self.physics_client,
            )
            p.setCollisionFilterGroupMask(
                w_id, -1, CollisionGroup.PARTICLE, CollisionMask.CONTAINER, physicsClientId=self.physics_client
            )
            p.resetBaseVelocity(w_id, [0.0, 0.0, -1.0], [0.0, 0.0, 0.0], physicsClientId=self.physics_client)
            self.particle_body_ids.append(w_id)
        self.active_count += to_activate
        return to_activate

    def spawn_all_at_positions(self, positions: list[tuple[float, float, float]]) -> None:
        """Spawn all fluid particles at specified 3D positions."""
        for pos in positions:
            w_id = p.createMultiBody(
                baseMass=self.particle_mass,
                baseCollisionShapeIndex=self.sphere_col,
                baseVisualShapeIndex=self.circle_vis,
                basePosition=pos,
                physicsClientId=self.physics_client,
            )
            p.changeDynamics(
                w_id,
                -1,
                linearDamping=self.linear_damping,
                angularDamping=self.angular_damping,
                lateralFriction=self.lateral_friction,
                restitution=self.restitution,
                physicsClientId=self.physics_client,
            )
            p.setCollisionFilterGroupMask(
                w_id, -1, CollisionGroup.PARTICLE, CollisionMask.CONTAINER, physicsClientId=self.physics_client
            )
            p.resetBaseVelocity(w_id, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0], physicsClientId=self.physics_client)
            self.particle_body_ids.append(w_id)
        self.active_count = len(self.particle_body_ids)

    def get_positions_and_velocities(
        self, fallen_ids: set[int] | None = None
    ) -> tuple[list[tuple[float, float, float]], list[tuple[float, float, float]]]:
        """Return positions and velocities of all particles, padding unspawned ones to keep constant shapes."""
        positions = []
        velocities = []
        if fallen_ids is None:
            fallen_ids = set()

        for w_id in self.particle_body_ids:
            if w_id in fallen_ids:
                positions.append((0.0, 0.0, 1000.0))
                velocities.append((0.0, 0.0, 0.0))
            else:
                pos, _ = p.getBasePositionAndOrientation(w_id, physicsClientId=self.physics_client)
                vel, _ = p.getBaseVelocity(w_id, physicsClientId=self.physics_client)
                positions.append(pos)
                velocities.append(vel)

        unspawned_count = self.n_particles - len(self.particle_body_ids)
        if unspawned_count > 0:
            positions.extend([(0.0, 0.0, 1000.0)] * unspawned_count)
            velocities.extend([(0.0, 0.0, 0.0)] * unspawned_count)

        return positions, velocities


@dataclass(frozen=True)
class VoxelVolumeReconstructor:
    """Reconstructs solid voxel fluid volumes from point-particle positions near boundary constraints."""

    nx: int
    ny: int
    nz: int
    origin: tuple[float, float, float]
    dx: float
    iz_floor: int
    processed_boundaries: ProcessedBoundaries

    def reconstruct(self, positions: Any) -> Any:
        """Map particle positions to grid and return dilated voxel centers outside solid boundaries.

        Args:
            positions: List or NumPy array of particle coordinates.

        Returns:
            NumPy array of voxel coordinates representing the reconstructed fluid volume.
        """
        if positions is None or len(positions) == 0:
            return np.empty((0, 3), dtype=np.float32)

        pos_arr = np.asarray(positions)
        x_min = self.origin[0]
        x_max = self.origin[0] + self.nx * self.dx
        y_min = self.origin[1]
        y_max = self.origin[1] + self.ny * self.dx
        z_min_bound = self.origin[2]
        z_max_bound = self.origin[2] + self.nz * self.dx

        # 1. Bounds check and filter valid positions
        x = pos_arr[:, 0]
        y = pos_arr[:, 1]
        z = pos_arr[:, 2]
        valid = (x >= x_min) & (x < x_max) & (y >= y_min) & (y < y_max) & (z >= z_min_bound) & (z < z_max_bound)
        if not np.any(valid):
            return np.empty((0, 3), dtype=np.float32)

        pos_valid = pos_arr[valid]

        # 2. Grid index mapping
        ixs = np.floor((pos_valid[:, 0] - x_min) / self.dx).astype(np.int32)
        iys = np.floor((pos_valid[:, 1] - y_min) / self.dx).astype(np.int32)
        izs = np.floor((pos_valid[:, 2] - z_min_bound) / self.dx).astype(np.int32)

        # 3. 3D neighborhood dilation around every particle (spherical 3D kernel)
        d_offsets = np.array(
            [
                (d_x, d_y, d_z)
                for d_x in [-1, 0, 1]
                for d_y in [-1, 0, 1]
                for d_z in [-1, 0, 1]
                if d_x**2 + d_y**2 + d_z**2 <= 2
            ],
            dtype=np.int32,
        )

        dilated_3d_x = (ixs[:, None] + d_offsets[:, 0]).ravel()
        dilated_3d_y = (iys[:, None] + d_offsets[:, 1]).ravel()
        dilated_3d_z = (izs[:, None] + d_offsets[:, 2]).ravel()

        # Clip to valid grid bounds
        valid_3d = (
            (dilated_3d_x >= 0)
            & (dilated_3d_x < self.nx)
            & (dilated_3d_y >= 0)
            & (dilated_3d_y < self.ny)
            & (dilated_3d_z >= 0)
            & (dilated_3d_z < self.nz)
        )
        all_ixs = dilated_3d_x[valid_3d]
        all_iys = dilated_3d_y[valid_3d]
        all_izs = dilated_3d_z[valid_3d]

        # Unique grid coordinates using flat 1D integer indexing (orders of magnitude faster than void struct sort)
        flat_idx = (all_ixs.astype(np.int64) * self.ny + all_iys.astype(np.int64)) * self.nz + all_izs.astype(np.int64)
        flat_unique = np.unique(flat_idx)

        iz_u = flat_unique % self.nz
        rem = flat_unique // self.nz
        iy_u = rem % self.ny
        ix_u = rem // self.ny

        cx = self.origin[0] + (ix_u + 0.5) * self.dx
        cy = self.origin[1] + (iy_u + 0.5) * self.dx
        cz = self.origin[2] + (iz_u + 0.5) * self.dx

        # Check solid boundaries vectorially via ProcessedBoundaries
        is_solid = self.processed_boundaries.is_solid(cx, cy, cz)

        # Keep voxels that are outside solid boundaries
        non_solid = ~is_solid
        if not np.any(non_solid):
            return np.empty((0, 3), dtype=np.float32)

        return np.column_stack((cx[non_solid], cy[non_solid], cz[non_solid]))


@dataclass(frozen=True)
class FluidPostProcessor:
    """Post-processing stage for transforming physics states into visualization geometries."""

    nx: int
    ny: int
    nz: int
    origin: tuple[float, float, float]
    dx: float
    voxel_radius_scale: float
    processed_boundaries: ProcessedBoundaries

    @property
    def iz_floor(self) -> int:
        """Get the Z grid index representing the reservoir floor boundary."""
        z_offset = self.processed_boundaries.cavity_z_offset
        return max(0, int(math.floor((z_offset - self.origin[2]) / self.dx)))

    def process_voxels(self, positions: Any) -> Any:
        """Run volume reconstruction to generate dense water voxels without boundary gaps.

        Args:
            positions: Current particle positions.

        Returns:
            NumPy array of voxel coordinates representing the reconstructed fluid volume.
        """
        reconstructor = VoxelVolumeReconstructor(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            origin=self.origin,
            dx=self.dx,
            iz_floor=self.iz_floor,
            processed_boundaries=self.processed_boundaries,
        )
        return reconstructor.reconstruct(positions)

    @cached_property
    def _grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute and cache the 3D voxel coordinate meshgrid."""
        x_coords = self.origin[0] + (np.arange(self.nx) + 0.5) * self.dx
        y_coords = self.origin[1] + (np.arange(self.ny) + 0.5) * self.dx
        z_coords = self.origin[2] + (np.arange(self.nz) + 0.5) * self.dx
        return np.meshgrid(x_coords, y_coords, z_coords, indexing="ij")

    @cached_property
    def _static_voxels(self) -> dict[str, np.ndarray]:
        """Compute and cache static boundary voxels (bowl, casing, tube, lid)."""
        gx, gy, gz = self._grid
        static_voxels: dict[str, np.ndarray] = {}
        for b in self.processed_boundaries.boundaries:
            if isinstance(b, ImpellerBoundary):
                continue
            label = b.__class__.__name__.replace("Boundary", "").lower()
            if hasattr(b, "is_solid_vectorized"):
                solid_mask = b.is_solid_vectorized(gx, gy, gz)
                if np.any(solid_mask):
                    pts = np.stack([gx[solid_mask], gy[solid_mask], gz[solid_mask]], axis=-1)
                    if label in static_voxels:
                        static_voxels[label] = np.concatenate([static_voxels[label], pts], axis=0)
                    else:
                        static_voxels[label] = pts
        return static_voxels

    def get_boundary_voxels(self, target_omega: float = 0.0, t_curr: float = 0.0) -> dict[str, np.ndarray]:
        """Extract voxel centers for each boundary element labeled by type.

        Args:
            target_omega: Optional rotational velocity for dynamic boundaries (e.g. impeller).
            t_curr: Current simulation time.

        Returns:
            Dictionary mapping boundary label to (M, 3) array of voxel coordinates.
        """
        gx, gy, gz = self._grid
        result: dict[str, np.ndarray] = {k: v.copy() for k, v in self._static_voxels.items()}
        angle = target_omega * t_curr

        for b in self.processed_boundaries.boundaries:
            if isinstance(b, ImpellerBoundary):
                label = b.__class__.__name__.replace("Boundary", "").lower()
                solid_mask = b.is_solid_vectorized(gx, gy, gz, angle=angle)
                if np.any(solid_mask):
                    pts = np.stack([gx[solid_mask], gy[solid_mask], gz[solid_mask]], axis=-1)
                    if label in result:
                        result[label] = np.concatenate([result[label], pts], axis=0)
                    else:
                        result[label] = pts
        return result


class Fluid:
    """Handles SPH fluid dynamics simulation for fluid particles in PyBullet using JAX."""

    PARTICLE_COLOR = [0.5, 0.8, 1.0, 1.0]
    LINEAR_DAMPING = 0.05
    ANGULAR_DAMPING = 0.05
    LATERAL_FRICTION = 0.1
    RESTITUTION = 0.0
    REST_DENSITY = 1000.0
    VISCOSITY = 0.02
    STIFFNESS = 100.0
    DEACTIVATION_BOX_FACTOR = 50.0
    VOXEL_RADIUS_SCALE = 0.54
    MAX_JOINT_VELOCITY = 200.0

    def __init__(
        self,
        config: Optional[FluidConfig] = None,
        provider: Optional[Provider] = None,
        body_id: Optional[int] = None,
        physics_client: Optional[int] = None,
        state_tracker: Optional[Any] = None,
        link_indices: Optional[dict[LinkType, Optional[int]]] = None,
    ):
        """Initialize the fluid simulation constants and state using a FluidConfig."""
        from model import BoundaryConfig, FluidConfig

        if config is None:
            config = FluidConfig()

        self.provider: Optional[Provider] = provider

        self.r_s = config.r_s
        self.particle_radius = config.particle_radius

        self.target_volume = config.target_volume
        self.spawn_buffer = config.spawn_buffer
        self.rest_density = config.rest_density
        self.viscosity = config.viscosity
        self.nx = config.nx
        self.ny = config.ny
        self.nz = config.nz
        self.dx = config.dx
        self.origin = config.origin

        self.stiffness = config.stiffness
        self.k = self.stiffness

        self.smoothing_factor = config.smoothing_factor
        self.sphere_vol_factor = config.sphere_vol_factor
        self.poly6_coeff_numerator = config.poly6_coeff_numerator
        self.poly6_coeff_denominator = config.poly6_coeff_denominator
        self.spiky_grad_coeff = config.spiky_grad_coeff
        self.visc_lap_coeff = config.visc_lap_coeff
        self.pressure_avg_factor = config.pressure_avg_factor
        self.min_distance_threshold = config.min_distance_threshold
        self.stiffness_boundary = config.stiffness_boundary
        self.damping_boundary = config.damping_boundary
        self.surface_smooth_center_coeff = config.surface_smooth_center_coeff
        self.surface_smooth_neighbor_coeff = config.surface_smooth_neighbor_coeff
        self.surface_smooth_diagonal_coeff = config.surface_smooth_diagonal_coeff

        self.volume_threshold_liters = config.volume_threshold_liters
        self.fallen_threshold_liters = config.fallen_threshold_liters
        self.recycle_fluid = config.recycle_fluid
        self.high_damping_value = config.high_damping_value

        # SPH values computed from settings
        self.h = self.smoothing_factor * self.r_s
        self.mass = self.sphere_vol_factor * math.pi * (self.r_s**3) * self.rest_density

        # Precompute kernel constants
        self.poly6_factor = self.poly6_coeff_numerator / (self.poly6_coeff_denominator * math.pi * (self.h**9))
        self.spiky_grad_factor = self.spiky_grad_coeff / (math.pi * (self.h**6))
        self.visc_lap_factor = self.visc_lap_coeff / (math.pi * (self.h**6))

        # Simulator state variables (formerly from FluidSimulator)
        self.spawner = None
        self.f_lbm = None
        self.body_id = body_id
        self.physics_client = physics_client
        self.boundaries: dict[LinkType, BoundaryConfig] = {}
        self.boundary_list: list[BoundaryConfig] = []
        self.pos_jax = None
        self.vel_jax = None
        vol_s = (4.0 / 3.0) * math.pi * (self.r_s**3)
        self.n_particles = int(round(self.target_volume / vol_s))
        self.last_positions: list[list[float]] = []
        self.last_velocities: list[list[float]] = []
        self.current_sim_time = 0.0
        self.torques: list[float] = []
        # Motor configurations are consolidated into BoundaryConfig
        self.spout_water_ids = ParticleSet(self.n_particles)
        self.fallen_out_water_ids = ParticleSet(self.n_particles)
        self.total_fallen_water_ids = ParticleSet(self.n_particles)
        self.state_tracker = state_tracker

        self._cached_active_indices = None
        self._cached_mapper = None
        self._cached_self_active_indices = None
        self.spawn_xy_coords: list[tuple[float, float]] = []

        self.link_indices = link_indices if link_indices is not None else {}

        self.gravity = config.gravity

        # Parse boundaries metadata using BoundaryConfig Pydantic model
        self.boundary_list = []
        if config.boundaries is not None:
            for label, val in config.boundaries.items():
                vals = val if isinstance(val, list) else [val]
                for item in vals:
                    b_info = item if isinstance(item, BoundaryConfig) else BoundaryConfig.model_validate(item)
                    b_info._label = label
                    self.boundary_list.append(b_info)

        self.processed_boundaries = BoundaryProcessor.process(
            self.boundary_list,
            body_id=self.body_id,
            physics_client=self.physics_client,
            default_idx_map=self.link_indices,
        )

        self.boundaries = {}
        for b in self.boundary_list:
            match b.link_type:
                case LinkType.BASE:
                    self.boundaries[LinkType.BASE] = b
                case LinkType.TUBE:
                    self.boundaries[LinkType.TUBE] = b
                case LinkType.IMPELLER:
                    self.boundaries[LinkType.IMPELLER] = b

        self.characteristic_length = self.processed_boundaries.cavity_inner_radius
        self.neighbor_list_box = 2.0 * self.processed_boundaries.cavity_inner_radius
        self.base_idx = self.processed_boundaries.base_idx

        self.post_processor = FluidPostProcessor(
            nx=self.nx,
            ny=self.ny,
            nz=self.nz,
            origin=self.origin,
            dx=self.dx,
            voxel_radius_scale=self.VOXEL_RADIUS_SCALE,
            processed_boundaries=self.processed_boundaries,
        )
        self.fluid_body_tracker = FluidBodyTracker(r_s=self.r_s)

        if physics_client is not None and body_id is not None and config.boundaries is not None:
            if state_tracker is not None:
                state_tracker.has_fluid_simulator = True
                self._update_state_tracker()

            self.spawner = FluidSpawner(
                physics_client=physics_client,
                r_s=self.r_s,
                n_particles=self.n_particles,
                particle_mass=self.particle_mass,
                particle_color=self.PARTICLE_COLOR,
                linear_damping=0.05,
                angular_damping=0.05,
                lateral_friction=0.1,
                restitution=0.0,
            )
            self.spawner.active_count = self.n_particles

            self.default_idx_map = np.arange(self.n_particles)[:, None]
            self.default_idx_map = np.repeat(self.default_idx_map, 64, axis=1)

            # Generate grid points to initially fill the cavity
            grid_points: list[tuple[float, float, float]] = []
            spacing = (self.vol_s) ** (1.0 / 3.0)

            hc_x = self.processed_boundaries.tube_x
            hc_y = self.processed_boundaries.tube_y
            hc_r = self.processed_boundaries.tube_inner_radius
            cavity_inner_radius = self.processed_boundaries.cavity_inner_radius
            cavity_z_offset = self.processed_boundaries.cavity_z_offset
            casing_x = self.processed_boundaries.casing_x
            casing_y = self.processed_boundaries.casing_y
            casing_radius = self.processed_boundaries.casing_radius
            casing_thickness = self.processed_boundaries.casing_thickness

            max_r_sq = (cavity_inner_radius - self.spawn_buffer) ** 2
            min_r_sq = (hc_r + self.spawn_buffer) ** 2
            casing_height = self.processed_boundaries.casing_height

            all_xy_coords = []
            lim = int(math.ceil(cavity_inner_radius / spacing))
            for ix in range(-lim, lim + 1):
                for iy in range(-lim, lim + 1):
                    x = ix * spacing
                    y = iy * spacing

                    # Inside base cavity boundary
                    if x**2 + y**2 >= max_r_sq:
                        continue
                    # Outside tube boundary
                    if (x - hc_x) ** 2 + (y - hc_y) ** 2 <= min_r_sq:
                        continue

                    all_xy_coords.append((x, y))

            all_xy_coords.sort(key=lambda pt: pt[0] ** 2 + pt[1] ** 2)
            self.spawn_xy_coords = all_xy_coords

            # Spawn particles above bottom plate layer by layer
            z = cavity_z_offset + self.r_s + self.spawn_buffer
            inner_casing_r = casing_radius - casing_thickness
            inner_casing_r_sq = (inner_casing_r + self.spawn_buffer) ** 2
            outer_casing_r_sq = (casing_radius + self.spawn_buffer) ** 2

            while len(grid_points) < self.n_particles:
                is_in_casing_layer = (casing_radius > 0.0) and (z <= cavity_z_offset + casing_height)
                for x, y in all_xy_coords:
                    if len(grid_points) >= self.n_particles:
                        break
                    # Only exclude casing wall thickness within the casing height
                    if is_in_casing_layer:
                        dist_casing_sq = (x - casing_x) ** 2 + (y - casing_y) ** 2
                        if inner_casing_r_sq <= dist_casing_sq <= outer_casing_r_sq:
                            continue
                    grid_points.append((x, y, z))
                z += spacing

            # Transform spawned points from local coordinates to world coordinates
            bowl_pos, bowl_orn = self._get_base_link_origin(body_id, physics_client)
            world_grid_points = []
            for pt in grid_points:
                wpt, _ = p.multiplyTransforms(bowl_pos, bowl_orn, pt, [0.0, 0.0, 0.0, 1.0])
                world_grid_points.append(wpt)

            self.pos_jax = jnp.array(world_grid_points, dtype=jnp.float32)
            self.vel_jax = jnp.zeros((self.n_particles, 3), dtype=jnp.float32)
            self.last_positions = world_grid_points

    @property
    def particle_mass(self) -> float:
        """Calculate physical mass of a particle."""
        vol_s = (4.0 / 3.0) * math.pi * (self.r_s**3)
        return float(vol_s * self.rest_density)

    @property
    def particle_body_ids(self) -> list[int]:
        """Return dummy IDs for compatibility."""
        return list(range(self.n_particles))

    @property
    def active_count(self) -> int:
        """Return number of particles."""
        return self.n_particles

    @property
    def vol_s(self) -> float:
        """Return volume of a particle."""
        return (4.0 / 3.0) * math.pi * (self.r_s**3)

    def _get_base_link_origin(
        self, body_id: int, physics_client: int
    ) -> tuple[tuple[float, float, float], tuple[float, float, float, float]]:
        """Get the true base link origin by subtracting the local inertia offset."""
        if _is_real_physics_client(physics_client):
            base_pos, base_orn = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
            dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=physics_client)
            local_inertia_pos = dynamics[3]
            local_inertia_orn = dynamics[4]
            inv_inertia_pos, inv_inertia_orn = p.invertTransform(local_inertia_pos, local_inertia_orn)
            return p.multiplyTransforms(base_pos, base_orn, inv_inertia_pos, inv_inertia_orn)
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def get_raw_particle_positions(self) -> np.ndarray:
        """Return the raw particle positions array."""
        if self.pos_jax is not None:
            self.pos_jax.block_until_ready()
            return np.asarray(self.pos_jax)
        return np.empty((0, 3), dtype=np.float32)

    def get_raw_particle_velocities(self) -> np.ndarray:
        """Return the raw particle velocities array."""
        if self.vel_jax is not None:
            self.vel_jax.block_until_ready()
            return np.asarray(self.vel_jax)
        return np.empty((0, 3), dtype=np.float32)

    def get_particle_positions(self) -> Any:
        """Return voxelized grid-based volume positions representing the water.

        Returns:
            NumPy array of voxel coordinates representing the reconstructed fluid volume.
        """
        if self.last_positions is None or len(self.last_positions) == 0:
            return np.empty((0, 3), dtype=np.float32)
        voxel_centers = self.post_processor.process_voxels(self.last_positions)
        self.last_voxel_positions = voxel_centers
        self._last_voxel_count = len(voxel_centers)
        return voxel_centers

    def get_particle_colors(self) -> list[list[float]]:
        """Return particle colors for logger."""
        count = getattr(self, "_last_voxel_count", self.n_particles)
        return [self.PARTICLE_COLOR] * count

    def get_particle_radii(self) -> list[float]:
        """Return particle radii for logger."""
        count = getattr(self, "_last_voxel_count", self.n_particles)
        return [self.dx * self.VOXEL_RADIUS_SCALE] * count

    def get_boundary_forces(self) -> np.ndarray:
        """Return the 3D forces applied by moving and static boundaries to all fluid particles.

        Returns:
            NumPy array of shape (N, 3) representing (Fx, Fy, Fz) in Newtons for each particle.
        """
        if hasattr(self, "last_boundary_forces") and self.last_boundary_forces is not None:
            return np.asarray(self.last_boundary_forces, dtype=np.float32)
        return np.zeros((self.n_particles, 3), dtype=np.float32)

    def get_voxel_forces(self) -> np.ndarray:
        """Return the 3D boundary interaction forces mapped to the reconstructed voxel volume.

        Returns:
            NumPy array of shape (V, 3) representing forces in Newtons at each voxel center.
        """
        voxel_positions = getattr(self, "last_voxel_positions", None)
        if voxel_positions is None or len(voxel_positions) == 0:
            return np.empty((0, 3), dtype=np.float32)
        if not hasattr(self, "last_boundary_forces") or self.last_boundary_forces is None:
            return np.zeros((len(voxel_positions), 3), dtype=np.float32)
        if self.last_positions is None or len(self.last_positions) == 0:
            return np.zeros((len(voxel_positions), 3), dtype=np.float32)

        tree = cKDTree(self.last_positions)
        _, indices = tree.query(voxel_positions, k=1)
        return np.asarray(self.last_boundary_forces[indices], dtype=np.float32)

    def get_boundary_reaction_force(self) -> np.ndarray:
        """Return the net 3D reaction force vector exerted by the fluid on the moving boundary bodies.

        Returns:
            NumPy array [Fx, Fy, Fz] in Newtons representing the net reaction force (opposite to boundary forces).
        """
        if not hasattr(self, "last_boundary_forces") or self.last_boundary_forces is None:
            return np.zeros(3, dtype=np.float32)
        return -np.sum(self.last_boundary_forces, axis=0)

    def get_boundary_reaction_torque(self) -> float:
        """Return the net reaction torque in N*m exerted by the fluid on the rotating impeller vanes.

        Returns:
            Reaction torque in N*m.
        """
        if len(self.torques) > 0:
            return float(self.torques[-1])
        return 0.0

    def get_fluid_bodies(self) -> list[FluidBody]:
        """Return the list of dynamic fluid bodies with recomputed physical shapes and move/split/merge tracking.

        Returns:
            List of active FluidBody instances.
        """
        if not hasattr(self, "fluid_body_tracker"):
            self.fluid_body_tracker = FluidBodyTracker(r_s=self.r_s)

        return self.fluid_body_tracker.update_bodies(
            self.last_positions,
            self.last_velocities,
            cad_context=self.processed_boundaries.fluid_context,
        )

    def get_boundary_voxels(self) -> dict[str, np.ndarray]:
        """Compute 3D world-space voxel coordinates for each boundary type in the simulation grid.

        Returns:
            Dictionary mapping boundary type name (e.g. 'bowl', 'casingwall', 'tubewall', 'casinglid', 'lid', 'impeller')
            to an (M, 3) NumPy array of voxel center coordinates.
        """
        impeller_b = self.boundaries.get(LinkType.IMPELLER, None)
        target_omega = impeller_b.target_omega if impeller_b is not None else 0.0
        return self.post_processor.get_boundary_voxels(
            target_omega=target_omega,
            t_curr=getattr(self, "current_sim_time", 0.0),
        )

    def compute_forces_jax(
        self,
        pos_jax: jnp.ndarray,
        vel_jax: jnp.ndarray,
    ) -> jnp.ndarray:
        """Compute SPH forces for all particles returning a JAX array."""
        return _compute_forces_jax(
            pos_jax,
            vel_jax,
            self.mass,
            self.h,
            self.rest_density,
            self.viscosity,
            self.stiffness,
            self.poly6_factor,
            self.spiky_grad_factor,
            self.visc_lap_factor,
            self.pressure_avg_factor,
            self.min_distance_threshold,
        )

    def compute_forces(
        self,
        positions: list[tuple[float, float, float]],
        velocities: list[tuple[float, float, float]],
    ) -> list[list[float]]:
        """Compute SPH forces for all particles using JAX."""
        n = len(positions)
        if n == 0:
            return []

        # Convert to JAX arrays
        pos_jax = jnp.array(positions, dtype=jnp.float32)
        vel_jax = jnp.array(velocities, dtype=jnp.float32)

        forces_jax = self.compute_forces_jax(pos_jax, vel_jax)

        # Convert back to standard list of lists
        return forces_jax.tolist()

    def compute_knudsen_number(
        self,
        positions: list[tuple[float, float, float]],
        characteristic_length: Optional[float] = None,
    ) -> float:
        """Calculate the Knudsen number (Kn = mean_free_path / L)."""
        if characteristic_length is None:
            characteristic_length = self.characteristic_length
        n = len(positions)
        if n < 2:
            return 0.0

        pos_jax = jnp.array(positions, dtype=jnp.float32)

        diff = pos_jax[:, None, :] - pos_jax[None, :, :]
        r2 = jnp.sum(diff * diff, axis=-1)

        self_mask_large = jnp.eye(n) * 1e10
        r2_masked = r2 + self_mask_large

        min_dists = jnp.sqrt(jnp.min(r2_masked, axis=1))
        mean_free_path = jnp.mean(min_dists)

        return float(mean_free_path / characteristic_length)

    @property
    def radii(self) -> dict[LinkType, float]:
        """Return dict mapping LinkType keys to their float radius values."""
        hc_info = self.boundaries.get(LinkType.TUBE)
        hc_r = hc_info.radius if hc_info is not None else 0.0

        base_info = self.boundaries.get(LinkType.BASE)
        cavity_r = float(base_info.radius + base_info.thickness) if base_info is not None else 0.0

        impeller_info = self.boundaries.get(LinkType.IMPELLER)
        vanes_clearance = float(impeller_info.radius + 0.003) if impeller_info is not None else 0.0

        fallen_r = cavity_r + 0.010
        return {
            LinkType.TUBE: hc_r,
            LinkType.BASE: cavity_r,
            LinkType.IMPELLER: vanes_clearance,
            LinkType.FALLEN: fallen_r,
        }

    @property
    def thresholds(self) -> dict[LinkType, float]:
        """Return dict mapping LinkType keys to their float thresholds."""
        outlet_idx = self.link_indices.get(LinkType.OUTLET)
        has_outlet_link = outlet_idx is not None and outlet_idx != -1
        offset_val = (5.0 / 3.0) * self.r_s
        min_h = 0.0
        hc_info = self.boundaries.get(LinkType.TUBE)
        hc_height = hc_info.height if hc_info is not None else 0.0

        if self.body_id is not None and _is_real_physics_client(self.physics_client):
            if has_outlet_link:
                state = p.getLinkState(self.body_id, outlet_idx, physicsClientId=self.physics_client)
                min_h = float(state[4][2] + hc_height - offset_val)
            else:
                base_pos, _ = self._get_base_link_origin(self.body_id, self.physics_client)
                hc_z = hc_info.xyz[2] if hc_info is not None else 0.0
                min_h = float(base_pos[2] + hc_z + hc_height - offset_val)

        max_y = 0.0
        if self.body_id is not None and _is_real_physics_client(self.physics_client):
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            if has_outlet_link and outlet_idx < num_joints:
                aabb = p.getAABB(self.body_id, outlet_idx, physicsClientId=self.physics_client)
                max_y = float(aabb[1][1] + offset_val)
            else:
                base_pos, _ = self._get_base_link_origin(self.body_id, self.physics_client)
                hc_y = hc_info.xyz[1] if hc_info is not None else 0.0
                hc_r = hc_info.radius if hc_info is not None else 0.0
                max_y = float(base_pos[1] + hc_y + hc_r + offset_val)

        offset_mm = 0.0
        hc_idx = self.link_indices.get(LinkType.TUBE)
        has_tube_link = hc_idx is not None and hc_idx != -1
        if self.body_id is not None and _is_real_physics_client(self.physics_client):
            num_joints = p.getNumJoints(self.body_id, physicsClientId=self.physics_client)
            if has_tube_link and hc_idx < num_joints:
                info = p.getJointInfo(self.body_id, hc_idx, physicsClientId=self.physics_client)
                hc_y = info[14][1]
            else:
                hc_y = hc_info.xyz[1] if hc_info is not None else 0.0
            cavity_r_mm = self.radii[LinkType.BASE] * 1000.0
            hc_r_mm = self.radii[LinkType.TUBE] * 1000.0
            hc_y_mm = hc_y * 1000.0
            offset_mm = float(cavity_r_mm - hc_r_mm - hc_y_mm)

        return {
            LinkType.OUTLET: min_h,
            LinkType.OUTLET_MAX_Y: max_y,
            LinkType.TUBE: offset_mm,
        }

    def calculate_magnetic_drag(self, drag_config: Optional[MagneticDragConfig]) -> float:
        """Calculate the axial magnetic coupling drag torque.

        Args:
            drag_config: Explicit configuration parameters for magnetic drag.
                         If None, returns 0.0 (direct coupling).

        Returns:
            float: The calculated magnetic drag friction torque in N*m.
        """
        if drag_config is None or drag_config.magnet_count <= 0:
            return 0.0

        # Delegate computation to the compiled JAX function
        return float(
            calculate_magnetic_drag_jax(
                drag_config.magnet_radius,
                drag_config.magnet_thickness,
                drag_config.pump_well_wall,
                drag_config.magnet_count,
                drag_config.impeller_shaft_radius,
            )
        )

    def calculate_bearing_and_viscous_drag(self, omega: float) -> float:
        """Calculate additional mechanical drag torque (bearing and viscous).

        Args:
            omega: Impeller angular velocity in rad/s.

        Returns:
            float: Total bearing and viscous drag friction torque in N*m.
        """
        impeller_b = self.boundaries.get(LinkType.IMPELLER)
        if impeller_b is None:
            return 0.0

        shaft_r = getattr(impeller_b, "impeller_shaft_radius", None)
        radius = getattr(impeller_b, "radius", None)

        if shaft_r is None or radius is None:
            raise ValueError(
                "Required URDF boundary metadata (radius or impeller_shaft_radius) is missing for the impeller."
            )

        # Delegate computation to the compiled JAX function
        return float(
            calculate_bearing_and_viscous_drag_jax(
                omega,
                shaft_r,
                radius,
            )
        )

    def _apply_joint_velocity(
        self,
        body_id: int,
        physics_client: int,
        link_key: LinkType | str,
        target_velocity: float,
        max_force: Optional[float] = None,
        velocity_gain: Optional[float] = None,
    ) -> None:
        """Apply velocity control to a specific joint in PyBullet."""
        joint_idx = self.link_indices.get(link_key, -1)
        if (
            joint_idx != -1
            and _is_real_physics_client(physics_client)
            and joint_idx < p.getNumJoints(body_id, physicsClientId=physics_client)
        ):
            p.changeDynamics(
                bodyUniqueId=body_id,
                linkIndex=joint_idx,
                maxJointVelocity=self.MAX_JOINT_VELOCITY,
                physicsClientId=physics_client,
            )
            p.setJointMotorControl2(
                bodyUniqueId=body_id,
                jointIndex=joint_idx,
                controlMode=p.VELOCITY_CONTROL,
                targetVelocity=target_velocity,
                force=max_force if max_force is not None else 0.0,
                velocityGain=velocity_gain if velocity_gain is not None else 1.0,
                physicsClientId=physics_client,
            )

    def _detect_dynamic_spheres(
        self,
        physics_client: int,
        b_pos_list: list[tuple[float, float, float]],
        b_orn_list: list[tuple[float, float, float, float]],
        b_static_list: list[BoundaryConfig],
    ) -> None:
        """Detect dynamic plastic spheres in the simulation and append them as sphere boundaries."""
        from model import BoundaryConfig, ShapeType, BoundaryType

        num_bodies = p.getNumBodies(physicsClientId=physics_client)
        for i in range(num_bodies):
            if i == self.body_id:
                continue

            b_pos, b_orn = p.getBasePositionAndOrientation(i, physicsClientId=physics_client)
            shapes = p.getCollisionShapeData(i, -1, physicsClientId=physics_client)
            if len(shapes) > 0:
                shape_type = shapes[0][2]
                radius = shapes[0][3][0]
            else:
                raise ValueError(f"No collision shape data found for body ID {i}")

            if shape_type == p.GEOM_SPHERE:
                b_cfg = BoundaryConfig(
                    shape=ShapeType.SPHERE,
                    type=BoundaryType.SOLID,
                    radius=radius,
                    xyz=b_pos,
                    rpy=p.getEulerFromQuaternion(b_orn),
                    link_type=LinkType.BASE,
                    link_idx=-1,
                )
                b_pos_list.append(b_pos)
                b_orn_list.append(b_orn)
                b_static_list.append(b_cfg)

    def update(
        self,
        body_id: int,
        physics_client: int,
        damping: Optional[float] = None,
        target_omega: Optional[float] = None,
        max_force: Optional[float] = None,
        motor_power: Optional[float] = None,
        velocity_gain: Optional[float] = None,
    ) -> None:
        """Step simulation and manage deactivation."""
        self.body_id = body_id
        self.physics_client = physics_client
        impeller_b = self.boundaries.get(LinkType.IMPELLER)
        if impeller_b is not None:
            if target_omega is not None:
                # Dynamically apply torque speed limit if motor power is specified
                if len(self.torques) > 0 and motor_power is not None:
                    last_torque = abs(self.torques[-1])
                    if last_torque > 1e-5:
                        omega = min(target_omega, motor_power / last_torque)
                    else:
                        omega = target_omega
                else:
                    omega = target_omega
                impeller_b.target_omega = omega
            else:
                omega = 0.0
                impeller_b.target_omega = 0.0

            if max_force is not None:
                impeller_b.max_force = max_force

            is_magnetic = MagneticDragConfig.is_magnetic_coupling(impeller_b)

            if is_magnetic:
                # For magnetic coupling, drive both the dry-side drive hub and the impeller
                self._apply_joint_velocity(body_id, physics_client, LinkType.DRIVE_HUB, omega, max_force, velocity_gain)
                self._apply_joint_velocity(body_id, physics_client, LinkType.IMPELLER, omega, max_force, velocity_gain)
            else:
                # For direct coupling, the motor directly drives the impeller
                self._apply_joint_velocity(body_id, physics_client, LinkType.IMPELLER, omega, max_force, velocity_gain)

        if not self.spawner:
            raise RuntimeError("Fluid spawner is not initialized.")

        damping_val = damping if damping is not None else -1.0

        b_static_list = list(self.boundary_list)
        # Detect dynamic spheres and append them as boundaries
        if self.body_id is not None and _is_real_physics_client(physics_client):
            self._detect_dynamic_spheres(physics_client, [], [], b_static_list)

        self.processed_boundaries = BoundaryProcessor.process(
            b_static_list,
            body_id=self.body_id,
            physics_client=physics_client,
            default_idx_map=self.link_indices,
        )

        b_pos_arr = jnp.array(self.processed_boundaries.b_pos_arr, dtype=jnp.float32)
        b_orn_arr = jnp.array(self.processed_boundaries.b_orn_arr, dtype=jnp.float32)

        if not hasattr(self, "f_lbm") or self.f_lbm is None:
            self.f_lbm = jnp.zeros((15, self.nx, self.ny, self.nz), dtype=jnp.float32)
            for i, w in enumerate(_weights):
                self.f_lbm = self.f_lbm.at[i].set(w * 1.0)

        if self.body_id is not None and _is_real_physics_client(physics_client):
            bowl_vel, _ = p.getBaseVelocity(self.body_id, physicsClientId=physics_client)
        else:
            bowl_vel = [0.0, 0.0, 0.0]
        base_vel_arr = jnp.array(bowl_vel, dtype=jnp.float32)

        if not hasattr(self, "physics_config") or self.physics_config is None:
            self.physics_config = PhysicsConfig(
                mass=self.particle_mass,
                dt_sub=1.0 / (240.0 * 5),
                n_substeps=5,
                processed_boundaries=self.processed_boundaries,
                gravity=tuple(map(float, self.gravity)),
                base_idx=self.base_idx,
                K_boundary=self.stiffness_boundary,
                D_boundary=self.damping_boundary,
                r_s=self.r_s,
                high_damping_value=self.high_damping_value,
                nx=self.nx,
                ny=self.ny,
                nz=self.nz,
                dx=self.dx,
                origin=tuple(map(float, self.origin)),
            )
        else:
            self.physics_config.K_boundary = self.stiffness_boundary
            self.physics_config.D_boundary = self.damping_boundary

        config = self.physics_config

        self.pos_jax, self.vel_jax, self.f_lbm, torque_accum, b_forces_accum = _physics_step_jax(
            self.pos_jax,
            self.vel_jax,
            self.f_lbm,
            b_pos_arr,
            b_orn_arr,
            base_vel_arr,
            impeller_b.target_omega if impeller_b is not None else 0.0,
            self.current_sim_time,
            damping_val,
            config,
        )
        self.last_boundary_forces = b_forces_accum
        # Add magnetic coupling attractive drag friction torque dynamically.
        # This models the axial attraction force of the neodymium disc magnet pairs
        # acting across the thin well partition floor, generating friction torque.
        mag_friction = 0.0
        bearing_viscous_drag = 0.0
        if impeller_b is not None and abs(impeller_b.target_omega) > 1e-3:
            if not hasattr(self, "_cached_mag_friction"):
                drag_config = None
                if MagneticDragConfig.is_magnetic_coupling(impeller_b):
                    try:
                        drag_config = MagneticDragConfig(
                            magnet_radius=impeller_b.magnet_radius,
                            magnet_thickness=impeller_b.magnet_thickness,
                            pump_well_wall=impeller_b.pump_well_wall,
                            magnet_count=impeller_b.magnet_count,
                            impeller_shaft_radius=impeller_b.impeller_shaft_radius,
                        )
                    except Exception as e:
                        raise ValueError(f"Invalid magnetic drag configuration: {e}") from e

                self._cached_mag_friction = self.calculate_magnetic_drag(drag_config)
                self._cached_bearing_drag = self.calculate_bearing_and_viscous_drag(impeller_b.target_omega)
            mag_friction = self._cached_mag_friction
            bearing_viscous_drag = self._cached_bearing_drag

        is_tuning = getattr(self.provider, "is_tuning", False)
        if not is_tuning:
            avg_step_torque = (float(torque_accum) / 5) + mag_friction + bearing_viscous_drag
            self.torques.append(avg_step_torque)
        else:
            self.torques.append(0.0)

        # Ensure JAX device computations are resolved before converting to NumPy views when needed
        if not is_tuning:
            if self.pos_jax is not None:
                self.pos_jax.block_until_ready()
                pos_np = np.asarray(self.pos_jax)
                self.last_positions = pos_np
            else:
                pos_np = np.empty((0, 3), dtype=np.float32)

            if self.vel_jax is not None:
                self.vel_jax.block_until_ready()
                vel_np = np.asarray(self.vel_jax)
                self.last_velocities = vel_np
            else:
                vel_np = np.empty((0, 3), dtype=np.float32)

            # Check for LBM numerical instability (bulk particle speeds exceeding physical limits)
            if len(pos_np) > 0 and len(vel_np) > 0:
                active_mask = pos_np[:, 2] < 100.0
                active_vels = vel_np[active_mask]
                if len(active_vels) > 0:
                    avg_speed = float(np.mean(np.linalg.norm(active_vels, axis=1)))
                    if avg_speed > 1.5:
                        msg = (
                            f"WARNING: LBM Simulation numerical instability detected! "
                            f"Average particle speed is {avg_speed:.2f} m/s (limit is 1.5 m/s). "
                            f"Please check boundary damping and stiffness coefficients."
                        )
                        if self.provider and getattr(self.provider, "logger", None) is not None:
                            self.provider.logger.print(msg, symbol="⚠️")
                        else:
                            print(f"⚠️ {msg}")

            if len(pos_np) > 0:
                xs = pos_np[:, 0]
                ys = pos_np[:, 1]
                zs = pos_np[:, 2]

                active_mask = zs < 100.0

                # Spout/outlet indices
                spout_indices = np.where(
                    active_mask
                    & (zs >= self.thresholds[LinkType.OUTLET])
                    & (ys < self.thresholds[LinkType.OUTLET_MAX_Y])
                )[0]
                if len(spout_indices) > 0:
                    self.spout_water_ids.add_multiple(spout_indices)

                # Fallen indices computed dynamically from base cavity floor and boundaries in world frame
                base_thick = (
                    float(self.processed_boundaries.boundaries[0].thickness)
                    if self.processed_boundaries.boundaries
                    else 0.0035
                )
                z_min = float("inf")
                z_max = float("-inf")
                for i, b in enumerate(self.boundary_list):
                    surf = b.compute_surface_bounds()
                    z_start = float(self.processed_boundaries.b_pos_arr[i, 2])
                    top = z_start + max(surf.z_top, float(b.radius), float(b.height))
                    bot = z_start + min(surf.z_bottom - base_thick, -float(b.radius), -float(b.height))
                    if top > z_max:
                        z_max = top
                    if bot < z_min:
                        z_min = bot

                # Apply buffers to allow physical oscillations/boundary penetration
                z_min -= 0.020
                z_max += 0.020

                fallen_indices = np.where(
                    active_mask & ((zs < z_min) | (zs > z_max) | (xs**2 + ys**2 > (self.radii[LinkType.FALLEN]) ** 2))
                )[0]

                if len(fallen_indices) > 0:
                    self.total_fallen_water_ids.add_multiple(fallen_indices)
                    pos_arr = np.array(self.pos_jax)
                    vel_arr = np.array(self.vel_jax)
                    if not self.recycle_fluid:
                        self.fallen_out_water_ids.add_multiple(fallen_indices)
                        for idx in fallen_indices:
                            pos_arr[idx] = [float(idx) * 10.0, 0.0, 1000.0]
                            vel_arr[idx] = [0.0, 0.0, 0.0]
                    else:
                        bowl_pos, bowl_orn = self._get_base_link_origin(self.body_id, physics_client)
                        for idx in fallen_indices:
                            if self.spawn_xy_coords:
                                x, y = random.choice(self.spawn_xy_coords)
                            else:
                                x, y = 0.0, 0.0
                            z_local = (
                                self.processed_boundaries.cavity_z_offset
                                + self.r_s
                                + self.spawn_buffer
                                + random.uniform(0.0, 0.010)
                            )
                            wpt, _ = p.multiplyTransforms(bowl_pos, bowl_orn, [x, y, z_local], [0.0, 0.0, 0.0, 1.0])
                            pos_arr[idx] = wpt
                            vel_arr[idx] = [0.0, 0.0, 0.0]
                    self.pos_jax = jnp.array(pos_arr)
                    self.vel_jax = jnp.array(vel_arr)

        self._update_state_tracker()
        self.current_sim_time += 1.0 / 240.0

    def get_water_meshes(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Compute watertight 3D triangle meshes for all active dynamic fluid bodies.

        Returns:
            Dictionary mapping fluid body identifier (e.g. 'pool_1', 'stream_2', 'sheet_3')
            to a tuple of (vertices_array, faces_array).
        """
        bodies = self.get_fluid_bodies()
        meshes = {}
        for body in bodies:
            name = body.display_name
            verts, faces = body.to_mesh()
            if len(verts) > 0 and len(faces) > 0:
                meshes[name] = (verts, faces)
        return meshes

    def _update_state_tracker(self) -> None:
        """Synchronize particle positions, colors, radii, water meshes, and boundary voxels to state tracker."""
        if self.state_tracker is not None:
            if get_env_bool("SHOW_WATER_VOXELS", False):
                self.state_tracker.particle_positions = self.get_particle_positions()
                self.state_tracker.particle_colors = self.get_particle_colors()
                self.state_tracker.particle_radii = self.get_particle_radii()
            else:
                self.state_tracker.particle_positions = []
                self.state_tracker.particle_colors = []
                self.state_tracker.particle_radii = []

            self.state_tracker.fluid_bodies = self.get_fluid_bodies()
            self.state_tracker.water_meshes = self.get_water_meshes()

            if get_env_bool("SHOW_BOUNDARY_VOXELS", False):
                self.state_tracker.boundary_voxels = self.get_boundary_voxels()
            else:
                self.state_tracker.boundary_voxels = None
