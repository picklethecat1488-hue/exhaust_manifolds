"""Intermediate boundary domain models and boundary processing stage."""

import math
from dataclasses import dataclass
from typing import Any, Optional, Sequence
import numpy as np
import pybullet as p
from model.boundary_config import BoundaryConfig, BoundaryParam, ShapeType, BoundaryType, LinkType
from model.fluid_body import CADFeature, CADFeatureType, FluidCADContext


def _is_real_physics_client(physics_client: Any) -> bool:
    """Check if the given physics client ID is connected to a real physics server."""
    if not isinstance(physics_client, int) or physics_client < 0:
        return False
    try:
        return p.isConnected(physicsClientId=physics_client) == 1
    except Exception:
        return False


# Shape enumeration integer codes for JAX tensor encoding
SHAPE_NONE = 0
SHAPE_CYLINDER = 1
SHAPE_BOX = 2
SHAPE_PLANE = 3
SHAPE_IMPELLER = 4
SHAPE_TUBE = 5
SHAPE_SPHERE = 6
SHAPE_CASING = 7

SHAPE_NAME_TO_INT: dict[ShapeType, int] = {
    ShapeType.CYLINDER: SHAPE_CYLINDER,
    ShapeType.BOX: SHAPE_BOX,
    ShapeType.PLANE: SHAPE_PLANE,
    ShapeType.IMPELLER: SHAPE_IMPELLER,
    ShapeType.TUBE: SHAPE_TUBE,
    ShapeType.SPHERE: SHAPE_SPHERE,
    ShapeType.CASING: SHAPE_CASING,
}


@dataclass(frozen=True)
class BowlBoundary:
    """Intermediate boundary representation for the reservoir bowl cavity."""

    radius: float
    z_floor: float
    height: float = 0.0
    thickness: float = 0.0035
    friction: float = 0.20
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate whether points lie inside the solid structure of the bowl container."""
        thick = self.thickness if self.thickness > 0.0 else 0.0035
        dist_sq = (x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2
        is_wall = (
            (z >= self.z_floor)
            & (z <= self.z_floor + self.height)
            & (dist_sq >= self.radius**2)
            & (dist_sq <= (self.radius + thick) ** 2)
        )
        is_floor = (z >= self.z_floor - thick) & (z <= self.z_floor) & (dist_sq <= (self.radius + thick) ** 2)
        return is_wall | is_floor

    def is_solid(self, x: float, y: float, z: float) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]))[0])


@dataclass(frozen=True)
class CasingWallBoundary:
    """Intermediate boundary representation for the pump casing cylinder wall."""

    x: float
    y: float
    r_inner: float
    r_outer: float
    z_min: float
    z_max: float
    slot_height: float
    slot_width: float
    ceiling_thickness: float = 0.0035
    friction: float = 0.20
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate whether points lie inside the solid pump casing wall."""
        in_z = (self.z_min <= z) & (z <= self.z_max)
        dist_sq = (x - self.x) ** 2 + (y - self.y) ** 2
        in_wall = (self.r_inner**2 <= dist_sq) & (dist_sq <= self.r_outer**2)
        in_cutout = (
            ((z - self.z_min) < self.slot_height)
            & ((y - self.y) > 0.0)
            & (np.abs(x - self.x) < (self.slot_width / 2.0))
        )
        return in_z & in_wall & (~in_cutout)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]))[0])


@dataclass(frozen=True)
class TubeWallBoundary:
    """Intermediate boundary representation for the vertical fluid tube wall."""

    x: float
    y: float
    r_inner: float
    r_outer: float
    z_min: float
    z_max: float
    slot_height: float
    slot_width: float
    spout_radius: float = 0.014
    spout_height: float = 0.049
    friction: float = 0.20
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate whether points lie inside the solid vertical tube wall."""
        in_z = (self.z_min <= z) & (z <= self.z_max)
        dist_sq = (x - self.x) ** 2 + (y - self.y) ** 2
        in_wall = (self.r_inner**2 <= dist_sq) & (dist_sq <= self.r_outer**2)
        in_cutout = (
            ((z - self.z_min) < self.slot_height)
            & ((y - self.y) < 0.0)
            & (np.abs(x - self.x) < (self.slot_width / 2.0))
        )
        return in_z & in_wall & (~in_cutout)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]))[0])


@dataclass(frozen=True)
class CasingLidBoundary:
    """Intermediate boundary representation for the pump cover plate with snout/tube pass-through."""

    x: float
    y: float
    radius: float
    z_min: float
    z_max: float
    tube_x: float
    tube_y: float
    tube_r_inner: float
    friction: float = 0.20
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate whether points lie inside the pump cover plate."""
        in_z = (self.z_min <= z) & (z <= self.z_max)
        dist_sq = (x - self.x) ** 2 + (y - self.y) ** 2
        in_lid = dist_sq <= self.radius**2
        dist_snout_sq = x**2 + y**2
        dist_tube_hole_sq = (x - self.tube_x) ** 2 + (y - self.tube_y) ** 2
        in_cutout = (dist_snout_sq < 0.0065**2) | (dist_tube_hole_sq < self.tube_r_inner**2)
        return in_z & in_lid & (~in_cutout)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]))[0])


@dataclass(frozen=True)
class LidBoundary:
    """Intermediate boundary representation for the top drinking lid with drinking tray, terrace, and drain hole."""

    r_outer: float
    r_pocket: float
    z_base: float
    z_floor: float
    z_top: float
    tube_x: float
    tube_y: float
    tube_r: float
    drain_y: float
    drain_r: float
    terrace_r: float
    terrace_z_max: float
    friction: float = 0.20
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    orn: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate whether points lie inside the solid structure of the lid."""
        in_z = (self.z_base <= z) & (z <= max(self.z_top, self.terrace_z_max))
        dist_sq = (x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2
        in_outer = dist_sq <= self.r_outer**2

        dist_terrace_sq = (x - self.tube_x) ** 2 + (y - self.tube_y) ** 2
        in_platform = dist_terrace_sq <= max(self.terrace_r, 0.030) ** 2
        in_drain = ((x**2 + (y - self.drain_y) ** 2) < self.drain_r**2) & (~in_platform)
        in_tube = dist_terrace_sq < self.tube_r**2
        is_hole = in_drain | in_tube

        in_base = (self.z_base <= z) & (z <= self.z_floor)
        in_rim = (self.z_floor < z) & (z <= self.z_top) & (dist_sq >= self.r_pocket**2)
        in_terrace = (self.z_floor < z) & (z <= self.terrace_z_max) & in_platform

        return in_z & in_outer & (~is_hole) & (in_base | in_rim | in_terrace)

    def is_solid(self, x: float, y: float, z: float) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]))[0])


@dataclass(frozen=True)
class ImpellerBoundary:
    """Intermediate boundary representation for the rotating impeller."""

    radius: float
    height: float
    thickness: float
    vane_thickness: float
    num_vanes: float
    vane_twist_rad: float
    target_omega: float
    max_force: float
    friction: float
    pos: tuple[float, float, float]
    orn: tuple[float, float, float, float]

    def is_solid_vectorized(
        self,
        x: np.ndarray,
        y: np.ndarray,
        z: np.ndarray,
        angle: float = 0.0,
    ) -> np.ndarray:
        """Evaluate points within the rotating impeller hub and helical vanes."""
        dx_loc = x - self.pos[0]
        dy_loc = y - self.pos[1]
        dz_loc = z - self.pos[2]
        r_sq = dx_loc**2 + dy_loc**2

        hub_r = self.thickness
        in_z = (dz_loc >= 0.0) & (dz_loc <= self.height)
        hub_mask = (r_sq <= hub_r**2) & in_z

        safe_height = self.height if self.height > 0.0 else 1.0
        pitch = self.vane_twist_rad / safe_height
        total_angle = angle + dz_loc * pitch

        xr = dx_loc * np.cos(total_angle) + dy_loc * np.sin(total_angle)
        yr = -dx_loc * np.sin(total_angle) + dy_loc * np.cos(total_angle)

        theta = np.arctan2(yr, xr)
        num_vanes = max(1.0, self.num_vanes)
        angle_sep = 2.0 * np.pi / num_vanes
        theta_mod = (theta + np.pi) % angle_sep - (angle_sep / 2.0)

        dist_to_blade = np.sqrt(r_sq) * np.sin(theta_mod)
        vane_thick = max(self.vane_thickness, 0.002)
        blades_mask = (np.abs(dist_to_blade) <= vane_thick / 2.0) & (r_sq >= hub_r**2) & (r_sq <= self.radius**2) & in_z
        return hub_mask | blades_mask

    def is_solid(self, x: float, y: float, z: float, angle: float = 0.0) -> bool:
        """Evaluate scalar point inside solid structure."""
        return bool(self.is_solid_vectorized(np.array([x]), np.array([y]), np.array([z]), angle=angle)[0])


@dataclass(frozen=True)
class SphereBoundary:
    """Intermediate boundary representation for a spherical boundary obstacle."""

    radius: float
    friction: float
    pos: tuple[float, float, float]
    orn: tuple[float, float, float, float]

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate points within spherical boundary."""
        dist_sq = (x - self.pos[0]) ** 2 + (y - self.pos[1]) ** 2 + (z - self.pos[2]) ** 2
        return dist_sq <= self.radius**2


@dataclass(frozen=True)
class PlaneBoundary:
    """Intermediate boundary representation for a planar boundary floor/wall."""

    thickness: float
    friction: float
    pos: tuple[float, float, float]
    orn: tuple[float, float, float, float]

    def is_solid_vectorized(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Evaluate points within planar slab boundary."""
        dz = z - self.pos[2]
        return (dz >= -self.thickness) & (dz <= 0.0)


@dataclass
class ProcessedBoundaries:
    """Structured container holding packed boundary tensor arrays and intermediate models."""

    b_shapes: np.ndarray
    b_types: np.ndarray
    b_params: np.ndarray
    b_pos_arr: np.ndarray
    b_orn_arr: np.ndarray
    b_vel_arr: np.ndarray
    base_idx: int
    boundaries: list[Any]

    @property
    def base(self) -> Optional[BowlBoundary]:
        """Get the base bowl cavity boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, BowlBoundary):
                return b
        return None

    @property
    def casing_wall(self) -> Optional[CasingWallBoundary]:
        """Get the pump casing wall boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, CasingWallBoundary):
                return b
        return None

    @property
    def tube_wall(self) -> Optional[TubeWallBoundary]:
        """Get the vertical tube wall boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, TubeWallBoundary):
                return b
        return None

    @property
    def casing_lid(self) -> Optional[CasingLidBoundary]:
        """Get the pump casing cover boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, CasingLidBoundary):
                return b
        return None

    @property
    def lid(self) -> Optional[LidBoundary]:
        """Get the top drinking lid boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, LidBoundary):
                return b
        return None

    @property
    def impeller(self) -> Optional[ImpellerBoundary]:
        """Get the rotating impeller boundary model if present."""
        for b in self.boundaries:
            if isinstance(b, ImpellerBoundary):
                return b
        return None

    @property
    def cavity_pos(self) -> tuple[float, float, float]:
        """Get base container world position."""
        if self.base_idx >= 0 and self.base_idx < len(self.b_pos_arr):
            return tuple(float(x) for x in self.b_pos_arr[self.base_idx])  # type: ignore[return-value]
        return (0.0, 0.0, 0.0)

    @property
    def cavity_orn(self) -> tuple[float, float, float, float]:
        """Get base container world orientation quaternion."""
        if self.base_idx >= 0 and self.base_idx < len(self.b_orn_arr):
            return tuple(float(x) for x in self.b_orn_arr[self.base_idx])  # type: ignore[return-value]
        return (0.0, 0.0, 0.0, 1.0)

    @property
    def cavity_z_offset(self) -> float:
        """Get base container bottom floor Z offset in world coordinates."""
        if self.base_idx >= 0 and self.base_idx < len(self.b_pos_arr):
            return float(
                self.b_pos_arr[self.base_idx, BoundaryParam.THICKNESS]
                + self.b_params[self.base_idx, BoundaryParam.Z_OFFSET]
            )
        return 0.0

    @property
    def cavity_inner_radius(self) -> float:
        """Get base container inner radius."""
        if self.base_idx >= 0 and self.base_idx < len(self.b_params):
            return float(self.b_params[self.base_idx, BoundaryParam.RADIUS])
        return 0.0

    @property
    def base_height(self) -> float:
        """Get base container height."""
        if self.base_idx >= 0 and self.base_idx < len(self.b_params):
            return float(self.b_params[self.base_idx, BoundaryParam.HEIGHT])
        return 0.0

    @property
    def fluid_context(self) -> FluidCADContext:
        """Construct dynamic CAD geometry boundaries context for multi-tier fluid simulation tracking."""
        return FluidCADContext.from_processed_boundaries(self)

    @property
    def tube_idx(self) -> int:
        """Find the index of the tube boundary element."""
        indices = np.where(self.b_shapes == SHAPE_TUBE)[0]
        return int(indices[0]) if len(indices) > 0 else -1

    @property
    def tube_x(self) -> float:
        """Get tube X coordinate in world space."""
        t_idx = self.tube_idx
        return float(self.b_pos_arr[t_idx, 0]) if t_idx >= 0 else 0.0

    @property
    def tube_y(self) -> float:
        """Get tube Y coordinate in world space."""
        t_idx = self.tube_idx
        return float(self.b_pos_arr[t_idx, 1]) if t_idx >= 0 else 0.0

    @property
    def tube_radius(self) -> float:
        """Get tube outer radius."""
        t_idx = self.tube_idx
        return float(self.b_params[t_idx, BoundaryParam.RADIUS]) if t_idx >= 0 else 0.0

    @property
    def tube_inner_radius(self) -> float:
        """Get tube inner bore radius."""
        t_idx = self.tube_idx
        if t_idx >= 0:
            return float(self.b_params[t_idx, BoundaryParam.RADIUS] - self.b_params[t_idx, BoundaryParam.THICKNESS])
        return 0.0

    @property
    def casing_idx(self) -> int:
        """Find the index of the pump casing boundary element."""
        indices = np.where(self.b_shapes == SHAPE_CASING)[0]
        return int(indices[0]) if len(indices) > 0 else -1

    @property
    def casing_x(self) -> float:
        """Get casing center X coordinate."""
        c_idx = self.casing_idx
        return float(self.b_pos_arr[c_idx, 0]) if c_idx >= 0 else 0.0

    @property
    def casing_y(self) -> float:
        """Get casing center Y coordinate."""
        c_idx = self.casing_idx
        return float(self.b_pos_arr[c_idx, 1]) if c_idx >= 0 else 0.0

    @property
    def casing_radius(self) -> float:
        """Get casing outer radius."""
        c_idx = self.casing_idx
        return float(self.b_params[c_idx, BoundaryParam.RADIUS]) if c_idx >= 0 else 0.0

    @property
    def casing_thickness(self) -> float:
        """Get casing wall thickness."""
        c_idx = self.casing_idx
        return float(self.b_params[c_idx, BoundaryParam.THICKNESS]) if c_idx >= 0 else 0.0

    @property
    def casing_height(self) -> float:
        """Get casing height."""
        c_idx = self.casing_idx
        return float(self.b_params[c_idx, BoundaryParam.HEIGHT]) if c_idx >= 0 else 0.0

    @property
    def b_pos_list(self) -> list[tuple[float, float, float]]:
        """Get list of 3D positions for all boundaries."""
        return [tuple(float(v) for v in p) for p in self.b_pos_arr]  # type: ignore[misc]

    @property
    def b_orn_list(self) -> list[tuple[float, float, float, float]]:
        """Get list of orientation quaternions for all boundaries."""
        return [tuple(float(v) for v in o) for o in self.b_orn_arr]  # type: ignore[misc]

    @property
    def tube_params(self) -> tuple[bool, float, float, float]:
        """Get packed tube geometry tuple."""
        t_idx = self.tube_idx
        if t_idx >= 0:
            has_tube = True
            base_pos = self.cavity_pos
            tb_x = float(self.b_pos_arr[t_idx, 0] - base_pos[0])
            tb_y = float(self.b_pos_arr[t_idx, 1] - base_pos[1])
            tb_r = float(self.b_params[t_idx, BoundaryParam.RADIUS] - self.b_params[t_idx, BoundaryParam.THICKNESS])
            return (has_tube, tb_x, tb_y, tb_r)
        return (False, 0.0, 0.0, 0.0)

    @property
    def damping_params(self) -> tuple[float, float, float, bool, float, float, float, float, float, float]:
        """Get packed damping parameters tuple."""
        t_idx = self.tube_idx
        has_tube = t_idx >= 0
        tube_yb = self.tube_y if has_tube else 0.0
        tube_rb = self.tube_inner_radius if has_tube else 0.0
        spout_r = float(self.b_params[t_idx, BoundaryParam.RADIUS]) if has_tube else 0.014
        tube_h = float(self.b_params[t_idx, BoundaryParam.HEIGHT]) if has_tube else 0.0
        influence_r = tube_rb + spout_r
        influence_h = tube_h + 0.049
        return (
            self.cavity_inner_radius,
            self.cavity_z_offset,
            self.base_height,
            has_tube,
            tube_yb,
            tube_rb,
            influence_r,
            influence_h,
            0.998,
            0.50,
        )

    def is_solid(self, x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        """Vectorized collision check evaluating all intermediate boundaries in union.

        Args:
            x: NumPy array of X coordinates.
            y: NumPy array of Y coordinates.
            z: NumPy array of Z coordinates.

        Returns:
            Boolean array indicating whether each (x, y, z) coordinate lies inside a solid boundary.
        """
        is_sol = np.zeros(len(x), dtype=bool)
        for b in self.boundaries:
            if hasattr(b, "is_solid_vectorized"):
                is_sol |= b.is_solid_vectorized(x, y, z)
        return is_sol


class BoundaryProcessor:
    """Processing stage converting raw BoundaryConfig lists into structured ProcessedBoundaries."""

    @classmethod
    def process(
        cls,
        boundary_list: Sequence[BoundaryConfig],
        body_id: Optional[int] = None,
        physics_client: Optional[int] = None,
        base_link_origin: Optional[tuple[tuple[float, float, float], tuple[float, float, float, float]]] = None,
        default_idx_map: Optional[dict[LinkType, Optional[int]]] = None,
    ) -> ProcessedBoundaries:
        """Process and normalize boundary configurations into typed intermediate models and packed arrays.

        Args:
            boundary_list: List of BoundaryConfig objects from configuration or provider metadata.
            body_id: Optional PyBullet multi-body ID for dynamic joint state tracking.
            physics_client: PyBullet client ID.
            base_link_origin: Pre-calculated base link world transform tuple (pos, orn).
            default_idx_map: Dictionary mapping LinkType to PyBullet link indices.

        Returns:
            ProcessedBoundaries containing structured boundary models and packed simulation tensors.
        """
        if default_idx_map is None:
            default_idx_map = {}

        if base_link_origin is not None:
            base_pos, base_orn = base_link_origin
        elif body_id is not None and physics_client is not None and _is_real_physics_client(physics_client):
            base_p, base_o = p.getBasePositionAndOrientation(body_id, physicsClientId=physics_client)
            dynamics = p.getDynamicsInfo(body_id, -1, physicsClientId=physics_client)
            inv_p, inv_o = p.invertTransform(dynamics[3], dynamics[4])
            base_pos, base_orn = p.multiplyTransforms(base_p, base_o, inv_p, inv_o)
        else:
            base_pos, base_orn = (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

        # 1. Resolve world coordinates, orientations, and linear velocities for each boundary element
        b_pos_list: list[tuple[float, float, float]] = []
        b_orn_list: list[tuple[float, float, float, float]] = []
        b_vel_list: list[tuple[float, float, float]] = []
        b_shapes_list: list[int] = []
        b_types_list: list[int] = []
        b_params_list: list[list[float]] = []

        base_boundary_idx = -1
        base_info: Optional[BoundaryConfig] = None
        impeller_info: Optional[BoundaryConfig] = None

        for idx, b in enumerate(boundary_list):
            link_idx = b.link_idx
            if link_idx == -1 and default_idx_map and b.link_type in default_idx_map:
                mapped_idx = default_idx_map[b.link_type]
                if mapped_idx is not None:
                    link_idx = mapped_idx

            if (
                link_idx != -1
                and body_id is not None
                and physics_client is not None
                and _is_real_physics_client(physics_client)
            ):
                state = p.getLinkState(body_id, link_idx, computeLinkVelocity=1, physicsClientId=physics_client)
                if state is not None:
                    parent_pos, parent_orn = state[4], state[5]
                    parent_vel = state[6]
                else:
                    parent_pos, parent_orn = base_pos, base_orn
                    parent_vel, _ = p.getBaseVelocity(body_id, physicsClientId=physics_client)
            elif body_id is not None and physics_client is not None and _is_real_physics_client(physics_client):
                parent_pos, parent_orn = base_pos, base_orn
                parent_vel, _ = p.getBaseVelocity(body_id, physicsClientId=physics_client)
            else:
                parent_pos, parent_orn = base_pos, base_orn
                parent_vel = (0.0, 0.0, 0.0)

            local_xyz = b.xyz
            local_rpy = b.rpy
            local_orn = p.getQuaternionFromEuler(local_rpy)
            b_world_pos, b_world_orn = p.multiplyTransforms(parent_pos, parent_orn, local_xyz, local_orn)

            b_pos_list.append(b_world_pos)
            b_orn_list.append(b_world_orn)
            b_vel_list.append(tuple(parent_vel))

        # Precompute maximum fountain top / ceiling Z relative to base link origin
        base_h = 0.0
        for b in boundary_list:
            if b.link_type == LinkType.BASE:
                base_h = float(b.height)
                break

        fountain_top_z = base_h
        for idx, b in enumerate(boundary_list):
            b_z = b_pos_list[idx][2] - base_pos[2] + max(float(b.height), float(b.radius))
            fountain_top_z = max(fountain_top_z, b_z)

        has_sph = any(b.shape == ShapeType.SPHERE for b in boundary_list)
        if has_sph:
            sph_idx = [idx for idx, b in enumerate(boundary_list) if b.shape == ShapeType.SPHERE][0]
            sph_top_z = b_pos_list[sph_idx][2] - base_pos[2] + float(boundary_list[sph_idx].radius) + 0.002
            max_ceiling_z = sph_top_z
        else:
            max_ceiling_z = fountain_top_z

        # Precompute casing, impeller, and lid geometry
        casing_top_z = 0.0
        impeller_radius = 0.0
        slot_w = 0.0
        slot_h = 0.0
        tube_r_eff = 0.0
        lid_slope_ratio = 0.0
        for b in boundary_list:
            if b.shape == ShapeType.CASING or b.link_type == LinkType.CASING:
                casing_top_z = float(b.height) + float(b.ceiling_thickness)
                slot_w = float(b.slot_width)
                slot_h = float(b.slot_height)
            elif b.shape == ShapeType.IMPELLER or b.link_type == LinkType.IMPELLER:
                impeller_radius = float(b.radius)
            elif b.shape == ShapeType.TUBE or b.link_type == LinkType.TUBE:
                tube_r_eff = max(0.0, float(b.radius) - float(b.thickness))
            elif (b.shape == ShapeType.CYLINDER or b.link_type == LinkType.LID) and b.has_drain:
                if float(b.radius) > 0.0:
                    lid_slope_ratio = float(b.height) / float(b.radius)

        a_slot = slot_w * slot_h
        a_tube = math.pi * (tube_r_eff**2) + 1e-6
        slot_constriction_ratio = min(a_slot / a_tube, 1.0) if tube_r_eff > 0.0 else 1.0

        for idx, b in enumerate(boundary_list):
            shape_int = SHAPE_NAME_TO_INT.get(b.shape, SHAPE_NONE) if b.shape is not None else SHAPE_NONE
            b_shapes_list.append(shape_int)

            type_int = 1 if (b.type == BoundaryType.CAVITY or b.type == BoundaryType.SOLID_CAVITY) else 0
            b_types_list.append(type_int)

            friction_val = float(b.boundary_friction) if b.boundary_friction is not None else 0.0
            lid_cavity_depth = (
                max(0.0, b_pos_list[idx][2] - base_pos[2]) if (b.link_type == LinkType.LID or b.has_drain) else 0.0
            )
            surf = b.compute_surface_bounds(
                max_ceiling_z=max_ceiling_z,
                casing_top_z=casing_top_z,
                impeller_radius=impeller_radius,
                slot_constriction_ratio=slot_constriction_ratio,
                lid_slope_ratio=lid_slope_ratio,
                lid_cavity_depth=lid_cavity_depth,
            )
            params = [
                float(b.radius),
                float(b.height),
                float(b.thickness),
                float(b.z_offset) if b.z_offset is not None else 0.0,
                float(b.slot_height),
                float(b.slot_width),
                float(b.ceiling_thickness),
                float(b.vane_thickness) if b.vane_thickness is not None else 0.0,
                float(b.num_vanes) if b.num_vanes is not None else 0.0,
                float(b.vane_twist_rad),
                float(b.cutoff_y) if b.cutoff_y is not None else 0.0,
                1.0 if b.has_tube else 0.0,
                1.0 if b.has_drain else 0.0,
                float(b.tube_radius),
                float(b.drain_hole_y) if b.drain_hole_y is not None else 0.0,
                float(b.drain_hole_radius) if b.drain_hole_radius is not None else 0.0,
                friction_val,
                surf.z_bottom,
                surf.z_top,
                surf.r_inner,
                surf.r_outer,
                surf.tray_z_min,
                surf.tray_z_max,
                surf.suction_z_min,
                surf.suction_z_max,
                surf.spout_z_min,
                surf.drain_target_z,
                surf.drain_influence_radius,
                surf.max_ceiling_z,
                surf.wall_band_r_max,
                surf.casing_top_z,
                surf.impeller_radius,
                surf.slot_constriction_ratio,
                surf.lid_slope_ratio,
                surf.drain_edge_r_min,
                surf.drain_edge_r_max,
                surf.shelf_depth,
                surf.has_intake,
                surf.intake_pos_x,
                surf.intake_pos_y,
                surf.intake_pos_z,
                surf.intake_normal_x,
                surf.intake_normal_y,
                surf.intake_normal_z,
                surf.intake_radius,
                surf.drain_pos_x,
                surf.drain_pos_y,
                surf.drain_pos_z,
                surf.drain_normal_x,
                surf.drain_normal_y,
                surf.drain_normal_z,
                surf.drain_radius,
                surf.tube_pos_x,
                surf.tube_pos_y,
                surf.tube_pos_z,
                surf.tube_normal_x,
                surf.tube_normal_y,
                surf.tube_normal_z,
                surf.tube_port_radius,
                surf.is_submerged,
                surf.pool_max_z,
            ]
            b_params_list.append(params)

            if b.link_type == LinkType.BASE and base_boundary_idx == -1:
                base_boundary_idx = idx
                base_info = b
            elif b.link_type == LinkType.IMPELLER:
                impeller_info = b

        # 2. Build intermediate structured domain models
        intermediate_boundaries: list[Any] = []
        z_floor = base_info.xyz[2] if (base_info is not None and base_info.xyz is not None) else 0.0
        r_bowl = base_info.radius if base_info is not None else 0.0
        base_h = base_info.height if base_info is not None else 0.0
        base_thick = base_info.thickness if base_info is not None else 0.0

        if base_info is not None:
            intermediate_boundaries.append(
                BowlBoundary(
                    radius=r_bowl,
                    z_floor=z_floor,
                    height=base_h,
                    thickness=base_thick,
                    friction=float(base_info.boundary_friction or 0.0),
                    pos=b_pos_list[base_boundary_idx] if base_boundary_idx != -1 else base_pos,
                    orn=b_orn_list[base_boundary_idx] if base_boundary_idx != -1 else base_orn,
                )
            )

        # Casing
        casing_x, casing_y, casing_r, casing_thick, casing_h = 0.0, 0.0, 0.0, 0.0, 0.0
        casing_ceiling_thick, casing_slot_h, casing_slot_w = 0.0, 0.0, 0.0
        casing_fric = 0.0
        casing_pos_t = (0.0, 0.0, 0.0)
        casing_orn_t = (0.0, 0.0, 0.0, 1.0)
        for i, b in enumerate(boundary_list):
            if b.shape == ShapeType.CASING or b.link_type == LinkType.CASING:
                if b.xyz is not None:
                    casing_x, casing_y = b.xyz[0], b.xyz[1]
                casing_r = b.radius
                casing_thick = b.thickness if b.thickness is not None else 0.0
                casing_h = b.height
                casing_ceiling_thick = getattr(b, "ceiling_thickness", 0.0) or 0.0
                casing_slot_h = getattr(b, "slot_height", 0.0) or 0.0
                casing_slot_w = getattr(b, "slot_width", 0.0) or 0.0
                casing_fric = float(b.boundary_friction or 0.0)
                casing_pos_t = b_pos_list[i]
                casing_orn_t = b_orn_list[i]
                break

        if casing_r > 0.0:
            intermediate_boundaries.append(
                CasingWallBoundary(
                    x=casing_x,
                    y=casing_y,
                    r_inner=casing_r - casing_thick,
                    r_outer=casing_r,
                    z_min=z_floor,
                    z_max=z_floor + casing_h,
                    slot_height=casing_slot_h,
                    slot_width=casing_slot_w,
                    ceiling_thickness=casing_ceiling_thick,
                    friction=casing_fric,
                    pos=casing_pos_t,
                    orn=casing_orn_t,
                )
            )

        # Tube
        tube_x, tube_y, tube_r, tube_thick, tube_h = 0.0, 0.0, 0.0, 0.0, 0.0
        tube_slot_h, tube_slot_w = 0.0, 0.0
        tube_spout_r, tube_spout_h = 0.0, 0.0
        tube_fric = 0.0
        tube_pos_t = (0.0, 0.0, 0.0)
        tube_orn_t = (0.0, 0.0, 0.0, 1.0)
        for i, b in enumerate(boundary_list):
            if b.link_type == LinkType.TUBE:
                if b.xyz is not None:
                    tube_x, tube_y = b.xyz[0], b.xyz[1]
                tube_r = b.radius
                tube_thick = b.thickness if b.thickness is not None else 0.0
                tube_h = b.height
                tube_slot_h = getattr(b, "slot_height", 0.0) or 0.0
                tube_slot_w = getattr(b, "slot_width", 0.0) or 0.0
                tube_spout_r = getattr(b, "spout_radius", 0.0) or 0.0
                tube_spout_h = getattr(b, "spout_height", 0.0) or 0.0
                tube_fric = float(b.boundary_friction or 0.0)
                tube_pos_t = b_pos_list[i]
                tube_orn_t = b_orn_list[i]
                break

        if tube_r > 0.0:
            intermediate_boundaries.append(
                TubeWallBoundary(
                    x=tube_x,
                    y=tube_y,
                    r_inner=tube_r - tube_thick,
                    r_outer=tube_r,
                    z_min=z_floor,
                    z_max=z_floor + tube_h,
                    slot_height=tube_slot_h,
                    slot_width=tube_slot_w,
                    spout_radius=tube_spout_r,
                    spout_height=tube_spout_h,
                    friction=tube_fric,
                    pos=tube_pos_t,
                    orn=tube_orn_t,
                )
            )

        # Casing Lid (pump cover)
        if casing_r > 0.0 and casing_ceiling_thick > 0.0:
            intermediate_boundaries.append(
                CasingLidBoundary(
                    x=casing_x,
                    y=casing_y,
                    radius=casing_r,
                    z_min=z_floor + (casing_h - casing_ceiling_thick),
                    z_max=z_floor + casing_h,
                    tube_x=tube_x,
                    tube_y=tube_y,
                    tube_r_inner=max(0.0, tube_r - tube_thick),
                    friction=casing_fric,
                    pos=casing_pos_t,
                    orn=casing_orn_t,
                )
            )

        # Top Drinking Lid
        lid_pocket_r = 0.0
        drain_y, drain_r = 0.0, 0.0
        lid_pocket_h, lid_thickness = 0.0, 0.0
        terrace_r = 0.0
        terrace_h = 0.0
        lid_fric = 0.0
        lid_pos_t = (0.0, 0.0, 0.0)
        lid_orn_t = (0.0, 0.0, 0.0, 1.0)
        has_lid = False
        for i, b in enumerate(boundary_list):
            if b.link_type == LinkType.LID:
                has_lid = True
                if (
                    getattr(b, "has_drain", False)
                    and getattr(b, "drain_hole_radius", 0.0) > 0.0
                    and b.radius < r_bowl
                    and b.radius > 0.0
                ):
                    lid_pocket_r = b.radius
                    lid_pocket_h = b.height if b.height is not None else 0.0
                    lid_thickness = b.thickness if b.thickness is not None else 0.0
                    drain_y = getattr(b, "drain_hole_y", 0.0) or 0.0
                    drain_r = getattr(b, "drain_hole_radius", 0.0) or 0.0
                    lid_fric = float(b.boundary_friction or 0.0)
                    lid_pos_t = b_pos_list[i]
                    lid_orn_t = b_orn_list[i]
                    if getattr(b, "has_intake", False) and getattr(b, "intake_pos", None) is not None:
                        terrace_h = b.intake_pos[2]
                    elif getattr(b, "shelf_depth", None) is not None:
                        terrace_h = b.shelf_depth
                    if getattr(b, "intake_radius", 0.0) > 0.0:
                        terrace_r = b.intake_radius
                elif getattr(b, "has_tube", False) and not getattr(b, "has_drain", False):
                    if b.radius > 0.0:
                        terrace_r = b.radius

        if has_lid or (lid_pocket_r > 0.0 and tube_h > 0.0):
            lid_pocket_z = z_floor + tube_h
            intermediate_boundaries.append(
                LidBoundary(
                    r_outer=r_bowl,
                    r_pocket=lid_pocket_r,
                    z_base=lid_pocket_z - lid_thickness,
                    z_floor=lid_pocket_z,
                    z_top=lid_pocket_z + lid_pocket_h,
                    tube_x=tube_x,
                    tube_y=tube_y,
                    tube_r=max(0.0, tube_r - tube_thick),
                    drain_y=drain_y,
                    drain_r=drain_r,
                    terrace_r=terrace_r,
                    terrace_z_max=lid_pocket_z + terrace_h,
                    friction=lid_fric,
                    pos=lid_pos_t,
                    orn=lid_orn_t,
                )
            )

        # Impeller
        if impeller_info is not None:
            imp_idx = boundary_list.index(impeller_info)
            intermediate_boundaries.append(
                ImpellerBoundary(
                    radius=impeller_info.radius,
                    height=impeller_info.height,
                    thickness=impeller_info.thickness,
                    vane_thickness=getattr(impeller_info, "vane_thickness", 0.0) or 0.0,
                    num_vanes=getattr(impeller_info, "num_vanes", 0.0) or 0.0,
                    vane_twist_rad=impeller_info.vane_twist_rad,
                    target_omega=impeller_info.target_omega,
                    max_force=impeller_info.max_force,
                    friction=float(impeller_info.boundary_friction or 0.0),
                    pos=b_pos_list[imp_idx],
                    orn=b_orn_list[imp_idx],
                )
            )

        return ProcessedBoundaries(
            b_shapes=np.array(b_shapes_list, dtype=np.int32),
            b_types=np.array(b_types_list, dtype=np.int32),
            b_params=np.array(b_params_list, dtype=np.float64),
            b_pos_arr=np.array(b_pos_list, dtype=np.float64),
            b_orn_arr=np.array(b_orn_list, dtype=np.float64),
            b_vel_arr=np.array(b_vel_list, dtype=np.float64),
            base_idx=base_boundary_idx,
            boundaries=intermediate_boundaries,
        )
