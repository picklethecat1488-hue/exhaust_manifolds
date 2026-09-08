"""Boundary configuration data models."""

from typing import Any, ClassVar, Literal, Optional, Tuple, Union, Sequence
from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from enum import StrEnum, IntEnum
import math
import numpy as np


class LinkType(IntEnum):
    """Bullet link types."""

    BASE = -1
    OUTLET = 0
    TUBE = 1
    IMPELLER = 2
    FALLEN = -2
    OUTLET_MAX_Y = -3
    LID = 3
    DRIVE_HUB = 4
    PUMP_COVER = 5
    CASING = 6


class ShapeType(StrEnum):
    """Supported boundary shape types."""

    CYLINDER = "cylinder"
    BOX = "box"
    PLANE = "plane"
    IMPELLER = "impeller"
    TUBE = "tube"
    CANOPY = "canopy"
    CASING = "casing"


class ShapeCode(IntEnum):
    """Integer codes for shapes used in JAX tensor solvers."""

    CYLINDER = 0
    CANOPY = 1
    TUBE = 2
    IMPELLER = 3
    BOX = 4
    PLANE = 5
    CASING = 6


class BoundaryType(StrEnum):
    """Boundary collision classification types."""

    CAVITY = "cavity"
    SOLID = "solid"
    OBSTACLE = "obstacle"
    SOLID_CAVITY = "solid_cavity"


class BoundaryCADConformance(BaseModel):
    """Struct representing 3D CAD boolean intersection and geometric conformance for a URDF boundary."""

    shape: Optional[ShapeType] = Field(default=None, description="Analytical shape type of the evaluated boundary")
    type: Optional[BoundaryType] = Field(default=None, description="Barrier type (solid, cavity, or solid_cavity)")
    solid_volume: float = Field(
        default=0.0, description="Total 3D volume of the reconstructed solid barrier geometry in mm³"
    )
    solid_intersection_volume: float = Field(
        default=0.0, description="Volume of overlap between the solid CAD model and the reconstructed barrier in mm³"
    )
    solid_conformance_ratio: float = Field(
        default=0.0, description="Ratio of CAD solid intersection volume to reconstructed solid barrier volume"
    )
    cavity_volume: float = Field(
        default=0.0, description="Total 3D volume of the reconstructed fluid flow cavity in mm³"
    )
    cavity_intersection_volume: float = Field(
        default=0.0, description="Volume of overlap between the solid CAD model and the open cavity volume in mm³"
    )


class Position3D(BaseModel):
    """3D Cartesian vector or coordinate (x, y, z) in meters."""

    x: float = Field(default=0.0, description="X coordinate in meters")
    y: float = Field(default=0.0, description="Y coordinate in meters")
    z: float = Field(default=0.0, description="Z coordinate in meters")

    @model_validator(mode="before")
    @classmethod
    def parse_input(cls, v: Any) -> Any:
        """Parse tuple, list, or string into x, y, z floats."""
        if isinstance(v, Position3D):
            return v
        if isinstance(v, (tuple, list)):
            if len(v) != 3:
                raise ValueError("Position3D must contain exactly 3 components.")
            return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}
        if isinstance(v, str):
            parts = [float(p) for p in v.strip().split()]
            if len(parts) != 3:
                raise ValueError("Position3D string must contain exactly 3 components.")
            return {"x": parts[0], "y": parts[1], "z": parts[2]}
        return v

    def to_tuple(self) -> Tuple[float, float, float]:
        """Convert to (x, y, z) float tuple."""
        return (self.x, self.y, self.z)

    def to_array(self) -> np.ndarray:
        """Convert to numpy array."""
        return np.array([self.x, self.y, self.z], dtype=np.float32)

    def __iter__(self):
        """Allow unpacking: x, y, z = pos."""
        return iter((self.x, self.y, self.z))

    def __getitem__(self, idx: int) -> float:
        """Index access: pos[0], pos[1], pos[2]."""
        return (self.x, self.y, self.z)[idx]

    def __hash__(self) -> int:
        """Calculate hash."""
        return hash((self.x, self.y, self.z))

    def __eq__(self, other: object) -> bool:
        """Compare equality."""
        if isinstance(other, Position3D):
            return (self.x, self.y, self.z) == (other.x, other.y, other.z)
        if isinstance(other, (tuple, list)) and len(other) == 3:
            return (self.x, self.y, self.z) == tuple(other)
        return False


class BoundaryParam(IntEnum):
    """Named indices for boundary parameter columns in b_params tensor."""

    RADIUS = 0
    HEIGHT = 1
    THICKNESS = 2
    Z_OFFSET = 3
    SLOT_HEIGHT = 4
    SLOT_WIDTH = 5
    CEILING_THICKNESS = 6
    VANE_THICKNESS = 7
    NUM_VANES = 8
    VANE_TWIST_RAD = 9
    CUTOFF_Y = 10
    HAS_TUBE = 11
    HAS_DRAIN = 12
    TUBE_RADIUS = 13
    DRAIN_HOLE_Y = 14
    DRAIN_HOLE_RADIUS = 15
    BOUNDARY_FRICTION = 16
    Z_BOTTOM = 17
    Z_TOP = 18
    R_INNER = 19
    R_OUTER = 20
    TRAY_Z_MIN = 21
    TRAY_Z_MAX = 22
    SUCTION_Z_MIN = 23
    SUCTION_Z_MAX = 24
    SPOUT_Z_MIN = 25
    DRAIN_TARGET_Z = 26
    DRAIN_INFLUENCE_RADIUS = 27
    MAX_CEILING_Z = 28
    WALL_BAND_R_MAX = 29
    CASING_TOP_Z = 30
    IMPELLER_RADIUS = 31
    SLOT_CONSTRICTION_RATIO = 32
    LID_SLOPE_RATIO = 33
    DRAIN_EDGE_R_MIN = 34
    DRAIN_EDGE_R_MAX = 35
    SHELF_DEPTH = 36
    HAS_INTAKE = 37
    INTAKE_POS_X = 38
    INTAKE_POS_Y = 39
    INTAKE_POS_Z = 40
    INTAKE_NORMAL_X = 41
    INTAKE_NORMAL_Y = 42
    INTAKE_NORMAL_Z = 43
    INTAKE_RADIUS = 44
    DRAIN_POS_X = 45
    DRAIN_POS_Y = 46
    DRAIN_POS_Z = 47
    DRAIN_NORMAL_X = 48
    DRAIN_NORMAL_Y = 49
    DRAIN_NORMAL_Z = 50
    DRAIN_RADIUS = 51
    TUBE_POS_X = 52
    TUBE_POS_Y = 53
    TUBE_POS_Z = 54
    TUBE_NORMAL_X = 55
    TUBE_NORMAL_Y = 56
    TUBE_NORMAL_Z = 57
    TUBE_PORT_RADIUS = 58
    IS_SUBMERGED = 59
    POOL_MAX_Z = 60

    # Vec3 Block Base Indices
    INTAKE_POS = 38
    INTAKE_NORMAL = 41
    DRAIN_POS = 45
    DRAIN_NORMAL = 48
    TUBE_POS = 52
    TUBE_NORMAL = 55


class SurfaceBounds(BaseModel):
    """Precomputed geometric surface boundaries and interaction extents for a physical boundary."""

    z_bottom: float = Field(description="Bottom surface / floor height in local Z (meters)")
    z_top: float = Field(description="Top surface / ceiling height in local Z (meters)")
    r_inner: float = Field(description="Inside wall radius in meters")
    r_outer: float = Field(description="Outside wall radius in meters")
    tray_z_min: float = Field(default=0.0, description="Active tray interaction lower Z bound (meters)")
    tray_z_max: float = Field(default=0.0, description="Active tray interaction upper Z bound (meters)")
    suction_z_min: float = Field(default=0.0, description="Active suction intake lower Z bound (meters)")
    suction_z_max: float = Field(default=0.0, description="Active suction intake upper Z bound (meters)")
    spout_z_min: float = Field(default=0.0, description="Active spout deflection lower Z bound (meters)")
    drain_target_z: float = Field(default=0.0, description="Target centroid Z for drain funneling (meters)")
    drain_influence_radius: float = Field(
        default=0.030, description="Radial influence threshold for drain funneling (meters)"
    )
    max_ceiling_z: float = Field(default=0.0, description="Maximum fountain containment ceiling Z (meters)")
    wall_band_r_max: float = Field(default=0.0, description="Outer wall containment buffer radius (meters)")
    casing_top_z: float = Field(default=0.0, description="Casing top ceiling height in local Z (meters)")
    impeller_radius: float = Field(default=0.0, description="Impeller outer radius (meters)")
    slot_constriction_ratio: float = Field(default=1.0, description="Impeller slot to tube area ratio")
    lid_slope_ratio: float = Field(default=0.0, description="Ratio of lid height to radius (slope gradient)")
    drain_edge_r_min: float = Field(
        default=0.0, description="Minimum radius for perimeter edge drainage cascade (meters)"
    )
    drain_edge_r_max: float = Field(
        default=0.0, description="Maximum radius for perimeter edge drainage cascade (meters)"
    )
    shelf_depth: float = Field(
        default=0.0, description="Downward solid shelf barrier depth to prevent tunneling (meters)"
    )
    has_intake: float = Field(default=0.0, description="Flag indicating if the boundary has an intake port")
    intake_pos_x: float = Field(default=0.0, description="X coordinate of the intake port in local frame (meters)")
    intake_pos_y: float = Field(default=0.0, description="Y coordinate of the intake port in local frame (meters)")
    intake_pos_z: float = Field(default=0.0, description="Z coordinate of the intake port in local frame (meters)")
    intake_normal_x: float = Field(default=0.0, description="X surface normal of the intake port")
    intake_normal_y: float = Field(default=0.0, description="Y surface normal of the intake port")
    intake_normal_z: float = Field(default=1.0, description="Z surface normal of the intake port")
    intake_radius: float = Field(default=0.0, description="Radius of the intake port (meters)")

    has_drain: float = Field(default=0.0, description="Flag indicating if the boundary has a drain/exhaust port")
    drain_pos_x: float = Field(default=0.0, description="X coordinate of the drain port in local frame (meters)")
    drain_pos_y: float = Field(default=0.0, description="Y coordinate of the drain port in local frame (meters)")
    drain_pos_z: float = Field(default=0.0, description="Z coordinate of the drain port in local frame (meters)")
    drain_normal_x: float = Field(default=0.0, description="X surface normal of the drain port")
    drain_normal_y: float = Field(default=0.0, description="Y surface normal of the drain port")
    drain_normal_z: float = Field(default=1.0, description="Z surface normal of the drain port")
    drain_radius: float = Field(default=0.0, description="Radius of the drain port (meters)")

    has_tube: float = Field(default=0.0, description="Flag indicating if the boundary has a tube port")
    tube_pos_x: float = Field(default=0.0, description="X coordinate of the tube port in local frame (meters)")
    tube_pos_y: float = Field(default=0.0, description="Y coordinate of the tube port in local frame (meters)")
    tube_pos_z: float = Field(default=0.0, description="Z coordinate of the tube port in local frame (meters)")
    tube_normal_x: float = Field(default=0.0, description="X surface normal of the tube port")
    tube_normal_y: float = Field(default=0.0, description="Y surface normal of the tube port")
    tube_normal_z: float = Field(default=1.0, description="Z surface normal of the tube port")
    tube_port_radius: float = Field(default=0.0, description="Radius of the tube port (meters)")

    is_submerged: float = Field(
        default=0.0, description="Flag indicating if boundary is submerged in liquid (1.0) or exposed to air (0.0)"
    )
    pool_max_z: float = Field(
        default=0.0, description="Maximum Z elevation of the reservoir pool cavity under the lid (meters)"
    )


class BoundaryConfig(BaseModel):
    """Pydantic model representing boundary geometry and properties.

    Enforces shape-specific field validation to ensure properties match the ShapeType.
    """

    # Dictionary mapping each ShapeType to the fields it supports/uses
    SHAPE_SUPPORTED_FIELDS: ClassVar[dict[ShapeType, set[str]]] = {
        ShapeType.CYLINDER: {
            "radius",
            "height",
            "thickness",
            "z_offset",
            "has_tube",
            "tube_pos",
            "tube_normal",
            "tube_radius",
            "tube_y",
            "has_drain",
            "drain_pos",
            "drain_normal",
            "drain_radius",
            "drain_hole_y",
            "drain_hole_radius",
            "shelf_depth",
            "has_intake",
            "intake_pos",
            "intake_normal",
            "intake_radius",
            "intake_hole_y",
            "intake_hole_radius",
            "intake_hole_z",
            "pool_max_z",
        },
        ShapeType.CANOPY: {
            "radius",
            "thickness",
        },
        ShapeType.TUBE: {
            "radius",
            "height",
            "thickness",
            "slot_height",
            "slot_width",
            "spout_radius",
            "spout_height",
            "has_intake",
            "intake_pos",
            "intake_normal",
            "intake_radius",
            "intake_hole_y",
            "intake_hole_radius",
            "intake_hole_z",
            "has_drain",
            "drain_pos",
            "drain_normal",
            "drain_radius",
            "drain_hole_y",
            "drain_hole_radius",
            "has_tube",
            "tube_pos",
            "tube_normal",
            "tube_radius",
            "tube_y",
        },
        ShapeType.IMPELLER: {
            "radius",
            "height",
            "thickness",
            "vane_thickness",
            "num_vanes",
            "vane_twist",
            "target_omega",
            "max_force",
            "magnet_radius",
            "magnet_thickness",
            "pump_well_wall",
            "magnet_count",
            "impeller_shaft_radius",
        },
        ShapeType.BOX: {
            "height",
        },
        ShapeType.PLANE: {
            "thickness",
        },
        ShapeType.CASING: {
            "radius",
            "height",
            "thickness",
            "slot_height",
            "slot_width",
            "tube_y",
            "cutoff_y",
            "ceiling_thickness",
            "has_intake",
            "intake_pos",
            "intake_normal",
            "intake_radius",
            "intake_hole_y",
            "intake_hole_radius",
            "intake_hole_z",
            "has_drain",
            "drain_pos",
            "drain_normal",
            "drain_radius",
            "drain_hole_y",
            "drain_hole_radius",
            "has_tube",
            "tube_pos",
            "tube_normal",
            "tube_radius",
        },
    }

    @model_validator(mode="after")
    def validate_shape_fields(self) -> "BoundaryConfig":
        """Validate shape-specific fields and ensure unsupported fields are not configured."""
        if self.shape is None:
            return self

        supported = self.SHAPE_SUPPORTED_FIELDS.get(self.shape, set())
        common = {"shape", "type", "link_type", "link_idx", "xyz", "rpy", "boundary_friction", "is_submerged"}

        for field_name in self.model_fields_set:
            if field_name not in common and field_name not in supported:
                raise ValueError(f"Field '{field_name}' is not supported for shape type '{self.shape.value}'.")

        # Value constraints validation:
        if self.shape is not None and "radius" in self.SHAPE_SUPPORTED_FIELDS.get(self.shape, set()):
            if self.radius <= 0.0:
                raise ValueError(f"{self.shape.name} shape requires a positive radius.")

        return self

    # ----------------------------------------------------
    # Core Identity & Link Metadata (Required, No Defaults)
    # ----------------------------------------------------
    link_type: LinkType = Field(description="Enum type of the link")
    link_idx: int = Field(description="Associated PyBullet multi-body link index")

    # ----------------------------------------------------
    # Boundary Categorization
    # ----------------------------------------------------
    _label: Optional[str] = PrivateAttr(default=None)
    shape: Optional[ShapeType] = Field(default=None, description="Geometry shape of the boundary element")
    type: Optional[BoundaryType] = Field(
        default=None, description="Collision type (cavity container, solid obstacle, or solid cavity)"
    )
    boundary_friction: Optional[float] = Field(default=0.0, ge=0.0, description="Boundary friction coefficient")
    is_submerged: Optional[bool] = Field(
        default=False, description="Flag indicating if the boundary is submerged in liquid"
    )

    # ----------------------------------------------------
    # Spatial Transform Fields
    # ----------------------------------------------------
    xyz: Tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Local translation offset [x, y, z]")
    rpy: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Local orientation roll-pitch-yaw [r, p, y]"
    )
    z_offset: Optional[float] = Field(default=0.0, description="Computed or explicit Z offset")

    # ----------------------------------------------------
    # Common Geometric Parameters
    # ----------------------------------------------------
    radius: float = Field(default=0.0, ge=0.0, description="Radius parameter (applicable for cylinders)")
    height: float = Field(default=0.0, ge=0.0, description="Height parameter (applicable for cylinders or boxes)")
    thickness: float = Field(default=0.0, description="Wall/plate thickness parameter if applicable")
    shelf_depth: Optional[float] = Field(
        default=None, description="Downward solid shelf barrier depth to prevent tunneling (meters)"
    )
    pool_max_z: Optional[float] = Field(
        default=None, description="Maximum reservoir pool depth / liquid fill height in local Z (meters)"
    )

    # ----------------------------------------------------
    # Cylinder / Cavity / Port Specific Parameters (3D Coordinates & Surface Normals)
    # ----------------------------------------------------
    has_drain: Optional[bool] = Field(
        default=False, description="Flag indicating if the boundary has a drain/exhaust hole"
    )
    drain_pos: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="3D coordinate (x, y, z) of the drain/exhaust port in local frame (meters)"
    )
    drain_normal: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0), description="3D surface normal (nx, ny, nz) of the drain/exhaust port"
    )
    drain_radius: Optional[float] = Field(default=0.0, description="Radius of the drain/exhaust hole in meters")

    has_intake: Optional[bool] = Field(default=False, description="Flag indicating if the boundary has an intake port")
    intake_pos: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="3D coordinate (x, y, z) of the intake port in local frame (meters)"
    )
    intake_normal: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0), description="3D surface normal (nx, ny, nz) of the intake port"
    )
    intake_radius: Optional[float] = Field(default=0.0, description="Radius of the intake port in meters")

    has_tube: Optional[bool] = Field(default=False, description="Flag indicating if the boundary has a tube hole")
    tube_pos: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="3D coordinate (x, y, z) of the tube port in local frame (meters)"
    )
    tube_normal: Tuple[float, float, float] = Field(
        default=(0.0, 0.0, 1.0), description="3D surface normal (nx, ny, nz) of the tube port"
    )
    tube_radius: float = Field(default=0.008, description="Tube hole radius")

    @property
    def drain_hole_y(self) -> float:
        """Legacy Y coordinate of the drain hole."""
        return self.drain_pos[1]

    @property
    def drain_hole_radius(self) -> float:
        """Legacy radius of the drain hole."""
        return float(self.drain_radius or 0.0)

    @property
    def intake_hole_y(self) -> float:
        """Legacy Y coordinate of the intake hole."""
        return self.intake_pos[1]

    @property
    def intake_hole_z(self) -> float:
        """Legacy Z coordinate of the intake hole."""
        return self.intake_pos[2]

    @property
    def intake_hole_radius(self) -> float:
        """Legacy radius of the intake hole."""
        return float(self.intake_radius or 0.0)

    @property
    def tube_y(self) -> float:
        """Legacy Y coordinate of the tube."""
        return self.tube_pos[1]

    # ----------------------------------------------------
    # Tube/Casing Specific Parameters
    # ----------------------------------------------------
    slot_height: float = Field(default=0.015, description="Height of pump slots if applicable")
    slot_width: float = Field(default=0.008, description="Width of pump slots if applicable")
    spout_radius: float = Field(default=0.014, description="Spout deflection radius allowance at the top of the tube")
    spout_height: float = Field(default=0.049, description="Spout deflection height allowance at the top of the tube")
    cutoff_y: Optional[float] = Field(default=None, description="Y cutoff coordinate for casing slot connection")
    ceiling_thickness: float = Field(default=0.002, description="Casing ceiling thickness in grid masks")

    # ----------------------------------------------------
    # Impeller Specific Parameters
    # ----------------------------------------------------
    vane_thickness: Optional[float] = Field(default=0.0015, description="Impeller/propeller vane thickness")
    num_vanes: Optional[float] = Field(default=4.0, description="Number of vanes on impeller/propeller")
    vane_twist: float = Field(
        default=-1080.0, description="Total twist angle of the rotary vanes (impeller blades) in degrees."
    )
    target_omega: float = Field(default=15.0, description="Target motor speed/angular velocity")
    max_force: float = Field(default=10.0, description="Maximum motor force/torque limit")

    # ----------------------------------------------------
    # Magnetic Coupling Specific Parameters
    # ----------------------------------------------------
    magnet_radius: Optional[float] = Field(default=None, description="Radius of coupling magnets in mm")
    magnet_thickness: Optional[float] = Field(default=None, description="Thickness of coupling magnets in mm")
    pump_well_wall: Optional[float] = Field(default=None, description="Pump well wall thickness in mm")
    magnet_count: Optional[int] = Field(default=None, description="Number of coupling magnet pairs")
    impeller_shaft_radius: Optional[float] = Field(default=None, description="Radius of the impeller shaft in mm")

    @model_validator(mode="before")
    @classmethod
    def migrate_legacy_ports(cls, data: Any) -> Any:
        """Migrate legacy 1D coordinate scalars into unified 3D coordinates and normals."""
        if isinstance(data, dict):
            # Migrate intake fields
            if "intake_hole_radius" in data and "intake_radius" not in data:
                data["intake_radius"] = data["intake_hole_radius"]
            if "intake_pos" not in data and ("intake_hole_y" in data or "intake_hole_z" in data):
                y = float(data.get("intake_hole_y", 0.0) or 0.0)
                z = float(data.get("intake_hole_z", 0.0) or 0.0)
                data["intake_pos"] = (0.0, y, z)
            # Migrate drain fields
            if "drain_hole_radius" in data and "drain_radius" not in data:
                data["drain_radius"] = data["drain_hole_radius"]
            if "drain_pos" not in data and "drain_hole_y" in data:
                y = float(data.get("drain_hole_y", 0.0) or 0.0)
                data["drain_pos"] = (0.0, y, 0.0)
            # Migrate tube_y
            if "tube_pos" not in data and "tube_y" in data:
                y = float(data.get("tube_y", 0.0) or 0.0)
                data["tube_pos"] = (0.0, y, 0.0)
        return data

    def __hash__(self) -> int:
        """Return a hash value calculated from model properties."""
        return hash(
            (
                self.shape,
                self.type,
                self.link_type,
                self.radius,
                self.height,
                self.xyz,
                self.rpy,
                self.link_idx,
                self.thickness,
                self.z_offset,
                self.slot_height,
                self.cutoff_y,
                self.vane_thickness,
                self.num_vanes,
                self.vane_twist,
                self.has_drain,
                self.drain_pos,
                self.drain_normal,
                self.drain_radius,
                self.has_intake,
                self.intake_pos,
                self.intake_normal,
                self.intake_radius,
                self.has_tube,
                self.tube_pos,
                self.tube_normal,
                self.tube_radius,
                self.target_omega,
                self.max_force,
                self.magnet_radius,
                self.magnet_thickness,
                self.pump_well_wall,
                self.magnet_count,
                self.impeller_shaft_radius,
                self.is_submerged,
            )
        )

    def __eq__(self, other: object) -> bool:
        """Compare equality with another BoundaryConfig based on properties."""
        if not isinstance(other, BoundaryConfig):
            return NotImplemented
        return (
            self.shape == other.shape
            and self.type == other.type
            and self.link_type == other.link_type
            and self.radius == other.radius
            and self.height == other.height
            and self.xyz == other.xyz
            and self.rpy == other.rpy
            and self.link_idx == other.link_idx
            and self.has_intake == other.has_intake
            and self.intake_pos == other.intake_pos
            and self.intake_normal == other.intake_normal
            and self.intake_radius == other.intake_radius
            and self.has_drain == other.has_drain
            and self.drain_pos == other.drain_pos
            and self.drain_normal == other.drain_normal
            and self.drain_radius == other.drain_radius
            and self.has_tube == other.has_tube
            and self.tube_pos == other.tube_pos
            and self.tube_normal == other.tube_normal
            and self.tube_radius == other.tube_radius
            and self.is_submerged == other.is_submerged
            and self.thickness == other.thickness
            and self.z_offset == other.z_offset
            and self.slot_height == other.slot_height
            and self.cutoff_y == other.cutoff_y
            and self.vane_thickness == other.vane_thickness
            and self.num_vanes == other.num_vanes
            and self.vane_twist == other.vane_twist
            and self.target_omega == other.target_omega
            and self.max_force == other.max_force
            and self.magnet_radius == other.magnet_radius
            and self.magnet_thickness == other.magnet_thickness
            and self.pump_well_wall == other.pump_well_wall
            and self.magnet_count == other.magnet_count
            and self.impeller_shaft_radius == other.impeller_shaft_radius
        )

    @field_validator(
        "xyz",
        "rpy",
        "intake_pos",
        "intake_normal",
        "drain_pos",
        "drain_normal",
        "tube_pos",
        "tube_normal",
        mode="before",
    )
    @classmethod
    def parse_string_to_tuple(
        cls, v: Union[str, Tuple[float, float, float], list[float]]
    ) -> Tuple[float, float, float]:
        """Convert space-separated string configurations to a tuple of floats."""
        if isinstance(v, str):
            parts = [float(x) for x in v.strip().split()]
            if len(parts) != 3:
                raise ValueError("xyz/rpy string must contain exactly 3 float values")
            return (parts[0], parts[1], parts[2])
        if isinstance(v, (list, tuple)):
            if len(v) != 3:
                raise ValueError("xyz/rpy must have exactly 3 values")
            return (float(v[0]), float(v[1]), float(v[2]))
        return v

    @property
    def vane_twist_rad(self) -> float:
        """Calculate and return vane_twist in radians."""
        return float(math.radians(self.vane_twist))

    def compute_surface_bounds(
        self,
        max_ceiling_z: float = 0.0,
        casing_top_z: float = 0.0,
        impeller_radius: float = 0.0,
        slot_constriction_ratio: float = 1.0,
        lid_slope_ratio: float = 0.0,
        lid_cavity_depth: float = 0.0,
    ) -> SurfaceBounds:
        """Precompute topological surfaces (tops, bottoms, inner and outer walls) and interaction bounds."""
        z_off = float(self.z_offset or 0.0)
        h = float(self.height)
        r = float(self.radius)
        thick = float(self.thickness)
        ceil_thick = float(self.ceiling_thickness)
        dr_r = float(self.drain_hole_radius or 0.0)

        z_bottom = z_off
        z_top = z_off + h
        r_inner = max(0.0, r - thick)
        r_outer = r
        if self.shelf_depth is not None:
            shelf_depth = float(self.shelf_depth)
        else:
            shelf_depth = thick

        tray_z_min = (
            -max(lid_cavity_depth - shelf_depth, thick)
            if (self.has_drain or self.link_type == LinkType.LID) and lid_cavity_depth > 0.0
            else -thick
        )
        tray_z_max = h
        suction_z_min = h - ceil_thick
        suction_z_max = h + ceil_thick
        spout_z_min = h - thick
        drain_target_z = -thick
        drain_influence_radius = dr_r
        wall_band_r_max = r + thick

        drain_edge_r_min = max(0.0, r - thick * 2.0) if self.has_drain else 0.0
        drain_edge_r_max = (r + thick) if self.has_drain else 0.0

        cad_derived_max_z = max(0.0, (z_off + h) - shelf_depth)
        pool_cap_z = float(self.pool_max_z) if self.pool_max_z is not None else cad_derived_max_z
        pool_max_z = (
            min(cad_derived_max_z, pool_cap_z)
            if (self.type == BoundaryType.CAVITY or self.link_type == LinkType.BASE)
            else 0.0
        )

        return SurfaceBounds(
            z_bottom=z_bottom,
            z_top=z_top,
            r_inner=r_inner,
            r_outer=r_outer,
            tray_z_min=tray_z_min,
            tray_z_max=tray_z_max,
            suction_z_min=suction_z_min,
            suction_z_max=suction_z_max,
            spout_z_min=spout_z_min,
            drain_target_z=drain_target_z,
            drain_influence_radius=drain_influence_radius,
            max_ceiling_z=max_ceiling_z,
            wall_band_r_max=wall_band_r_max,
            casing_top_z=casing_top_z,
            impeller_radius=impeller_radius,
            slot_constriction_ratio=slot_constriction_ratio,
            lid_slope_ratio=lid_slope_ratio,
            drain_edge_r_min=drain_edge_r_min,
            drain_edge_r_max=drain_edge_r_max,
            shelf_depth=shelf_depth,
            has_intake=1.0 if self.has_intake else 0.0,
            intake_pos_x=float(self.intake_pos[0]),
            intake_pos_y=float(self.intake_pos[1]),
            intake_pos_z=float(self.intake_pos[2]),
            intake_normal_x=float(self.intake_normal[0]),
            intake_normal_y=float(self.intake_normal[1]),
            intake_normal_z=float(self.intake_normal[2]),
            intake_radius=float(self.intake_radius or 0.0),
            has_drain=1.0 if self.has_drain else 0.0,
            drain_pos_x=float(self.drain_pos[0]),
            drain_pos_y=float(self.drain_pos[1]),
            drain_pos_z=float(self.drain_pos[2]),
            drain_normal_x=float(self.drain_normal[0]),
            drain_normal_y=float(self.drain_normal[1]),
            drain_normal_z=float(self.drain_normal[2]),
            drain_radius=float(self.drain_radius or 0.0),
            has_tube=1.0 if self.has_tube else 0.0,
            tube_pos_x=float(self.tube_pos[0]),
            tube_pos_y=float(self.tube_pos[1]),
            tube_pos_z=float(self.tube_pos[2]),
            tube_normal_x=float(self.tube_normal[0]),
            tube_normal_y=float(self.tube_normal[1]),
            tube_normal_z=float(self.tube_normal[2]),
            tube_port_radius=float(self.tube_radius or 0.0),
            is_submerged=1.0 if self.is_submerged else 0.0,
            pool_max_z=pool_max_z,
        )


class ResolvedBoundaries(BaseModel):
    """Container for resolved link indices and boundaries linked to physical URDF joints."""

    link_indices: dict[LinkType, int] = Field(default_factory=dict)
    boundaries: dict[str, list[dict[str, Any]]] = Field(default_factory=dict)

    @classmethod
    def from_link_names(
        cls,
        boundaries: dict[str, Any],
        link_names: Sequence[str] = (),
    ) -> "ResolvedBoundaries":
        """Resolve boundaries given a sequence of link/joint names indexed by link ID.

        Args:
            boundaries: Dictionary mapping component labels to URDF boundary metadata.
            link_names: Sequence of joint/link name strings.

        Returns:
            ResolvedBoundaries containing structured link_indices and resolved boundaries.
        """
        link_indices: dict[LinkType, int] = {}
        for i, link_name in enumerate(link_names):
            match link_name:
                case name if "tube" in name:
                    link_indices[LinkType.TUBE] = i
                    link_indices[LinkType.OUTLET] = i
                case name if "impeller" in name:
                    link_indices[LinkType.IMPELLER] = i
                case name if "drive_hub" in name:
                    link_indices[LinkType.DRIVE_HUB] = i
                case name if "lid" in name:
                    link_indices[LinkType.LID] = i
                case name if "pump_cover" in name:
                    link_indices[LinkType.PUMP_COVER] = i
        return cls.resolve(boundaries=boundaries, link_indices=link_indices)

    @classmethod
    def resolve(
        cls,
        boundaries: dict[str, Any],
        link_indices: Optional[dict[LinkType, int]] = None,
    ) -> "ResolvedBoundaries":
        """Resolve raw provider boundary geometries using provided link indices.

        Args:
            boundaries: Dictionary mapping component labels to URDF boundary metadata.
            link_indices: Optional dictionary mapping LinkType enums to joint link IDs.

        Returns:
            ResolvedBoundaries containing structured link_indices and resolved boundaries.
        """
        if link_indices is None:
            link_indices = {}

        resolved_boundaries: dict[str, list[dict[str, Any]]] = {}
        for label, val in boundaries.items():
            vals = val if isinstance(val, list) else [val]
            resolved_vals = []
            for item in vals:
                item_dict = item.model_dump(exclude_defaults=True) if hasattr(item, "model_dump") else dict(item)
                base_label = label.split("/")[-1]
                match base_label:
                    case "bowl":
                        match item_dict.get("link_type"):
                            case LinkType.TUBE | "tube":
                                item_dict["link_type"] = LinkType.TUBE
                                item_dict["link_idx"] = -1
                            case LinkType.CASING | "casing":
                                item_dict["link_type"] = LinkType.CASING
                                item_dict["link_idx"] = -1
                            case LinkType.LID | "lid":
                                item_dict["link_type"] = LinkType.LID
                                item_dict["link_idx"] = -1
                            case _:
                                item_dict["link_type"] = LinkType.BASE
                                item_dict["link_idx"] = -1
                    case "tube":
                        item_dict["link_type"] = LinkType.TUBE
                        item_dict["link_idx"] = link_indices.get(LinkType.TUBE, -1)
                    case "impeller":
                        item_dict["link_type"] = LinkType.IMPELLER
                        item_dict["link_idx"] = link_indices.get(LinkType.IMPELLER, -1)
                    case "lid":
                        item_dict["link_type"] = LinkType.LID
                        item_dict["link_idx"] = link_indices.get(LinkType.LID, -1)
                    case "pump_cover":
                        item_dict["link_type"] = LinkType.PUMP_COVER
                        item_dict["link_idx"] = link_indices.get(LinkType.PUMP_COVER, -1)
                    case _:
                        item_dict["link_type"] = item_dict.get("link_type", LinkType.BASE)
                        item_dict["link_idx"] = link_indices.get(item_dict["link_type"], -1)
                resolved_vals.append(item_dict)
            resolved_boundaries[label] = resolved_vals

        return cls(link_indices=link_indices, boundaries=resolved_boundaries)
