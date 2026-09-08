"""Models package root."""

import warnings
import os
from pathlib import Path

# Load .env file into os.environ before JAX or other packages are imported
env_file = Path(__file__).resolve().parents[2] / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip()
                if k not in os.environ:
                    os.environ[k] = v
                if k.startswith("APP_"):
                    unprefixed = k[4:]
                    if unprefixed not in os.environ:
                        os.environ[unprefixed] = v

from .utils import (
    discover_provider,
    load_manifest,
    get_rgba_color,
    initialize_jax_environment,
    rerun_is_enabled,
    str_to_bool,
    get_env_bool,
)

# Initialize JAX environment configuration deterministically
initialize_jax_environment()

import OCP.TopoDS  # type: ignore

# Monkey-patch TopoDS_Shape to resolve Pydantic validation errors.
# Pydantic 2 probes for a HashCode method when validating OCP-wrapped types.
if not hasattr(OCP.TopoDS.TopoDS_Shape, "HashCode"):
    OCP.TopoDS.TopoDS_Shape.HashCode = lambda self, upper: id(self) % upper  # type: ignore

from .provider import (
    Provider,
    URDFMetadata,
    URDFBoundary,
    IntakePort,
    DrainPort,
    TubePort,
    FlowSlot,
    SpoutDeflection,
    ImpellerVanes,
    MagneticCoupling,
)
from .types import (
    Mode,
    Section,
    MODES,
    ColorType,
    SUBASSEMBLIES,
    MATERIAL,
    EXPORT,
    Simulate,
    URDFShape,
    URDFCollisionType,
    URDFCollisionShapeType,
    URDFBoundaryType,
    URDFJointType,
    URDFMotorType,
    COLOR,
    DAEMON_LOGGERS,
)
from .target_list import TargetList
from .room import Room
from .bullet import Bullet, LinkType
from .boundary import (
    BowlBoundary,
    CasingWallBoundary,
    TubeWallBoundary,
    CasingLidBoundary,
    LidBoundary,
    ImpellerBoundary,
    SphereBoundary,
    PlaneBoundary,
    ProcessedBoundaries,
    BoundaryProcessor,
)
from .transforms import (
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
)
from .cad_boundary import (
    reconstruct_boundary_cad_solid,
    reconstruct_boundary_cad_cavity,
    evaluate_boundary_cad_conformance,
    validate_room_urdf_boundaries,
)
from .fluid import Fluid
from .provider_router import ProviderRouter
from .provider_manager import ProviderManager
from .utils import load_manifest, discover_provider
from .wiring_diagram import WiringDiagram
from .blender import BlenderRenderer, CameraViewport, RenderConfig
from model.wiring import Wiring
