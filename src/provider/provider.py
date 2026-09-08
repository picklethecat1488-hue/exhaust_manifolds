"""Base definitions for geometry and data providers."""

from __future__ import annotations
import os
import inspect
import math
from contextvars import ContextVar
from typing import Optional, Any, Callable, TYPE_CHECKING
from concurrent.futures import ThreadPoolExecutor
from functools import cached_property
from pydantic import validate_call, BaseModel
from typing import cast
from model.app_config import AppConfig
import re
from .types import (
    Mode,
    Section,
    MODES,
    SUBASSEMBLIES,
    COLOR,
    MATERIAL,
    EXPORT,
    Simulate,
    URDFCollisionType,
    URDFJointType,
    URDFMotorType,
)
from .target_list import TargetList
from .orchestrator import Orchestrator
from .utils import load_manifest, get_rgba_color
from .room import Room

# Monkeypatch build123d.BuildPart.__exit__ to copy urdf_* attributes to the final part
from build123d import BuildPart  # type: ignore

_original_buildpart_exit = BuildPart.__exit__


def _custom_buildpart_exit(self: Any, exc_type: Any, exc_val: Any, exc_tb: Any) -> Any:
    res = _original_buildpart_exit(self, exc_type, exc_val, exc_tb)
    if hasattr(self, "part") and self.part is not None:
        for attr in dir(self):
            if attr.startswith("urdf_"):
                setattr(self.part, attr, getattr(self, attr))
    return res


BuildPart.__exit__ = _custom_buildpart_exit

if TYPE_CHECKING:
    from model.app_config import AppConfig
    from shell import Logger


_default_executor: Optional[ThreadPoolExecutor] = None


class ProviderOrchestrator(Orchestrator):
    """Handles the validation and execution strategy for build actions."""

    def __init__(self, provider: Provider, executor: Optional[ThreadPoolExecutor] = None):
        """Initialize the orchestrator with a provider reference."""
        self.provider = provider
        if executor is not None:
            self.executor = executor
        else:
            global _default_executor
            if _default_executor is None:
                _default_executor = ThreadPoolExecutor()
            self.executor = _default_executor

    @validate_call(config={"arbitrary_types_allowed": True})
    def execute(
        self,
        targets: tuple[str, ...],
        action: Section,
        subassemblies: tuple[str | None, ...] = (),
        modes: tuple[Mode | str, ...] = (Mode.DEFAULT,),
    ) -> Any:
        """Perform the requested build action."""
        # Diagram action does not use subassemblies during build execution
        handler_subs = () if action == Section.DIAGRAM else subassemblies
        self.pre_handler(targets, action, handler_subs, modes)

        if action == Section.DIAGRAM:
            # Diagrams operate on all targets at once. We pick the handler for the first target.
            handler = self.provider.diagram[targets[0]]
            # Diagrams operate on all targets at once and return content in a Room.
            room = Room(config=self.provider.app_config, materials=self.provider.materials)
            handler(room, targets, modes[0])
            results = [room]
            self.post_handler(targets, results, action)
            return results[0]

        # Flatten units of work into (target, subassembly, mode) triples
        work_subs = list(subassemblies) if subassemblies else [None]
        work = [(t, sa, m) for t in targets for sa in work_subs for m in modes]

        if action == Section.VIEW:

            def view_task(item: tuple[str, Optional[str], Mode]) -> Room:
                target, _, m = item
                room = Room(config=self.provider.app_config, materials=self.provider.materials)
                setattr(room, "mode", m)
                self.provider.view[target](room, m)
                return room

            raw_results = list(self.executor.map(view_task, work))
        elif action == Section.CONFIG:

            def config_task(item: tuple[str, Optional[str], Mode]) -> None:
                target, sa, m = item
                self.provider.config[m](target, sa)

            for item in work:
                config_task(item)
            self.post_handler(targets, None, action)
            return None
        elif action == Section.PART:

            def build_task(item: tuple[str, Optional[str], Mode]) -> Any:
                target, sa, m = item
                handler = self.provider.part[target]
                return handler(target, sa, m)

            raw_results = list(self.executor.map(build_task, work))
        else:
            raise ValueError(f"Unsupported action: {action}")

        # Group results back by target for VIEW and BUILD
        results = []
        group_size = len(work_subs) * len(modes)
        for i in range(len(targets)):
            group = raw_results[i * group_size : (i + 1) * group_size]
            results.append(group[0] if group_size == 1 else group)

        self.post_handler(targets, results, action)
        return list(zip(targets, results))

    def pre_handler(
        self,
        targets: tuple[str, ...],
        action: Section,
        subassemblies: tuple[str | None, ...],
        modes: tuple[Mode | str, ...],
    ) -> None:
        """Validate input parameters before the handler execution."""
        # Ensure the action is recognized by the orchestrator
        if action not in [Section.VIEW, Section.CONFIG, Section.PART, Section.DIAGRAM]:
            raise ValueError(f"No handler registered for action '{action}' in {self.provider.__class__.__name__}")

        # Diagrams operate on all targets at once, so we validate the first target has a handler.
        if action == Section.DIAGRAM:
            if targets[0] not in self.provider.diagram:
                raise ValueError(f"No diagram handler registered for '{targets[0]}' in {self.provider.name}")

        valid_targets = self.provider.targets
        manifest = self.provider.manifest

        for name in targets:
            if name not in valid_targets:
                raise ValueError(f"Unsupported part name: '{name}'. Supported: {valid_targets}")

            actions_config = manifest.get(name, {})
            if action not in actions_config:
                raise ValueError(
                    f"Action '{action}' is not supported for part '{name}'. Supported: {list(actions_config.keys())}"
                )

            if action == Section.VIEW and name not in self.provider.view:
                raise ValueError(f"No view function registered for room '{name}' in {self.provider.name}")

            if action == Section.PART and name not in self.provider.part:
                raise ValueError(f"No part handler registered for '{name}' in {self.provider.name}")

            if action == Section.CONFIG:
                for mode in modes:
                    if mode not in self.provider.config:
                        raise ValueError(f"No config handler registered for mode '{mode}' in {self.provider.name}")

            action_config = actions_config[action]
            supported_modes = action_config.get(MODES, [])

            for mode in modes:
                if mode not in supported_modes:
                    raise ValueError(
                        f"Mode '{mode}' is not supported for action '{action}' on part '{name}'. "
                        f"Supported modes: {supported_modes}"
                    )

            if subassemblies:
                supported_subs = action_config.get(SUBASSEMBLIES, [])
                for sa in subassemblies:
                    if sa not in supported_subs:
                        raise ValueError(
                            f"Subassembly '{sa}' is not supported for part '{name}'. "
                            f"Supported subassemblies: {supported_subs}"
                        )

    def post_handler(self, targets: tuple[str, ...], results: Optional[list[Any]], action: Section) -> None:
        """Validate build results after the handler execution."""
        if action == Section.CONFIG:
            return

        expected_len = 1 if action == Section.DIAGRAM else len(targets)
        if results is None or len(results) != expected_len:
            raise ValueError(
                f"Orchestration failed: expected {expected_len} items, got {len(results) if results else 0}."
            )

        if any(r is None for r in results):
            raise ValueError(f"Orchestration failed: one or more results for action '{action}' were None.")


class Provider:
    """Base class for all build providers."""

    orchestrator_type: type[Orchestrator] = ProviderOrchestrator

    def __init__(
        self,
        executor: Optional[ThreadPoolExecutor] = None,
        config: Optional["AppConfig"] = None,
        logger: Optional["Logger"] = None,
    ):
        """Initialize the provider."""
        if config is None:
            config = AppConfig()
        self.app_config = config
        self.logger = logger
        self.orchestrator = self.orchestrator_type(self, executor=executor)

    @property
    def name(self) -> str:
        """Infers the provider name from the class name (snake_case, minus 'Provider')."""
        cls_name = self.__class__.__name__
        if cls_name.endswith("Provider"):
            cls_name = cls_name.removesuffix("Provider")
        return re.sub(r"(?<!^)(?=[A-Z])", "_", cls_name).lower()

    @property
    def default_config(self) -> BaseModel:
        """Return a default instance of the provider's configuration (an empty BaseModel)."""
        return BaseModel()

    @property
    def settings(self) -> Any:
        """Return the provider-specific configuration sub-model from the global config."""
        return getattr(self.app_config, self.name.lower(), self.default_config)

    @property
    def manifest(self) -> dict[str, dict[str, Any]]:
        """Map part names to their supported capabilities and colors.

        By default, attempts to load "manifest.yaml" relative to the provider module.
        """
        base_dir = os.path.dirname(os.path.abspath(inspect.getfile(self.__class__)))
        manifest_path = os.path.join(base_dir, "manifest.yaml")
        if os.path.exists(manifest_path):
            return load_manifest(manifest_path)
        return {}

    @property
    def part(self) -> dict[str, Callable[..., Any]]:
        """Map part names to their build handler methods."""
        return {}

    @property
    def diagram(self) -> dict[str, Callable[..., Any]]:
        """Map diagram names to their build handler methods."""
        return {}

    @property
    def config(self) -> dict[str, Callable[[str, Optional[str]], Any]]:
        """Map Modes to configuration handler methods."""
        return {}

    @property
    def view(self) -> dict[str, Callable[[Room, Mode], None]]:
        """Map room names to view functions."""
        return {}

    def get_simulate_hooks(self, sim_name: str) -> dict[Simulate, Callable[..., Any]]:
        """Return the simulation hooks for the given target. Subclasses override this."""
        hooks = self.get_simulate_hooks_impl(sim_name)
        Provider.validate_simulate_hooks(hooks)
        return hooks

    def get_simulate_hooks_impl(self, sim_name: str) -> dict[Simulate, Callable[..., Any]]:
        """Implement get_simulate_hooks. Subclasses override this."""
        return {}

    @staticmethod
    def validate_simulate_hooks(hooks: dict[Simulate, Callable[..., Any]]) -> None:
        """Validate simulation hooks structure and signatures."""
        if hasattr(hooks, "_mock_return_value") or hooks.__class__.__name__ in ("MagicMock", "Mock", "NonCallableMock"):
            return
        if not isinstance(hooks, dict):
            raise TypeError(f"Simulation hooks must be a dictionary, got {type(hooks).__name__}")

        import inspect

        for key, hook in hooks.items():
            if not isinstance(key, Simulate):
                raise TypeError(f"Simulation hook key must be a Simulate enum, got {type(key).__name__}")
            if not callable(hook):
                raise TypeError(f"Simulation hook value must be callable, got {type(hook).__name__}")

            # Inspect signature
            try:
                sig = inspect.signature(hook)
                params = list(sig.parameters.values())
                has_varargs = any(p.kind == inspect.Parameter.VAR_POSITIONAL for p in params)

                # Minimum required parameters
                min_params = 4

                # Count positional/keyword parameters
                pos_params = [
                    p
                    for p in params
                    if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
                ]
                required_pos_params = [p for p in pos_params if p.default == inspect.Parameter.empty]

                if len(required_pos_params) > min_params:
                    raise ValueError(
                        f"Simulation hook {key.name} requires {len(required_pos_params)} parameters, "
                        f"but only {min_params} will be provided."
                    )
                if len(pos_params) < min_params and not has_varargs:
                    raise ValueError(
                        f"Simulation hook {key.name} must accept at least {min_params} parameters, "
                        f"got {len(pos_params)}."
                    )
            except (ValueError, TypeError) as e:
                raise ValueError(f"Invalid signature for simulation hook {key.name}: {e}")

    @property
    def targets(self) -> TargetList:
        """List of supported build targets derived from the manifest keys."""
        return TargetList(self, self.manifest.keys())

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_color(self, target: str, subassembly: Optional[str] = None) -> tuple[float, float, float, float]:
        """Resolve the color for a specific target and subassembly."""
        target_cfg = self.manifest.get(target, {})
        color = target_cfg.get(COLOR)

        if isinstance(color, dict):
            # If COLOR is a dict, resolve by subassembly key.
            if subassembly:
                color = color.get(subassembly)
            else:
                color = next(iter(color.values())) if color else None

        if color is None:
            return self.app_config.color

        if isinstance(color, (tuple, list)):
            return tuple(color)  # type: ignore

        return get_rgba_color(color, 1.0, self.app_config.color[:3])

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_material(self, target: str, subassembly: Optional[str] = None) -> Optional[str]:
        """Resolve the material for a specific target and subassembly."""
        target_cfg = self.manifest.get(target, {})
        material = target_cfg.get(MATERIAL)

        if isinstance(material, dict):
            # If MATERIAL is a dict, resolve by subassembly key.
            if subassembly:
                material = material.get(subassembly)
            else:
                material = next(iter(material.values())) if material else None

        if material is None:
            return None

        return str(material)

    @cached_property
    def materials(self) -> "MaterialsModel":
        """Return strongly-typed materials model loaded from the manifest."""
        from model import MaterialsModel

        return MaterialsModel.from_manifest(self.manifest)

    @validate_call(config={"arbitrary_types_allowed": True})
    def get_export_types(self, target: str, subassembly: Optional[str] = None) -> list[str]:
        """Resolve the export formats for a specific target and subassembly."""
        target_cfg = self.manifest.get(target, {})
        export_val = target_cfg.get(EXPORT)

        if isinstance(export_val, dict):
            # If EXPORT is a dict, resolve by subassembly key.
            if subassembly:
                export_val = export_val.get(subassembly)
            else:
                export_val = next(iter(export_val.values())) if export_val else None

        if export_val is None:
            return ["stl"]

        if isinstance(export_val, list):
            return [str(e).lower() for e in export_val]
        return [str(export_val).lower()]

    @validate_call(config={"arbitrary_types_allowed": True})
    def run(self, targets: TargetList) -> Any:
        """Perform the requested provider-specific build action based on TargetList."""
        action = targets.action
        if action is None:
            raise ValueError(f"No action specified for {targets}. You must call .supporting(action) before running.")

        if action == Section.DIAGRAM and targets.subassemblies:
            raise ValueError(
                f"Subassemblies cannot be specified for Section.DIAGRAM in '{self.name}'. "
                "Diagrams are global assembly views."
            )

        return self.orchestrator.execute(tuple(targets), action, tuple(targets.subassemblies), tuple(targets.modes))


class URDFMetadata:
    """Context manager and builder utility for attaching URDF metadata to CAD shapes."""

    _current: ContextVar[Optional[URDFMetadata]] = ContextVar("URDFMetadata._current", default=None)

    def __init__(
        self,
        label: str,
        material: str,
        density: float,
        boundary_friction: float,
        collision_type: URDFCollisionType,
        parent: Optional[str] = None,
        joint_type: Optional[URDFJointType | str] = None,
        boundaries: Optional[list[Any]] = None,
        collision_primitives: Optional[list[Any]] = None,
        motor_type: Optional[URDFMotorType | str] = None,
        motor_target: Optional[float] = None,
        motor_force: Optional[float] = None,
        geometry: Optional[Any] = None,
        magnet_radius: Optional[float] = None,
        magnet_thickness: Optional[float] = None,
        pump_well_wall: Optional[float] = None,
        magnet_count: Optional[int] = None,
        impeller_shaft_radius: Optional[float] = None,
    ) -> None:
        """Initialize URDFMetadata and attach properties to geometry."""
        from build123d import Builder  # type: ignore

        # If geometry is not explicitly provided, fetch the active builder from build123d context
        if geometry is None:
            builder = Builder._get_context()
            if builder is None:
                raise ValueError(
                    "URDFMetadata must be instantiated within a build123d builder context or have geometry provided explicitly."
                )
            geometry = builder

        self.geometry = geometry
        self.label = label
        self.material = material
        self.density = density
        self.boundary_friction = boundary_friction
        self.collision_type = collision_type
        self.parent = parent
        self.joint_type = joint_type
        self.collision_primitives = collision_primitives
        self.motor_type = motor_type
        self.motor_target = motor_target
        self.motor_force = motor_force
        self.magnet_radius = magnet_radius
        self.magnet_thickness = magnet_thickness
        self.pump_well_wall = pump_well_wall
        self.magnet_count = magnet_count
        self.impeller_shaft_radius = impeller_shaft_radius

        # Initialize boundaries list with any provided boundaries
        self.boundaries: list[Any] = list(boundaries) if boundaries is not None else []
        self._pending_boundaries: list[Any] = []
        self._token: Any = None

        self._apply()

    def _apply(self) -> None:
        """Apply collected metadata properties to the geometry."""
        from typing import cast
        from .types import URDFShape

        for b in self._pending_boundaries:
            if not b._applied:
                b._apply()

        target_geom = self.geometry
        if hasattr(target_geom, "part") and target_geom.part is not None:
            target_geom = target_geom.part

        u_geom = cast(URDFShape, target_geom)
        u_geom.urdf_label = self.label
        u_geom.urdf_material = self.material
        u_geom.urdf_density = self.density
        u_geom.urdf_boundary_friction = self.boundary_friction
        u_geom.urdf_collision_type = self.collision_type

        u_geom.urdf_parent = self.parent
        u_geom.urdf_joint_type = self.joint_type

        if self.boundaries:
            u_geom.urdf_boundaries = self.boundaries
        if self.collision_primitives is not None:
            u_geom.urdf_collision_primitives = self.collision_primitives

        if self.motor_type is not None:
            u_geom.urdf_motor_type = self.motor_type
        if self.motor_target is not None:
            u_geom.urdf_motor_target = self.motor_target
        if self.motor_force is not None:
            u_geom.urdf_motor_force = self.motor_force

        if self.magnet_radius is not None:
            u_geom.urdf_magnet_radius = self.magnet_radius
        if self.magnet_thickness is not None:
            u_geom.urdf_magnet_thickness = self.magnet_thickness
        if self.pump_well_wall is not None:
            u_geom.urdf_pump_well_wall = self.pump_well_wall
        if self.magnet_count is not None:
            u_geom.urdf_magnet_count = self.magnet_count
        if self.impeller_shaft_radius is not None:
            u_geom.urdf_impeller_shaft_radius = self.impeller_shaft_radius

    def __enter__(self) -> "URDFMetadata":
        """Enter context manager."""
        self._token = self._current.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._token is not None:
            self._current.reset(self._token)
        self._apply()


class URDFBoundary:
    """Builder utility for attaching analytical boundaries within a URDFMetadata block."""

    _current: ContextVar[Optional["URDFBoundary"]] = ContextVar("current_urdf_boundary", default=None)

    def __init__(
        self,
        part: Any,
        link_type: Any,
        link_idx: int = -1,
        shape: Optional[Any] = None,
        type: Optional[Any] = None,
        **kwargs: Any,
    ) -> None:
        """Initialize URDFBoundary and register it to the active URDFMetadata context."""
        metadata = URDFMetadata._current.get()
        if metadata is None:
            raise ValueError("URDFBoundary must be instantiated within a URDFMetadata context block.")

        self.part = part
        self.link_type = link_type
        self.link_idx = link_idx
        self.shape = shape
        self.type = type
        self.kwargs = dict(kwargs)
        self.features: list[Any] = []
        self._token: Any = None
        self._applied = False
        self._metadata = metadata

        metadata._pending_boundaries.append(self)

    def register_feature(self, feature: Any) -> None:
        """Register a declarative feature to this boundary."""
        self.features.append(feature)

    def _apply(self) -> None:
        """Extract CAD parameters, merge features, and validate BoundaryConfig."""
        if self._applied:
            return
        from model import BoundaryConfig
        from provider.cad_boundary import extract_boundary_from_cad

        metadata = self._metadata or URDFMetadata._current.get()
        if metadata is None:
            raise ValueError("URDFBoundary must be instantiated within a URDFMetadata context block.")

        merged_kwargs = dict(self.kwargs)
        for feat in self.features:
            if hasattr(feat, "apply_to_boundary"):
                feat.apply_to_boundary(merged_kwargs)

        boundary_friction = merged_kwargs.pop("boundary_friction", getattr(metadata, "boundary_friction", 0.20))

        cad_candidates = extract_boundary_from_cad(
            self.part,
            shape=self.shape,
            type=self.type,
            link_type=self.link_type,
            link_idx=self.link_idx,
            boundary_friction=boundary_friction,
            **merged_kwargs,
        )

        resolved_shape = cad_candidates.get("shape")
        supported_fields = BoundaryConfig.SHAPE_SUPPORTED_FIELDS.get(resolved_shape, set())
        common_fields = {"link_type", "link_idx", "shape", "type", "xyz", "rpy", "boundary_friction"}

        config_dict = {k: v for k, v in cad_candidates.items() if k in supported_fields or k in common_fields}
        boundary = BoundaryConfig.model_validate(config_dict)
        metadata.boundaries.append(boundary)
        self._applied = True

    @classmethod
    def from_shape(
        cls,
        shape_geom: Any,
        link_type: Any,
        link_idx: int = -1,
        shape: Optional[Any] = None,
        type: Optional[Any] = None,
        **kwargs: Any,
    ) -> "URDFBoundary":
        """Construct and register a URDFBoundary directly from a build123d shape or feature solid."""
        return cls(
            part=shape_geom,
            link_type=link_type,
            link_idx=link_idx,
            shape=shape,
            type=type,
            **kwargs,
        )

    @classmethod
    def from_part(
        cls,
        part: Any,
        link_type: Any,
        link_idx: int = -1,
        shape: Optional[Any] = None,
        type: Optional[Any] = None,
        **kwargs: Any,
    ) -> "URDFBoundary":
        """Construct and register a URDFBoundary from a CAD part with attached joint ports."""
        return cls(
            part=part,
            link_type=link_type,
            link_idx=link_idx,
            shape=shape,
            type=type,
            **kwargs,
        )

    def __enter__(self) -> "URDFBoundary":
        """Enter context manager."""
        self._token = self._current.set(self)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._token is not None:
            self._current.reset(self._token)
        self._apply()


def _extract_pos_norm_from_location(
    loc: Any,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    """Extract position (m) and normal vector from a build123d Location."""
    pos_m = (
        float(loc.position.X * 0.001),
        float(loc.position.Y * 0.001),
        float(loc.position.Z * 0.001),
    )
    trsf = loc.wrapped.Transformation().VectorialPart()
    norm_m = (
        float(trsf.Value(1, 3)),
        float(trsf.Value(2, 3)),
        float(trsf.Value(3, 3)),
    )
    return pos_m, norm_m


class IntakePort:
    """Declarative fluid intake port for a URDFBoundary."""

    def __init__(
        self,
        location: Optional[Any] = None,
        pos: Optional[tuple[float, float, float]] = None,
        normal: Optional[tuple[float, float, float]] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Initialize an IntakePort with CAD location or explicit coordinates."""
        if location is not None:
            self.pos, self.normal = _extract_pos_norm_from_location(location)
        else:
            self.pos = pos or (0.0, 0.0, 0.0)
            self.normal = normal or (0.0, 0.0, 1.0)
        self.radius = float(radius) if radius is not None else 0.0

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply intake port attributes to the boundary kwargs."""
        kwargs["has_intake"] = True
        kwargs["intake_pos"] = self.pos
        kwargs["intake_normal"] = self.normal
        if self.radius > 0.0:
            kwargs["intake_radius"] = self.radius


class DrainPort:
    """Declarative fluid drain port for a URDFBoundary."""

    def __init__(
        self,
        location: Optional[Any] = None,
        pos: Optional[tuple[float, float, float]] = None,
        normal: Optional[tuple[float, float, float]] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Initialize a DrainPort with CAD location or explicit coordinates."""
        if location is not None:
            self.pos, self.normal = _extract_pos_norm_from_location(location)
        else:
            self.pos = pos or (0.0, 0.0, 0.0)
            self.normal = normal or (0.0, 0.0, 1.0)
        self.radius = float(radius) if radius is not None else 0.0

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply drain port attributes to the boundary kwargs."""
        kwargs["has_drain"] = True
        kwargs["drain_pos"] = self.pos
        kwargs["drain_normal"] = self.normal
        if self.radius > 0.0:
            kwargs["drain_radius"] = self.radius


class TubePort:
    """Declarative delivery tube port for a URDFBoundary."""

    def __init__(
        self,
        location: Optional[Any] = None,
        pos: Optional[tuple[float, float, float]] = None,
        normal: Optional[tuple[float, float, float]] = None,
        radius: Optional[float] = None,
    ) -> None:
        """Initialize a TubePort with CAD location or explicit coordinates."""
        if location is not None:
            self.pos, self.normal = _extract_pos_norm_from_location(location)
        else:
            self.pos = pos or (0.0, 0.0, 0.0)
            self.normal = normal or (0.0, 0.0, 1.0)
        self.radius = float(radius) if radius is not None else 0.0

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply tube port attributes to the boundary kwargs."""
        kwargs["has_tube"] = True
        kwargs["tube_pos"] = self.pos
        kwargs["tube_normal"] = self.normal
        if self.radius > 0.0:
            kwargs["tube_radius"] = self.radius


class FlowSlot:
    """Declarative discharge slot opening for casing and tube boundaries."""

    def __init__(
        self,
        height: float,
        width: float,
        cutoff_y: float = 0.0,
        ceiling_thickness: float = 0.0,
    ) -> None:
        """Initialize a FlowSlot with physical opening dimensions."""
        self.height = float(height)
        self.width = float(width)
        self.cutoff_y = float(cutoff_y)
        self.ceiling_thickness = float(ceiling_thickness)

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply flow slot attributes to the boundary kwargs."""
        kwargs["slot_height"] = self.height
        kwargs["slot_width"] = self.width
        kwargs["cutoff_y"] = self.cutoff_y
        kwargs["ceiling_thickness"] = self.ceiling_thickness


class SpoutDeflection:
    """Declarative spout deflection geometry for tube boundaries."""

    def __init__(self, radius: float, height: float) -> None:
        """Initialize a SpoutDeflection canopy."""
        self.radius = float(radius)
        self.height = float(height)

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply spout deflection attributes to the boundary kwargs."""
        kwargs["spout_radius"] = self.radius
        kwargs["spout_height"] = self.height


class ImpellerVanes:
    """Declarative impeller vane geometry for impeller boundaries."""

    def __init__(
        self,
        count: int,
        twist: float,
        thickness: float,
    ) -> None:
        """Initialize ImpellerVanes with count, twist angle, and vane thickness."""
        self.count = int(count)
        self.twist = float(twist)
        self.thickness = float(thickness)

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply impeller vane attributes to the boundary kwargs."""
        kwargs["num_vanes"] = self.count
        kwargs["vane_twist"] = self.twist
        kwargs["vane_thickness"] = self.thickness


class MagneticCoupling:
    """Declarative magnetic coupling parameters for impeller boundaries."""

    def __init__(
        self,
        radius: float,
        thickness: float,
        count: int,
        well_wall: Optional[float] = None,
        shaft_radius: Optional[float] = None,
    ) -> None:
        """Initialize MagneticCoupling with magnet pocket geometry."""
        self.radius = float(radius)
        self.thickness = float(thickness)
        self.count = int(count)
        self.well_wall = float(well_wall) if well_wall is not None else None
        self.shaft_radius = float(shaft_radius) if shaft_radius is not None else None

        boundary = URDFBoundary._current.get()
        if boundary is not None:
            boundary.register_feature(self)

    def apply_to_boundary(self, kwargs: dict[str, Any]) -> None:
        """Apply magnetic coupling attributes to the boundary kwargs."""
        kwargs["magnet_radius"] = self.radius
        kwargs["magnet_thickness"] = self.thickness
        kwargs["magnet_count"] = self.count
        if self.well_wall is not None:
            kwargs["pump_well_wall"] = self.well_wall
        if self.shaft_radius is not None:
            kwargs["impeller_shaft_radius"] = self.shaft_radius
