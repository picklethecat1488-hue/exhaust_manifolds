"""Specialized container for visualization items."""

import io
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Union, Optional, TYPE_CHECKING, cast, Callable
from build123d import (
    BoundBox,
    BuildPart,
    BuildSketch,
    BuildLine,
    Compound,
    Edge,
    Vector,
    Plane,
    Text,
    Rectangle,
    Align,
    Location,
    Sketch,
    RigidJoint,
    RevoluteJoint,
    LinearJoint,
    BallJoint,
    LineType,
    Shape,
)
from build123d.exporters import ExportSVG, Drawing
from .utils import get_env_bool


def resolve_shape(val: Any) -> Optional[Shape]:
    """Resolve a value to a build123d Shape if possible."""
    if isinstance(val, Shape):
        return val
    if hasattr(val, "wrapped") and isinstance(val.wrapped, Shape):
        return val
    if hasattr(val, "part") and isinstance(val.part, Shape):
        return val.part
    return None


from ezdxf.colors import RGB
import rerun as rr
import numpy as np
import socket
from model import DiagramOptions, TextArgs
from .types import (
    ColorType,
    URDFShape,
    Simulate,
    CollisionGroup,
    CollisionMask,
    URDFCollisionType,
    URDFCollisionShapeType,
)
from .utils import get_rgba_color

if TYPE_CHECKING:
    from model.app_config import AppConfig


from .bullet import BulletStateTracker, _is_real_physics_client


class Room(dict[str, tuple[Any, tuple[float, float, float, float]]]):
    """
    A dictionary-like container for CAD geometry and its visualization metadata.

    Keys are unique names for the items, and values are tuples of (geometry, rgba_tuple).
    """

    def __init__(self, config: Optional["AppConfig"] = None, is_simulate: bool = False):
        """Initialize the room with an optional application configuration."""
        super().__init__()
        self.config = config
        self.is_simulate = is_simulate
        self._labels: list[tuple[str, str, Any, TextArgs]] = []
        self.gravity: tuple[float, float, float] = (0.0, 0.0, -9.81)
        self.line_weights: dict[str, float] = {}

    def add(
        self,
        name: str,
        geometry: Any,
        color: Optional[Union[str, ColorType, tuple[float, float, float]]] = None,
        alpha: float = 1.0,
        line_weight: Optional[float] = None,
    ) -> None:
        """
        Add a geometry item (Part, Sketch, Wire, or Builder) to the room.

        Args:
            name: Unique name for the item.
            geometry: The CAD object or build123d builder.
            color: The color name, enum member, RGB 3-tuple, or None to use default.
            alpha: Transparency value from 0.0 to 1.0.
            line_weight: Custom thickness scaling factor (e.g. 2.0).
        """
        if name in self:
            raise ValueError(f"An item with the name '{name}' already exists in the Room.")

        # Support BuildPart objects by extracting the underlying geometry
        if isinstance(geometry, BuildPart):
            geometry = geometry.part
        elif isinstance(geometry, BuildSketch):
            geometry = geometry.sketch
        elif isinstance(geometry, BuildLine):
            geometry = geometry.line

        default_rgb = self.config.color[:3] if self.config else (1.0, 1.0, 1.0)

        final_color: Union[str, ColorType, tuple[float, float, float]] = ColorType.GREY
        final_alpha = alpha

        if color is not None:
            final_color = color
        elif self.config:
            final_color = self.config.color[:3]
            if alpha == 1.0:
                final_alpha = self.config.color[3]

        rgba = get_rgba_color(final_color, final_alpha, default_rgb)
        self[name] = (geometry, rgba)
        if line_weight is not None:
            self.line_weights[name] = line_weight

    def add_label(self, name: str, text: str, location: Any, options: Optional[TextArgs] = None) -> None:
        """Add a camera-aligned text annotation at a 3D location."""
        self._labels.append((name, text, Vector(location), options or TextArgs()))

    @property
    def compound(self) -> Compound:
        """Return the room contents as a single build123d Compound."""
        children = []

        for name, (obj, rgba) in sorted(self.items()):
            if hasattr(obj, "label"):
                # Setting the label allows visualization tools like ocp_vscode to identify parts.
                setattr(obj, "label", name)
                # Setting the color metadata for visualization.
                if hasattr(obj, "color"):
                    setattr(obj, "color", rgba)
                children.append(obj)
            else:
                raise ValueError(f"Item '{name}' in Room could not be converted to a build123d object.")

        return Compound(children=children)

    def export_diagram(self, path: Union[str, Path, io.BytesIO], options: Optional["DiagramOptions"] = None) -> None:
        """Export the room contents to a file, inferring format from extension."""
        if isinstance(path, (str, Path)) and not str(path).lower().endswith(".svg"):
            raise ValueError(f"Unsupported diagram format for '{path}'. Only .svg is supported.")

        svg_opts = self._parse_options(options)
        style = svg_opts.get("style", "outline")
        color_mode = style in ("color", "color_hidden")
        show_hidden = style in ("hidden", "color_hidden")

        look_from, look_at, up_vector = self._get_projection_vectors(svg_opts)
        drawing = Drawing(
            self.compound,
            look_from=look_from,
            look_at=look_at,
            look_up=up_vector,
            with_hidden=show_hidden,
        )
        visible_shapes = [drawing.visible_lines]

        # Get the bounding box of the projected geometry to calculate triad placement.
        model_bb = drawing.visible_lines.bounding_box()

        # Add debug axes if requested, placed in the corner and scaled to the viewpane.
        if svg_opts.get("show_axes"):
            axes = self._create_debug_axes(model_bb, look_from, look_at, up_vector)
            if axes:
                visible_shapes.append(axes)

        # Define the view plane for projecting annotations to match the Drawing's 2D space.
        # The view direction is the vector from the target back to the camera.
        view_dir = (look_from - look_at).normalized()
        view_x = up_vector.cross(view_dir).normalized()
        view_plane = Plane(origin=look_at, x_dir=view_x, z_dir=view_dir)

        visible_shapes.extend(self._project_labels(view_plane))

        # Include hidden lines in bounding box calculations if they are shown
        all_shapes = list(visible_shapes)
        if show_hidden:
            all_shapes.append(drawing.hidden_lines)

        final_drawing = Compound(children=all_shapes)
        bb = final_drawing.bounding_box()

        # ExportSVG uses 'scale' to determine physical dimensions from model units.
        width = bb.max.X - bb.min.X
        height = bb.max.Y - bb.min.Y
        if "width" in svg_opts and width > 0:
            svg_opts["scale"] = svg_opts.pop("width") / width
        elif "height" in svg_opts and height > 0:
            svg_opts["scale"] = svg_opts.pop("height") / height

        # Filter for fields explicitly supported by build123d's ExportSVG class.
        supported_fields = {
            "unit",
            "scale",
            "margin",
            "fit_to_stroke",
            "precision",
            "fill_color",
            "line_weight",
            "line_color",
            "line_type",
            "dot_length",
        }
        filtered_opts = {k: v for k, v in svg_opts.items() if k in supported_fields}

        exporter = ExportSVG(**filtered_opts)

        if color_mode:
            shapes_by_layer = {}
            for name, (val, rgba) in self.items():
                geom = resolve_shape(val)
                if geom is None:
                    continue

                color_key = tuple(round(c, 3) for c in rgba[:3])
                line_weight = self.line_weights.get(name, 1.0)

                layer_key = (color_key, line_weight)
                if layer_key not in shapes_by_layer:
                    shapes_by_layer[layer_key] = []
                shapes_by_layer[layer_key].append(geom)

            visible_weight_default = filtered_opts.get("line_weight", 1.0)

            # Add each layer group as a separate layer
            for (color_key, weight_mult), geoms in shapes_by_layer.items():
                geom_comp = Compound(children=geoms)
                drawing = Drawing(
                    geom_comp,
                    look_from=look_from,
                    look_at=look_at,
                    look_up=up_vector,
                    with_hidden=show_hidden,
                )
                r_int, g_int, b_int = (int(c * 255) for c in color_key)
                color_val = RGB(r_int, g_int, b_int)

                # Format layer name to include weight to keep it unique
                weight_str = str(weight_mult).replace(".", "_")
                layer_name = f"Layer_{r_int}_{g_int}_{b_int}_{weight_str}"

                layer_weight = visible_weight_default * weight_mult

                exporter.add_layer(layer_name, line_color=color_val, line_weight=layer_weight)
                exporter.add_shape(drawing.visible_lines, layer=layer_name)

                if show_hidden:
                    hidden_layer_name = f"Hidden_{r_int}_{g_int}_{b_int}_{weight_str}"
                    hidden_color = RGB(
                        int(color_val.r + (255 - color_val.r) * 0.71),
                        int(color_val.g + (255 - color_val.g) * 0.71),
                        int(color_val.b + (255 - color_val.b) * 0.71),
                    )
                    hidden_weight = layer_weight * 0.5
                    exporter.add_layer(
                        hidden_layer_name,
                        line_color=hidden_color,
                        line_weight=hidden_weight,
                        line_type=LineType.ISO_DOT,
                    )
                    exporter.add_shape(drawing.hidden_lines, layer=hidden_layer_name)

            # Project and add labels in a separate Labels layer
            labels = self._project_labels(view_plane)
            if labels:
                label_color = RGB(0, 0, 0)
                label_weight = filtered_opts.get("line_weight", 1.0)
                exporter.add_layer("Labels", line_color=label_color, line_weight=label_weight)
                exporter.add_shape(Compound(children=labels), layer="Labels")
        elif show_hidden:
            visible_color = filtered_opts.get("line_color", (0, 0, 0))
            if isinstance(visible_color, tuple):
                visible_color = RGB(*visible_color)
            visible_weight = filtered_opts.get("line_weight", 1.0)

            # Base hidden color off visible color (e.g. blend with white to make it lighter)
            if visible_color.r == 0 and visible_color.g == 0 and visible_color.b == 0:
                hidden_color = RGB(180, 180, 180)  # Lighter grey
            else:
                hidden_color = RGB(
                    int(visible_color.r + (255 - visible_color.r) * 0.71),
                    int(visible_color.g + (255 - visible_color.g) * 0.71),
                    int(visible_color.b + (255 - visible_color.b) * 0.71),
                )
            hidden_weight = visible_weight * 0.5  # Thinner line weight

            exporter.add_layer("Visible", line_color=visible_color, line_weight=visible_weight)
            exporter.add_layer("Hidden", line_color=hidden_color, line_weight=hidden_weight, line_type=LineType.ISO_DOT)

            exporter.add_shape(Compound(children=visible_shapes), layer="Visible")
            exporter.add_shape(drawing.hidden_lines, layer="Hidden")
        else:
            exporter.add_shape(Compound(children=visible_shapes))

        exporter.write(cast(Any, path))

    def _parse_options(self, options: Optional["DiagramOptions"]) -> dict[str, Any]:
        """Extract and normalize diagram options from Pydantic models or mock objects."""
        if not options:
            return {}

        if hasattr(options, "model_dump"):
            # Use aliases to map stroke_width/color to line_weight/color
            return options.model_dump(by_alias=True, exclude_none=True)

        # Fallback for simple objects or mocks
        fields = [
            "width",
            "height",
            "stroke_width",
            "stroke_color",
            "margin",
            "view_from",
            "projection_dir",
            "projection_origin",
            "show_axes",
            "style",
            "look_up",
        ]
        svg_opts = {}
        for field in fields:
            if hasattr(options, field):
                val = getattr(options, field)
                if val is not None:
                    svg_opts[field] = val

        # Manual mapping for mock objects if they don't use aliases
        if "stroke_width" in svg_opts:
            svg_opts["line_weight"] = svg_opts.pop("stroke_width")
        if "stroke_color" in svg_opts:
            svg_opts["line_color"] = svg_opts.pop("stroke_color")

        return svg_opts

    def _get_projection_vectors(self, svg_opts: dict[str, Any]) -> tuple[Vector, Vector, Vector]:
        # Calculate the camera projection direction and a stable "up" vector.
        look_at = Vector(self.compound.center())
        if svg_opts.get("projection_dir") is not None:
            look_at = Vector(svg_opts["projection_dir"])

        view_dir = Vector(-1, -1, 1)  # Default ISO
        view_from = svg_opts.get("view_from", "iso").lower()

        mapping = {
            "top": Vector(0, 0, 1),
            "bottom": Vector(0, 0, -1),
            "front": Vector(0, -1, 0),
            "rear": Vector(0, 1, 0),
            "left": Vector(-1, 0, 0),
            "right": Vector(1, 0, 0),
            "iso": Vector(-1, -1, 1),
        }

        if view_from != "iso":
            parts = view_from.replace(",", " ").split()
            combined = Vector(0, 0, 0)
            for p in parts:
                if p in mapping:
                    combined += mapping[p]
            if combined.length > 0:
                view_dir = combined

        # Scale the view direction based on the bounding box to ensure the camera
        # is outside the model. Using 2x the diagonal provides a safe distance.
        bb = self.compound.bounding_box()
        distance = bb.diagonal * 2 if bb.diagonal > 0 else 1000.0
        look_from = look_at + (view_dir.normalized() * distance)
        if svg_opts.get("projection_origin") is not None:
            look_from = Vector(svg_opts["projection_origin"])

        view_dir_norm = (look_from - look_at).normalized()

        z_axis = Vector(0, 0, 1)
        up_vector = Vector(svg_opts.get("look_up", z_axis))
        if abs(view_dir_norm.dot(up_vector)) > 0.99 and "look_up" not in svg_opts:
            up_vector = Vector(0, 1, 0)
        return look_from, look_at, up_vector

    def _project_labels(self, view_plane: Plane) -> list[Sketch]:
        """Project 3D annotations into 2D sketches aligned with the view plane."""
        projected_sketches = []
        for _, text, loc, opts in self._labels:
            local_loc = cast(Vector, view_plane.to_local_coords(loc))
            l_loc = cast(Vector, local_loc)
            with BuildSketch() as s:
                Text(
                    text,
                    font_size=opts.font_size,
                    font=opts.font,
                    font_style=opts.font_style,
                    align=opts.align,
                )

            # Place the 2D text at the projected X/Y coordinates.
            projected_sketches.append(s.sketch.moved(Location((l_loc.X, l_loc.Y))))
        return projected_sketches

    def _create_debug_axes(
        self,
        model_bb: BoundBox,
        look_from: Vector,
        look_at: Vector,
        up_vector: Vector,
    ) -> Optional[Any]:
        """Create a projected coordinate system triad scaled and positioned for the viewpane."""
        # Create our own axes and arrows to ensure correct orientation and avoid issues with make_triad.
        # The base_axis_length will be scaled later by axes_scale.
        base_axis_length = 1.0
        arrow_length_factor = 0.15
        arrow_width_factor = 0.07

        axes = [Vector(1, 0, 0), Vector(0, 1, 0), Vector(0, 0, 1)]
        custom_triad_parts = []
        for direction in axes:
            tip = direction * base_axis_length
            custom_triad_parts.append(Edge.make_line(Vector(0, 0, 0), tip))
            custom_triad_parts.extend(self._create_arrowhead(tip, direction, arrow_length_factor, arrow_width_factor))

        axes_triad = Compound(children=custom_triad_parts).moved(Location(look_at))
        axes_drawing = Drawing(axes_triad, look_from=look_from, look_at=look_at, look_up=up_vector)
        projected_axes = axes_drawing.visible_lines

        # Calculate projection for labels to ensure they face the camera.
        view_dir = (look_from - look_at).normalized()
        view_x = up_vector.cross(view_dir).normalized()
        view_plane = Plane(origin=look_at, x_dir=view_x, z_dir=view_dir)

        def project_to_2d(p3d: Vector) -> Vector:
            # Project world vectors (relative to triad origin) to the view plane.
            local = cast(Vector, view_plane.to_local_coords(look_at + p3d))
            return Vector(local.X, local.Y)

        # Create 2D text labels at the projected endpoints of the 3D unit vectors.
        # This ensures they are always flat to the viewpane (facing the camera).
        labels = []
        label_margin_factor = 0.1  # Small margin beyond the arrow tip
        label_offset_for_direct_view = 0.5  # Offset factor relative to font_size
        label_font_size = 0.2  # From Text constructor

        for text, vec in [
            ("X", Vector(base_axis_length + label_margin_factor, 0, 0)),
            ("Y", Vector(0, base_axis_length + label_margin_factor, 0)),
            ("Z", Vector(0, 0, base_axis_length + label_margin_factor)),
        ]:
            p2d = cast(Vector, project_to_2d(vec))
            # If the axis is pointing almost directly at the camera, apply a small 2D offset
            if view_dir.dot(vec.normalized()) > 0.95:  # Check 3D vector alignment with view direction
                p2d += Vector(
                    label_font_size * label_offset_for_direct_view, label_font_size * label_offset_for_direct_view, 0
                )

            with BuildSketch() as s:
                Text(text, font_size=label_font_size, align=(Align.CENTER, Align.CENTER))
            labels.append(s.sketch.moved(Location((p2d.X, p2d.Y))))

        # Combine the projected lines with the camera-facing labels.
        projected_axes = Compound(children=[projected_axes] + labels)

        # Scale to ~15% of the model's larger dimension.
        m_width = model_bb.max.X - model_bb.min.X
        m_height = model_bb.max.Y - model_bb.min.Y
        model_dim = max(m_width, m_height)

        axes_bb = projected_axes.bounding_box()
        a_width = axes_bb.max.X - axes_bb.min.X
        a_height = axes_bb.max.Y - axes_bb.min.Y
        axes_dim = max(a_width, a_height)

        # Size of the debug axes as a ratio of model size
        projection_scale = 0.15

        if axes_dim > 0:
            axes_scale = (model_dim * projection_scale) / axes_dim
            projected_axes = projected_axes.scale(axes_scale)

            # Move to bottom-left corner.
            new_axes_bb = projected_axes.bounding_box()
            offset = Vector(
                model_bb.min.X - new_axes_bb.min.X,
                model_bb.min.Y - new_axes_bb.min.Y,
            )
            return projected_axes.translate(offset)
        return None

    def _create_arrowhead(self, tip: Vector, direction: Vector, length: float, width: float) -> list[Edge]:
        """Create a V-shaped arrowhead pointing along direction at tip."""
        # Find a vector perpendicular to direction to define the 'width' of the V.
        # We try (0,0,1) first, unless direction is also along Z.
        z_axis = Vector(0, 0, 1)
        cross_p = z_axis.cross(direction)
        if cross_p.length < 1e-5:
            cross_p = Vector(0, 1, 0).cross(direction)
        perp = cross_p.normalized()

        p1 = tip - (direction * length) + (perp * width)
        p2 = tip - (direction * length) - (perp * width)
        return [Edge.make_line(tip, p1), Edge.make_line(tip, p2)]

    def disconnect_joints(self) -> None:
        """Clear all URDF joint properties from shapes in the room as a precautionary step."""
        for geom, _ in self.values():
            u_geom = cast(URDFShape, geom)
            if hasattr(u_geom, "urdf_parent"):
                u_geom.urdf_parent = None
            if hasattr(u_geom, "urdf_joint_type"):
                u_geom.urdf_joint_type = None
            if hasattr(u_geom, "urdf_joint_axis"):
                u_geom.urdf_joint_axis = None
            if hasattr(u_geom, "urdf_joint_lower"):
                u_geom.urdf_joint_lower = None
            if hasattr(u_geom, "urdf_joint_upper"):
                u_geom.urdf_joint_upper = None

    def translate_joints(self) -> None:
        """Map build123d joints defined on shapes in the room to URDF-compatible properties."""
        self.disconnect_joints()

        def format_coord(v: float) -> str:
            return str(int(v)) if v.is_integer() else f"{v:.6f}".rstrip("0").rstrip(".")

        # Map from python object id of geometry to its label/name in the room
        geom_to_label = {}
        for name, (geom, _) in self.items():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", name)
            geom_to_label[id(u_geom)] = label

        # Also populate urdf_label for all room geometries if not already set, to make sure
        # they are recognized as URDF links.
        for name, (geom, _) in self.items():
            u_geom = cast(URDFShape, geom)
            if not getattr(u_geom, "urdf_label", None):
                u_geom.urdf_label = name

        for name, (geom, _) in self.items():
            u_geom = cast(URDFShape, geom)
            joints = getattr(u_geom, "joints", None)
            if not isinstance(joints, dict):
                continue

            for joint in joints.values():
                if getattr(joint, "connected_to", None) is not None:
                    parent_geom = u_geom
                    child_geom = cast(URDFShape, joint.connected_to.parent)

                    parent_label = geom_to_label.get(id(parent_geom))
                    child_label = geom_to_label.get(id(child_geom))

                    if parent_label is not None and child_label is not None:
                        child_geom.urdf_parent = parent_label
                        child_joint = joint.connected_to

                        # Map joint types and attributes
                        self._map_joint_properties(child_geom, child_joint, format_coord)

    def _map_joint_properties(self, child_geom: URDFShape, child_joint: Any, format_coord: Any) -> None:
        """Map build123d joint attributes or custom overrides to child geometry properties."""
        if hasattr(child_joint, "urdf_joint_type"):
            u_joint = cast(URDFShape, child_joint)
            child_geom.urdf_joint_type = u_joint.urdf_joint_type
            if hasattr(u_joint, "urdf_joint_axis"):
                child_geom.urdf_joint_axis = u_joint.urdf_joint_axis
            if hasattr(u_joint, "urdf_joint_lower"):
                child_geom.urdf_joint_lower = u_joint.urdf_joint_lower
            if hasattr(u_joint, "urdf_joint_upper"):
                child_geom.urdf_joint_upper = u_joint.urdf_joint_upper
        else:
            match child_joint:
                case RigidJoint():
                    child_geom.urdf_joint_type = "fixed"
                case RevoluteJoint():
                    is_continuous = False
                    if getattr(child_joint, "angular_range", None) is not None:
                        lower, upper = child_joint.angular_range
                        if abs(upper - lower) >= 360.0:
                            is_continuous = True

                    if is_continuous:
                        child_geom.urdf_joint_type = "continuous"
                    else:
                        child_geom.urdf_joint_type = "revolute"
                        if getattr(child_joint, "angular_range", None) is not None:
                            child_geom.urdf_joint_lower = math.radians(child_joint.angular_range[0])
                            child_geom.urdf_joint_upper = math.radians(child_joint.angular_range[1])

                    if getattr(child_joint, "relative_axis", None) is not None:
                        dir_vec = child_joint.relative_axis.direction
                        child_geom.urdf_joint_axis = (
                            f"{format_coord(dir_vec.X)} {format_coord(dir_vec.Y)} {format_coord(dir_vec.Z)}"
                        )
                case LinearJoint():
                    child_geom.urdf_joint_type = "prismatic"
                    if getattr(child_joint, "relative_axis", None) is not None:
                        dir_vec = child_joint.relative_axis.direction
                        child_geom.urdf_joint_axis = (
                            f"{format_coord(dir_vec.X)} {format_coord(dir_vec.Y)} {format_coord(dir_vec.Z)}"
                        )
                    if getattr(child_joint, "linear_range", None) is not None:
                        child_geom.urdf_joint_lower = child_joint.linear_range[0] * 0.001
                        child_geom.urdf_joint_upper = child_joint.linear_range[1] * 0.001
                case BallJoint():
                    child_geom.urdf_joint_type = "spherical"

    def export_urdf(self, path: Union[str, Path, io.StringIO], project_name: str) -> None:
        """Export a combined URDF from a Room object."""
        import yaml
        from build123d import CenterOf

        self.translate_joints()

        # Build map of labels to geometries to resolve relative joint coordinates
        label_to_geom = {}
        for geom, _ in self.values():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", None)
            if label:
                label_to_geom[label] = u_geom

        template_path = Path(__file__).parent.parent / "urdf_template.yaml"
        with open(template_path, "r") as f:
            templates = yaml.safe_load(f)
        robot_template = templates["robot_template"].strip()
        link_template = templates["link_template"].strip()
        joint_template = templates["joint_template"].strip()
        axis_limit_template = templates["axis_limit_template"].strip()
        collision_template = templates["collision_template"].strip()
        primitive_collision_template = templates["primitive_collision_template"].strip()
        analytical_boundary_template = templates["analytical_boundary_template"].strip()

        links_info = []
        for geom, rgba in self.values():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", None)
            if not label:
                continue

            parent = getattr(u_geom, "urdf_parent", None)
            joint_type = getattr(u_geom, "urdf_joint_type", "fixed")
            joint_axis = getattr(u_geom, "urdf_joint_axis", "0 0 1")
            density = getattr(u_geom, "urdf_density", 1.0)

            # --- COLLISION RESOLUTION COMPARISON ---
            # | Approach | Step Time (s) | Escapes | Explosions | Stability |
            # | --- | --- | --- | --- | --- |
            # | 1. Concave Mesh (PyBullet) | 20.0 - 40.0 | 0 | 0 | High |
            # | 2. Primitive Compound (PyBullet) | 5.0 - 15.0 | Rare | Frequent | Low-Medium |
            # | 3. Analytical SPH (JAX) | 0.001 - 0.003 | 0 | 0 | Very High |
            # ---------------------------------------
            collision_type = getattr(u_geom, "urdf_collision_type", "convex")

            # Compute relative location if the link has a parent
            parent_label = getattr(u_geom, "urdf_parent", None)
            if parent_label and parent_label in label_to_geom:
                parent_geom = label_to_geom[parent_label]
                rel_loc = parent_geom.location.inverse() * geom.location
            else:
                rel_loc = geom.location

            pos = rel_loc.position
            xyz = [pos.X * 0.001, pos.Y * 0.001, pos.Z * 0.001]

            # Convert build123d rotation matrix (extrinsic ZYX) to URDF extrinsic XYZ (roll, pitch, yaw)
            trsf = rel_loc.wrapped.Transformation().VectorialPart()
            m = [
                [trsf.Value(1, 1), trsf.Value(1, 2), trsf.Value(1, 3)],
                [trsf.Value(2, 1), trsf.Value(2, 2), trsf.Value(2, 3)],
                [trsf.Value(3, 1), trsf.Value(3, 2), trsf.Value(3, 3)],
            ]
            sin_theta = -m[2][0]
            sin_theta = max(-1.0, min(1.0, sin_theta))
            theta = math.asin(sin_theta)
            if abs(math.cos(theta)) > 1e-6:
                phi = math.atan2(m[2][1], m[2][2])
                psi = math.atan2(m[1][0], m[0][0])
            else:
                phi = math.atan2(-m[1][2], m[1][1])
                psi = 0.0
            rpy = [phi, theta, psi]

            # Invert the translation to get the local shape centered at local origin

            local_shape = geom.location.inverse() * u_geom

            # Mass and inertia calculations using local shape and density
            density_g_mm3 = density * 1e-3
            volume_mm3 = local_shape.volume
            mass_kg = volume_mm3 * density_g_mm3 * 1e-3

            com = local_shape.center(CenterOf.MASS)
            com_m = [com.X * 0.001, com.Y * 0.001, com.Z * 0.001]

            raw_inertia = local_shape.matrix_of_inertia
            scale_factor = density * 1e-12

            com_m = [com.X * 0.001, com.Y * 0.001, com.Z * 0.001]

            raw_inertia = local_shape.matrix_of_inertia
            scale_factor = density * 1e-12

            ixx = raw_inertia[0][0] * scale_factor
            ixy = raw_inertia[0][1] * scale_factor
            ixz = raw_inertia[0][2] * scale_factor
            iyy = raw_inertia[1][1] * scale_factor
            iyz = raw_inertia[1][2] * scale_factor
            izz = raw_inertia[2][2] * scale_factor

            rgba_str = f"{rgba[0]:.6f} {rgba[1]:.6f} {rgba[2]:.6f} {rgba[3]:.6f}"
            joint_lower = getattr(u_geom, "urdf_joint_lower", -3.14159)
            joint_upper = getattr(u_geom, "urdf_joint_upper", 3.14159)

            links_info.append(
                {
                    "name": label,
                    "parent": parent,
                    "joint_type": joint_type,
                    "joint_axis": joint_axis,
                    "joint_lower": joint_lower,
                    "joint_upper": joint_upper,
                    "xyz": xyz,
                    "rpy": rpy,
                    "mass_kg": mass_kg,
                    "com_x": com_m[0],
                    "com_y": com_m[1],
                    "com_z": com_m[2],
                    "ixx": ixx,
                    "ixy": ixy,
                    "ixz": ixz,
                    "iyy": iyy,
                    "iyz": iyz,
                    "izz": izz,
                    "obj_filename": getattr(u_geom, "urdf_obj_filename", None) or f"{label}.obj",
                    "rgba_str": rgba_str,
                    "collision_type": collision_type,
                    "collision_primitives": getattr(u_geom, "urdf_collision_primitives", None),
                    "boundaries": getattr(u_geom, "urdf_boundaries", None),
                }
            )

        if not links_info:
            return

        # Validate that the URDF has exactly one root link
        roots = [link["name"] for link in links_info if link["parent"] is None]
        if len(roots) > 1:
            raise ValueError(f"URDF file with multiple root links found: {' '.join(roots)}")
        elif len(roots) == 0:
            raise ValueError("URDF file with no root links found (contains cycles).")

        links_strings = []
        for link in links_info:
            c_type = link["collision_type"]
            c_prims = link.get("collision_primitives")
            if c_type == URDFCollisionType.NONE:
                collision_section = ""
            elif c_type == URDFCollisionType.ANALYTICAL:
                boundaries_list = link.get("boundaries")
                if boundaries_list:
                    collision_strings = []
                    for b in boundaries_list:
                        b_shape = b.shape if hasattr(b, "shape") else b.get("shape", "cylinder")
                        b_type = b.type if hasattr(b, "type") else b.get("type", "cavity")
                        b_radius = b.radius if hasattr(b, "radius") else b.get("radius", 0.0)
                        b_height = b.height if hasattr(b, "height") else b.get("height", 0.0)

                        b_xyz = b.xyz if hasattr(b, "xyz") else b.get("xyz", (0.0, 0.0, 0.0))
                        b_rpy = b.rpy if hasattr(b, "rpy") else b.get("rpy", (0.0, 0.0, 0.0))

                        xyz_str = f"{b_xyz[0]:.6f} {b_xyz[1]:.6f} {b_xyz[2]:.6f}"
                        rpy_str = f"{b_rpy[0]:.6f} {b_rpy[1]:.6f} {b_rpy[2]:.6f}"

                        collision_strings.append(
                            analytical_boundary_template.format(
                                boundary_shape=b_shape,
                                boundary_type=b_type,
                                boundary_radius=b_radius,
                                boundary_height=b_height,
                                boundary_xyz=xyz_str,
                                boundary_rpy=rpy_str,
                            )
                        )
                    collision_section = "\n".join(collision_strings) + "\n"
                else:
                    collision_section = ""
            elif c_prims is not None:
                collision_strings = []
                for prim in c_prims:
                    p_type = prim["type"]
                    p_xyz = prim.get("xyz", [0.0, 0.0, 0.0])
                    p_rpy = prim.get("rpy", [0.0, 0.0, 0.0])
                    xyz_str = f"{p_xyz[0]:.6f} {p_xyz[1]:.6f} {p_xyz[2]:.6f}"
                    rpy_str = f"{p_rpy[0]:.6f} {p_rpy[1]:.6f} {p_rpy[2]:.6f}"

                    match p_type:
                        case URDFCollisionShapeType.BOX:
                            sz = prim["size"]
                            geom_str = f'<box size="{sz[0]:.6f} {sz[1]:.6f} {sz[2]:.6f}"/>'
                        case URDFCollisionShapeType.CYLINDER:
                            geom_str = f'<cylinder radius="{prim["radius"]:.6f}" length="{prim["length"]:.6f}"/>'
                        case URDFCollisionShapeType.SPHERE:
                            geom_str = f'<sphere radius="{prim["radius"]:.6f}"/>'
                        case _:
                            raise ValueError(f"Unsupported collision primitive type: {p_type}")

                    collision_strings.append(
                        primitive_collision_template.format(
                            collision_type=c_type,
                            xyz=xyz_str,
                            rpy=rpy_str,
                            geometry=geom_str,
                        )
                    )
                collision_section = "\n".join(collision_strings) + "\n"
            else:
                collision_section = (
                    collision_template.format(
                        project_name=project_name,
                        obj_filename=link["obj_filename"],
                        collision_type=c_type,
                    )
                    + "\n"
                )

            links_strings.append(
                link_template.format(
                    link_name=link["name"],
                    com_x=link["com_x"],
                    com_y=link["com_y"],
                    com_z=link["com_z"],
                    mass_kg=link["mass_kg"],
                    ixx=link["ixx"],
                    ixy=link["ixy"],
                    ixz=link["ixz"],
                    iyy=link["iyy"],
                    iyz=link["iyz"],
                    izz=link["izz"],
                    project_name=project_name,
                    obj_filename=link["obj_filename"],
                    rgba=link["rgba_str"],
                    collision_section=collision_section,
                )
            )

        joints_strings = []
        for link in links_info:
            if link["parent"] is not None:
                axis_limit_str = ""
                if link["joint_type"] in ("revolute", "prismatic"):
                    axis_limit_str = (
                        axis_limit_template.format(
                            joint_axis=link["joint_axis"], lower=link["joint_lower"], upper=link["joint_upper"]
                        )
                        + "\n  "
                    )
                elif link["joint_type"] in ("continuous", "planar"):
                    axis_limit_str = f'  <axis xyz="{link["joint_axis"]}"/>\n  '

                joints_strings.append(
                    joint_template.format(
                        parent_name=link["parent"],
                        child_name=link["name"],
                        joint_type=link["joint_type"],
                        xyz_x=link["xyz"][0],
                        xyz_y=link["xyz"][1],
                        xyz_z=link["xyz"][2],
                        rpy_r=link["rpy"][0],
                        rpy_p=link["rpy"][1],
                        rpy_y=link["rpy"][2],
                        axis_limit=axis_limit_str,
                    )
                )

        urdf_content = robot_template.format(
            robot_name=project_name,
            links="\n".join(links_strings),
            joints="\n".join(joints_strings),
        )

        if isinstance(path, (str, Path)):
            with open(path, "w") as f:
                f.write(urdf_content)
        else:
            path.write(urdf_content)

    def _log_rerun(
        self,
        transforms: dict[str, tuple[list[float], list[float]]],
        particle_positions: list[list[float]],
        particle_colors: Optional[list[list[float]]] = None,
        particle_radii: Optional[list[float]] = None,
        boundary_voxels: Optional[dict[str, Any]] = None,
        water_meshes: Optional[dict[str, Any]] = None,
        step_idx: Optional[int] = None,
    ) -> None:
        """Log the given state data to Rerun."""
        if step_idx is not None:
            rr.set_time("step", sequence=step_idx)

        # Log link transforms
        for label, (pos, orn) in transforms.items():
            rr.log(
                f"world/{label}",
                rr.Transform3D(
                    translation=pos,
                    rotation=rr.Quaternion(xyzw=orn),
                    scale=0.001,
                ),
            )

        # 1. Log transparent CAD solid water meshes
        if water_meshes is not None and len(water_meshes) > 0:
            for name, (verts, faces) in water_meshes.items():
                if len(verts) > 0 and len(faces) > 0:
                    rr.log(
                        f"world/water/{name}",
                        rr.Mesh3D(
                            vertex_positions=verts,
                            triangle_indices=faces,
                            albedo_factor=[50, 160, 240, 110],  # Transparent aquatic blue (alpha ~ 0.43)
                        ),
                    )

        # 2. Log particles using vectorized numpy operations (gated on SHOW_WATER_VOXELS environment variable)
        enable_water_voxels = get_env_bool("SHOW_WATER_VOXELS", False)
        if enable_water_voxels and particle_positions is not None and len(particle_positions) > 0:
            pos_arr = np.asarray(particle_positions)
            active_mask = pos_arr[:, 2] < 100.0
            if np.any(active_mask):
                filtered_positions = pos_arr[active_mask]

                if particle_radii is not None and len(particle_radii) > 0:
                    filtered_radii = np.asarray(particle_radii)[active_mask]
                else:
                    filtered_radii = None

                colors_arg = [128, 204, 255, 178]
                if particle_colors is not None and len(particle_colors) > 0:
                    colors_arr = np.asarray(particle_colors)[active_mask]
                    if colors_arr.size > 0:
                        if np.issubdtype(colors_arr.dtype, np.floating) or np.max(colors_arr) <= 1.0:
                            colors_arg = (colors_arr * 255.0).round().astype(np.uint8)
                        else:
                            colors_arg = colors_arr.astype(np.uint8)

                rr.log(
                    "world/particles",
                    rr.Points3D(
                        positions=filtered_positions,
                        radii=filtered_radii,
                        colors=colors_arg,
                    ),
                )

        # Log boundary voxels labeled for each boundary type (gated on environment variable)
        enable_boundary_voxels = get_env_bool("SHOW_BOUNDARY_VOXELS", False)
        if enable_boundary_voxels and boundary_voxels is not None and len(boundary_voxels) > 0:
            boundary_color_map = {
                "bowl": [180, 180, 190, 80],
                "casingwall": [255, 160, 50, 100],
                "casing_wall": [255, 160, 50, 100],
                "casing": [255, 160, 50, 100],
                "tubewall": [50, 200, 100, 100],
                "tube_wall": [50, 200, 100, 100],
                "tube": [50, 200, 100, 100],
                "casinglid": [160, 100, 220, 100],
                "casing_lid": [160, 100, 220, 100],
                "lid": [140, 90, 200, 100],
                "impeller": [220, 50, 50, 140],
                "sphere": [100, 180, 220, 100],
                "plane": [150, 150, 150, 80],
            }
            default_color = [160, 160, 160, 90]

            for label, vox_positions in boundary_voxels.items():
                if vox_positions is not None and len(vox_positions) > 0:
                    pos_arr = np.asarray(vox_positions, dtype=np.float32)
                    color = boundary_color_map.get(label.lower(), default_color)
                    rr.log(
                        f"world/boundaries/{label}",
                        rr.Points3D(
                            positions=pos_arr,
                            radii=0.0018,
                            colors=color,
                        ),
                    )

    def simulate(
        self,
        provider_hooks: dict[Simulate, Callable[..., Any]],
        proj_name: str,
        sim_target: str,
        steps: int,
        manager: Any,
        logger: Any,
        build_dir: str = "build",
        save_rrd: Optional[str] = None,
        rerun_port: Optional[int] = None,
        spawn_viewer: bool = True,
        stage_window_size: Optional[int] = None,
    ) -> None:
        """
        Run a PyBullet physics simulation for the room geometries.
        """
        from .provider import Provider
        from .bullet import Bullet

        logger.print("Running Simulation (Ctrl-C to exit)...", symbol="🤖")

        if not self:
            raise ValueError("Cannot simulate an empty Room.")

        Provider.validate_simulate_hooks(provider_hooks)

        bullet_sim = Bullet(
            self,
            provider_hooks,
            proj_name,
            sim_target,
            steps,
            manager,
            logger,
            build_dir,
            save_rrd,
            rerun_port,
            spawn_viewer,
            stage_window_size=stage_window_size,
        )
        bullet_sim.run()

        # Combine all staged recordings if any were created and save_rrd was not already written directly
        if stage_window_size and save_rrd:
            import glob
            import os
            import re
            from .utils import merge_rrd_recordings

            if not os.path.exists(save_rrd):
                base_dir = os.path.dirname(save_rrd) or "."
                base_name = os.path.splitext(os.path.basename(save_rrd))[0]
                prefix = re.sub(r"_\d+$", "", base_name)
                stage_pattern = os.path.join(base_dir, f"{prefix}_*.rrd")
                staged_files = sorted(glob.glob(stage_pattern))
                staged_files = [
                    f
                    for f in staged_files
                    if os.path.abspath(f) != os.path.abspath(save_rrd)
                    and re.match(rf"^{re.escape(prefix)}_\d+\.rrd$", os.path.basename(f))
                ]
                if staged_files:
                    staged_files.sort(
                        key=lambda x: (
                            int(re.search(r"_(\d+)\.rrd$", os.path.basename(x)).group(1))
                            if re.search(r"_(\d+)\.rrd$", os.path.basename(x))
                            else 0
                        )
                    )
                    merge_rrd_recordings(staged_files, save_rrd)
                    logger.print(f"Combined {len(staged_files)} staged recording(s) into {save_rrd}", symbol="✨")
