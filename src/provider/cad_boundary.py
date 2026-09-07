"""CAD reconstruction and boolean intersection analysis for analytical URDF boundaries."""

import math
from typing import Any, Optional
import build123d as b3d
from model.boundary_config import BoundaryType, ShapeType, BoundaryCADConformance
from .room import Room


def reconstruct_boundary_cad_solid(b: Any, parent_location: Optional[b3d.Location] = None) -> Optional[Any]:
    """
    Reconstruct the 3D solid barrier geometry for an analytical URDF boundary.

    Args:
        b: Analytical boundary configuration or URDFBoundary instance.
        parent_location: Optional parent part location to transform into world coordinates.

    Returns:
        The reconstructed solid or compound as a build123d object, or None if pure cavity.
    """
    shape = getattr(b, "shape", None)
    b_type = getattr(b, "type", None)
    radius_m = getattr(b, "radius", 0.0) or 0.0
    thickness_m = getattr(b, "thickness", 0.0) or 0.0
    height_m = getattr(b, "height", 0.0) or 0.0
    xyz_m = getattr(b, "xyz", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    rpy_rad = getattr(b, "rpy", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)

    # Convert coordinates and dimensions from meters to millimeters for CAD booleans
    r_out = radius_m * 1000.0
    thick = thickness_m * 1000.0
    r_in = max(0.0, r_out - thick)
    h = height_m * 1000.0
    pos = (xyz_m[0] * 1000.0, xyz_m[1] * 1000.0, xyz_m[2] * 1000.0)
    rot = (math.degrees(rpy_rad[0]), math.degrees(rpy_rad[1]), math.degrees(rpy_rad[2]))
    local_loc = b3d.Location(pos, rot)
    loc = parent_location * local_loc if parent_location is not None else local_loc

    # Pure CAVITY boundaries represent open fluid space, not solid CAD features
    if b_type == BoundaryType.CAVITY:
        return None

    match shape:
        case ShapeType.TUBE:
            with b3d.BuildPart() as p:
                b3d.Cylinder(radius=r_out, height=h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
                if r_in > 0.0:
                    b3d.Cylinder(
                        radius=r_in,
                        height=h + 0.01,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                        mode=b3d.Mode.SUBTRACT,
                    )
            return p.part.located(loc)

        case ShapeType.CYLINDER:
            match b_type:
                case BoundaryType.SOLID:
                    with b3d.BuildPart() as p:
                        b3d.Cylinder(radius=r_out, height=h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
                    return p.part.located(loc)
                case BoundaryType.SOLID_CAVITY:
                    with b3d.BuildPart() as p:
                        b3d.Cylinder(radius=r_out, height=h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
                        if r_in > 0.0:
                            b3d.Cylinder(
                                radius=r_in,
                                height=h + 0.01,
                                align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                                mode=b3d.Mode.SUBTRACT,
                            )
                    return p.part.located(loc)
                case _:
                    return None

        case ShapeType.SPHERE:
            with b3d.BuildPart() as p:
                b3d.Sphere(radius=r_out)
                if r_in > 0.0:
                    b3d.Sphere(radius=r_in, mode=b3d.Mode.SUBTRACT)
                with b3d.Locations((0, 0, -r_out)):
                    b3d.Cylinder(
                        radius=r_out + 1.0,
                        height=r_out,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                        mode=b3d.Mode.SUBTRACT,
                    )
            return p.part.located(loc)

        case ShapeType.CASING:
            ceil_thick = (getattr(b, "ceiling_thickness", 0.0) or 0.0) * 1000.0
            with b3d.BuildPart() as p:
                b3d.Cylinder(radius=r_out, height=h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
                if r_in > 0.0:
                    cut_h = h - ceil_thick if ceil_thick > 0.0 else h + 0.01
                    b3d.Cylinder(
                        radius=r_in,
                        height=cut_h,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                        mode=b3d.Mode.SUBTRACT,
                    )
            return p.part.located(loc)

        case ShapeType.IMPELLER:
            num_vanes = int(getattr(b, "num_vanes", 0) or 0)
            vane_t = (getattr(b, "vane_thickness", 0.0) or 0.0) * 1000.0
            shaft_r = (getattr(b, "thickness", 0.0) or 0.0) * 1000.0
            hub_h = h * 0.5 if h > 0.0 else 0.0
            hub_r = max(0.0, r_out - 4.0)
            with b3d.BuildPart() as p:
                # Hub body base
                b3d.Cylinder(
                    radius=hub_r,
                    height=hub_h,
                    align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                )
                # Central guide post sleeve
                if shaft_r > 0.0:
                    b3d.Cylinder(
                        radius=shaft_r + 2.0,
                        height=h,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                    )
                    # Central shaft hole
                    b3d.Cylinder(
                        radius=shaft_r,
                        height=h + 0.01,
                        align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
                        mode=b3d.Mode.SUBTRACT,
                    )
                # Radial blades extending from central sleeve to hub perimeter
                sleeve_r = shaft_r + 2.0 if shaft_r > 0.0 else 4.5
                if num_vanes > 0 and vane_t > 0.0 and hub_h < h:
                    blade_len = max(0.0, hub_r - sleeve_r)
                    with b3d.Locations((0, 0, hub_h)):
                        for v_idx in range(num_vanes):
                            v_angle = (360.0 / num_vanes) * v_idx
                            with b3d.Locations(b3d.Rot(0, 0, v_angle)):
                                with b3d.Locations((sleeve_r, 0, 0)):
                                    b3d.Box(
                                        blade_len,
                                        vane_t,
                                        h - hub_h,
                                        align=(b3d.Align.MIN, b3d.Align.CENTER, b3d.Align.MIN),
                                    )
            return p.part.located(loc)

        case _:
            return None


def reconstruct_boundary_cad_cavity(b: Any, parent_location: Optional[b3d.Location] = None) -> Optional[Any]:
    """
    Reconstruct the 3D fluid cavity (open flow volume) for an analytical URDF boundary.

    Args:
        b: Analytical boundary configuration or URDFBoundary instance.
        parent_location: Optional parent part location to transform into world coordinates.

    Returns:
        The reconstructed fluid cavity volume as a build123d object, or None if pure solid.
    """
    shape = getattr(b, "shape", None)
    b_type = getattr(b, "type", None)
    radius_m = getattr(b, "radius", 0.0) or 0.0
    thickness_m = getattr(b, "thickness", 0.0) or 0.0
    height_m = getattr(b, "height", 0.0) or 0.0
    xyz_m = getattr(b, "xyz", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)
    rpy_rad = getattr(b, "rpy", (0.0, 0.0, 0.0)) or (0.0, 0.0, 0.0)

    # Pure solid boundaries do not have a fluid cavity
    if b_type == BoundaryType.SOLID:
        return None

    r_out = radius_m * 1000.0
    thick = thickness_m * 1000.0
    r_in = max(0.0, r_out - thick) if thick > 0.0 else r_out
    h = height_m * 1000.0
    pos = (xyz_m[0] * 1000.0, xyz_m[1] * 1000.0, xyz_m[2] * 1000.0)
    rot = (math.degrees(rpy_rad[0]), math.degrees(rpy_rad[1]), math.degrees(rpy_rad[2]))
    local_loc = b3d.Location(pos, rot)
    loc = parent_location * local_loc if parent_location is not None else local_loc

    match shape:
        case ShapeType.TUBE | ShapeType.CYLINDER | ShapeType.CASING:
            if r_in <= 0.0 or h <= 0.0:
                return None
            with b3d.BuildPart() as p:
                b3d.Cylinder(radius=r_in, height=h, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
            return p.part.located(loc)

        case ShapeType.SPHERE:
            if r_in <= 0.0:
                return None
            with b3d.BuildPart() as p:
                b3d.Sphere(radius=r_in)
            return p.part.located(loc)

        case _:
            return None


def evaluate_boundary_cad_conformance(
    cad_solid: Any, boundary: Any, parent_location: Optional[b3d.Location] = None
) -> BoundaryCADConformance:
    """
    Perform 3D boolean intersection checks between a CAD part and its URDF boundary.

    Args:
        cad_solid: The CAD part solid or compound.
        boundary: The analytical URDFBoundary or BoundaryConfig object.
        parent_location: Optional location override for the CAD part.

    Returns:
        A structured BoundaryCADConformance model containing boolean intersection volumes and conformance ratios.
    """
    loc = parent_location if parent_location is not None else getattr(cad_solid, "location", None)

    solid_geom = reconstruct_boundary_cad_solid(boundary, parent_location=loc)
    solid_vol = float(solid_geom.volume) if solid_geom is not None and hasattr(solid_geom, "volume") else 0.0
    solid_inter_vol = 0.0
    solid_ratio = 0.0

    if solid_geom is not None and solid_vol > 0.0:
        inter = cad_solid.intersect(solid_geom)
        solid_inter_vol = float(sum(s.volume for s in inter.solids()) if inter else 0.0)
        solid_ratio = float(solid_inter_vol / solid_vol)

    cavity_geom = reconstruct_boundary_cad_cavity(boundary, parent_location=loc)
    cavity_vol = float(cavity_geom.volume) if cavity_geom is not None and hasattr(cavity_geom, "volume") else 0.0
    cavity_inter_vol = 0.0

    if cavity_geom is not None and cavity_vol > 0.0:
        cav_inter = cad_solid.intersect(cavity_geom)
        cavity_inter_vol = float(sum(s.volume for s in cav_inter.solids()) if cav_inter else 0.0)

    return BoundaryCADConformance(
        shape=getattr(boundary, "shape", None),
        type=getattr(boundary, "type", None),
        solid_volume=solid_vol,
        solid_intersection_volume=solid_inter_vol,
        solid_conformance_ratio=solid_ratio,
        cavity_volume=cavity_vol,
        cavity_intersection_volume=cavity_inter_vol,
    )


def validate_room_urdf_boundaries(
    room: Room, min_conformance_ratio: float = 0.70
) -> list[tuple[str, BoundaryCADConformance]]:
    """
    Validate all URDF boundaries across all parts in a Room against their physical CAD geometries.

    Args:
        room: The Room containing built parts and attached URDF boundaries.
        min_conformance_ratio: Minimum ratio of CAD solid intersection to reconstructed boundary volume.

    Returns:
        A list of (part_name, BoundaryCADConformance) tuples for all registered boundaries.
    """
    conformance_results: list[tuple[str, BoundaryCADConformance]] = []

    for part_name, (geom, _) in room.items():
        boundaries = getattr(geom, "urdf_boundaries", None)
        if not boundaries:
            continue
        cad_solid = getattr(geom, "part", geom)

        for b in boundaries:
            conf = evaluate_boundary_cad_conformance(cad_solid, b)
            conformance_results.append((part_name, conf))

            if conf.solid_volume > 0.0:
                if conf.solid_conformance_ratio < min_conformance_ratio:
                    raise ValueError(
                        f"Part {part_name} boundary {conf.shape}/{conf.type} has insufficient CAD volume conformance: "
                        f"{conf.solid_conformance_ratio:.2%} (expected >= {min_conformance_ratio:.2%})"
                    )

            if conf.cavity_volume > 0.0:
                if conf.cavity_volume <= 0.0:
                    raise ValueError(f"Part {part_name} cavity boundary {conf.shape} has non-positive volume")

    return conformance_results
