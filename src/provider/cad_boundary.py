"""CAD reconstruction and boolean intersection analysis for analytical URDF boundaries."""

import math
from typing import Any, Optional
import build123d as b3d
from model.boundary_config import BoundaryType, ShapeType, BoundaryCADConformance, LinkType
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

        case ShapeType.CANOPY:
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

        case ShapeType.CANOPY:
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


def extract_boundary_from_cad(
    part: Any,
    shape: Optional[ShapeType] = None,
    type: Optional[BoundaryType] = None,
    **overrides: Any,
) -> dict[str, Any]:
    """
    Extract analytical URDF boundary configuration parameters directly from a CAD part, solid, or compound.

    Inspects B-Rep face topology (cylinders, spheres, planes), bounding box bounds, center locations,
    and attached port joints (intake, drain, tube).

    Args:
        part: build123d Part, Solid, Compound, or Shape instance.
        shape: Optional explicit ShapeType. If None, derived from B-Rep face topology.
        type: Optional BoundaryType (e.g. SOLID, CAVITY, SOLID_CAVITY).
        **overrides: Optional parameter overrides that take precedence over CAD-derived defaults.

    Returns:
        Dictionary of boundary configuration parameters ready for BoundaryConfig validation.
    """
    # 1. Resolve solid geometry
    if hasattr(part, "part") and part.part is not None:
        solid = part.part
    elif hasattr(part, "solid") and callable(part.solid) and part.solid() is not None:
        solid = part.solid()
    else:
        solid = part

    # 2. Query bounding box
    solid_any: Any = solid
    bbox = solid_any.bounding_box()
    dx = bbox.max.X - bbox.min.X
    dy = bbox.max.Y - bbox.min.Y
    dz = bbox.max.Z - bbox.min.Z

    center_x = (bbox.max.X + bbox.min.X) * 0.5
    center_y = (bbox.max.Y + bbox.min.Y) * 0.5
    center_z = (bbox.max.Z + bbox.min.Z) * 0.5

    # 3. Query B-Rep faces if available
    cyl_faces = []
    sph_faces = []
    plane_faces = []
    if hasattr(solid, "faces") and callable(solid.faces):
        faces_list = solid.faces()
        if hasattr(faces_list, "filter_by"):
            cyl_faces = list(faces_list.filter_by(b3d.GeomType.CYLINDER))
            sph_faces = list(faces_list.filter_by(b3d.GeomType.SPHERE))
            plane_faces = list(faces_list.filter_by(b3d.GeomType.PLANE))
        else:
            cyl_faces = [f for f in faces_list if getattr(f, "geom_type", None) == b3d.GeomType.CYLINDER]
            sph_faces = [f for f in faces_list if getattr(f, "geom_type", None) == b3d.GeomType.SPHERE]
            plane_faces = [f for f in faces_list if getattr(f, "geom_type", None) == b3d.GeomType.PLANE]

    # 4. Infer shape type if not explicitly provided
    if shape is None:
        if len(sph_faces) > 0 and len(cyl_faces) == 0:
            shape = ShapeType.CANOPY
        elif len(cyl_faces) > 0:
            shape = ShapeType.CYLINDER
        elif abs(dx - dy) < 1e-3 and max(dx, dy) > 0.0:
            shape = ShapeType.CYLINDER
        else:
            shape = ShapeType.BOX

    default_radius = 0.0
    default_height = 0.0
    default_thickness = 0.0
    default_xyz = (0.0, 0.0, 0.0)

    match shape:
        case ShapeType.CYLINDER | ShapeType.TUBE | ShapeType.CASING | ShapeType.IMPELLER:
            radii = [float(f.radius) * 0.001 for f in cyl_faces if hasattr(f, "radius") and f.radius is not None]
            if radii:
                radii.sort()
                default_radius = radii[-1]
                default_thickness = float(max(0.0, radii[-1] - radii[0])) if len(radii) > 1 else 0.0
            else:
                default_radius = float(max(dx, dy) / 2.0 * 0.001)
                default_thickness = 0.0

            z_planar = [
                float(f.center().Z) * 0.001 for f in plane_faces if hasattr(f, "center") and f.center() is not None
            ]
            if z_planar:
                z_min = min(z_planar)
                z_max = max(z_planar)
                default_height = max(0.0, z_max - z_min)
                default_xyz = (float(center_x * 0.001), float(center_y * 0.001), float(z_min))
            else:
                default_height = float(dz * 0.001)
                default_xyz = (float(center_x * 0.001), float(center_y * 0.001), float(bbox.min.Z * 0.001))

        case ShapeType.CANOPY:
            sph_radii = [float(f.radius) * 0.001 for f in sph_faces if hasattr(f, "radius") and f.radius is not None]
            if sph_radii:
                sph_radii.sort()
                default_radius = sph_radii[-1]
                default_thickness = float(max(0.0, sph_radii[-1] - sph_radii[0])) if len(sph_radii) > 1 else 0.0
            else:
                default_radius = float(max(dx, dy, dz) / 2.0 * 0.001)
                default_thickness = 0.0
            default_height = float(dz * 0.001)
            default_xyz = (float(center_x * 0.001), float(center_y * 0.001), float(center_z * 0.001))

        case ShapeType.BOX:
            default_radius = 0.0
            default_thickness = 0.0
            default_height = float(dz * 0.001)
            default_xyz = (float(center_x * 0.001), float(center_y * 0.001), float(bbox.min.Z * 0.001))

        case ShapeType.PLANE:
            default_radius = float(max(dx, dy) / 2.0 * 0.001)
            default_thickness = float(dz * 0.001)
            default_height = 0.0
            default_xyz = (float(center_x * 0.001), float(center_y * 0.001), float(bbox.min.Z * 0.001))

    # 5. Extract rotation / orientation
    solid_location = getattr(solid, "location", None)
    if solid_location is not None:
        trsf = solid_location.wrapped.Transformation().VectorialPart()
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
        default_rpy = (float(phi), float(theta), float(psi))
    else:
        default_rpy = (0.0, 0.0, 0.0)

    # 6. Extract joint ports (intake, drain, tube) attached to part or solid
    joint_params: dict[str, Any] = {}
    part_joints = getattr(part, "joints", None) or getattr(solid, "joints", None)
    if part_joints and isinstance(part_joints, dict):
        for j_name, j_obj in part_joints.items():
            name_lower = str(j_name).lower()
            j_loc = getattr(j_obj, "location", None) or getattr(j_obj, "local_location", None)
            if j_loc is None:
                continue
            pos_m = (
                float(j_loc.position.X * 0.001),
                float(j_loc.position.Y * 0.001),
                float(j_loc.position.Z * 0.001),
            )
            trsf = j_loc.wrapped.Transformation().VectorialPart()
            norm_m = (
                float(trsf.Value(1, 3)),
                float(trsf.Value(2, 3)),
                float(trsf.Value(3, 3)),
            )

            if "intake" in name_lower and "intake_pos" not in overrides:
                joint_params["has_intake"] = True
                joint_params["intake_pos"] = pos_m
                joint_params["intake_normal"] = norm_m
                if hasattr(j_obj, "radius") and j_obj.radius is not None and "intake_radius" not in overrides:
                    joint_params["intake_radius"] = float(j_obj.radius)
            elif "drain" in name_lower and "drain_pos" not in overrides:
                joint_params["has_drain"] = True
                joint_params["drain_pos"] = pos_m
                joint_params["drain_normal"] = norm_m
                if hasattr(j_obj, "radius") and j_obj.radius is not None and "drain_radius" not in overrides:
                    joint_params["drain_radius"] = float(j_obj.radius)
            elif ("tube" in name_lower or "spout" in name_lower) and "tube_pos" not in overrides:
                joint_params["has_tube"] = True
                joint_params["tube_pos"] = pos_m
                joint_params["tube_normal"] = norm_m
                if hasattr(j_obj, "radius") and j_obj.radius is not None and "tube_radius" not in overrides:
                    joint_params["tube_radius"] = float(j_obj.radius)
            elif (
                "shelf" in name_lower or "seat" in name_lower or "pocket" in name_lower
            ) and "shelf_depth" not in overrides:
                if pos_m[2] > 0.0 and default_height > pos_m[2]:
                    joint_params["shelf_depth"] = float(default_height - pos_m[2])
                else:
                    joint_params["shelf_depth"] = float(abs(pos_m[2])) if abs(pos_m[2]) > 0.0 else default_thickness

            for slot_attr in ["slot_height", "slot_width", "cutoff_y", "ceiling_thickness"]:
                if hasattr(j_obj, slot_attr) and getattr(j_obj, slot_attr) is not None and slot_attr not in overrides:
                    joint_params[slot_attr] = float(getattr(j_obj, slot_attr))

    # 7. Auto-derive shelf_depth from cylinder wall/floor thickness if not otherwise specified
    if shape == ShapeType.CYLINDER and "shelf_depth" not in overrides and "shelf_depth" not in joint_params:
        eff_thick = overrides.get("thickness", default_thickness)
        if eff_thick is not None and float(eff_thick) > 0.0:
            joint_params["shelf_depth"] = float(eff_thick)

    # 7. Check for secondary off-center cylinder faces (e.g. integrated tube column in base reservoir)
    if (
        shape == ShapeType.CYLINDER
        and "has_tube" not in joint_params
        and "has_tube" not in overrides
        and overrides.get("link_type") == LinkType.BASE
    ):
        for f in cyl_faces:
            if hasattr(f, "radius") and f.radius is not None:
                c = f.center()
                dist_xy = math.sqrt(c.X * c.X + c.Y * c.Y) * 0.001
                r_f = float(f.radius) * 0.001
                if dist_xy > 0.010 and r_f < 0.020:
                    joint_params["has_tube"] = True
                    joint_params["tube_pos"] = (float(c.X * 0.001), float(c.Y * 0.001), float(default_xyz[2]))
                    joint_params["tube_normal"] = (0.0, 0.0, 1.0)
                    if "tube_radius" not in overrides and "tube_radius" not in joint_params:
                        joint_params["tube_radius"] = r_f
                    break

    # 8. Extract direct shape/part instance metadata attributes if present
    shape_metadata_attrs = [
        "num_vanes",
        "vane_twist",
        "vane_thickness",
        "magnet_radius",
        "magnet_thickness",
        "pump_well_wall",
        "magnet_count",
        "impeller_shaft_radius",
        "slot_height",
        "slot_width",
        "spout_radius",
        "spout_height",
        "cutoff_y",
        "ceiling_thickness",
        "is_submerged",
        "shelf_depth",
        "tube_radius",
        "intake_radius",
        "drain_radius",
    ]
    shape_params: dict[str, Any] = {}
    for attr in shape_metadata_attrs:
        val = None
        if hasattr(part, "__dict__") and attr in part.__dict__:
            val = part.__dict__[attr]
        elif hasattr(solid, "__dict__") and attr in solid.__dict__:
            val = solid.__dict__[attr]
        elif hasattr(part, attr) and not hasattr(type(part), attr):
            val = getattr(part, attr, None)
        elif hasattr(solid, attr) and not hasattr(type(solid), attr):
            val = getattr(solid, attr, None)

        if val is not None and attr not in overrides and attr not in joint_params:
            shape_params[attr] = val

    # 9. Merge defaults with overrides
    candidates: dict[str, Any] = {
        "shape": shape,
        "type": type,
        "radius": default_radius,
        "height": default_height,
        "thickness": default_thickness,
        "xyz": default_xyz,
        "rpy": default_rpy,
        **shape_params,
        **joint_params,
        **overrides,
    }
    return candidates
