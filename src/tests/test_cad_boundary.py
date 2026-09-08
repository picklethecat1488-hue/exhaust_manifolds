"""Unit tests for CAD-to-URDF analytical boundary derivation and conformance validation."""

import math
import numpy as np
import build123d as b3d
from model.boundary_config import BoundaryType, ShapeType
from provider import (
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
from provider.cad_boundary import (
    extract_boundary_from_cad,
    evaluate_boundary_cad_conformance,
    reconstruct_boundary_cad_solid,
    reconstruct_boundary_cad_cavity,
)
from provider.types import LinkType, URDFCollisionType


def test_extract_boundary_from_cylinder():
    """Verify analytical boundary parameter extraction from build123d Cylinder primitive."""
    r_mm = 25.0
    h_mm = 50.0
    cyl = b3d.Cylinder(radius=r_mm, height=h_mm, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    params = extract_boundary_from_cad(cyl, shape=ShapeType.CYLINDER, type=BoundaryType.SOLID)

    assert params["shape"] == ShapeType.CYLINDER
    assert params["type"] == BoundaryType.SOLID
    assert math.isclose(params["radius"], 0.025, abs_tol=1e-5)
    assert math.isclose(params["height"], 0.050, abs_tol=1e-5)
    assert math.isclose(params["thickness"], 0.0, abs_tol=1e-5)
    assert math.isclose(params["xyz"][0], 0.0, abs_tol=1e-5)
    assert math.isclose(params["xyz"][1], 0.0, abs_tol=1e-5)
    assert math.isclose(params["xyz"][2], 0.0, abs_tol=1e-5)


def test_extract_boundary_from_hollow_tube():
    """Verify extraction of outer radius and wall thickness from hollow tube CAD solid."""
    r_out_mm = 6.0
    r_in_mm = 4.5
    h_mm = 80.0
    thick_mm = r_out_mm - r_in_mm

    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=r_out_mm, height=h_mm, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))
        b3d.Cylinder(
            radius=r_in_mm,
            height=h_mm + 1.0,
            align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN),
            mode=b3d.Mode.SUBTRACT,
        )

    tube_solid = p.part
    params = extract_boundary_from_cad(tube_solid, shape=ShapeType.TUBE, type=BoundaryType.SOLID_CAVITY)

    assert params["shape"] == ShapeType.TUBE
    assert math.isclose(params["radius"], r_out_mm * 0.001, abs_tol=1e-5)
    assert math.isclose(params["thickness"], thick_mm * 0.001, abs_tol=1e-5)
    assert math.isclose(params["height"], h_mm * 0.001, abs_tol=1e-5)


def test_extract_boundary_from_sphere():
    """Verify extraction of sphere radius and center coordinates from build123d Sphere."""
    sph_r_mm = 7.5
    center_pos = (0.0, 28.0, 105.0)

    with b3d.BuildPart() as p:
        with b3d.Locations(center_pos):
            b3d.Sphere(radius=sph_r_mm)

    sph_solid = p.part
    params = extract_boundary_from_cad(sph_solid, shape=ShapeType.SPHERE, type=BoundaryType.SOLID)

    assert params["shape"] == ShapeType.SPHERE
    assert math.isclose(params["radius"], sph_r_mm * 0.001, abs_tol=1e-5)
    assert math.isclose(params["xyz"][0], center_pos[0] * 0.001, abs_tol=1e-5)
    assert math.isclose(params["xyz"][1], center_pos[1] * 0.001, abs_tol=1e-5)
    assert math.isclose(params["xyz"][2], center_pos[2] * 0.001, abs_tol=1e-5)


def test_extract_boundary_with_joint_ports():
    """Verify that intake, drain, and tube fluid ports are automatically resolved from CAD RigidJoints."""
    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=30.0, height=10.0, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    # Attach joints for intake port facing down (-Z) and drain port facing up (+Z)
    b3d.RigidJoint("intake_port", p.part, b3d.Location((0, 28.0, 10.0), (0, 180, 0)))
    b3d.RigidJoint("drain_port", p.part, b3d.Location((0, -20.0, 0.0), (0, 0, 0)))

    params = extract_boundary_from_cad(p.part, shape=ShapeType.CYLINDER, type=BoundaryType.CAVITY)

    assert params.get("has_intake") is True
    assert math.isclose(params["intake_pos"][0], 0.0, abs_tol=1e-5)
    assert math.isclose(params["intake_pos"][1], 0.028, abs_tol=1e-5)
    assert math.isclose(params["intake_pos"][2], 0.010, abs_tol=1e-5)
    assert math.isclose(params["intake_normal"][2], -1.0, abs_tol=1e-4)

    assert params.get("has_drain") is True
    assert math.isclose(params["drain_pos"][0], 0.0, abs_tol=1e-5)
    assert math.isclose(params["drain_pos"][1], -0.020, abs_tol=1e-5)
    assert math.isclose(params["drain_pos"][2], 0.0, abs_tol=1e-5)
    assert math.isclose(params["drain_normal"][2], 1.0, abs_tol=1e-4)


def test_urdf_boundary_from_shape_context_registration():
    """Verify URDFBoundary.from_shape registers directly to the active URDFMetadata context."""
    with b3d.BuildPart() as p:
        b3d.Cylinder(radius=40.0, height=20.0, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    with URDFMetadata(
        label="test_link",
        material="petg",
        density=1270.0,
        boundary_friction=0.20,
        collision_type=URDFCollisionType.ANALYTICAL,
        geometry=p.part,
    ) as meta:
        URDFBoundary.from_shape(
            p.part,
            link_type=LinkType.BASE,
            type=BoundaryType.SOLID,
        )

    assert len(meta.boundaries) == 1
    b_cfg = meta.boundaries[0]
    assert b_cfg.link_type == LinkType.BASE
    assert b_cfg.shape == ShapeType.CYLINDER
    assert math.isclose(b_cfg.radius, 0.040, abs_tol=1e-5)
    assert math.isclose(b_cfg.height, 0.020, abs_tol=1e-5)


def test_boundary_cad_boolean_conformance():
    """Verify 3D boolean intersection evaluation between a CAD solid and its derived boundary."""
    r_mm = 20.0
    h_mm = 30.0
    cyl = b3d.Cylinder(radius=r_mm, height=h_mm, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    with URDFMetadata(
        label="bowl",
        material="petg",
        density=1270.0,
        boundary_friction=0.20,
        collision_type=URDFCollisionType.ANALYTICAL,
        geometry=cyl,
    ) as meta:
        URDFBoundary.from_shape(cyl, link_type=LinkType.BASE, type=BoundaryType.SOLID)

    boundary = meta.boundaries[0]
    conformance = evaluate_boundary_cad_conformance(cyl, boundary)

    assert conformance.solid_volume > 0.0
    assert math.isclose(conformance.solid_conformance_ratio, 1.0, abs_tol=1e-3)


def test_declarative_port_and_feature_context_managers():
    """Verify that IntakePort, DrainPort, TubePort, and FlowSlot populate boundary metadata and shelf_depth is auto-derived."""
    cyl = b3d.Cylinder(radius=30.0, height=20.0, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    with URDFMetadata(
        label="test_casing",
        material="petg",
        density=1270.0,
        boundary_friction=0.20,
        collision_type=URDFCollisionType.ANALYTICAL,
        geometry=cyl,
    ) as meta:
        with URDFBoundary.from_shape(
            cyl,
            link_type=LinkType.CASING,
            shape=ShapeType.CASING,
            type=BoundaryType.SOLID_CAVITY,
            thickness=0.002,
        ):
            IntakePort(location=b3d.Location((0.0, -25.0, 5.0), (0, -90, 0)), radius=0.004)
            DrainPort(location=b3d.Location((0.0, 20.0, 0.0), (0, 90, 0)), radius=0.0035)
            TubePort(location=b3d.Location((0.0, 20.0, 0.0), (0, 0, 0)), radius=0.003)
            FlowSlot(height=0.009, width=0.008, cutoff_y=0.0, ceiling_thickness=0.001)

        with URDFBoundary.from_shape(
            cyl,
            link_type=LinkType.LID,
            shape=ShapeType.CYLINDER,
            type=BoundaryType.CAVITY,
            thickness=0.003,
        ):
            pass

    assert len(meta.boundaries) == 2
    b_casing = meta.boundaries[0]
    assert b_casing.link_type == LinkType.CASING
    assert b_casing.has_intake is True
    assert math.isclose(b_casing.intake_pos[0], 0.0, abs_tol=1e-5)
    assert math.isclose(b_casing.intake_pos[1], -0.025, abs_tol=1e-5)
    assert math.isclose(b_casing.intake_pos[2], 0.005, abs_tol=1e-5)
    assert math.isclose(b_casing.intake_radius, 0.004, abs_tol=1e-5)

    assert b_casing.has_drain is True
    assert math.isclose(b_casing.drain_pos[0], 0.0, abs_tol=1e-5)
    assert math.isclose(b_casing.drain_pos[1], 0.020, abs_tol=1e-5)
    assert math.isclose(b_casing.drain_radius, 0.0035, abs_tol=1e-5)

    assert b_casing.has_tube is True
    assert math.isclose(b_casing.tube_pos[0], 0.0, abs_tol=1e-5)
    assert math.isclose(b_casing.tube_pos[1], 0.020, abs_tol=1e-5)
    assert math.isclose(b_casing.tube_radius, 0.003, abs_tol=1e-5)

    assert math.isclose(b_casing.slot_height, 0.009, abs_tol=1e-5)
    assert math.isclose(b_casing.slot_width, 0.008, abs_tol=1e-5)

    b_lid = meta.boundaries[1]
    assert b_lid.link_type == LinkType.LID
    assert math.isclose(b_lid.shelf_depth, 0.003, abs_tol=1e-5)


def test_declarative_impeller_features_and_spout_deflection():
    """Verify that SpoutDeflection, ImpellerVanes, and MagneticCoupling populate boundary parameters."""
    cyl = b3d.Cylinder(radius=15.0, height=8.0, align=(b3d.Align.CENTER, b3d.Align.CENTER, b3d.Align.MIN))

    with URDFMetadata(
        label="test_impeller",
        material="petg",
        density=1270.0,
        boundary_friction=0.20,
        collision_type=URDFCollisionType.ANALYTICAL,
        geometry=cyl,
    ) as meta:
        with URDFBoundary.from_shape(
            cyl,
            link_type=LinkType.IMPELLER,
            shape=ShapeType.IMPELLER,
            type=BoundaryType.SOLID,
            thickness=0.001,
        ):
            ImpellerVanes(count=6, twist=45.0, thickness=0.0012)
            MagneticCoupling(
                radius=0.006,
                thickness=0.003,
                count=4,
                well_wall=0.0015,
                shaft_radius=0.001,
            )

    assert len(meta.boundaries) == 1
    b = meta.boundaries[0]
    assert b.num_vanes == 6
    assert math.isclose(b.vane_twist, 45.0, abs_tol=1e-5)
    assert math.isclose(b.vane_thickness, 0.0012, abs_tol=1e-5)
    assert math.isclose(b.magnet_radius, 0.006, abs_tol=1e-5)
    assert math.isclose(b.magnet_thickness, 0.003, abs_tol=1e-5)
    assert b.magnet_count == 4
    assert math.isclose(b.pump_well_wall, 0.0015, abs_tol=1e-5)
    assert math.isclose(b.impeller_shaft_radius, 0.001, abs_tol=1e-5)
