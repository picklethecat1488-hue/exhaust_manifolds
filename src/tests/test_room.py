"""Unit tests for the Room container."""

import os
import shutil
import tempfile
from unittest.mock import MagicMock, patch
import math
import numpy as np
from typing import cast, Any
import pybullet as p
import pytest
from build123d import Compound, Box, Vector, RigidJoint, RevoluteJoint, Axis, Location, LinearJoint, BallJoint
from model import AppConfig, TextArgs, DiagramOptions, BoundaryType
from provider import Simulate, ColorType, URDFShape, URDFCollisionType, URDFCollisionShapeType, Mode
from provider.room import Room, BulletStateTracker


def test_room_add_success():
    """Verify items can be added to the room."""
    room = Room()
    room.add("item1", "geom1", color=ColorType.RED, alpha=0.5)
    assert "item1" in room
    assert room["item1"] == ("geom1", (1.0, 0.0, 0.0, 0.5))


def test_room_is_simulate_attribute():
    """Verify is_simulate parameter sets the attribute correctly."""
    room_default = Room()
    assert not room_default.is_simulate

    room_sim = Room(is_simulate=True)
    assert room_sim.is_simulate


def test_room_add_duplicate_name_raises_error():
    """Verify that adding a duplicate name raises a ValueError."""
    room = Room()
    room.add("duplicate", "geom1")
    with pytest.raises(ValueError, match="An item with the name 'duplicate' already exists"):
        room.add("duplicate", "geom2")


def test_room_add_default_values():
    """Verify default values are applied when adding items."""
    room = Room()
    room.add("default_item", "geom")
    assert room["default_item"] == ("geom", (0.5, 0.5, 0.5, 1.0))


def test_room_add_with_config():
    """Verify that default color is read from AppConfig when color is None."""
    mock_config = MagicMock(spec=AppConfig)
    mock_config.color = (0.1, 0.2, 0.3, 0.4)

    room = Room(config=mock_config)
    room.add("item", "geom")
    assert room["item"] == ("geom", (0.1, 0.2, 0.3, 0.4))


def test_room_add_with_rgb_tuple():
    """Verify that an RGB 3-tuple can be used as a color."""
    room = Room()
    rgb = (0.5, 0.5, 0.5)
    room.add("item", "geom", color=rgb)
    assert room["item"] == ("geom", (0.5, 0.5, 0.5, 1.0))


def test_room_compound_property():
    """Verify that the room can be converted to a single build123d Compound."""
    room = Room()
    box = Box(1, 1, 1)
    room.add("item1", box, color=ColorType.RED)

    assy = room.compound
    assert isinstance(assy, Compound)
    # build123d Compound children are identified by labels
    assert any(child.label == "item1" for child in assy.children)


def test_room_export_diagram():
    """Verify that export_diagram correctly maps options and calls ExportSVG for SVG paths."""
    room = Room()

    mock_exporter_cls = MagicMock()
    with patch.dict(Room.export_diagram.__globals__, {"ExportSVG": mock_exporter_cls}):
        mock_instance = mock_exporter_cls.return_value

        # Define a mock options object with snake_case attributes
        class MockOptions:
            width = 1000
            show_axes = False
            stroke_width = 0.5
            margin = 10
            view_from = "front"

        options = MockOptions()

        # Add geometry to trigger scale calculation
        from build123d import Box

        room.add("item1", Box(100, 100, 100))

        room.export_diagram("test_output.svg", options)  # type: ignore

        # Verify that width is converted to scale and unsupported fields are removed
        expected_opts = {
            "scale": 10.0,  # 1000 / 100
            "line_weight": 0.5,
            "margin": 10,
        }

        mock_exporter_cls.assert_called_once_with(**expected_opts)
        mock_instance.write.assert_called_once_with("test_output.svg")

        # Verify add_shape was called with a Compound containing our item.
        # We check the captured argument directly to avoid calling room.compound
        # again, which would deparent the children from the recorded call.
        mock_instance.add_shape.assert_called_once()
        passed_compound = mock_instance.add_shape.call_args[0][0]
        assert isinstance(passed_compound, Compound)
        # One child for the manifold lines
        assert len(passed_compound.children) == 1


def test_room_add_label():
    """Verify annotations can be added to the room."""
    room = Room()
    loc = (10, 20, 30)
    room.add_label("label1", "Hello", loc, options=TextArgs(font_size=12))
    assert len(room._labels) == 1
    name, text, pos, opts = room._labels[0]
    assert text == "Hello"
    assert pos == Vector(10, 20, 30)
    assert opts.font_size == 12


def test_room_parse_options_view_from():
    """Verify that view_from is correctly parsed from DiagramOptions."""
    room = Room()
    options = DiagramOptions(view_from="top rear")
    opts = room._parse_options(options)
    assert opts["view_from"] == "top rear"


def test_room_get_projection_vectors_named_views():
    """Verify that named view expressions map to correct direction vectors."""
    room = Room()
    room.add("box", Box(1, 1, 1))  # Centered at (0,0,0)

    # Test simple 'top' view
    look_from, look_at, _ = room._get_projection_vectors({"view_from": "top"})
    assert look_at == Vector(0, 0, 0)
    assert (look_from - look_at).normalized() == Vector(0, 0, 1)

    # Test composite 'top rear' view: top (0,0,1) + rear (0,1,0) -> (0,1,1)
    look_from, look_at, _ = room._get_projection_vectors({"view_from": "top rear"})
    expected_dir = Vector(0, 1, 1).normalized()
    assert (look_from - look_at).normalized() == expected_dir


def test_room_get_projection_vectors_dynamic_distance():
    """Verify that the camera distance is calculated based on the bounding box diagonal."""
    room = Room()
    room.add("box", Box(100, 100, 100))
    bb = room.compound.bounding_box()
    expected_distance = bb.diagonal * 2

    look_from, look_at, _ = room._get_projection_vectors({"view_from": "top"})
    assert (look_from - look_at).length == pytest.approx(expected_distance)


def test_room_get_projection_vectors_overrides():
    """Verify that projection_origin and projection_dir override automatic calculations."""
    room = Room()
    room.add("box", Box(1, 1, 1))
    opts = {"projection_origin": (10, 10, 10), "projection_dir": (1, 1, 1)}
    look_from, look_at, _ = room._get_projection_vectors(opts)
    assert look_from == Vector(10, 10, 10)
    assert look_at == Vector(1, 1, 1)


def test_room_translate_joints():
    """Verify that translate_joints correctly maps build123d joints to URDF attributes on shapes."""
    room = Room()

    parent = Box(10, 10, 10)
    parent_shape = cast(URDFShape, parent)
    parent_shape.urdf_label = "parent_link"
    room.add("parent_link", parent)

    child = Box(5, 5, 5)
    child_shape = cast(URDFShape, child)
    child_shape.urdf_label = "child_link"
    room.add("child_link", child)

    pj = RigidJoint("joint_p", parent, Location((0, 0, 5)))
    cj = RevoluteJoint("joint_c", child, Axis((0, 0, 0), (0, 1, 0)), angular_range=(0, 180))
    pj.connect_to(cj)

    assert getattr(child, "urdf_parent", None) is None

    room.translate_joints()

    assert child_shape.urdf_parent == "parent_link"
    assert child_shape.urdf_joint_type == "revolute"
    assert child_shape.urdf_joint_axis == "0 1 0"
    assert child_shape.urdf_joint_lower is not None
    assert math.isclose(child_shape.urdf_joint_lower, 0.0)
    assert child_shape.urdf_joint_upper is not None
    assert math.isclose(child_shape.urdf_joint_upper, math.radians(180))


def test_room_disconnect_joints():
    """Verify that disconnect_joints clears all URDF joint connection properties on shapes."""
    room = Room()

    child = Box(5, 5, 5)
    child_shape = cast(URDFShape, child)
    child_shape.urdf_parent = "parent_link"
    child_shape.urdf_joint_type = "revolute"
    child_shape.urdf_joint_axis = "0 1 0"
    child_shape.urdf_joint_lower = 0.0
    child_shape.urdf_joint_upper = 3.14
    room.add("child", child)

    assert child_shape.urdf_parent == "parent_link"

    room.disconnect_joints()

    assert child_shape.urdf_parent is None
    assert child_shape.urdf_joint_type is None
    assert child_shape.urdf_joint_axis is None
    assert child_shape.urdf_joint_lower is None
    assert child_shape.urdf_joint_upper is None


def test_room_urdf_pybullet_integration():
    """Verify that PyBullet can successfully load the exported URDF and parses all joint types correctly."""
    room = Room()

    # Create parent link
    parent = Box(10, 10, 10)
    p_shape = cast(URDFShape, parent)
    p_shape.urdf_label = "parent_link"
    room.add("parent_link", parent)

    temp_dir = tempfile.mkdtemp()
    proj_dir = os.path.join(temp_dir, "test_robot")
    os.makedirs(proj_dir, exist_ok=True)

    try:
        # Mock .obj files
        with open(os.path.join(proj_dir, "parent_link.obj"), "w") as f:
            f.write("# empty obj\n")

        # 1. Fixed (RigidJoint)
        child_fixed = Box(2, 2, 2)
        room.add("child_fixed", child_fixed)
        with open(os.path.join(proj_dir, "child_fixed.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_fixed = RigidJoint("j_p_fixed", parent, Location((15, 0, 0)))
        cj_fixed = RigidJoint("j_c_fixed", child_fixed, Location((0, 0, 0)))
        pj_fixed.connect_to(cj_fixed)

        # 2. Revolute
        child_rev = Box(2, 2, 2)
        room.add("child_rev", child_rev)
        with open(os.path.join(proj_dir, "child_rev.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_rev = RigidJoint("j_p_rev", parent, Location((30, 0, 0)))
        cj_rev = RevoluteJoint("j_c_rev", child_rev, Axis((0, 0, 0), (0, 1, 0)), angular_range=(0, 180))
        pj_rev.connect_to(cj_rev)

        # 3. Continuous (RevoluteJoint with >= 360 angular range)
        child_cont = Box(2, 2, 2)
        room.add("child_cont", child_cont)
        with open(os.path.join(proj_dir, "child_cont.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_cont = RigidJoint("j_p_cont", parent, Location((45, 0, 0)))
        cj_cont = RevoluteJoint("j_c_cont", child_cont, Axis((0, 0, 0), (0, 1, 0)), angular_range=(0, 360))
        pj_cont.connect_to(cj_cont)

        # 4. Prismatic (LinearJoint)
        child_prism = Box(2, 2, 2)
        room.add("child_prism", child_prism)
        with open(os.path.join(proj_dir, "child_prism.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_prism = RigidJoint("j_p_prism", parent, Location((60, 0, 0)))
        cj_prism = LinearJoint("j_c_prism", child_prism, Axis((0, 0, 0), (0, 0, 1)), linear_range=(0, 100))
        pj_prism.connect_to(cj_prism)

        # 5. Ball/Spherical (BallJoint)
        child_ball = Box(2, 2, 2)
        room.add("child_ball", child_ball)
        with open(os.path.join(proj_dir, "child_ball.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_ball = RigidJoint("j_p_ball", parent, Location((75, 0, 0)))
        cj_ball = BallJoint("j_c_ball", child_ball, Location((0, 0, 0)))
        pj_ball.connect_to(cj_ball)

        # 6. Planar (using RigidJoint + urdf_joint_type override)
        child_planar = Box(2, 2, 2)
        room.add("child_planar", child_planar)
        with open(os.path.join(proj_dir, "child_planar.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_planar = RigidJoint("j_p_planar", parent, Location((90, 0, 0)))
        cj_planar = RigidJoint("j_c_planar", child_planar, Location((0, 0, 0)))
        cast(Any, cj_planar).urdf_joint_type = "planar"
        cast(Any, cj_planar).urdf_joint_axis = "0 1 1"
        pj_planar.connect_to(cj_planar)

        # 7. Floating (using RigidJoint + urdf_joint_type override)
        child_float = Box(2, 2, 2)
        room.add("child_float", child_float)
        with open(os.path.join(proj_dir, "child_float.obj"), "w") as f:
            f.write("# empty obj\n")
        pj_float = RigidJoint("j_p_float", parent, Location((105, 0, 0)))
        cj_float = RigidJoint("j_c_float", child_float, Location((0, 0, 0)))
        cast(Any, cj_float).urdf_joint_type = "floating"
        pj_float.connect_to(cj_float)

        urdf_path = os.path.join(temp_dir, "test.urdf")
        room.export_urdf(urdf_path, "test_robot")

        # Load in PyBullet
        physics_client = p.connect(p.DIRECT)
        try:
            body_id = p.loadURDF(urdf_path)
            assert body_id >= 0, "PyBullet failed to load the URDF file"

            num_joints = p.getNumJoints(body_id)
            assert num_joints == 7, f"Expected 7 joints, found {num_joints}"

            # Verify each joint type
            joints_info = {}
            for i in range(num_joints):
                info = p.getJointInfo(body_id, i)
                name = info[1].decode("utf-8")
                j_type = info[2]
                lower = info[8]
                upper = info[9]
                axis = info[13]
                joints_info[name] = {"type": j_type, "lower": lower, "upper": upper, "axis": axis}

            # 1. Fixed
            fixed_info = joints_info["parent_link_to_child_fixed"]
            assert fixed_info["type"] == p.JOINT_FIXED

            # 2. Revolute
            rev_info = joints_info["parent_link_to_child_rev"]
            assert rev_info["type"] == p.JOINT_REVOLUTE
            assert math.isclose(rev_info["lower"], 0.0, abs_tol=1e-5)
            assert math.isclose(rev_info["upper"], math.radians(180), abs_tol=1e-5)
            assert rev_info["axis"] == (0.0, 1.0, 0.0)

            # 3. Continuous
            cont_info = joints_info["parent_link_to_child_cont"]
            assert cont_info["type"] == p.JOINT_REVOLUTE
            # Continuous joints do not have upper/lower limits in PyBullet
            assert cont_info["lower"] == 0.0
            assert cont_info["upper"] == -1.0
            assert cont_info["axis"] == (0.0, 1.0, 0.0)

            # 4. Prismatic
            prism_info = joints_info["parent_link_to_child_prism"]
            assert prism_info["type"] == p.JOINT_PRISMATIC
            assert math.isclose(prism_info["lower"], 0.0, abs_tol=1e-5)
            assert math.isclose(prism_info["upper"], 0.1, abs_tol=1e-5)  # 100mm = 0.1m
            assert prism_info["axis"] == (0.0, 0.0, 1.0)

            # 5. Ball/Spherical
            ball_info = joints_info["parent_link_to_child_ball"]
            assert ball_info["type"] == p.JOINT_SPHERICAL

            # 6. Planar
            planar_info = joints_info["parent_link_to_child_planar"]
            assert planar_info["type"] == p.JOINT_PLANAR

            # 7. Floating (loaded as fixed by PyBullet internally but parsed without error)
            float_info = joints_info["parent_link_to_child_float"]
            assert float_info["type"] == p.JOINT_FIXED

        finally:
            p.disconnect(physicsClientId=physics_client)

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def test_export_urdf_multiple_root_links():
    """Verify that export_urdf raises ValueError if multiple root links are found."""
    room = Room()
    # Create two disconnected shapes, each with an urdf_label
    box1 = Box(1, 1, 1)
    box1_part = cast(URDFShape, box1)
    box1_part.urdf_label = "link1"

    box2 = Box(1, 1, 1)
    box2_part = cast(URDFShape, box2)
    box2_part.urdf_label = "link2"

    room.add("box1", box1)
    room.add("box2", box2)

    import io

    output = io.StringIO()
    with pytest.raises(ValueError, match="URDF file with multiple root links found: link1 link2"):
        room.export_urdf(output, "test_project")


def test_bullet_reset_camera():
    """Verify that reset_camera successfully calls resetDebugVisualizerCamera."""
    from provider.bullet import Bullet

    room = Room()
    box = Box(10, 10, 10)
    room.add("box", box)

    physics_client = p.connect(p.DIRECT)
    try:
        with patch("pybullet.resetDebugVisualizerCamera") as mock_reset:
            bullet_sim = Bullet(room, {}, "", "", 0, None, None)
            bullet_sim.reset_camera(physics_client, view_from="top rear")
            mock_reset.assert_called_once()
            args, kwargs = mock_reset.call_args
            assert kwargs["cameraDistance"] > 0
            assert math.isclose(kwargs["cameraYaw"], 90.0)
            assert math.isclose(kwargs["cameraPitch"], -44.5)
            assert len(kwargs["cameraTargetPosition"]) == 3
    finally:
        p.disconnect(physicsClientId=physics_client)


def test_bullet_copy_project_assets():
    """Verify that copy_project_assets successfully copies existing OBJ files or raises FileNotFoundError."""
    from provider.bullet import Bullet

    room = Room()
    parent = Box(1, 1, 1)
    cast(Any, parent).urdf_label = "parent"
    room.add("parent", parent)

    with tempfile.TemporaryDirectory() as temp_dir:
        build_proj_dir = os.path.join(temp_dir, "build")
        proj_dir = os.path.join(temp_dir, "proj")
        os.makedirs(build_proj_dir, exist_ok=True)
        os.makedirs(proj_dir, exist_ok=True)

        bullet_sim = Bullet(room, {}, "test_proj", "parent", 0, None, None)

        # 1. Raises FileNotFoundError if OBJ does not exist
        with pytest.raises(FileNotFoundError, match="Required OBJ file not found"):
            bullet_sim._copy_project_assets(build_proj_dir, proj_dir)

        # 2. Successfully copies if OBJ exists
        real_obj_path = os.path.join(build_proj_dir, "parent.obj")
        with open(real_obj_path, "w") as f:
            f.write("# parent obj")

        bullet_sim._copy_project_assets(build_proj_dir, proj_dir)
        temp_obj_path = os.path.join(proj_dir, "parent.obj")
        assert os.path.exists(temp_obj_path)
        with open(temp_obj_path, "r") as f:
            assert f.read() == "# parent obj"


@patch("rerun.init")
@patch("rerun.spawn")
@patch("rerun.connect_grpc")
def test_bullet_init_rerun(mock_connect_grpc, mock_spawn, mock_init):
    """Verify Rerun connection initialization logic."""
    from provider.bullet import Bullet

    room = Room()

    # 1. spawn_viewer is False -> just call init
    bullet_sim = Bullet(room, {}, "test_proj", "", 0, None, None, spawn_viewer=False)
    bullet_sim._init_rerun()
    mock_init.assert_called_once_with("test_proj")
    mock_spawn.assert_not_called()
    mock_connect_grpc.assert_not_called()

    mock_init.reset_mock()
    mock_spawn.reset_mock()
    mock_connect_grpc.reset_mock()

    # Mock is_port_in_use to simulate free port
    with patch("socket.socket") as mock_sock:
        # connect_ex returns 1 (meaning port is NOT in use)
        mock_sock.return_value.__enter__.return_value.connect_ex.return_value = 1

        # 2. spawn_viewer is True, port is free -> spawn
        bullet_sim = Bullet(room, {}, "test_proj", "", 0, None, None, spawn_viewer=True, rerun_port=9876)
        bullet_sim._init_rerun()
        mock_init.assert_called_once_with("test_proj")
        mock_spawn.assert_called_once_with(port=9876)
        mock_connect_grpc.assert_not_called()

    mock_init.reset_mock()
    mock_spawn.reset_mock()
    mock_connect_grpc.reset_mock()

    # Mock is_port_in_use to simulate busy port
    with patch("socket.socket") as mock_sock:
        # connect_ex returns 0 (meaning port IS in use)
        mock_sock.return_value.__enter__.return_value.connect_ex.return_value = 0

        # 3. spawn_viewer is True, port is busy -> connect
        bullet_sim = Bullet(room, {}, "test_proj", "", 0, None, None, spawn_viewer=True, rerun_port=9876)
        bullet_sim._init_rerun()
        mock_init.assert_called_once_with("test_proj")
        mock_spawn.assert_not_called()
        mock_connect_grpc.assert_called_once_with("rerun+http://127.0.0.1:9876/proxy")


def test_bullet_state_tracker_update_state():
    """Verify that BulletStateTracker.update_state fetches states and particle colors from PyBullet."""
    room = Room()
    parent = Box(1, 1, 1)
    cast(Any, parent).urdf_label = "parent"
    room.add("parent", parent)

    physics_client = p.connect(p.DIRECT)
    try:
        # Create a sphere visual shape to test color extraction
        sphere_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.003, rgbaColor=[1.0, 0.0, 0.0, 1.0], physicsClientId=physics_client
        )
        body_id = p.createMultiBody(
            baseMass=0.0,
            baseVisualShapeIndex=sphere_vis,
            physicsClientId=physics_client,
        )

        # Create a particle body with positive mass
        particle_col = p.createCollisionShape(p.GEOM_SPHERE, radius=0.003, physicsClientId=physics_client)
        particle_vis = p.createVisualShape(
            p.GEOM_SPHERE, radius=0.003, rgbaColor=[0.0, 0.5, 1.0, 0.8], physicsClientId=physics_client
        )
        particle_id = p.createMultiBody(
            baseMass=0.001,
            baseCollisionShapeIndex=particle_col,
            baseVisualShapeIndex=particle_vis,
            physicsClientId=physics_client,
        )

        label_to_link_idx = {"parent": -1}
        state_tracker = BulletStateTracker(body_id, physics_client, label_to_link_idx)
        state_tracker.update_state()
        transforms = state_tracker.transforms
        positions = state_tracker.particle_positions
        colors = state_tracker.particle_colors

        assert "parent" in transforms
        assert len(positions) == 1
        assert len(colors) == 1
        assert len(state_tracker.particle_radii) == 1
        # Extract rgbaColor from visual data list (floats)
        assert math.isclose(colors[0][0], 0.0, abs_tol=1e-5)
        assert math.isclose(colors[0][1], 0.5, abs_tol=1e-5)
        assert math.isclose(colors[0][2], 1.0, abs_tol=1e-5)
        assert math.isclose(colors[0][3], 0.8, abs_tol=1e-5)
        assert math.isclose(state_tracker.particle_radii[0], 0.003, abs_tol=1e-5)

    finally:
        p.disconnect(physicsClientId=physics_client)


def test_bullet_state_tracker_error_handling():
    """Verify that BulletStateTracker.update_state ignores bodies that cause exceptions."""
    physics_client = p.connect(p.DIRECT)
    try:
        body_id = p.createMultiBody(physicsClientId=physics_client)
        # Add a body index that doesn't exist to simulate getDynamicsInfo failure
        state_tracker = BulletStateTracker(body_id, physics_client, {})
        # Force tracker to think there are more bodies
        state_tracker._last_checked_num_bodies = 0
        with patch("pybullet.getDynamicsInfo", side_effect=p.error("Fake Error")):
            # Should not raise exception
            state_tracker.update_state()
            assert len(state_tracker.particle_positions) == 0
    finally:
        p.disconnect(physicsClientId=physics_client)


def test_room_simulate_loop_optimization():
    """Verify that Room.simulate skips state tracking and rerun logging when logging is disabled."""
    room = Room()
    room.add("item", Box(1, 1, 1))

    # Mock setup, step, teardown hooks
    hooks = {
        Simulate.SETUP: MagicMock(),
        Simulate.STEP: MagicMock(return_value=None),
    }

    from provider.bullet import Bullet

    with (
        patch("provider.bullet.p.loadURDF", return_value=1) as mock_load_urdf,
        patch("provider.bullet.p.stepSimulation") as mock_step_sim,
        patch("provider.bullet.BulletStateTracker") as mock_tracker_cls,
        patch("provider.bullet.os.path.exists", return_value=True),
        patch("provider.bullet.shutil.copy"),
        patch.object(Bullet, "_copy_project_assets") as mock_copy,
        patch.object(Bullet, "_init_simulation_objects", return_value={}) as mock_init_objects,
        patch.object(Bullet, "_init_rerun") as mock_init_rerun,
        patch.object(room, "_log_rerun") as mock_log_rerun,
    ):
        mock_tracker = mock_tracker_cls.return_value

        # Run with spawn_viewer=False and save_rrd=None (logging disabled)
        room.simulate(
            provider_hooks=hooks,
            proj_name="test_proj",
            sim_target="item",
            steps=5,
            manager=MagicMock(),
            logger=MagicMock(),
            spawn_viewer=False,
            save_rrd=None,
        )

        # BulletStateTracker.update_state and room._log_rerun should NOT be called at all
        mock_tracker.update_state.assert_not_called()
        mock_log_rerun.assert_not_called()
        assert mock_step_sim.call_count == 5

        # Reset mocks
        mock_tracker.reset_mock()
        mock_log_rerun.reset_mock()
        mock_step_sim.reset_mock()

        # Run with spawn_viewer=True (logging enabled)
        room.simulate(
            provider_hooks=hooks,
            proj_name="test_proj",
            sim_target="item",
            steps=5,
            manager=MagicMock(),
            logger=MagicMock(),
            spawn_viewer=True,
            save_rrd=None,
        )

        # BulletStateTracker.update_state and room._log_rerun should be called
        assert mock_tracker.update_state.call_count == 5
        assert mock_log_rerun.call_count == 5
        assert mock_step_sim.call_count == 5


def test_room_translate_joints_relative():
    """Verify that Room.translate_joints correctly computes parent-relative joint coordinates."""
    room = Room()

    parent = Box(10, 10, 10)
    cast(URDFShape, parent).urdf_label = "parent_link"
    parent.locate(Location((0, 0, 0)))
    room.add("parent", parent)

    child = Box(5, 5, 5)
    cast(URDFShape, child).urdf_label = "child_link"
    # Position child offset in X
    child.locate(Location((20, 0, 0)))
    room.add("child", child)

    pj = RigidJoint("j_p", parent, Location((0, 0, 0)))
    cj = RigidJoint("j_c", child, Location((0, 0, 0)))
    pj.connect_to(cj)

    # Let's export or check relative coordinates in URDF
    import io

    output = io.StringIO()
    room.export_urdf(output, "test_proj")
    urdf_text = output.getvalue()

    # The joint translation from parent to child should be (20, 0, 0)
    # Wait, the shape coordinates are mm, URDF is meters, so X offset should be 0.02
    assert 'xyz="0.02 0 0"' in urdf_text or 'xyz="0.020000 0.000000 0.000000"' in urdf_text or 'xyz="0.02' in urdf_text


def test_room_export_urdf_primitive_collisions():
    """Verify that Room.export_urdf exports primitive collision tags when collision_primitives is set."""
    room = Room()

    bowl = Box(10, 10, 10)
    u_bowl = cast(URDFShape, bowl)
    u_bowl.urdf_label = "bowl"
    u_bowl.urdf_collision_type = URDFCollisionType.COMPOUND
    u_bowl.urdf_collision_primitives = [
        {
            "type": URDFCollisionShapeType.BOX,
            "size": [0.152, 0.152, 0.004],
            "xyz": [0.0, 0.0, 0.002],
            "rpy": [0.0, 0.0, 0.0],
        },
        {
            "type": URDFCollisionShapeType.CYLINDER,
            "radius": 0.008,
            "length": 0.100,
            "xyz": [0.0, 0.0, 0.05],
            "rpy": [0.0, 0.0, 1.57],
        },
        {"type": URDFCollisionShapeType.SPHERE, "radius": 0.005, "xyz": [0.0, 0.0, 0.0], "rpy": [0.0, 0.0, 0.0]},
    ]
    u_bowl.locate(Location((0, 0, 0)))
    room.add("bowl_part", bowl)

    import io

    output = io.StringIO()
    room.export_urdf(output, "test_proj")
    urdf_text = output.getvalue()

    # Verify that collision type is compound
    assert "<collision_type>compound</collision_type>" in urdf_text

    # Verify that box geom is exported
    assert '<box size="0.152000 0.152000 0.004000"/>' in urdf_text
    assert 'xyz="0.000000 0.000000 0.002000"' in urdf_text

    # Verify that cylinder geom is exported
    assert '<cylinder radius="0.008000" length="0.100000"/>' in urdf_text
    assert 'xyz="0.000000 0.000000 0.050000"' in urdf_text
    assert 'rpy="0.000000 0.000000 1.570000"' in urdf_text

    # Verify that sphere geom is exported
    assert '<sphere radius="0.005000"/>' in urdf_text


def test_cat_fountain_provider_bowl_collision_types():
    """Verify that CatFountainProvider sets bowl collision type depending on simulation mode."""
    from projects.cat_fountain.provider import CatFountainProvider

    provider = CatFountainProvider()

    # 1. Test simulation mode (analytical)
    bowl_sim = provider.build_bowl("bowl", mode=Mode.SIMULATE)
    u_bowl_sim = cast(URDFShape, bowl_sim.part)
    assert u_bowl_sim.urdf_collision_type == URDFCollisionType.ANALYTICAL
    assert len(u_bowl_sim.urdf_boundaries) > 0  # type: ignore
    assert u_bowl_sim.urdf_boundaries[0].type == BoundaryType.CAVITY  # type: ignore

    # 2. Test default/other mode (also analytical)
    bowl_def = provider.build_bowl("bowl", mode=Mode.DEFAULT)
    u_bowl_def = cast(URDFShape, bowl_def.part)
    assert u_bowl_def.urdf_collision_type == URDFCollisionType.ANALYTICAL


def test_urdf_shape_derived_properties():
    """Verify that build123d.Shape classes have derived urdf_height and urdf_thickness properties."""
    box = Box(10.0, 20.0, 30.0)
    u_box = cast(URDFShape, box)

    # height should be Z dimension (30.0) in meters -> 0.03
    assert math.isclose(u_box.urdf_height, 0.03, abs_tol=1e-6)

    # thickness should be minimum dimension (10.0) in meters -> 0.01
    assert math.isclose(u_box.urdf_thickness, 0.01, abs_tol=1e-6)


def test_merge_rrd_recordings(tmp_path):
    """Verify that merge_rrd_recordings combines multiple RRD files cleanly."""
    import rerun as rr
    from provider.utils import merge_rrd_recordings

    # Create two dummy RRD recordings
    file1 = tmp_path / "part1.rrd"
    file2 = tmp_path / "part2.rrd"
    out_file = tmp_path / "combined.rrd"

    rec1 = rr.RecordingStream(application_id="test1")
    rec1.save(str(file1))
    rec1.log("test/point1", rr.Points3D([[0.0, 0.0, 0.0]]))
    rec1.flush()

    rec2 = rr.RecordingStream(application_id="test2")
    rec2.save(str(file2))
    rec2.log("test/point2", rr.Points3D([[1.0, 1.0, 1.0]]))
    rec2.flush()

    assert file1.exists()
    assert file2.exists()

    result = merge_rrd_recordings([file1, file2], out_file)
    assert result == out_file
    assert out_file.exists()
    assert out_file.stat().st_size > 0


def test_room_simulate_combines_staged_recordings(tmp_path):
    """Verify that Room.simulate triggers merge_rrd_recordings for staged checkpoints."""
    from unittest.mock import MagicMock, patch
    from provider.room import Room

    room = Room(is_simulate=True)
    room.add("test_box", Box(10, 10, 10))

    mock_logger = MagicMock()
    mock_manager = MagicMock()

    staged_file1 = tmp_path / "sim_1000.rrd"
    staged_file2 = tmp_path / "sim_2000.rrd"
    staged_file1.write_text("dummy1")
    staged_file2.write_text("dummy2")

    save_rrd_path = tmp_path / "sim_7200.rrd"

    with (
        patch("provider.bullet.Bullet.run") as mock_bullet_run,
        patch("provider.utils.merge_rrd_recordings") as mock_merge,
    ):
        room.simulate(
            provider_hooks={},
            proj_name="test_proj",
            sim_target="test_target",
            steps=7200,
            manager=mock_manager,
            logger=mock_logger,
            save_rrd=str(save_rrd_path),
            stage_window_size=1000,
        )

        mock_bullet_run.assert_called_once()
        mock_merge.assert_called_once()
        args, _ = mock_merge.call_args
        assert sorted(args[0]) == [str(staged_file1), str(staged_file2)]
        assert args[1] == str(save_rrd_path)


def test_room_log_rerun_show_boundary_voxels(monkeypatch):
    """Verify that Room._log_rerun logs boundary voxels only when SHOW_BOUNDARY_VOXELS is enabled."""
    room = Room(is_simulate=True)
    boundary_voxels = {"lid": [[0.0, 0.0, 0.105], [0.01, 0.01, 0.105]]}

    with patch("rerun.log") as mock_rr_log:
        # Default: disabled
        monkeypatch.delenv("SHOW_BOUNDARY_VOXELS", raising=False)
        room._log_rerun(
            transforms={},
            particle_positions=None,
            particle_colors=None,
            particle_radii=None,
            boundary_voxels=boundary_voxels,
            step_idx=0,
        )
        logged_paths = [call[0][0] for call in mock_rr_log.call_args_list]
        assert not any("world/boundaries" in p for p in logged_paths)

    with patch("rerun.log") as mock_rr_log:
        # Enabled: True
        monkeypatch.setenv("SHOW_BOUNDARY_VOXELS", "1")
        room._log_rerun(
            transforms={},
            particle_positions=None,
            particle_colors=None,
            particle_radii=None,
            boundary_voxels=boundary_voxels,
            step_idx=0,
        )
        logged_paths = [call[0][0] for call in mock_rr_log.call_args_list]
        assert any("world/boundaries/lid" in p for p in logged_paths)


def test_room_log_rerun_show_water_voxels_and_solid_meshes(monkeypatch):
    """Verify that Room._log_rerun logs solid water meshes and gates voxel particles on SHOW_WATER_VOXELS."""
    import rerun as rr

    room = Room(is_simulate=True)
    water_meshes = {
        "pool_1": (
            np.array([[0.0, 0.0, 0.041], [0.05, 0.0, 0.041], [0.0, 0.05, 0.078]], dtype=np.float32),
            np.array([[0, 1, 2]], dtype=np.uint32),
        )
    }
    particles = [[0.0, 0.0, 0.050], [0.01, 0.01, 0.052]]

    # 1. Default: Solid water meshes logged, discrete voxel particles suppressed
    with patch("rerun.log") as mock_rr_log:
        monkeypatch.delenv("SHOW_WATER_VOXELS", raising=False)
        room._log_rerun(
            transforms={},
            particle_positions=particles,
            particle_colors=None,
            particle_radii=None,
            water_meshes=water_meshes,
            step_idx=0,
        )
        logged_calls = {call[0][0]: call[0][1] for call in mock_rr_log.call_args_list}
        assert "world/water/pool_1" in logged_calls
        mesh_arg = logged_calls["world/water/pool_1"]
        assert isinstance(mesh_arg, rr.Mesh3D)
        assert mesh_arg.albedo_factor is not None
        assert "world/particles" not in logged_calls

    # 2. SHOW_WATER_VOXELS=1: Both solid meshes and discrete particles logged
    with patch("rerun.log") as mock_rr_log:
        monkeypatch.setenv("SHOW_WATER_VOXELS", "1")
        room._log_rerun(
            transforms={},
            particle_positions=particles,
            particle_colors=None,
            particle_radii=None,
            boundary_voxels=None,
            water_meshes=water_meshes,
            step_idx=0,
        )
        logged_calls = {call[0][0]: call[0][1] for call in mock_rr_log.call_args_list}
        assert "world/water/pool_1" in logged_calls
        assert "world/particles" in logged_calls
