"""Unit tests for the Viewer class."""

import fnmatch
import pytest
import pybullet
from unittest.mock import MagicMock, patch, PropertyMock, call
from view import Viewer, BuildPart, BuildSketch, BuildLine, Part, Sketch, Wire
from provider import Section, TargetList, Mode, Simulate, Room, URDFShape
from build123d import Box, RigidJoint, RevoluteJoint, Location, Axis
from typing import cast


class TestViewer:
    """Viewer unit tests."""

    @pytest.fixture
    def mock_manager(self):
        """Mock ProviderManager fixture."""
        manager = MagicMock()
        # Configure targets mock to support chaining and basic list behavior
        targets_mock = MagicMock(spec=TargetList)
        targets_mock.provider = MagicMock()
        targets_mock.modes = [Mode.DEFAULT]
        targets_mock.for_targets.return_value = targets_mock
        targets_mock.supporting.return_value = targets_mock
        # Ensure the mock is truthy for the 'if not targets' validation check
        targets_mock.__len__.return_value = 1
        targets_mock.__iter__.return_value = iter(["mock/target"])
        manager.router.targets = targets_mock
        return manager

    @pytest.fixture
    def mock_logger(self):
        """Mock Logger fixture."""
        return MagicMock()

    @pytest.fixture
    def viewer(self, mock_manager, mock_logger):
        """Viewer fixture."""
        return Viewer(mock_manager, mock_logger)

    @pytest.fixture(autouse=True)
    def mock_pybullet_rerun_functions(self):
        """Mock Rerun-related Room state-querying/logging functions to avoid connection errors."""
        with (
            patch("provider.bullet.BulletStateTracker.update_state"),
            patch.object(Room, "_log_rerun"),
            patch("provider.bullet.Bullet._init_rerun"),
            patch("rerun.log"),
            patch("rerun.Asset3D"),
        ):
            yield

    def test_get_summary_truncation(self, viewer):
        """Verify summary logic for short and long lists."""
        assert viewer.get_summary(["a", "b"]) == "a, b"
        long_list = [str(i) for i in range(20)]
        summary = viewer.get_summary(long_list)
        assert summary.startswith("0, 1, 2, 3, 4, 5, 6, 7")
        assert "(20 items)" in summary

    @patch("view.show")
    def test_show_view_metadata_alignment(self, mock_show, viewer):
        """Verify color/alpha lists are aligned even with mixed metadata."""
        t1, t2 = "p1/part", "p1/diagram"

        # Ensure mocks returned by for_targets support iteration AND have a provider attribute
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m1.__iter__.return_value = iter([t1])
        m1.__len__.return_value = 1
        m1.__getitem__.return_value = t1
        m1.subassemblies = []

        m2 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m2.__iter__.return_value = iter([t2])
        m2.__len__.return_value = 1
        m2.__getitem__.return_value = t2
        m2.subassemblies = []

        viewer.target_parser.resolve = MagicMock(side_effect=[None, m1, m2])
        viewer.manager.router.manifest = {t1: {Section.PART: {}}, t2: {Section.DIAGRAM: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            # Part has color, Diagram does not (None)
            mock_geom_1 = MagicMock(spec=Part, label=None)
            mock_geom_1.wrapped = "fake_wrapped_1"
            mock_diag_room = Room()
            mock_diag_geom = MagicMock(spec=Part, label=None, wrapped="fake_wrapped_diag")
            mock_diag_room.add("diagram", mock_diag_geom)

            viewer.manager.router.run.side_effect = [
                [(t1, mock_geom_1)],  # Result for t1
                [("p1", mock_diag_room)],  # Result for t2
            ]
            viewer.manager.router.get_color.return_value = (1, 0, 0, 0.5)

            viewer.show_view([t1, t2])

            assert mock_show.called
            assy = mock_show.call_args[0][0]
            # Verify labels are present in the compound
            labels = [c.label for c in assy.children]
            assert "p1_part" in labels
            assert "p1_diagram" in labels

    @patch("view.show")
    def test_show_view_room(self, mock_show, viewer):
        """Verify show_view handles Section.VIEW targets."""
        target_name = "tube/wire"
        mock_targets = MagicMock(spec=TargetList)
        mock_targets.__iter__.return_value = iter([target_name])
        mock_targets.__len__.return_value = 1
        mock_targets.subassemblies = []
        viewer.target_parser.resolve = MagicMock(side_effect=[mock_targets, None, None])
        viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            mock_geom = MagicMock(label=None)
            del mock_geom.toCompound
            mock_geom.wrapped = "fake_wrapped"
            # VIEW actions return a list of (target, Room) tuples
            mock_room = Room()
            mock_room.add("item_0", mock_geom, color=(1, 0, 1))
            viewer.manager.router.run.return_value = [(target_name, mock_room)]

            viewer.show_view([target_name])

            mock_show.assert_called_once()
            assy = mock_show.call_args[0][0]
            assert any(c.label == "tube_wire_item_0" for c in assy.children)

    @patch("view.show")
    def test_show_view_part(self, mock_show, viewer):
        """Verify show_view handles Section.PART targets."""
        target_name = "tube/driver"
        mock_targets = MagicMock(spec=TargetList)
        mock_targets.__iter__.return_value = iter([target_name])
        mock_targets.__len__.return_value = 1
        mock_targets.subassemblies = []
        viewer.target_parser.resolve = MagicMock(side_effect=[None, mock_targets, None])
        viewer.manager.router.manifest = {target_name: {Section.PART: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            mock_geom = MagicMock(spec=Part, label=None)
            mock_geom.wrapped = "fake_wrapped"
            viewer.manager.router.run.return_value = [(target_name, mock_geom)]
            viewer.manager.router.get_color.return_value = (1, 0, 0, 1)

            viewer.show_view([target_name])

            mock_show.assert_called_once()
            assy = mock_show.call_args[0][0]
            assert any(c.label == "tube_driver" for c in assy.children)

    @patch("view.show")
    def test_show_view_diagram(self, mock_show, viewer):
        """Verify show_view handles Section.DIAGRAM targets."""
        target_name = "tube/diagram"
        mock_targets = MagicMock(spec=TargetList)
        mock_targets.__iter__.return_value = iter([target_name])
        mock_targets.__len__.return_value = 1
        viewer.target_parser.resolve = MagicMock(side_effect=[None, None, mock_targets])
        viewer.manager.router.manifest = {target_name: {Section.DIAGRAM: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            mock_diag_room = Room()
            mock_diag_geom = MagicMock(spec=Part, label=None, wrapped="fake_wrapped_diag")
            mock_diag_room.add("item", mock_diag_geom)
            viewer.manager.router.run.return_value = [("tube", mock_diag_room)]

            viewer.show_view([target_name])

            mock_show.assert_called_once()
            assy = mock_show.call_args[0][0]
            # Verify diagram label is present in the compound
            assert any(c.label == "tube_diagram" for c in assy.children)

    @patch("view.show")
    def test_show_view_multiple_targets(self, mock_show, viewer):
        """Verify show_view handles multiple targets at once."""
        t1, t2 = "tube/driver", "tube/passenger"

        # Return configured mocks from for_targets to satisfy iteration and attribute access
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m1.__iter__.return_value = iter([t1])
        m1.__len__.return_value = 1
        m1.subassemblies = []

        m2 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m2.__iter__.return_value = iter([t2])
        m2.__len__.return_value = 1
        m2.subassemblies = []

        # resolve is called for [VIEW, PART, DIAGRAM] for each target.
        # For t1: resolve(t1, VIEW) -> None, resolve(t1, PART) -> m1. break.
        # For t2: resolve(t2, VIEW) -> None, resolve(t2, PART) -> m2. break.
        viewer.target_parser.resolve = MagicMock(side_effect=[None, m1, None, m2])

        viewer.manager.router.manifest = {t1: {Section.PART: {}}, t2: {Section.PART: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            # Return one geom for each target run
            g1, g2 = MagicMock(spec=Part, label=None), MagicMock(spec=Part, label=None)
            g1.wrapped, g2.wrapped = "w1", "w2"
            viewer.manager.router.run.side_effect = [[(t1, g1)], [(t2, g2)]]
            viewer.manager.router.get_color.return_value = (1, 1, 1, 1)

            viewer.show_view([t1, t2])

            assert mock_show.call_count == 1
            assy = mock_show.call_args[0][0]
            names = [c.label for c in assy.children]
            assert "tube_driver" in names
            assert "tube_passenger" in names

    @patch("view.show")
    def test_show_view_name_collision_deduplication(self, mock_show, viewer):
        """Verify that items with colliding names are de-duplicated with suffixes."""
        t1, t2 = "p1/t1", "p2/t1"

        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m1.__iter__.return_value = iter([t1])
        m1.__len__.return_value = 1
        m1.subassemblies = []

        m2 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.DEFAULT])
        m2.__iter__.return_value = iter([t2])
        m2.__len__.return_value = 1
        m2.subassemblies = []

        # resolve is called for [VIEW, PART, DIAGRAM] for each target.
        viewer.target_parser.resolve = MagicMock(side_effect=[None, m1, None, m2])

        viewer.manager.router.manifest = {t1: {Section.PART: {}}, t2: {Section.PART: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            # Return one geom for each target run, forcing a name collision on "duplicate"
            g1, g2 = MagicMock(spec=Part, label=None), MagicMock(spec=Part, label=None)
            g1.wrapped, g2.wrapped = "w1", "w2"
            viewer.manager.router.run.side_effect = [[("duplicate", g1)], [("duplicate", g2)]]
            viewer.manager.router.get_color.return_value = (1, 1, 1, 1)

            viewer.show_view([t1, t2])

            assy = mock_show.call_args[0][0]
            names = [c.label for c in assy.children]
            assert "duplicate" in names
            assert "duplicate_1" in names

    def test_show_view_not_found(self, viewer):
        """Verify error when target is not found."""
        viewer.target_parser.resolve = MagicMock(
            side_effect=ValueError("No geometry generated for the specified targets.")
        )
        with pytest.raises(ValueError, match="No geometry generated for the specified targets."):
            viewer.show_view(["missing"])

    @patch("view.show")
    def test_show_view_unpacks_builders(self, mock_show, viewer):
        """Verify show_view unpacks BuildPart, BuildSketch, BuildLine objects."""
        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            # Mock BuildPart, BuildSketch, BuildLine and their .part/.sketch/.line attributes
            mock_part_obj = MagicMock(spec=Part, label=None)
            mock_part_obj.wrapped = "wp"
            mock_build_part = MagicMock(spec=BuildPart)
            mock_build_part.part = mock_part_obj

            mock_sketch_obj = MagicMock(spec=Sketch, label=None)
            mock_sketch_obj.wrapped = "ws"
            mock_build_sketch = MagicMock(spec=BuildSketch)
            mock_build_sketch.sketch = mock_sketch_obj

            mock_line_obj = MagicMock(spec=Wire, label=None)
            mock_line_obj.wrapped = "wl"
            mock_build_line = MagicMock(spec=BuildLine)
            mock_build_line.line = mock_line_obj

            # Mock a provider that returns these builder objects
            target_name = "mock/builders"
            mock_targets = MagicMock(spec=TargetList)
            mock_targets.__iter__.return_value = iter([target_name])
            mock_targets.__len__.return_value = 1
            viewer.target_parser.resolve = MagicMock(side_effect=[mock_targets, None, None])
            viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

            # Simulate _get_view_items returning builder objects inside a Room
            mock_room = Room()
            mock_room.add("part", mock_build_part, color=(1, 0, 0))
            mock_room.add("sketch", mock_build_sketch, color=(0, 1, 0))
            mock_room.add("line", mock_build_line, color=(0, 0, 1))
            viewer.manager.router.run.return_value = [(target_name, mock_room)]

            viewer.show_view([target_name])

            mock_show.assert_called_once()
            assy = mock_show.call_args[0][0]
            names = [c.label for c in assy.children]
            assert names == ["mock_builders_line", "mock_builders_part", "mock_builders_sketch"]

    def test_show_view_unsupported_action(self, viewer):
        """Verify error when target supports no visual actions."""
        target_name = "tube/config_only"
        viewer.manager.router.targets.__iter__.return_value = iter([target_name])
        viewer.manager.router.targets.__getitem__.return_value = target_name
        viewer.manager.router.manifest = {target_name: {Section.CONFIG: {}}}

        with pytest.raises(ValueError, match="No geometry generated for the specified targets"):
            viewer.show_view([target_name])

    @patch("provider.room.os.path.exists")
    @patch("provider.room.shutil.copy")
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.disconnect")
    @patch("pybullet.setGravity")
    def test_show_simulation_execution(
        self,
        mock_set_gravity,
        mock_disconnect,
        mock_step_simulation,
        mock_set_motor_control,
        mock_get_joint_info,
        mock_get_num_joints,
        mock_load_urdf,
        mock_connect,
        mock_copy,
        mock_exists,
        viewer,
    ):
        """Verify that show_simulation copies URDF/OBJs, connects to PyBullet, configures motors, and executes simulation hooks."""
        # 1. Setup mocks
        mock_exists.return_value = True
        mock_connect.return_value = 42
        mock_load_urdf.return_value = 100
        mock_get_num_joints.return_value = 1
        # Joint index 0, name b'parent_to_child'
        mock_get_joint_info.return_value = (0, b"parent_to_child", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b"child", (0, 0, 0))

        # 2. Setup mock provider with hooks
        mock_provider = MagicMock()
        mock_setup = MagicMock()
        mock_step = MagicMock(return_value=None)
        mock_provider.simulate = {
            Simulate.SETUP: mock_setup,
            Simulate.STEP: mock_step,
        }

        # 3. Setup room with parent and child geometry
        room = Room()

        parent = Box(1, 1, 1)
        p_shape = cast(URDFShape, parent)
        p_shape.urdf_label = "parent"
        room.add("parent", parent)

        child = Box(1, 1, 1)
        c_shape = cast(URDFShape, child)
        c_shape.urdf_label = "child"
        c_shape.urdf_motor_type = "velocity"
        c_shape.urdf_motor_target = 5.0
        c_shape.urdf_motor_force = 2.0
        room.add("child", child)

        pj = RigidJoint("j_p", parent, Location((0, 0, 0)))
        cj = RevoluteJoint("j_c", child, Axis((0, 0, 0), (0, 0, 1)))
        pj.connect_to(cj)

        # 4. Execute room.simulate with SMOKE_TEST enabled to run 10 steps
        room.simulate(
            provider_hooks=mock_provider.simulate,
            proj_name="mock",
            sim_target="mock/target",
            steps=10,
            manager=viewer.manager,
            logger=viewer.logger,
        )

        # 5. Assertions
        mock_exists.assert_any_call("build/obj/mock/parent.obj")
        mock_exists.assert_any_call("build/obj/mock/child.obj")
        mock_exists.assert_any_call("build/urdf/mock/target.urdf")
        assert mock_copy.call_count == 3
        mock_connect.assert_called_once_with(pybullet.DIRECT)
        mock_load_urdf.assert_called_once()
        mock_get_num_joints.assert_called_once_with(100, physicsClientId=42)
        mock_set_gravity.assert_called_once_with(0, 0, -9.81, physicsClientId=42)

        # Verify setJointMotorControl2 is called for the velocity motor
        mock_set_motor_control.assert_any_call(
            bodyUniqueId=100,
            jointIndex=0,
            controlMode=0,  # VELOCITY_CONTROL is 0 in PyBullet
            targetVelocity=5.0,
            force=2.0,
            physicsClientId=42,
        )

        # Verify hooks were called
        mock_setup.assert_called_once_with(100, 42, "mock/target", {})
        assert mock_step.call_count == 10
        actual_calls = [call.args for call in mock_step.call_args_list]
        expected_calls = [(100, 42, i, "mock/target") for i in range(10)]
        assert actual_calls == expected_calls
        mock_disconnect.assert_called_once_with(physicsClientId=42)

    @patch("provider.room.os.path.exists")
    @patch("provider.room.shutil.copy")
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.disconnect")
    @patch("pybullet.setGravity")
    def test_show_simulation_executio_terminate_early(
        self,
        mock_set_gravity,
        mock_disconnect,
        mock_step_simulation,
        mock_set_motor_control,
        mock_get_joint_info,
        mock_get_num_joints,
        mock_load_urdf,
        mock_connect,
        mock_copy,
        mock_exists,
        viewer,
    ):
        """Verify that show_simulation copies URDF/OBJs, connects to PyBullet, configures motors, and executes simulation hooks."""
        # 1. Setup mocks
        mock_exists.return_value = True
        mock_connect.return_value = 42
        mock_load_urdf.return_value = 100
        mock_get_num_joints.return_value = 1
        # Joint index 0, name b'parent_to_child'
        mock_get_joint_info.return_value = (0, b"parent_to_child", 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, b"child", (0, 0, 0))

        # 2. Setup mock provider with hooks
        mock_provider = MagicMock()
        mock_setup = MagicMock()
        mock_step = MagicMock(return_value=None)
        mock_step.side_effect = [None, None] + ["terminated early"] * 100
        mock_provider.simulate = {
            Simulate.SETUP: mock_setup,
            Simulate.STEP: mock_step,
        }

        # 3. Setup room with parent and child geometry
        room = Room()

        parent = Box(1, 1, 1)
        p_shape = cast(URDFShape, parent)
        p_shape.urdf_label = "parent"
        room.add("parent", parent)

        child = Box(1, 1, 1)
        c_shape = cast(URDFShape, child)
        c_shape.urdf_label = "child"
        c_shape.urdf_motor_type = "velocity"
        c_shape.urdf_motor_target = 5.0
        c_shape.urdf_motor_force = 2.0
        room.add("child", child)

        pj = RigidJoint("j_p", parent, Location((0, 0, 0)))
        cj = RevoluteJoint("j_c", child, Axis((0, 0, 0), (0, 0, 1)))
        pj.connect_to(cj)

        # 4. Execute room.simulate with SMOKE_TEST enabled to run 10 steps
        room.simulate(
            provider_hooks=mock_provider.simulate,
            proj_name="mock",
            sim_target="mock/target",
            steps=10,
            manager=viewer.manager,
            logger=viewer.logger,
        )

        # 5. Assertions
        mock_exists.assert_any_call("build/obj/mock/parent.obj")
        mock_exists.assert_any_call("build/obj/mock/child.obj")
        mock_exists.assert_any_call("build/urdf/mock/target.urdf")
        assert mock_copy.call_count == 3
        mock_connect.assert_called_once_with(pybullet.DIRECT)
        mock_load_urdf.assert_called_once()
        mock_get_num_joints.assert_called_once_with(100, physicsClientId=42)
        mock_set_gravity.assert_called_once_with(0, 0, -9.81, physicsClientId=42)

        # Verify setJointMotorControl2 is called for the velocity motor
        mock_set_motor_control.assert_any_call(
            bodyUniqueId=100,
            jointIndex=0,
            controlMode=0,  # VELOCITY_CONTROL is 0 in PyBullet
            targetVelocity=5.0,
            force=2.0,
            physicsClientId=42,
        )

        # Verify hooks were called with early termination
        mock_setup.assert_called_once_with(100, 42, "mock/target", {})
        assert mock_step.call_count == 3
        actual_calls = [call.args for call in mock_step.call_args_list]
        expected_calls = [(100, 42, i, "mock/target") for i in range(3)]
        assert actual_calls == expected_calls
        mock_disconnect.assert_called_once_with(physicsClientId=42)

    @patch("view.show")
    @patch("view.Builder")
    @patch("view.Room.simulate")
    def test_show_view_runs_simulation_in_simulate_mode(self, mock_simulate, mock_builder_class, mock_show, viewer):
        """Verify show_view runs simulation instead of standard view when SIMULATE mode is resolved."""
        target_name = "mock/target"
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.SIMULATE])
        m1.__iter__.return_value = iter([target_name])
        m1.__len__.return_value = 1
        m1.__getitem__.return_value = target_name
        m1.subassemblies = []
        m1.provider = viewer.manager.router.targets.provider

        viewer.target_parser.resolve = MagicMock(side_effect=[m1, None, None])
        viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

        # Simulate getting room from VIEW
        mock_room = Room()
        mock_geom = MagicMock(spec=Part, label=None, wrapped="fake_geom")
        mock_room.add("part", mock_geom)
        viewer.manager.router.run.return_value = [(target_name, mock_room)]

        viewer.show_view([target_name])

        # Verify standard ocp_vscode show is NOT called, but Room.simulate IS called
        mock_show.assert_not_called()
        mock_simulate.assert_called_once()
        kwargs = mock_simulate.call_args[1]
        assert kwargs.get("provider_hooks") == m1.provider.get_simulate_hooks.return_value
        assert kwargs.get("proj_name") == "mock"
        assert kwargs.get("sim_target") == "mock/target"
        assert kwargs.get("steps") == 2000

        # Verify Builder compiles parts and URDFs prior to simulating
        mock_builder_class.assert_called_once_with(viewer.manager, viewer.logger)
        mock_builder = mock_builder_class.return_value
        mock_builder.generate_parts.assert_called_once_with("build", names=["mock/*"], force_update=False)
        mock_builder.generate_urdfs.assert_called_once_with("build", names=["mock/*"], force_update=False)

    @patch("view.show")
    @patch("view.Builder")
    @patch("view.Room.simulate")
    def test_show_view_simulation_no_build(self, mock_simulate, mock_builder_class, mock_show, viewer):
        """Verify show_view skips compilation when no_build=True is specified."""
        target_name = "mock/target"
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.SIMULATE])
        m1.__iter__.return_value = iter([target_name])
        m1.__len__.return_value = 1
        m1.__getitem__.return_value = target_name
        m1.subassemblies = []
        m1.provider = viewer.manager.router.targets.provider

        viewer.target_parser.resolve = MagicMock(side_effect=[m1, None, None])
        viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

        # Simulate getting room from VIEW
        mock_room = Room()
        mock_geom = MagicMock(spec=Part, label=None, wrapped="fake_geom")
        mock_room.add("part", mock_geom)
        viewer.manager.router.run.return_value = [(target_name, mock_room)]

        viewer.show_view([target_name], no_build=True)

        # Verify simulate is called but Builder is never instantiated or run
        mock_simulate.assert_called_once()
        kwargs = mock_simulate.call_args[1]
        assert kwargs.get("provider_hooks") == m1.provider.get_simulate_hooks.return_value
        assert kwargs.get("proj_name") == "mock"
        assert kwargs.get("sim_target") == "mock/target"
        assert kwargs.get("steps") == 2000
        mock_builder_class.assert_not_called()

    @patch("view.show")
    @patch("view.Builder")
    @patch("view.Room.simulate")
    def test_show_view_simulation_custom_build_dir(self, mock_simulate, mock_builder_class, mock_show, viewer):
        """Verify show_view runs compilation in the specified custom build directory."""
        target_name = "mock/target"
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.SIMULATE])
        m1.__iter__.return_value = iter([target_name])
        m1.__len__.return_value = 1
        m1.__getitem__.return_value = target_name
        m1.subassemblies = []
        m1.provider = viewer.manager.router.targets.provider

        viewer.target_parser.resolve = MagicMock(side_effect=[m1, None, None])
        viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

        # Simulate getting room from VIEW
        mock_room = Room()
        mock_geom = MagicMock(spec=Part, label=None, wrapped="fake_geom")
        mock_room.add("part", mock_geom)
        viewer.manager.router.run.return_value = [(target_name, mock_room)]

        viewer.show_view([target_name], build_dir="custom_build")

        # Verify Builder compiles parts and URDFs in custom_build
        mock_builder_class.assert_called_once_with(viewer.manager, viewer.logger)
        mock_builder = mock_builder_class.return_value
        mock_builder.generate_parts.assert_called_once_with("custom_build", names=["mock/*"], force_update=False)
        mock_builder.generate_urdfs.assert_called_once_with("custom_build", names=["mock/*"], force_update=False)

        # Verify simulate is called with build_dir="custom_build"
        mock_simulate.assert_called_once()
        kwargs = mock_simulate.call_args[1]
        assert kwargs.get("provider_hooks") == m1.provider.get_simulate_hooks.return_value
        assert kwargs.get("proj_name") == "mock"
        assert kwargs.get("sim_target") == "mock/target"
        assert kwargs.get("steps") == 2000
        assert kwargs.get("build_dir") == "custom_build"

    def test_show_simulation_empty_room(self, viewer):
        """Verify show_simulation raises ValueError when room is empty."""
        room = Room()
        mock_provider = MagicMock()
        with pytest.raises(ValueError, match="Cannot simulate an empty Room."):
            room.simulate(
                provider_hooks=mock_provider.simulate,
                proj_name="mock",
                sim_target="mock/target",
                steps=10,
                manager=viewer.manager,
                logger=viewer.logger,
            )

    def test_show_simulation_missing_urdf_file(self, viewer):
        """Verify show_simulation raises FileNotFoundError when URDF file is missing."""
        room = Room()
        parent = Box(1, 1, 1)
        p_shape = cast(URDFShape, parent)
        p_shape.urdf_label = "parent"
        room.add("parent", parent)

        mock_provider = MagicMock()
        with (
            patch("provider.room.shutil.copy"),
            patch("provider.room.os.path.exists", side_effect=lambda path: not path.endswith(".urdf")),
        ):
            with pytest.raises(FileNotFoundError, match="Required URDF file not found for simulation"):
                room.simulate(
                    provider_hooks=mock_provider.simulate,
                    proj_name="mock",
                    sim_target="mock/target",
                    steps=10,
                    manager=viewer.manager,
                    logger=viewer.logger,
                )

    def test_show_simulation_missing_obj_file(self, viewer):
        """Verify show_simulation raises FileNotFoundError when an OBJ file is missing."""
        room = Room()
        parent = Box(1, 1, 1)
        p_shape = cast(URDFShape, parent)
        p_shape.urdf_label = "parent"
        room.add("parent", parent)

        mock_provider = MagicMock()
        with patch("provider.room.shutil.copy"), patch("provider.room.os.path.exists", return_value=False):
            with pytest.raises(FileNotFoundError, match="Required OBJ file not found for simulation"):
                room.simulate(
                    provider_hooks=mock_provider.simulate,
                    proj_name="mock",
                    sim_target="mock/target",
                    steps=10,
                    manager=viewer.manager,
                    logger=viewer.logger,
                )

    @patch("provider.room.os.path.exists")
    @patch("provider.room.shutil.copy")
    @patch("pybullet.connect")
    @patch("pybullet.loadURDF")
    @patch("pybullet.getNumJoints")
    @patch("pybullet.getJointInfo")
    @patch("pybullet.setJointMotorControl2")
    @patch("pybullet.stepSimulation")
    @patch("pybullet.disconnect")
    @patch("pybullet.setGravity")
    def test_show_simulation_custom_gravity(
        self,
        mock_set_gravity,
        mock_disconnect,
        mock_step_simulation,
        mock_set_motor_control,
        mock_get_joint_info,
        mock_get_num_joints,
        mock_load_urdf,
        mock_connect,
        mock_copy,
        mock_exists,
        viewer,
    ):
        """Verify that show_simulation configures gravity according to room.gravity."""
        mock_exists.return_value = True
        mock_connect.return_value = 42
        mock_load_urdf.return_value = 100
        mock_get_num_joints.return_value = 0

        room = Room()
        room.gravity = (1.0, 2.0, -3.0)
        parent = Box(1, 1, 1)
        p_shape = cast(URDFShape, parent)
        p_shape.urdf_label = "parent"
        room.add("parent", parent)

        mock_provider = MagicMock()
        room.simulate(
            provider_hooks=mock_provider.simulate,
            proj_name="mock",
            sim_target="mock/target",
            steps=10,
            manager=viewer.manager,
            logger=viewer.logger,
        )

        mock_set_gravity.assert_called_once_with(1.0, 2.0, -3.0, physicsClientId=42)

    @patch("view.show")
    @patch("view.Builder")
    @patch("view.Room.simulate")
    def test_show_view_no_gui_parameter(self, mock_simulate, mock_builder_class, mock_show, viewer):
        """Verify show_view passes spawn_viewer=False when no_gui=True is specified."""
        target_name = "p1/product:view/simulate"
        m1 = MagicMock(spec=TargetList, provider=viewer.manager.router.targets.provider, modes=[Mode.SIMULATE])
        m1.__iter__.return_value = iter([target_name])
        m1.__len__.return_value = 1
        m1.__getitem__.return_value = target_name
        m1.subassemblies = []

        viewer.target_parser.resolve = MagicMock(return_value=m1)
        viewer.manager.router.manifest = {target_name: {Section.VIEW: {}}}

        mock_room = Room()
        mock_geom = MagicMock(spec=Part)
        mock_room.add("geom", mock_geom, color=(1, 0, 0, 0.5))
        viewer.manager.router.run.return_value = [("product", mock_room)]

        viewer.show_view([target_name], no_gui=True, no_build=True)

        mock_simulate.assert_called_once()
        kwargs = mock_simulate.call_args[1]
        assert kwargs.get("spawn_viewer") is False

    def test_view_cli_no_gui(self):
        """Verify view.py CLI parses --no-gui flag correctly."""
        import sys
        from unittest.mock import patch
        from view import get_args

        test_args = ["view.py", "cat_fountain/product:view/simulate", "--no-gui"]
        with patch.object(sys, "argv", test_args):
            args = get_args()
            assert args.no_gui is True

        test_args_default = ["view.py", "cat_fountain/product:view/simulate"]
        with patch.object(sys, "argv", test_args_default):
            args = get_args()
            assert args.no_gui is False

    def test_view_cli_port(self):
        """Verify view.py CLI parses -p/--port correctly and rejects --rerun-port."""
        import sys
        from unittest.mock import patch
        from view import get_args

        # Test -p / --port parsing
        test_args = ["view.py", "cat_fountain/product:view/simulate", "-p", "1234"]
        with patch.object(sys, "argv", test_args):
            args = get_args()
            assert args.port == 1234

        test_args_long = ["view.py", "cat_fountain/product:view/simulate", "--port", "5678"]
        with patch.object(sys, "argv", test_args_long):
            args = get_args()
            assert args.port == 5678

        # Test that --rerun-port is consolidated/removed and raises an error
        test_args_removed = ["view.py", "cat_fountain/product:view/simulate", "--rerun-port", "1234"]
        with patch.object(sys, "argv", test_args_removed):
            with pytest.raises(SystemExit):
                get_args()

    def test_view_cli_stage_window(self):
        """Verify view.py CLI parses --stage-window and --stage-window-size correctly."""
        import sys
        from unittest.mock import patch
        from view import get_args

        test_args = ["view.py", "cat_fountain/product:view/simulate", "--stage-window", "1000"]
        with patch.object(sys, "argv", test_args):
            args = get_args()
            assert args.stage_window_size == 1000

        test_args_alias = ["view.py", "cat_fountain/product:view/simulate", "--stage-window-size", "2500"]
        with patch.object(sys, "argv", test_args_alias):
            args = get_args()
            assert args.stage_window_size == 2500

        test_args_alias2 = ["view.py", "cat_fountain/product:view/simulate", "--staging-frame-window-size", "500"]
        with patch.object(sys, "argv", test_args_alias2):
            args = get_args()
            assert args.stage_window_size == 500

    def test_view_cli_blender_flags(self):
        """Verify view.py CLI parses --view-from, --save-mp4, --fps, --resolution, and --samples."""
        import sys
        from unittest.mock import patch
        from view import get_args

        test_args = [
            "view.py",
            "cat_fountain/product",
            "--view-from",
            "front",
            "--save-mp4",
            "recordings/out.mp4",
            "--fps",
            "60",
            "--resolution",
            "3840x2160",
            "--samples",
            "64",
        ]
        with patch.object(sys, "argv", test_args):
            args = get_args()
            assert args.view_from == "front"
            assert args.save_mp4 == "recordings/out.mp4"
            assert args.fps == 60
            assert args.resolution == "3840x2160"
            assert args.samples == 64

        test_args_alias = [
            "view.py",
            "cat_fountain/product",
            "--export-mp4",
            "recordings/export.mp4",
        ]
        with patch.object(sys, "argv", test_args_alias):
            args = get_args()
            assert args.save_mp4 == "recordings/export.mp4"

    @patch("provider.blender.BlenderRenderer.render_turntable_to_mp4")
    def test_show_view_save_mp4_turntable(self, mock_render_turntable, viewer):
        """Verify show_view triggers turntable MP4 export via Blender when save_mp4 is passed on static target."""
        target_name = "tube/driver"
        mock_targets = MagicMock(spec=TargetList)
        mock_targets.__iter__.return_value = iter([target_name])
        mock_targets.__len__.return_value = 1
        mock_targets.subassemblies = []
        viewer.target_parser.resolve = MagicMock(side_effect=[None, mock_targets, None])
        viewer.manager.router.manifest = {target_name: {Section.PART: {}}}

        with patch("provider.room.Compound", side_effect=lambda children: MagicMock(children=children)):
            mock_geom = MagicMock(spec=Part, label=None)
            mock_geom.wrapped = "fake_wrapped"
            viewer.manager.router.run.return_value = [(target_name, mock_geom)]
            viewer.manager.router.get_color.return_value = (1, 0, 0, 1)

            viewer.show_view(
                [target_name],
                save_mp4="out.mp4",
                view_from="front",
                fps=60,
                resolution=(2560, 1440),
                samples=32,
                no_gui=True,
            )

            mock_render_turntable.assert_called_once()
            kwargs = mock_render_turntable.call_args[1]
            assert kwargs.get("output_mp4") == "out.mp4"
            assert kwargs.get("config").view_from == "front"
            assert kwargs.get("config").fps == 60
            assert kwargs.get("config").resolution == (2560, 1440)
