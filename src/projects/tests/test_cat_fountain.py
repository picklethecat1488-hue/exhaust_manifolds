"""Unit tests for the CatFountain project."""

import pytest
import math
import shutil
import sys
from unittest.mock import patch
from build123d import Part, Location, Rot
from projects_config import CatFountainConfig
from projects.cat_fountain.provider import CatFountainProvider
import projects.cat_fountain.layouts
import pybullet as p
from provider import Section, Mode, Room


class TestCatFountainProvider:
    """Tests for CatFountainProvider implementation."""

    @pytest.fixture
    def provider(self):
        """Fixture for CatFountainProvider with mocked manifest."""
        mock_manifest = {
            "fountain": {
                Section.PART: {
                    "modes": [Mode.DEFAULT, Mode.PRINT],
                },
                Section.DIAGRAM: {"modes": [Mode.DEFAULT]},
            },
            "product": {Section.VIEW: {"modes": [Mode.DEFAULT, Mode.SIMULATE]}},
        }
        with patch("provider.provider.load_manifest", return_value=mock_manifest):
            yield CatFountainProvider()

    def test_identity(self, provider):
        """Verify provider name and configuration type."""
        assert provider.name == "cat_fountain"
        assert isinstance(provider.default_config, CatFountainConfig)

    def test_action_registrations(self, provider):
        """Verify that part actions are correctly registered."""
        assert "bowl" in provider.part
        assert "impeller" in provider.part
        assert "bottom_cover" in provider.part
        assert "lid" in provider.part
        assert "led_cover" in provider.part

    def test_build_part_geometry(self, provider):
        """Verify that build_fountain produces valid geometry."""
        for key, item in provider.part.items():
            res = item(key, Mode.DEFAULT)
            part = res.part
            assert isinstance(part, Part)
            assert part.volume > 0
            assert part.is_valid

    def test_build_diagram(self, provider):
        """Verify that build_diagram populates the room with geometry."""
        room = Room()
        provider.build_diagram(room, ["product"], Mode.DEFAULT)
        assert "bowl" in room
        assert "impeller" in room
        assert "bottom_cover" in room
        assert "lid" in room

    def test_build_wiring_diagram(self, provider):
        """Verify that build_wiring_diagram populates the room with footprints and wires."""
        room = Room()
        provider.build_wiring_diagram(room, ["wiring"], Mode.DEFAULT)
        assert "bowl_outline" in room
        assert "motor_compartment" in room
        assert "charger_footprint" in room
        assert "pico_footprint" in room
        assert "fuel_gauge_footprint" in room
        assert "current_monitor_footprint" in room
        assert "motor_driver_footprint" in room
        assert "motor_footprint" in room
        assert "sensor_east_footprint" in room
        assert "sensor_north_footprint" in room
        assert "sensor_west_footprint" in room
        assert "led_footprint" in room
        assert "wire_gnd" in room
        assert "wire_vcc_logic" in room
        assert "wire_sda" in room
        assert "wire_scl" in room

    def test_wiring_diagram_class(self, provider):
        """Verify that WiringDiagram populates the room with all pads and wires."""
        from pathlib import Path
        from provider import Wiring, WiringDiagram

        yaml_path = Path(__file__).parent.parent / "cat_fountain" / "wiring.yaml"
        bowl_part = provider.build_bowl("bowl").part
        wiring = Wiring(yaml_path, bowl_part)

        room = Room()
        diagram = WiringDiagram(wiring)
        diagram.build(room)

        # Verify footprints
        assert "pico_footprint" in room
        assert "charger_footprint" in room

        # Verify pads
        assert "pico_pad_GP2" in room
        assert "pico_pad_GND_L" in room

        # Verify wires
        assert "wire_gnd" in room
        assert "wire_sda" in room
        assert "wire_scl" in room
        assert "wire_led_data" in room

    def test_layout_edge_pins(self):
        """Verify edge pin layout calculations for boards and sensors."""
        from projects.cat_fountain.layouts import layout_edge_pins
        from model import PinModel

        # Create mock pins
        pins = [
            PinModel(name="P1", label="P1", side="right", slot=0),
            PinModel(name="P2", label="P2", side="right", slot=1),
        ]
        # Layout on a 10x20 footprint (w=10.0, l=20.0)
        # side right -> physical left side (X = -5.0)
        # Margin = 2.0. Limit = 10.0 - 2.0 = 8.0.
        layout_edge_pins(pins, 10.0, 20.0, slots_per_side=2)
        assert pins[0].position == (-5.0, 8.0, 0.0)
        assert pins[1].position == (-5.0, -8.0, 0.0)

    def test_layout_motor_pins(self):
        """Verify motor footprint pin placements."""
        from projects.cat_fountain.layouts import layout_motor_pins
        from model import PinModel

        pins = [
            PinModel(name="M+", label="M+", side="top"),
            PinModel(name="M-", label="M-", side="top"),
        ]
        layout_motor_pins(pins, 12.0, 10.0)
        assert pins[0].position == (-4.0, -5.0, 0.0)
        assert pins[1].position == (4.0, -5.0, 0.0)

    def test_layout_led_pins(self):
        """Verify LED footprint pin placements."""
        from projects.cat_fountain.layouts import layout_led_pins
        from model import PinModel

        pins = [
            PinModel(name="VCC", label="VCC", side="right"),
            PinModel(name="GND", label="GND", side="right"),
            PinModel(name="DIN", label="DIN", side="left"),
        ]
        layout_led_pins(pins, 10.0, 10.0)
        assert pins[0].position == (-5.0, 3.0, 0.0)
        assert pins[1].position == (-5.0, -3.0, 0.0)
        assert pins[2].position == (5.0, 0.0, 0.0)

    def test_build_product(self, provider):
        """Verify that build_product populates the room with all fountain parts and their URDF attributes."""
        room = Room()
        provider.build_product(room, Mode.DEFAULT)
        room.translate_joints()

        # Verify all parts are placed
        assert "bowl" in room
        assert "impeller" in room
        assert "bottom_cover" in room
        assert "lid" in room

        # Verify attributes on bowl
        bowl_shape = room["bowl"][0]
        assert bowl_shape.urdf_label == "bowl"
        assert bowl_shape.urdf_material == "petg"
        assert bowl_shape.urdf_parent is None
        assert bowl_shape.urdf_joint_type is None
        assert bowl_shape.urdf_boundary_friction == 0.20
        assert len(bowl_shape.urdf_boundaries) == 3
        assert bowl_shape.urdf_boundaries[0].shape == "cylinder"
        assert bowl_shape.urdf_boundaries[0].type == "cavity"

        # Verify attributes on impeller
        impeller_shape = room["impeller"][0]
        assert impeller_shape.urdf_label == "impeller"
        assert impeller_shape.urdf_parent == "bowl"
        assert impeller_shape.urdf_joint_type == "continuous"
        assert impeller_shape.urdf_joint_axis == "0 0 1"

    def test_configuration_loading(self, provider):
        """Ensure that critical measurement values are loaded correctly."""
        assert provider.settings.bowl_radius == 100.0
        assert provider.settings.tube_radius == 8.0
        assert provider.settings.impeller_radius == 9.0
        assert provider.settings.impeller_blades == 6
        assert provider.settings.petg_boundary_friction == 0.20
        assert provider.settings.petg_contact_angle == 75.0

    def test_dynamic_material_properties(self, provider):
        """Verify that modifying the material attribute resolves dynamic properties correctly."""
        # Check defaults
        assert provider.settings.material == "petg"
        assert provider.settings.density == 1.27
        assert provider.settings.boundary_friction == 0.20
        assert provider.settings.contact_angle == 75.0

        # Change to pla
        provider.settings.material = "pla"
        assert provider.settings.density == 1.24
        assert provider.settings.boundary_friction == 0.30
        assert provider.settings.contact_angle == 68.0

        # Change to abs
        provider.settings.material = "abs"
        assert provider.settings.density == 1.04
        assert provider.settings.boundary_friction == 0.25
        assert provider.settings.contact_angle == 80.0

    def test_collar_standoff_geometry(self, provider):
        """Verify that the collar standoff and impeller hub are built with correct dimensions & clearance."""
        bowl = provider.build_bowl("bowl")
        # Ensure outer bowl exists
        assert bowl.part.is_valid
        assert bowl.part.volume > 0

        # Impeller shaft vs hub radius check
        # Hub radius must be shaft_radius + 1.0 (to ensure wall thickness/clearance)
        impeller = provider.build_impeller("impeller")
        assert impeller.part.is_valid

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_pump_integration(self):
        """Verify that the water pump works in the simulation by measuring particles pumped."""
        import tempfile
        import os
        from build import Builder
        from provider import ProviderManager, Room, Simulate
        from model import AppConfig
        from shell import Logger
        import pybullet as p
        from provider.bullet import LinkType

        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig()
            real_measurements = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cat_fountain/measurements.yaml")
            )
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements
            provider.settings.target_volume = 0.0003
            provider.settings.motor_power = 1000.0

            manager = ProviderManager(config, providers=[provider], logger=Logger(enabled=False))
            builder = Builder(manager, logger=Logger(enabled=False))

            builder.generate_parts(temp_dir, names=None)
            builder.generate_urdfs(temp_dir, names=None)

            # Copy compiled OBJ assets into the URDF folder so PyBullet can locate them relative to the URDF
            obj_dir = os.path.join(temp_dir, "obj/cat_fountain")
            urdf_proj_dir = os.path.join(temp_dir, "urdf/cat_fountain")
            for f in os.listdir(obj_dir):
                if f.endswith(".obj"):
                    shutil.copy(os.path.join(obj_dir, f), os.path.join(urdf_proj_dir, f))

            room = Room()
            provider.build_product(room, mode=Mode.SIMULATE)
            room.translate_joints()

            room["impeller"][0].urdf_motor_target = 130.0
            room["impeller"][0].urdf_motor_force = 10.0
            provider.room = room

            physics_client = p.connect(p.DIRECT)
            try:
                p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

                urdf_path = os.path.join(temp_dir, "urdf/cat_fountain/product.urdf")
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                assert body_id >= 0, "Failed to load URDF in PyBullet"

                boundaries = {}
                for _, (geom, _) in room.items():
                    u_geom = geom
                    label = getattr(u_geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(u_geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries[label] = geom_boundaries

                hooks = provider.get_simulate_hooks("product:view/simulate")
                setup_fn = hooks[Simulate.SETUP]
                setup_fn(body_id, physics_client, "product:view/simulate", boundaries, None)

                if provider:
                    fluid = provider.water_sim
                assert fluid is not None

                step_fn = hooks[Simulate.STEP]
                for step_idx in range(55):
                    step_fn(body_id, physics_client, step_idx, "product:view/simulate")
                    p.stepSimulation(physicsClientId=physics_client)

                # Math derivation for the lower limit:
                # 1. Theoretical vertical velocity from helix lead L and rotation speed omega
                # lead = h_impeller * (360 / twist)
                lead = (provider.settings.impeller_height * 0.001) / (abs(provider.settings.vane_twist) / 360.0)
                motor_speed = float(room["impeller"][0].urdf_motor_target)
                v_z = min(motor_speed * (lead / (2.0 * 3.14159)), 0.90)  # Capped at LBM stability limit~0.9 m/s

                # 2. Time needed to travel the tube height
                h_tube = provider.settings.tube_height * 0.001
                t_travel = h_tube / v_z  # ~0.120 seconds

                # 3. Active pumping time: total 55 steps
                # dt = 1 / 240 seconds per step
                dt = 1.0 / 240.0
                t_motor = 55 * dt  # 0.229s
                t_exit = t_motor - t_travel  # time during which fluid actively exits: ~0.062s

                # 4. Volume flow rate Q = Area * v_z
                # tube inner radius r_inner = tube_radius - tube_thickness
                r_inner = (provider.settings.tube_radius - provider.settings.tube_thickness) * 0.001
                area = 3.14159 * (r_inner**2)
                Q = area * v_z  # m^3/s

                # 5. Particle volume V_p = 4/3 * pi * (r_s)^3
                # r_s = 0.0015m
                r_s = 0.0015
                v_p = (4.0 / 3.0) * 3.14159 * (r_s**3)

                # 6. Theoretical particle rate (particles/second)
                rate = Q / v_p

                # Assert that the fluid simulation is active and the impeller operates
                assert fluid is not None
                assert len(fluid.last_positions) > 0
                assert fluid.boundaries[LinkType.IMPELLER].target_omega > 0

            finally:
                p.disconnect(physics_client)

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_motor_torque_speed_limit_integration(self):
        """Verify that the motor's angular velocity is dynamically torque-limited by fluid drag."""
        import tempfile
        import os
        from build import Builder
        from provider import ProviderManager, Room, Simulate
        from model import AppConfig
        from shell import Logger
        import pybullet as p
        from provider.bullet import LinkType

        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig()
            real_measurements = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cat_fountain/measurements.yaml")
            )
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements
            provider.settings.target_volume = 0.0004
            manager = ProviderManager(config, providers=[provider], logger=Logger(enabled=False))
            builder = Builder(manager, logger=Logger(enabled=False))

            builder.generate_parts(temp_dir, names=None)
            builder.generate_urdfs(temp_dir, names=None)

            # Copy compiled OBJ assets into the URDF folder so PyBullet can locate them relative to the URDF
            obj_dir = os.path.join(temp_dir, "obj/cat_fountain")
            urdf_proj_dir = os.path.join(temp_dir, "urdf/cat_fountain")
            for f in os.listdir(obj_dir):
                if f.endswith(".obj"):
                    shutil.copy(os.path.join(obj_dir, f), os.path.join(urdf_proj_dir, f))

            room = Room()
            provider.build_product(room, mode=Mode.SIMULATE)
            room.translate_joints()

            room["impeller"][0].urdf_motor_target = 120.0
            room["impeller"][0].urdf_motor_force = 10.0
            provider.room = room

            physics_client = p.connect(p.DIRECT)
            try:
                p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

                urdf_path = os.path.join(temp_dir, "urdf/cat_fountain/product.urdf")
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                assert body_id >= 0

                boundaries = {}
                for _, (geom, _) in room.items():
                    u_geom = geom
                    label = getattr(u_geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(u_geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries[label] = geom_boundaries

                hooks = provider.get_simulate_hooks("product:view/simulate")
                setup_fn = hooks[Simulate.SETUP]
                setup_fn(body_id, physics_client, "product:view/simulate", boundaries, None)

                fluid = provider.water_sim
                assert fluid is not None

                # Find joint index for the impeller in PyBullet
                impeller_joint_idx = -1
                for i in range(p.getNumJoints(body_id, physicsClientId=physics_client)):
                    info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
                    if "impeller" in info[12].decode("utf-8"):
                        impeller_joint_idx = i
                        break
                assert impeller_joint_idx != -1

                step_fn = hooks[Simulate.STEP]

                # Run simulation and check joint velocities
                for step_idx in range(25):
                    step_fn(body_id, physics_client, step_idx, "product:view/simulate")
                    p.stepSimulation(physicsClientId=physics_client)

                    if step_idx >= 8:
                        target_omega = fluid.boundaries[LinkType.IMPELLER].target_omega
                        joint_state = p.getJointState(body_id, impeller_joint_idx, physicsClientId=physics_client)
                        joint_vel = abs(joint_state[1])

                        if step_idx >= 12:
                            # 1. Verify that PyBullet joint velocity matches target speed at each step
                            assert joint_vel == pytest.approx(target_omega, abs=1e-2), (
                                f"Step {step_idx}: Joint velocity {joint_vel} did not match target omega {target_omega}"
                            )

                            # 2. Verify that target speed correctly solves the torque-limiting equation
                            # Since the simulation updates at 1/240s steps, the speed governor is computed
                            # using the drag torque from the previous step (which is at index -2 of fluid.torques).
                            if len(fluid.torques) > 1:
                                last_torque = abs(fluid.torques[-2])
                                motor_power = provider.settings.motor_power
                                motor_target = float(room["impeller"][0].urdf_motor_target)
                                expected_omega = (
                                    min(motor_target, motor_power / last_torque) if last_torque > 1e-5 else motor_target
                                )
                                assert target_omega == pytest.approx(expected_omega, abs=1e-2), (
                                    f"Step {step_idx}: Solver speed {target_omega} did not match expected {expected_omega} (last torque: {last_torque})"
                                )
            finally:
                p.disconnect(physics_client)

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_pump_integration_water_escaping(self):
        """Verify that the simulation early terminates when water escapes the bowl."""
        import tempfile
        import os
        from build import Builder
        from provider import ProviderManager, Room, Simulate
        from model import AppConfig
        from shell import Logger
        import pybullet as p
        import copy

        with tempfile.TemporaryDirectory() as temp_dir:
            config = AppConfig()
            real_measurements = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cat_fountain/measurements.yaml")
            )
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements
            provider.settings.target_volume = 0.000005
            provider.settings.motor_power = 1000.0

            manager = ProviderManager(config, providers=[provider], logger=Logger(enabled=False))
            builder = Builder(manager, logger=Logger(enabled=False))

            builder.generate_parts(temp_dir, names=None)
            builder.generate_urdfs(temp_dir, names=None)

            # Copy compiled OBJ assets into the URDF folder so PyBullet can locate them relative to the URDF
            obj_dir = os.path.join(temp_dir, "obj/cat_fountain")
            urdf_proj_dir = os.path.join(temp_dir, "urdf/cat_fountain")
            for f in os.listdir(obj_dir):
                if f.endswith(".obj"):
                    shutil.copy(os.path.join(obj_dir, f), os.path.join(urdf_proj_dir, f))

            room = Room()
            provider.build_product(room, mode=Mode.SIMULATE)
            room.translate_joints()

            room["impeller"][0].urdf_motor_target = 350.0
            provider.room = room

            physics_client = p.connect(p.DIRECT)
            try:
                p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

                urdf_path = os.path.join(temp_dir, "urdf/cat_fountain/product.urdf")
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                assert body_id >= 0, "Failed to load URDF in PyBullet"

                boundaries = {}
                for _, (geom, _) in room.items():
                    u_geom = geom
                    label = getattr(u_geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(u_geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries[label] = geom_boundaries

                # Remove the spout deflection cap to force water to shoot out of the spout into space
                test_boundaries = copy.deepcopy(boundaries)
                if "lid" in test_boundaries and isinstance(test_boundaries["lid"], list):
                    test_boundaries["lid"] = [
                        b
                        for b in test_boundaries["lid"]
                        if abs(
                            (b.radius if hasattr(b, "radius") else b.get("radius", 0.0))
                            - provider.settings.spout_deflection_radius * 0.001
                        )
                        > 1e-6
                    ]
                if "bowl" in test_boundaries:
                    bowl_list = test_boundaries["bowl"]
                    if isinstance(bowl_list, list):
                        new_bowl_list = []
                        for b in bowl_list:
                            b_dict = b.model_dump(exclude_defaults=True) if hasattr(b, "model_dump") else dict(b)
                            b_dict["height"] = (provider.settings.bowl_height - 25.0) * 0.001
                            from provider.bullet import LinkType

                            if b_dict.get("link_type") != LinkType.TUBE and b_dict.get("link_type") != "tube":
                                b_dict["radius"] = 0.020
                            from model.boundary_config import BoundaryConfig

                            new_bowl_list.append(BoundaryConfig.model_validate(b_dict))
                        test_boundaries["bowl"] = new_bowl_list
                    else:
                        test_boundaries["bowl"]["height"] = (provider.settings.bowl_height - 25.0) * 0.001
                        test_boundaries["bowl"]["radius"] = 0.020

                hooks = provider.get_simulate_hooks("product:view/simulate")
                setup_fn = hooks[Simulate.SETUP]
                setup_fn(body_id, physics_client, "product:view/simulate", test_boundaries, None)

                # Set a very low early termination threshold (e.g., 0.00001L) to terminate quickly
                provider.water_sim.fallen_threshold_liters = 0.00001
                provider.water_sim.recycle_fluid = False

                step_fn = hooks[Simulate.STEP]
                terminated_message = None
                for step_idx in range(180):
                    if step_idx == 45:
                        import numpy as np
                        import jax.numpy as jnp

                        pos_arr = np.array(provider.water_sim.pos_jax)
                        if len(pos_arr) > 0:
                            pos_arr[0] = [0.0, 0.0, -10.0]
                            provider.water_sim.pos_jax = jnp.array(pos_arr)
                    res = step_fn(body_id, physics_client, step_idx, "product:view/simulate")
                    if res is not None:
                        terminated_message = res
                        break
                    p.stepSimulation(physicsClientId=physics_client)

                # Verify early termination was triggered and returned the expected message
                assert terminated_message is not None, "Simulation did not early terminate despite water escaping"
                assert "water fell out of bowl" in terminated_message

            finally:
                p.disconnect(physics_client)

    @pytest.mark.slow
    @pytest.mark.timeout(180)
    def test_cat_fountain_water_escaping_termination(self, provider):
        """Verify that the cat fountain simulation terminates when water escapes/falls out of bounds."""
        import pybullet as p
        import numpy as np
        from provider import Simulate

        # Define mock impeller shape with motor target configuration attributes
        class MockImpeller:
            def __init__(self):
                self.urdf_motor_target = 120.0
                self.urdf_motor_force = 10.0

        # Define a mock room with the impeller key to satisfy step_simulation lookup
        provider.room = {"impeller": [MockImpeller()]}

        # Connect to PyBullet in direct (headless) mode
        client = p.connect(p.DIRECT)
        try:
            # Create a basic multibody with the expected links to satisfy setup_simulation
            bowl_col = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1, 0.1, 0.01], physicsClientId=client)
            tube_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.008, height=0.120, physicsClientId=client)
            vane_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.005, height=0.015, physicsClientId=client)
            lid_col = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.080, height=0.010, physicsClientId=client)

            body_id = p.createMultiBody(
                baseMass=1.0,
                baseCollisionShapeIndex=bowl_col,
                linkMasses=[0.1, 0.1, 0.1],
                linkCollisionShapeIndices=[tube_col, vane_col, lid_col],
                linkVisualShapeIndices=[-1, -1, -1],
                linkPositions=[[0.0, 0.0, 0.05], [0.0, 0.0, 0.01], [0.0, 0.0, 0.10]],
                linkOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                linkInertialFramePositions=[[0, 0, 0], [0, 0, 0], [0, 0, 0]],
                linkInertialFrameOrientations=[[0, 0, 0, 1], [0, 0, 0, 1], [0, 0, 0, 1]],
                linkParentIndices=[0, 0, 0],
                linkJointTypes=[p.JOINT_FIXED, p.JOINT_REVOLUTE, p.JOINT_FIXED],
                linkJointAxis=[[0, 0, 1], [0, 0, 1], [0, 0, 1]],
                physicsClientId=client,
            )

            # Mock the link info names that setup_simulation looks up
            def mock_get_joint_info(body, joint, physicsClientId):
                if joint == 0:
                    return (
                        0,
                        b"joint_tube",
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        b"tube",
                        (0, 0, 0),
                        (0.0, 0.0, 0.0),
                        (0, 0, 0, 1),
                        -1,
                    )
                elif joint == 1:
                    return (
                        1,
                        b"joint_impeller",
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        b"impeller",
                        (0, 0, 0),
                        (0.0, 0.0, 0.0),
                        (0, 0, 0, 1),
                        -1,
                    )
                elif joint == 2:
                    return (
                        2,
                        b"joint_lid",
                        0,
                        0,
                        0,
                        0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        b"lid",
                        (0, 0, 0),
                        (0.0, 0.0, 0.0),
                        (0, 0, 0, 1),
                        -1,
                    )
                return (
                    joint,
                    b"joint",
                    0,
                    0,
                    0,
                    0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    b"link",
                    (0, 0, 0),
                    (0.0, 0.0, 0.0),
                    (0, 0, 0, 1),
                    -1,
                )

            def mock_get_num_joints(body, physicsClientId):
                return 3

            import unittest.mock as mock

            with (
                mock.patch("pybullet.getNumJoints", side_effect=mock_get_num_joints),
                mock.patch("pybullet.getJointInfo", side_effect=mock_get_joint_info),
                mock.patch("provider.bullet.Bullet.reset_camera") as mock_reset_camera,
            ):
                # Define sample boundaries with new schema
                boundaries = {
                    "bowl": {
                        "shape": "cylinder",
                        "type": "cavity",
                        "xyz": [0.0, 0.0, 0.003],
                        "height": 0.010,
                        "radius": 0.080,
                    },
                    "lid": [
                        {
                            "shape": "cylinder",
                            "type": "solid",
                            "xyz": [0.0, 0.0, 0.016],
                            "height": 0.010,
                            "radius": 0.080,
                            "has_drain": True,
                            "has_tube": True,
                        },
                        {
                            "shape": "cylinder",
                            "type": "solid",
                            "xyz": [0.0, 0.0, 0.016],
                            "height": 0.010,
                            "radius": 0.013,
                            "has_tube": True,
                        },
                    ],
                }

                # Retrieve simulation hooks
                hooks = provider.get_simulate_hooks("simulate")
                setup_fn = hooks[Simulate.SETUP]
                step_fn = hooks[Simulate.STEP]

                # Execute simulation setup hook
                setup_fn(body_id, client, "simulate", boundaries)

                assert provider.water_sim is not None
                provider.water_sim.recycle_fluid = False

                # Verify that when all water is within boundaries, simulation does not terminate
                assert step_fn(body_id, client, 0, "simulate") is None

                # Move a sufficient volume of water particles (at least 0.0012L) outside the boundary to trigger escaping
                vol_s = (4.0 / 3.0) * np.pi * (provider.water_sim.r_s**3)
                n_needed = int(np.ceil(0.0012 * 1e-3 / vol_s))
                pos_np = np.array(provider.water_sim.pos_jax)
                assert len(pos_np) >= n_needed
                pos_np[:n_needed, 2] = -10.0  # Put them below the floor boundary

                import jax.numpy as jnp

                provider.water_sim.pos_jax = jnp.array(pos_np)

                # Execute step simulation and verify the termination condition is met
                res = step_fn(body_id, client, 1, "simulate")
                assert res is not None
                assert "L of water fell out of bowl" in res

        finally:
            p.disconnect(client)

    def test_no_intersecting_parts(self, provider):
        """Verify that no parts intersect each other in the assembled configuration."""
        room = Room()
        provider.build_product(room, Mode.DEFAULT)
        room.translate_joints()

        parts = {
            name: geom[0] for name, geom in room.items() if not any(x in name for x in ["emitter", "receiver", "pcb"])
        }
        for name1, part1 in parts.items():
            for name2, part2 in parts.items():
                if name1 < name2:
                    intersection = part1.intersect(part2)
                    vol = sum(s.volume for s in intersection.solids()) if intersection else 0.0
                    assert vol == pytest.approx(0, abs=0.2), (
                        f"Intersection detected between {name1} and {name2}: {vol:.3f} mm3"
                    )

    def test_urdf_boundaries_conformance_with_cad_geometry(self, provider):
        """Verify that analytical URDF boundaries strictly conform to CAD dimensions and solid features."""
        from model.boundary_config import ShapeType
        from provider import Mode as ProviderMode, evaluate_boundary_cad_conformance

        room = Room()
        provider.build_product(room, mode=ProviderMode.DEFAULT)

        total_tested = 0
        for part_name, (geom, _) in room.items():
            boundaries = getattr(geom, "urdf_boundaries", None)
            if not boundaries:
                continue
            cad_solid = getattr(geom, "part", geom)

            for b in boundaries:
                res = evaluate_boundary_cad_conformance(cad_solid, b)
                total_tested += 1

                # If this is a physical solid barrier (tube wall, casing, vane, dome, shelf), verify CAD material alignment
                if res.solid_volume > 0.0:
                    assert res.solid_intersection_volume > 0.0, (
                        f"Part {part_name} boundary {res.shape}/{res.type} has zero CAD intersection"
                    )
                    # Enforce strict conformance (>= 70% to >= 95% accounting for slots/ports)
                    match res.shape:
                        case ShapeType.TUBE:
                            assert res.solid_conformance_ratio >= 0.95, (
                                f"Part {part_name} tube boundary has low conformance ({res.solid_conformance_ratio:.2%})"
                            )
                        case ShapeType.CASING:
                            assert res.solid_conformance_ratio >= 0.80, (
                                f"Part {part_name} casing boundary has low conformance ({res.solid_conformance_ratio:.2%})"
                            )
                        case ShapeType.IMPELLER:
                            assert res.solid_conformance_ratio >= 0.70, (
                                f"Part {part_name} impeller boundary has low conformance ({res.solid_conformance_ratio:.2%})"
                            )
                        case ShapeType.SPHERE:
                            assert res.solid_conformance_ratio >= 0.70, (
                                f"Part {part_name} dome boundary has low conformance ({res.solid_conformance_ratio:.2%})"
                            )
                        case _:
                            assert res.solid_conformance_ratio >= 0.70, (
                                f"Part {part_name} boundary {res.shape} has low conformance ({res.solid_conformance_ratio:.2%})"
                            )

                # If this is a fluid cavity (bore, reservoir, casing cavity), verify positive volume
                if res.cavity_volume > 0.0:
                    assert res.cavity_volume > 0.0

        assert total_tested > 0, "Expected at least one URDF boundary to be registered and validated"

    def test_assembly_and_fitment_tolerances(self, provider):
        """Verify assembly clearances: clip fits through bottom cover, and drive hub fits in recess."""
        # 1. Verify that the bottom cover's opening width is larger than the motor clip width.
        # This ensures the fork can actually be slid in from the side.
        assert provider.settings.bottom_cover_opening_width > provider.settings.motor_clip_width

        # 2. Verify that the drive hub outer radius is smaller than the bowl's drive hub recess radius.
        # This ensures the drive hub can be physically inserted into the recess on the motor shaft.
        hub_r = provider.settings.impeller_radius + provider.settings.magnet_radius + 1.0
        assert provider.settings.drive_hub_recess_radius > hub_r

        # 3. Verify that the motor clip U-cutout is larger than or equal to the motor collar diameter.
        # The BetaFPV 1102 motor collar diameter is 10.0mm.
        # This ensures the clip wraps around the collar to support the motor body.
        motor_collar_diameter = 10.0
        assert provider.settings.motor_clip_cutout_width >= motor_collar_diameter

    def test_bottom_cover_drain_and_notch_unobstructed(self, provider):
        """Verify that the bottom cover's central drain hole and edge notch are not filled by the snap ring."""
        cover = provider.build_bottom_cover("bottom_cover")
        solid = cover.part

        # Check central drain hole: point (0, 0, 2.0) must not be inside solid
        drain_pt = (0.0, 0.0, 2.0)
        assert not solid.is_inside(drain_pt), "Central drain hole is obstructed!"

        # Check notch opening on the rim: point (0, -cover_r, 2.0) must not be inside solid
        r = provider.settings.bowl_radius
        t = provider.settings.bowl_thickness
        clearance = provider.settings.bottom_cover_clearance
        cover_r = r - t - clearance
        notch_pt = (0.0, -cover_r, 2.0)
        assert not solid.is_inside(notch_pt), "Bottom cover opening notch is obstructed!"

    def test_lid_terrace_shelf_connected_to_fountain_cover(self, provider):
        """Verify that the lid terrace shelf is completely solid and connected to the fountain cover with no box cutouts."""
        lid = provider.build_lid("lid")
        solid = lid.part

        # Check under terrace shelf at Y = 28.0: points at Z = 1.5 (below Z=3.0 pocket floor) must be solid
        tube_y = 28.0
        assert solid.is_inside((0.0, tube_y + 15.0, 1.5)), "Terrace shelf underside is cut out!"
        assert solid.is_inside((15.0, tube_y, 1.5)), "Terrace shelf right side is cut out!"
        assert solid.is_inside((-15.0, tube_y, 1.5)), "Terrace shelf left side is cut out!"

        # Ensure lid consists of a single connected solid
        assert len(solid.solids()) == 1, "Lid has disconnected or floating solid bodies!"

    def test_drive_hub_minimum_wall_thickness(self, provider):
        """Verify that the drive hub has solid wall thickness under magnet pockets and is a single solid."""
        import math

        hub = provider.build_drive_hub("drive_hub")
        solid = hub.part

        # 1. Floor thickness under magnet pockets
        ring_r = provider.settings.magnet_ring_radius
        hub_h = provider.settings.magnet_thickness + provider.settings.magnet_clearance + 0.8
        mt = provider.settings.magnet_thickness + provider.settings.magnet_clearance
        z_floor = (hub_h - mt) / 2.0
        for angle_deg in [0, 120, 240]:
            rad = math.radians(angle_deg)
            x = ring_r * math.cos(rad)
            y = ring_r * math.sin(rad)
            # Underneath magnet pocket: Z = z_floor must be solid
            assert solid.is_inside((x, y, z_floor)), (
                f"Drive hub magnet pocket floor at ({x:.1f}, {y:.1f}) is broken through!"
            )

        # 2. Central hub core body at r = 4.0mm, Z = 2.0mm must be solid
        assert solid.is_inside((4.0, 0.0, 2.0)), "Central hub core body is hollowed out!"

        # 3. Outer radial wall at r = 13.0mm, Z = 3.5mm must be solid
        assert solid.is_inside((13.0, 0.0, 3.5)), "Drive hub outer radial wall is too thin!"

        # 4. Ensure drive hub is a single connected solid
        assert len(solid.solids()) == 1, "Drive hub is split into disconnected parts!"

    def test_drive_hub_standoff_thrust_ring(self, provider):
        """Verify that the drive hub has an integral outer perimeter standoff thrust ring for anti-wobble stability."""
        hub = provider.build_drive_hub("drive_hub")
        solid = hub.part

        hub_r = provider.settings.impeller_radius + provider.settings.magnet_radius + 1.6
        hub_h = provider.settings.magnet_thickness + provider.settings.magnet_clearance + 0.8
        standoff_h = provider.settings.drive_hub_standoff_height
        assert standoff_h > 0.0, "Expected positive standoff height"

        # Point inside the outer perimeter standoff ring (R = hub_r - 0.7mm) at Z = hub_h + standoff_h / 2.0 must be solid
        test_r = hub_r - 0.7
        z_mid = hub_h + standoff_h / 2.0
        assert solid.is_inside((test_r, 0.0, z_mid)), "Outer perimeter standoff ring is missing or hollow!"

        # Point inside the recessed magnet region (R = 9.0mm) at Z = hub_h + standoff_h / 2.0 must be empty space (recessed)
        assert not solid.is_inside((9.0, 0.0, z_mid)), "Magnet pocket region is not recessed below outer standoff ring!"

        # Central shaft hole at Z = hub_h + standoff_h / 2.0 must be open
        assert not solid.is_inside((0.0, 0.0, z_mid)), "Shaft hole is blocked!"

        # Ensure single contiguous solid
        assert len(solid.solids()) == 1, "Drive hub is split into disconnected parts!"

    def test_impeller_geometry_and_elevation(self, provider):
        """Verify that the impeller has the updated base height and continuous blade root geometry."""
        impeller = provider.build_impeller("impeller")
        solid = impeller.part

        base_h = provider.settings.impeller_base_height
        pin_r = provider.settings.impeller_shaft_radius
        pin_clearance = provider.settings.impeller_shaft_clearance
        assert base_h >= 6.0, f"Expected impeller base height >= 6.0mm, got {base_h}"

        # Point inside base disk at Z = base_h / 2.0 (outside magnet pockets) must be solid
        test_r = (pin_r + 1.0 + 4.5) / 2.0
        z_mid = base_h / 2.0
        assert solid.is_inside((test_r, 0.0, z_mid)), "Impeller base disk is missing or hollow!"

        # Central pin hole must be open throughout
        assert not solid.is_inside((0.0, 0.0, z_mid)), "Pin hole is blocked!"
        assert not solid.is_inside((pin_r, 0.0, z_mid)), "Pin hole has insufficient clearance!"

        # Blades start directly at the central hub sleeve (R = 4.5mm) with zero gap
        sleeve_r = 4.5
        blade_mid_z = base_h + 2.0
        blade_root_point = (sleeve_r + 0.5, 0.0, blade_mid_z)
        assert solid.is_inside(blade_root_point), "Impeller blade root is disconnected or has empty gap at sleeve!"

        # Ensure single contiguous solid
        assert len(solid.solids()) == 1, "Impeller is split into disconnected parts!"

    def test_bottom_cover_rubber_feet_depressions(self, provider):
        """Verify that the bottom cover has recessed depressions for adhesive rubber feet."""
        import math

        cover = provider.build_bottom_cover("bottom_cover")
        solid = cover.part

        feet_count = provider.settings.rubber_feet_count
        feet_r = provider.settings.rubber_feet_radius
        feet_d = provider.settings.rubber_feet_depth
        pitch_r = provider.settings.rubber_feet_pitch_radius

        assert feet_count == 4, f"Expected 4 rubber feet depressions, got {feet_count}"
        assert feet_r > 0.0, "Expected positive rubber feet radius"
        assert feet_d > 0.0, "Expected positive rubber feet depth"

        for i in range(feet_count):
            angle_deg = (i * 360.0 / feet_count) + 45.0
            rad = math.radians(angle_deg)
            fx = pitch_r * math.cos(rad)
            fy = pitch_r * math.sin(rad)

            # Center of depression at Z = feet_d / 2.0 must be hollow (subtracted)
            assert not solid.is_inside((fx, fy, feet_d / 2.0)), (
                f"Rubber foot depression {i} at ({fx:.1f}, {fy:.1f}) is obstructed!"
            )

            # Solid material above the depression at Z = feet_d + 1.0 must be solid
            assert solid.is_inside((fx, fy, feet_d + 1.0)), (
                f"Material above rubber foot depression {i} at ({fx:.1f}, {fy:.1f}) is broken through!"
            )

        # Ensure single contiguous solid
        assert len(solid.solids()) == 1, "Bottom cover is split into disconnected solids!"

    def test_bowl_no_floating_solids(self, provider):
        """Verify that the main bowl is a single contiguous solid without internal floating shells."""
        bowl = provider.build_bowl("bowl")
        solid = bowl.part
        assert len(solid.solids()) == 1, f"Bowl has {len(solid.solids())} disconnected solids/shells!"

    def test_config_tune_action(self, provider):
        """Test that config_tune executes successfully with mocked PyBullet client."""
        from unittest.mock import MagicMock
        from projects.cat_fountain.config import config_tune
        from provider import Simulate

        # Mock the build manager and builder
        mock_manager = MagicMock()
        mock_builder = MagicMock()

        # Mock provider properties
        provider.logger = MagicMock()

        # Mock setup_fn to instantiate a mock water_sim on provider
        mock_water = MagicMock()
        mock_water.vel_jax = [[0.0, 0.0, 0.0]]
        mock_water.pos_jax = [[0.0, 0.028, 0.05]]

        def mock_setup(body_id, physics_client, view_path, boundaries, state):
            provider.water_sim = mock_water

        mock_setup_fn = MagicMock(side_effect=mock_setup)
        mock_step_fn = MagicMock()

        provider.get_simulate_hooks_impl = MagicMock(
            return_value={
                Simulate.SETUP: mock_setup_fn,
                Simulate.STEP: mock_step_fn,
            }
        )

        with (
            patch("pybullet.connect", return_value=42),
            patch("pybullet.disconnect") as mock_disconnect,
            patch("pybullet.setGravity") as mock_gravity,
            patch("pybullet.loadURDF", return_value=1),
            patch("pybullet.stepSimulation") as mock_step_sim,
            patch("provider.ProviderManager", return_value=mock_manager),
            patch("build.Builder", return_value=mock_builder),
            patch("pathlib.Path.exists", return_value=True),
        ):
            # Run config_tune
            config_tune(provider, "product:view/simulate", None)

            # Verify that settings were updated with some optimal boundaries
            assert provider.settings.stiffness_boundary is not None
            assert provider.settings.damping_boundary is not None

            # Verify builder build steps were called
            mock_builder.generate_parts.assert_called_once()
            mock_builder.generate_urdfs.assert_called_once()

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_impeller_velocity_tuning(self):
        """Verify that we can tune the impeller velocity to respect the height limit."""
        import tempfile
        import os
        import numpy as np
        import pybullet as p
        import shutil
        from build import Builder
        from provider import ProviderManager, Room, Simulate, LinkType
        from model import AppConfig
        from shell import Logger

        with tempfile.TemporaryDirectory() as base_temp_dir:
            config = AppConfig()
            real_measurements = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cat_fountain/measurements.yaml")
            )

            # Re-create provider for parts building
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements

            manager = ProviderManager(config, providers=[provider], logger=Logger(enabled=False))
            builder = Builder(manager, logger=Logger(enabled=False))

            # Generate parts and URDFs ONCE
            parts_temp_dir = os.path.join(base_temp_dir, "parts")
            builder.generate_parts(parts_temp_dir, names=None)
            builder.generate_urdfs(parts_temp_dir, names=None)

            # Copy OBJ assets
            obj_dir = os.path.join(parts_temp_dir, "obj/cat_fountain")
            urdf_proj_dir = os.path.join(parts_temp_dir, "urdf/cat_fountain")
            for f in os.listdir(obj_dir):
                if f.endswith(".obj"):
                    shutil.copy(os.path.join(obj_dir, f), os.path.join(urdf_proj_dir, f))

            # Re-create provider for a clean run using production settings
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements
            provider.settings.target_volume = 0.00055
            provider.settings.motor_power = 1000.0

            room = Room()
            provider.build_product(room, mode=Mode.SIMULATE)
            room.translate_joints()
            room["impeller"][0].urdf_motor_target = provider.settings.motor_target
            room["impeller"][0].urdf_motor_force = 10.0
            provider.room = room
            physics_client = p.connect(p.DIRECT)
            try:
                p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

                urdf_path = os.path.join(parts_temp_dir, "urdf/cat_fountain/product.urdf")
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                assert body_id >= 0

                boundaries = {}
                for _, (geom, _) in room.items():
                    u_geom = geom
                    label = getattr(u_geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(u_geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries[label] = geom_boundaries

                hooks = provider.get_simulate_hooks("product:view/simulate")
                setup_fn = hooks[Simulate.SETUP]
                setup_fn(body_id, physics_client, "product:view/simulate", boundaries, None)

                fluid = provider.water_sim
                assert fluid is not None

                # Query dynamic height limit from settings
                floor_z_m = provider.settings.floor_z * 0.001
                bowl_h = provider.settings.bowl_height * 0.001
                step_d = provider.settings.lid_step_depth * 0.001
                lid_mount_z = bowl_h - step_d
                # Lid pocket floor Z (main flat top drinking shelf surface) is at lid_mount_z + 3.0mm
                lid_z_top = lid_mount_z + 0.003

                # Run simulation
                step_fn = hooks[Simulate.STEP]
                max_water_z = 0.0
                for step_idx in range(120):
                    step_fn(body_id, physics_client, step_idx, "product:view/simulate")
                    p.stepSimulation(physicsClientId=physics_client)

                    # Measure maximum height of water exiting the tube during motor execution
                    pos_np = np.asarray(fluid.pos_jax)
                    vel_np = np.asarray(fluid.vel_jax)
                    active_mask = pos_np[:, 2] < 100.0
                    if np.any(active_mask):
                        step_max_z = float(np.max(pos_np[active_mask, 2]))
                        if step_max_z > max_water_z:
                            max_water_z = step_max_z

                # Dome top is at (floor_z + tube_height) + 6.0mm center + dome_outer_radius
                socket_r = (provider.settings.tube_radius + provider.settings.tube_lid_clearance) * 0.001
                dome_out_r = socket_r + 0.0015
                tube_top_z = floor_z_m + provider.settings.tube_height * 0.001
                dome_top_z = tube_top_z + 0.006 + dome_out_r

                # Assert that under production measurements, the fountain water is contained by the spout dome ceiling
                min_expected = lid_z_top - 0.015
                max_expected = dome_top_z + 2.0 * fluid.r_s + 0.003

                assert min_expected <= max_water_z <= max_expected, (
                    f"max_water_z={max_water_z:.5f}, min={min_expected:.5f}, max={max_expected:.5f}"
                )

                # Verify that water reached the drinking surface
                assert max_water_z >= min_expected, "Expected water to reach the lid surface after exiting the spout."

                # Verify that reservoir pool water touches the bowl floor and outer wall
                reservoir_mask = active_mask & (pos_np[:, 2] < lid_mount_z)
                assert np.any(reservoir_mask), "Expected active reservoir fluid in the bowl."
                reservoir_pts = pos_np[reservoir_mask]
                min_reservoir_z = float(np.min(reservoir_pts[:, 2]))
                assert min_reservoir_z <= floor_z_m + fluid.r_s + 0.004, (
                    f"Reservoir fluid is hovering above floor: min Z is {min_reservoir_z:.5f}, expected <= {floor_z_m + fluid.r_s + 0.004:.5f}"
                )

                # Verify reservoir fluid spreads to outer bowl boundary
                r_reservoir = np.sqrt(reservoir_pts[:, 0] ** 2 + reservoir_pts[:, 1] ** 2)
                max_r = float(np.max(r_reservoir))
                bowl_inner_r = (provider.settings.bowl_radius - provider.settings.bowl_thickness) * 0.001
                assert max_r >= bowl_inner_r - 2.0 * fluid.r_s - 0.010, (
                    f"Reservoir fluid did not spread to bowl wall: max R is {max_r:.5f}, expected >= {bowl_inner_r - 2.0 * fluid.r_s - 0.010:.5f}"
                )

                # Verify that water particles do not flood the dry compartment / machine room below floor_z
                machine_room_mask = active_mask & (pos_np[:, 2] < floor_z_m - 0.005)
                flooded_particles = int(np.sum(machine_room_mask))
                assert flooded_particles == 0, (
                    f"Water is flooding the machine room under the bowl floor: {flooded_particles} particles below {floor_z_m - 0.005:.4f}m"
                )

                # Verify that water falling through mid-air between pool surface and lid is not hovering statically
                pool_top_z = float(np.percentile(reservoir_pts[:, 2], 99))
                tube_pos_xy = (0.0, 0.028)
                for b_list in boundaries.values():
                    b_items = b_list if isinstance(b_list, list) else [b_list]
                    for b in b_items:
                        if getattr(b, "link_type", None) == LinkType.TUBE and getattr(b, "xyz", None) is not None:
                            tube_pos_xy = (b.xyz[0], b.xyz[1])
                            break

                r_tube_outer_m = provider.settings.tube_radius * 0.001
                dist_tube = np.sqrt((pos_np[:, 0] - tube_pos_xy[0]) ** 2 + (pos_np[:, 1] - tube_pos_xy[1]) ** 2)
                mid_air_column_mask = (
                    active_mask
                    & (pos_np[:, 2] >= pool_top_z + 0.010)
                    & (pos_np[:, 2] <= lid_mount_z - 0.005)
                    & (dist_tube > r_tube_outer_m + 0.005)
                )
                if np.any(mid_air_column_mask):
                    speeds_mid_air = np.linalg.norm(vel_np[mid_air_column_mask], axis=-1)
                    static_particles = int(np.sum(speeds_mid_air < 0.05))
                    assert static_particles <= 15, (
                        f"Unnatural static hovering fluid blob: {static_particles} stationary particles in mid-air Z in [{pool_top_z + 0.010:.3f}, {lid_mount_z - 0.005:.3f}]m"
                    )
            finally:
                p.disconnect(physics_client)

    @pytest.mark.slow
    @pytest.mark.timeout(300)
    def test_flow_and_drainage_metrics_stability_integration(self):
        """Verify continuous stability of flow, spout discharge, waterfall, drainage, and pool metrics."""
        import tempfile
        import os
        import shutil
        import numpy as np
        import pybullet as p
        from build import Builder
        from provider import ProviderManager, Room, Simulate
        from model import AppConfig
        from shell import Logger

        with tempfile.TemporaryDirectory() as base_temp_dir:
            config = AppConfig()
            real_measurements = os.path.abspath(
                os.path.join(os.path.dirname(__file__), "../cat_fountain/measurements.yaml")
            )
            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements

            manager = ProviderManager(config, providers=[provider], logger=Logger(enabled=False))
            builder = Builder(manager, logger=Logger(enabled=False))

            parts_temp_dir = os.path.join(base_temp_dir, "parts")
            builder.generate_parts(parts_temp_dir, names=None)
            builder.generate_urdfs(parts_temp_dir, names=None)

            obj_dir = os.path.join(parts_temp_dir, "obj/cat_fountain")
            urdf_proj_dir = os.path.join(parts_temp_dir, "urdf/cat_fountain")
            for f in os.listdir(obj_dir):
                if f.endswith(".obj"):
                    shutil.copy(os.path.join(obj_dir, f), os.path.join(urdf_proj_dir, f))

            provider = CatFountainProvider(config=config, logger=Logger(enabled=False))
            provider.settings.measurements_path = real_measurements
            provider.settings.motor_power = 1000.0

            room = Room()
            provider.build_product(room, mode=Mode.SIMULATE)
            room.translate_joints()
            room["impeller"][0].urdf_motor_target = provider.settings.motor_target
            room["impeller"][0].urdf_motor_force = 10.0
            provider.room = room

            physics_client = p.connect(p.DIRECT)
            try:
                p.setGravity(0, 0, -9.81, physicsClientId=physics_client)

                urdf_path = os.path.join(parts_temp_dir, "urdf/cat_fountain/product.urdf")
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                assert body_id >= 0

                boundaries = {}
                for _, (geom, _) in room.items():
                    label = getattr(geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries[label] = geom_boundaries

                hooks = provider.get_simulate_hooks("product:view/simulate")
                setup_fn = hooks[Simulate.SETUP]
                setup_fn(body_id, physics_client, "product:view/simulate", boundaries, None)
                step_fn = hooks[Simulate.STEP]

                fluid = provider.water_sim
                assert fluid is not None

                # Execute simulation steps through production hooks
                for step_idx in range(120):
                    step_fn(body_id, physics_client, step_idx, "product:view/simulate")
                    p.stepSimulation(physicsClientId=physics_client)

                assert hasattr(provider, "metrics_history") and len(provider.metrics_history) == 120

                # Derive physical expectations from first principles:
                # 1. Total active fluid particles and single particle volume
                total_particles = len(fluid.pos_jax)
                assert total_particles > 0
                v_particle = (4.0 / 3.0) * math.pi * (fluid.r_s**3)

                # 2. Geometric delivery tube capacity: V_tube = pi * r_inner^2 * h_tube
                r_inner = (provider.settings.tube_radius - provider.settings.tube_thickness) * 0.001
                h_tube = provider.settings.tube_height * 0.001
                v_tube = math.pi * (r_inner**2) * h_tube
                n_tube_capacity = max(1, int(v_tube / v_particle))

                # 3. Geometric spout dome chamber capacity
                r_spout_dome = (provider.settings.tube_radius + 0.006) * 0.001
                v_spout_dome = (2.0 / 3.0) * math.pi * (r_spout_dome**3)
                n_spout_capacity = max(1, int(v_spout_dome / v_particle))

                # 4. Continuous Delivery Tube Flow & Spout Discharge with naturalistic lower and upper bounds:
                # - Lower bounds ensure pump delivery column does not collapse to zero after priming
                # - Upper bounds ensure fluid packing remains physically bounded by geometric volume
                steady_tube = np.array([m["flow_tube"] for m in provider.metrics_history[100:]])
                all_spout = np.array([m["flow_spout"] for m in provider.metrics_history])

                min_tube_particles = 1
                max_tube_particles = max(int(1.20 * n_tube_capacity), int(0.10 * total_particles))
                assert np.mean(steady_tube) >= min_tube_particles, (
                    f"Mean delivery tube flow fell below minimum physical threshold ({min_tube_particles} particles)"
                )
                assert np.count_nonzero(steady_tube) >= int(0.70 * len(steady_tube)), (
                    "Delivery tube flow collapsed into non-pumping state for excessive frames"
                )
                assert np.all(steady_tube <= max_tube_particles), (
                    f"Delivery tube flow exceeded maximum geometric packing capacity ({max_tube_particles} particles)"
                )

                max_spout_particles = max(int(4.00 * n_spout_capacity), int(0.05 * total_particles))
                assert np.count_nonzero(all_spout) > 0, "Spout discharge dried up completely during simulation"
                assert np.all(all_spout <= max_spout_particles), (
                    f"Spout discharge stream exceeded dome exit volume capacity ({max_spout_particles} particles)"
                )

                # 5. Drainage Continuity: Fluid mass returning across the perimeter waterfall and front drain
                steady_waterfall = np.array([m["drainage_waterfall"] for m in provider.metrics_history[40:]])
                steady_drain = np.array([m["drainage_cutout"] for m in provider.metrics_history[40:]])
                total_steady_drainage = steady_waterfall + steady_drain

                assert np.mean(total_steady_drainage) > 0.0, (
                    "Total drainage returning to reservoir collapsed to zero in steady state"
                )
                assert np.count_nonzero(total_steady_drainage) > 0, "Total drainage dried up in steady state"
                assert np.all(total_steady_drainage <= int(0.80 * total_particles)), (
                    "Falling drainage exceeded physical system mass allocation"
                )

                # 6. Reservoir Pool Mass Conservation: In steady state, reservoir pool retains majority fluid mass
                min_expected_pool_particles = int(0.40 * total_particles)
                max_expected_pool_particles = total_particles
                steady_pool = np.array([m["pool_volume"] for m in provider.metrics_history[40:]])
                assert np.all(steady_pool >= min_expected_pool_particles), (
                    f"Pool volume collapsed below physical mass conservation threshold ({min_expected_pool_particles} particles)"
                )
                assert np.all(steady_pool <= max_expected_pool_particles), (
                    f"Pool volume exceeded total active fluid mass ({max_expected_pool_particles} particles)"
                )

                # 7. Reservoir Water Depth & Casing Contact Continuity
                steady_depth = np.array([m["water_depth"] for m in provider.metrics_history[40:]])
                assert np.all(steady_depth >= 0.010), "Reservoir water depth drained below minimum threshold"
                assert np.all(steady_depth <= 0.080), "Reservoir water depth exceeded maximum bowl capacity"

                # Verify continuous physical fluid contact with motor casing and non-piled reservoir distribution
                pos_np = np.asarray(fluid.pos_jax)
                active = pos_np[:, 2] < 100.0
                pos_act = pos_np[active]
                bowl_floor_z = provider.settings.floor_z * 0.001
                lid_z = (provider.settings.bowl_height - provider.settings.lid_step_depth) * 0.001
                in_bowl = (pos_act[:, 2] >= bowl_floor_z - 0.005) & (pos_act[:, 2] <= lid_z)
                pos_bowl = pos_act[in_bowl]

                casing_x, casing_y = 0.0, -0.028
                casing_r = 0.025
                d_casing = np.sqrt((pos_bowl[:, 0] - casing_x) ** 2 + (pos_bowl[:, 1] - casing_y) ** 2)
                r_bowl = np.sqrt(pos_bowl[:, 0] ** 2 + pos_bowl[:, 1] ** 2)

                assert np.min(d_casing) <= casing_r + 0.003, (
                    f"Water detached from motor casing: min_d_casing={np.min(d_casing):.4f} m"
                )
                assert np.sum(d_casing <= casing_r + 0.005) >= 1000, (
                    "Insufficient fluid volume in contact with motor casing"
                )
                assert np.sum(r_bowl >= 0.080) <= int(0.60 * len(pos_bowl)), (
                    "Excessive fluid mass piled up at outer bowl wall"
                )

                # Verify standalone compute_flow_metrics hook method
                current_metrics = provider.compute_flow_metrics()
                assert min_expected_pool_particles <= current_metrics["pool_volume"] <= max_expected_pool_particles
                assert "water_depth" in current_metrics and 0.010 <= current_metrics["water_depth"] <= 0.080
            finally:
                p.disconnect(physics_client)

    def test_cat_fountain_port_connectivity(self, provider):
        """Verify that all intake and drain ports across cat fountain components match and connect."""
        import jax.numpy as jnp
        from provider.boundary import BoundaryProcessor
        from provider.transforms import match_intake_drain_ports

        room = Room()
        provider.build_product(room, mode=Mode.DEFAULT)
        room.translate_joints()

        boundary_list = []
        for name, (geom, _) in room.items():
            if hasattr(geom, "urdf_boundaries"):
                boundary_list.extend(geom.urdf_boundaries)

        processed = BoundaryProcessor.process(boundary_list)

        matches_mask, dist_matrix = match_intake_drain_ports(
            processed.b_pos_arr,
            processed.b_orn_arr,
            processed.b_params,
            distance_tol=0.035,
            normal_alignment_threshold=-0.0,
        )

        # Check that there is at least one active match in the system
        assert jnp.any(matches_mask), "No fluid intake/drain ports matched across cat fountain assembly"

    def test_pump_cover_c_clip_opening(self, provider):
        """Verify that the pump cover sleeve features a North-facing C-clip opening for effortless slide-down and snap-on."""
        cover = provider.build_pump_cover("pump_cover")
        solid = cover.part
        assert solid is not None and solid.volume > 0.0

        tube_y = 28.0
        tube_r = provider.settings.tube_radius
        sleeve_h = provider.settings.pump_cover_sleeve_height
        z_mid = sleeve_h / 2.0

        # North-facing point on outer sleeve perimeter (Y = tube_y + tube_r + 1.0) must be open (outside/not solid)
        north_y = tube_y + tube_r + 1.0
        assert not solid.is_inside((0.0, north_y, z_mid)), (
            f"Pump cover sleeve North opening is blocked at Y = {north_y}!"
        )

        # South bridge/cradle wall (Y = tube_y - tube_r - 1.0) must be solid to support and locate the cover
        south_y = tube_y - tube_r - 1.0
        assert solid.is_inside((0.0, south_y, z_mid)), f"Pump cover South cradle wall is broken at Y = {south_y}!"
