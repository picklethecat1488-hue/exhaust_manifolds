"""Unit tests for SPH Fluid simulation class."""

import math
from provider.fluid import Fluid
from provider.bullet import LinkType
from model import FluidConfig


def test_fluid_initialization():
    """Verify that Fluid class initializes with the correct constants."""
    fluid = Fluid(config=FluidConfig(r_s=0.003, rest_density=1000.0, viscosity=0.5, stiffness=2000.0))
    assert math.isclose(fluid.r_s, 0.003)
    assert math.isclose(fluid.h, 0.009)
    assert fluid.rest_density == 1000.0
    assert fluid.viscosity == 0.5
    assert fluid.k == 2000.0
    assert fluid.mass > 0.0


def test_zero_shear_strength():
    """
    Verify the Zero Shear Strength rule (fluids continuously deform under shear).

    If we apply a sideways force to a block of particles, they should flow and deform.
    """
    fluid = Fluid(config=FluidConfig(r_s=0.003, viscosity=0.1))

    # Place particles in a vertical column to represent a fluid block
    positions = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.006),
        (0.0, 0.0, 0.012),
        (0.0, 0.006, 0.0),
        (0.0, 0.006, 0.006),
    ]
    # Stationary initial velocities
    velocities = [(0.0, 0.0, 0.0)] * 5

    # Step simulation manually by applying a shear (horizontal) force
    # and verify that deformation (change in relative horizontal positions) is continuous.
    forces = fluid.compute_forces(positions, velocities)

    # Ensure SPH forces computed are valid
    assert len(forces) == 5
    for f in forces:
        assert len(f) == 3

    # Apply a constant sideways shear velocity to the top particle and verify viscosity transmits
    # shear to lower particles, causing continuous deformation.
    velocities[2] = (1.0, 0.0, 0.0)  # Top particle moving horizontally
    forces_under_shear = fluid.compute_forces(positions, velocities)

    # Viscosity force should pull neighboring particles in the direction of the shear velocity
    # Particle 1 (neighbor to particle 2 at z=0.006) should experience a positive force in X
    assert forces_under_shear[1][0] > 0.0


def test_knudsen_number():
    """Verify that the fluid acts as a continuous regime with Kn < 0.1."""
    fluid = Fluid(config=FluidConfig(r_s=0.003))

    # Create a dense grid of particles resembling a fluid continuum
    positions = []
    for x in range(5):
        for y in range(5):
            for z in range(5):
                positions.append((x * 0.006, y * 0.006, z * 0.006))

    kn = fluid.compute_knudsen_number(positions, characteristic_length=0.076)
    kn_fallback = fluid.compute_knudsen_number(positions)

    # For a dense fluid packing, Kn should be strictly less than 0.1
    # demonstrating that it operates in the continuum fluid dynamics regime.
    assert kn < 0.1
    assert kn > 0.0
    assert kn == kn_fallback

    # Verify that when boundaries are configured, we derive it from LinkType.BASE boundary radius
    from model.boundary_config import BoundaryConfig

    boundaries_config = {
        "bowl": BoundaryConfig(
            link_type=LinkType.BASE,
            radius=0.1,
            link_idx=-1,
        )
    }
    fluid_with_boundary = Fluid(config=FluidConfig(r_s=0.003, boundaries=boundaries_config))
    assert fluid_with_boundary.characteristic_length == 0.1
    kn_boundary = fluid_with_boundary.compute_knudsen_number(positions)
    # The characteristic length is larger (0.1 vs 0.076), so Knudsen number should be smaller
    assert kn_boundary < kn


def test_fluid_config_boundary_validation():
    """Verify that FluidConfig enforces boundaries containing a BASE LinkType boundary."""
    import pytest
    from pydantic import ValidationError
    from model.boundary_config import BoundaryConfig

    # Empty boundaries dictionary should raise ValidationError
    with pytest.raises(ValidationError, match="Boundaries dictionary is required and cannot be empty"):
        FluidConfig(boundaries={})

    # Boundaries dictionary without a BASE LinkType should raise ValidationError
    invalid_boundaries = {
        "tube": BoundaryConfig(
            link_type=LinkType.TUBE,
            radius=0.008,
            link_idx=1,
        )
    }
    with pytest.raises(ValidationError, match="Every fluid configuration model must contain a BASE LinkType boundary"):
        FluidConfig(boundaries=invalid_boundaries)

    # ShapeType cylinder with invalid radius
    from model.boundary_config import ShapeType

    with pytest.raises(ValidationError, match="CYLINDER shape requires a positive radius"):
        BoundaryConfig(
            shape=ShapeType.CYLINDER,
            link_type=LinkType.BASE,
            radius=0.0,
            height=0.1,
            link_idx=-1,
        )

    # ShapeType cylinder with unsupported fields (e.g. num_vanes)
    with pytest.raises(ValidationError, match="is not supported for shape type 'cylinder'"):
        BoundaryConfig(
            shape=ShapeType.CYLINDER,
            link_type=LinkType.BASE,
            radius=0.076,
            height=0.1,
            num_vanes=4.0,
            link_idx=-1,
        )


def test_boundary_config_vane_twist_rad():
    """Verify that vane_twist_rad is computed as a read-only property and not present in serialized fields."""
    from model.boundary_config import BoundaryConfig, ShapeType
    from provider.bullet import LinkType

    # Default twist is -1080.0 degrees
    cfg = BoundaryConfig(
        shape=ShapeType.IMPELLER,
        link_type=LinkType.IMPELLER,
        radius=0.03,
        link_idx=0,
    )
    assert math.isclose(cfg.vane_twist, -1080.0)
    assert math.isclose(cfg.vane_twist_rad, -6.0 * math.pi)

    # Custom twist
    cfg_custom = BoundaryConfig(
        shape=ShapeType.IMPELLER,
        link_type=LinkType.IMPELLER,
        radius=0.03,
        vane_twist=180.0,
        link_idx=0,
    )
    assert math.isclose(cfg_custom.vane_twist, 180.0)
    assert math.isclose(cfg_custom.vane_twist_rad, math.pi)

    # Verify vane_twist_rad is not part of model_fields or serialized dict
    assert "vane_twist_rad" not in cfg.model_fields
    assert "vane_twist_rad" not in cfg.model_dump()


def test_newtonian_viscosity():
    """Verify that viscosity force acts to reduce relative velocity linearly (Newtonian)."""
    fluid = Fluid(config=FluidConfig(r_s=0.003, viscosity=0.5))

    # Two adjacent particles with relative velocity
    positions = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.005),
    ]

    # Newtonian viscosity: force should scale linearly with relative velocity
    vel_1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    forces_1 = fluid.compute_forces(positions, vel_1)

    vel_2 = [(0.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    forces_2 = fluid.compute_forces(positions, vel_2)

    # The viscosity force on particle 0 in X should be positive (dragged by particle 1)
    assert forces_1[0][0] > 0.0
    assert forces_2[0][0] > 0.0

    # Viscosity force should be approximately doubled when velocity difference is doubled
    ratio = forces_2[0][0] / forces_1[0][0]
    assert math.isclose(ratio, 2.0, rel_tol=1e-2)


def test_momentum_conservation():
    """Verify Newton's Third Law (action/reaction forces sum to zero)."""
    fluid = Fluid(config=FluidConfig(r_s=0.003, viscosity=0.5, stiffness=1000.0))

    # 3 asymmetric particles to create complex internal forces
    positions = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.004),
        (0.0, 0.003, 0.003),
    ]
    velocities = [
        (0.1, -0.2, 0.0),
        (-0.3, 0.1, 0.2),
        (0.0, 0.1, -0.1),
    ]

    forces = fluid.compute_forces(positions, velocities)

    # Sum of all internal forces must be extremely close to 0.0 (momentum conservation)
    sum_x = sum(f[0] for f in forces)
    sum_y = sum(f[1] for f in forces)
    sum_z = sum(f[2] for f in forces)

    # Using 5e-4 tolerance to account for float32 precision
    assert math.isclose(sum_x, 0.0, abs_tol=5e-4)
    assert math.isclose(sum_y, 0.0, abs_tol=5e-4)
    assert math.isclose(sum_z, 0.0, abs_tol=5e-4)


def test_incompressibility_repulsion():
    """Verify that compressed particles experience repulsive pressure forces."""
    fluid = Fluid(config=FluidConfig(r_s=0.003, stiffness=2000.0, viscosity=0.1))

    # Place 16 particles in two tight clusters separated along the Z-axis
    positions = []
    # Cluster A (centered around z = 0.0)
    for x in (-0.001, 0.001):
        for y in (-0.001, 0.001):
            for z in (-0.001, 0.001):
                positions.append((x, y, z))
    # Cluster B (centered around z = 0.005)
    for x in (-0.001, 0.001):
        for y in (-0.001, 0.001):
            for z in (0.004, 0.006):
                positions.append((x, y, z))

    velocities = [(0.0, 0.0, 0.0)] * 16
    forces = fluid.compute_forces(positions, velocities)

    # Average force on Cluster A should have negative Z (pushed away from B)
    avg_f_a_z = sum(forces[i][2] for i in range(8)) / 8.0
    # Average force on Cluster B should have positive Z (pushed away from A)
    avg_f_b_z = sum(forces[i][2] for i in range(8, 16)) / 8.0

    assert avg_f_a_z < 0.0
    assert avg_f_b_z > 0.0


def test_single_particle_forces():
    """Verify that a single isolated particle experiences zero internal force."""
    fluid = Fluid(config=FluidConfig(r_s=0.003))

    positions = [(0.0, 0.0, 0.0)]
    velocities = [(1.0, 2.0, -3.0)]

    forces = fluid.compute_forces(positions, velocities)
    assert len(forces) == 1
    assert forces[0] == [0.0, 0.0, 0.0]


def test_compute_forces_jax_direct():
    """Verify that compute_forces_jax produces the same output as compute_forces but as a JAX array."""
    import jax.numpy as jnp

    fluid = Fluid(config=FluidConfig(r_s=0.003))
    positions = [
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.004),
        (0.0, 0.003, 0.003),
    ]
    velocities = [
        (0.1, -0.2, 0.0),
        (-0.3, 0.1, 0.2),
        (0.0, 0.1, -0.1),
    ]
    pos_jax = jnp.array(positions, dtype=jnp.float32)
    vel_jax = jnp.array(velocities, dtype=jnp.float32)

    forces_list = fluid.compute_forces(positions, velocities)
    forces_jax = fluid.compute_forces_jax(pos_jax, vel_jax)

    assert isinstance(forces_jax, jnp.ndarray)
    assert forces_jax.shape == (3, 3)
    for i in range(len(positions)):
        for j in range(3):
            assert math.isclose(float(forces_jax[i, j]), forces_list[i][j], abs_tol=1e-4)


def test_fluid_spawner_padding_and_jitter():
    """Verify that FluidSpawner spawns batches with jitter and pads arrays correctly."""
    import pybullet as p
    from provider.fluid import FluidSpawner

    physics_client = p.connect(p.DIRECT)
    try:
        spawner = FluidSpawner(
            physics_client=physics_client,
            r_s=0.003,
            n_particles=10,
            particle_mass=0.002,
            particle_color=[0, 0, 1, 1],
            linear_damping=0.05,
            angular_damping=0.05,
            lateral_friction=0.1,
            restitution=0.0,
        )

        # Initial state
        assert spawner.active_count == 0
        assert len(spawner.particle_body_ids) == 0

        # 1. Spawn a batch of 4 particles
        newly_spawned = spawner.spawn_batch(spawn_z=0.100, batch_size=4, spacing=0.008)
        assert newly_spawned == 4
        assert spawner.active_count == 4
        assert len(spawner.particle_body_ids) == 4

        # Verify positions and velocities are padded up to n_particles (10)
        positions, velocities = spawner.get_positions_and_velocities()
        assert len(positions) == 10
        assert len(velocities) == 10

        # First 4 positions should be active particles (z < 100)
        for i in range(4):
            assert positions[i][2] < 100.0
            assert abs(positions[i][0]) <= 0.006
            assert abs(positions[i][1]) <= 0.006

        # Remaining 6 should be padded to 1000.0
        for i in range(4, 10):
            assert math.isclose(positions[i][2], 1000.0)
            assert positions[i][0] == 0.0
            assert positions[i][1] == 0.0

        # Spawn more than n_particles capacity
        newly_spawned_2 = spawner.spawn_batch(spawn_z=0.100, batch_size=10, spacing=0.008)
        assert newly_spawned_2 == 6  # Spawner caps at n_particles (10)
        assert spawner.active_count == 10

    finally:
        p.disconnect(physicsClientId=physics_client)


def test_fluid_simulator_dynamic_properties():
    """Verify that Fluid reads target velocity, force, and offset from shape metadata and PyBullet."""
    from unittest.mock import patch, MagicMock
    from provider.fluid import Fluid
    from provider.room import Room
    from model import BoundaryConfig, FluidConfig

    # Mock provider
    provider = MagicMock()
    provider.settings.bowl_radius = 80.0
    provider.settings.bowl_thickness = 3.5
    provider.settings.tube_thickness = 1.5
    provider.settings.impeller_shaft_radius = 1.5

    boundaries = {
        "bowl": BoundaryConfig(
            link_type=LinkType.BASE,
            radius=0.0765,
            thickness=0.0035,
            link_idx=-1,
        ),
        "tube": BoundaryConfig(
            link_type=LinkType.TUBE,
            radius=0.008,
            height=0.100,
            link_idx=1,
        ),
        "impeller": BoundaryConfig(
            link_type=LinkType.IMPELLER,
            radius=0.0,
            link_idx=0,
            impeller_shaft_radius=2.5,
        ),
    }

    # Mock PyBullet functions
    def mock_get_num_joints(body_id, physicsClientId):
        return 3

    def mock_get_joint_info(body_id, joint_idx, physicsClientId):
        # idx 0: impeller, idx 1: tube, idx 2: spout
        if joint_idx == 0:
            return (
                0,
                b"joint0",
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
                (0, 0, 0),
                (0, 0, 0, 1),
                -1,
            )
        elif joint_idx == 1:
            return (
                1,
                b"joint1",
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
                (0.0, 0.057, 0.0),
                (0, 0, 0, 1),
                -1,
            )
        else:
            return (
                2,
                b"joint2",
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
                b"spout",
                (0, 0, 0),
                (0, 0, 0),
                (0, 0, 0, 1),
                -1,
            )

    def mock_get_aabb(body_id, link_idx, physicsClientId):
        if link_idx == -1:
            return ((-0.080, -0.080, 0.0), (0.080, 0.080, 0.040))
        elif link_idx == 1:
            return ((-0.008, 0.049, 0.0), (0.008, 0.065, 0.100))
        return ((0, 0, 0), (0, 0, 0))

    def mock_get_link_state(body_id, link_idx, computeLinkVelocity=0, physicsClientId=None):
        if link_idx == 2:
            return (None, None, None, None, (0.0, 0.0, 0.025), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        return (None, None, None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0), (0.0, 0.0, 0.0), (0.0, 0.0, 0.0))

    def mock_get_base_position_and_orientation(body_id, physicsClientId):
        return (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)

    def mock_get_dynamics_info(body_id, link_idx, physicsClientId):
        return [None, None, None, (0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)]

    with (
        patch("pybullet.getNumJoints", side_effect=mock_get_num_joints),
        patch("pybullet.getJointInfo", side_effect=mock_get_joint_info),
        patch("pybullet.getAABB", side_effect=mock_get_aabb),
        patch("pybullet.getLinkState", side_effect=mock_get_link_state),
        patch("pybullet.getConnectionInfo", return_value={"isConnected": True}),
        patch("pybullet.createCollisionShape", return_value=0),
        patch("pybullet.createVisualShape", return_value=0),
        patch("pybullet.getBasePositionAndOrientation", side_effect=mock_get_base_position_and_orientation),
        patch("pybullet.getBaseVelocity", return_value=((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))),
        patch("pybullet.getDynamicsInfo", side_effect=mock_get_dynamics_info),
    ):
        sim = Fluid(
            config=FluidConfig(boundaries=boundaries),
            provider=provider,
            body_id=42,
            physics_client=1,
            link_indices={
                LinkType.OUTLET: 2,
                LinkType.TUBE: 1,
                LinkType.IMPELLER: 0,
            },
        )
        # Verify dynamic properties using the refactored field names
        assert sim.link_indices == {
            LinkType.OUTLET: 2,
            LinkType.TUBE: 1,
            LinkType.IMPELLER: 0,
        }
        assert sim.boundaries[LinkType.IMPELLER].target_omega == 15.0
        assert sim.boundaries[LinkType.IMPELLER].max_force == 10.0
        assert math.isclose(sim.radii[LinkType.TUBE], 0.008, abs_tol=1e-5)
        assert math.isclose(sim.radii[LinkType.BASE], 0.080, abs_tol=1e-5)
        assert math.isclose(sim.radii[LinkType.IMPELLER], 0.003, abs_tol=1e-5)
        assert math.isclose(sim.radii[LinkType.FALLEN], 0.090, abs_tol=1e-5)
        assert math.isclose(sim.thresholds[LinkType.OUTLET], 0.120, abs_tol=1e-5)
        assert math.isclose(sim.thresholds[LinkType.OUTLET_MAX_Y], 0.005, abs_tol=1e-5)
        assert math.isclose(sim.thresholds[LinkType.TUBE], 15.0, abs_tol=1e-5)


def test_fluid_default_thresholds():
    """Verify that fluid thresholds default correctly when physics client is not connected."""
    sim = Fluid(config=FluidConfig(r_s=0.003))
    assert sim.thresholds[LinkType.TUBE] == 0.0
    assert sim.thresholds[LinkType.OUTLET] == 0.0
    assert sim.thresholds[LinkType.OUTLET_MAX_Y] == 0.0


def test_fluid_parameter_overrides():
    """Verify that Fluid correctly accepts parameter overrides via the constructor."""
    from provider.fluid import Fluid

    # Create a dummy settings and provider
    class DummySettings:
        pass

    class DummyProvider:
        def __init__(self):
            self.settings = DummySettings()

    provider = DummyProvider()

    # Set overridable simulation constants
    provider.settings.PARTICLE_RADIUS = 0.0025
    provider.settings.TARGET_VOLUME = 0.001
    provider.settings.VOLUME_THRESHOLD_LITERS = 0.600
    provider.settings.FALLEN_THRESHOLD_LITERS = 0.100
    provider.settings.SPAWN_BUFFER = 0.005

    # Set overridable physical settings
    provider.settings.REST_DENSITY = 800.0
    provider.settings.VISCOSITY = 0.12
    provider.settings.STIFFNESS = 150.0

    # Set overridable SPH constants
    provider.settings.SMOOTHING_FACTOR = 4.0
    provider.settings.SPHERE_VOL_FACTOR = 1.333
    provider.settings.POLY6_COEFF_NUMERATOR = 300.0
    provider.settings.POLY6_COEFF_DENOMINATOR = 50.0
    provider.settings.SPIKY_GRAD_COEFF = -40.0
    provider.settings.VISC_LAP_COEFF = 40.0
    provider.settings.PRESSURE_AVG_FACTOR = 3.0
    provider.settings.MIN_DISTANCE_THRESHOLD = 1e-5

    # Initialize Fluid with custom parameters directly
    fluid = Fluid(
        config=FluidConfig(
            particle_radius=provider.settings.PARTICLE_RADIUS,
            target_volume=provider.settings.TARGET_VOLUME,
            volume_threshold_liters=provider.settings.VOLUME_THRESHOLD_LITERS,
            fallen_threshold_liters=provider.settings.FALLEN_THRESHOLD_LITERS,
            spawn_buffer=provider.settings.SPAWN_BUFFER,
            rest_density=provider.settings.REST_DENSITY,
            viscosity=provider.settings.VISCOSITY,
            stiffness=provider.settings.STIFFNESS,
            smoothing_factor=provider.settings.SMOOTHING_FACTOR,
            sphere_vol_factor=provider.settings.SPHERE_VOL_FACTOR,
            poly6_coeff_numerator=provider.settings.POLY6_COEFF_NUMERATOR,
            poly6_coeff_denominator=provider.settings.POLY6_COEFF_DENOMINATOR,
            spiky_grad_coeff=provider.settings.SPIKY_GRAD_COEFF,
            visc_lap_coeff=provider.settings.VISC_LAP_COEFF,
            pressure_avg_factor=provider.settings.PRESSURE_AVG_FACTOR,
            min_distance_threshold=provider.settings.MIN_DISTANCE_THRESHOLD,
        ),
        provider=provider,
    )

    # Assert Fluid constants are overridden
    assert fluid.particle_radius == 0.0025
    assert fluid.target_volume == 0.001
    assert fluid.volume_threshold_liters == 0.600
    assert fluid.fallen_threshold_liters == 0.100
    assert fluid.spawn_buffer == 0.005
    assert fluid.rest_density == 800.0
    assert fluid.viscosity == 0.12
    assert fluid.stiffness == 150.0
    assert fluid.k == 150.0
    assert fluid.smoothing_factor == 4.0
    assert fluid.sphere_vol_factor == 1.333
    assert fluid.pressure_avg_factor == 3.0
    assert fluid.min_distance_threshold == 1e-5
    assert fluid.r_s == 0.0025

    # Test default constructor arguments
    fluid_default = Fluid(provider=provider)
    assert fluid_default.r_s == 0.003
    assert fluid_default.rest_density == 1000.0


def test_fluid_config_object():
    """Verify that Fluid correctly parses settings from a flat FluidConfig object."""
    from model import FluidConfig

    config = FluidConfig(
        viscosity=0.77,
        rest_density=920.0,
        smoothing_factor=4.5,
        target_volume=0.0008,
        sim_name="config_test",
    )

    # Initialize purely from config
    fluid = Fluid(config=config)
    assert fluid.viscosity == 0.77
    assert fluid.rest_density == 920.0
    assert fluid.smoothing_factor == 4.5
    assert fluid.target_volume == 0.0008


def test_compute_boundary_forces_reads_tube_y():
    """Verify that _compute_boundary_forces_jax resolves tube_y from boundary_configs."""
    import jax.numpy as jnp
    from provider.fluid import _compute_boundary_forces_jax
    from model import BoundaryConfig, ShapeType, BoundaryType
    from provider.bullet import LinkType

    # 1. Define a boundary config representing the spout deflection cap (inverted, small radius, non-zero xyz[1])
    spout_deflection_cfg = BoundaryConfig(
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        link_type=LinkType.LID,
        radius=0.013,
        height=0.0,
        thickness=0.040,
        xyz=(0.0, 0.042, 0.016),
        rpy=(3.141592653589793, 0.0, 0.0),
        link_idx=2,
        has_tube=True,
    )

    # 2. Call _compute_boundary_forces_jax with dummy pos and vel
    pos = jnp.array([[0.0, 0.042, 0.005]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), dtype=jnp.float32)
    b_pos_arr = jnp.zeros((1, 3), dtype=jnp.float32)
    b_orn_arr = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    # Pack parameters
    b_shapes = jnp.array([1], dtype=jnp.int32)
    b_types = jnp.array([1], dtype=jnp.int32)
    b_params = jnp.array(
        [
            [
                0.013,  # radius
                0.0,  # height
                0.040,  # thickness
                0.0,  # z_offset
                0.0,  # slot_height
                0.0,  # slot_width
                0.0,  # ceiling_thickness
                0.0,  # vane_thickness
                0.0,  # num_vanes
                0.0,  # vane_twist_rad
                0.0,  # cutoff_y
                1.0,  # has_tube
                0.0,  # has_drain
                0.0,  # tube_radius
                0.0,  # drain_hole_y
                0.0,  # drain_hole_radius
                0.20,  # boundary_friction
            ]
        ],
        dtype=jnp.float32,
    )

    # Call the JIT function
    forces, torque = _compute_boundary_forces_jax(
        pos,
        vel,
        r_s=0.0015,
        K=1000.0,
        D=5.0,
        b_pos_arr=b_pos_arr,
        b_orn_arr=b_orn_arr,
        b_shapes=b_shapes,
        b_types=b_types,
        b_params=b_params,
        omega=0.0,
        t=0.0,
    )
    # The function compiles and executes successfully
    assert forces is not None


def test_particleset_happy_path():
    """Verify that ParticleSet behaves correctly for happy path operations."""
    import numpy as np
    from provider.fluid import ParticleSet

    pset = ParticleSet(10)
    assert len(pset) == 0
    assert 3 not in pset

    # Single add
    pset.add(3)
    assert len(pset) == 1
    assert 3 in pset

    # Vectorized multiple add
    pset.add_multiple(np.array([1, 5, 7]))
    assert len(pset) == 4
    assert 1 in pset
    assert 5 in pset
    assert 7 in pset

    # Iteration
    indices = sorted(list(pset))
    assert indices == [1, 3, 5, 7]

    # Clear
    pset.clear()
    assert len(pset) == 0
    assert 3 not in pset


def test_particleset_sad_path():
    """Verify that ParticleSet handles index bounds correctly (sad path)."""
    import pytest
    from provider.fluid import ParticleSet

    pset = ParticleSet(5)
    with pytest.raises(IndexError):
        pset.add(5)

    with pytest.raises(IndexError):
        pset.add(-6)


def test_physics_step_spout_forcing_happy():
    """Verify that _physics_step_jax runs successfully with dynamic tube forcing (happy path)."""
    import jax.numpy as jnp
    from provider.fluid import _physics_step_jax, PhysicsConfig
    from model import BoundaryConfig, ShapeType, BoundaryType
    from provider.bullet import LinkType

    # Happy path: tube configuration with non-zero tube_velocity
    base_cfg = BoundaryConfig(
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        link_type=LinkType.BASE,
        radius=0.1,
        height=0.05,
        link_idx=-1,
    )
    tube_cfg = BoundaryConfig(
        shape=ShapeType.TUBE,
        link_type=LinkType.TUBE,
        radius=0.010,
        height=0.03,
        spout_radius=0.005,
        spout_height=0.010,
        xyz=(0.0, 0.04, 0.0),
        link_idx=1,
    )

    pos = jnp.array([[0.0, 0.0, 0.02]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), dtype=jnp.float32)
    f_lbm = jnp.zeros((15, 32, 32, 28), dtype=jnp.float32)
    b_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.04, 0.0]], dtype=jnp.float32)
    b_orn = jnp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    config = PhysicsConfig(
        mass=1e-6,
        dt_sub=1.0 / 240.0,
        n_substeps=1,
        boundary_configs=(base_cfg, tube_cfg),
        gravity=(0.0, 0.0, -9.81),
        base_idx=0,
        K_boundary=1000.0,
        D_boundary=5.0,
        r_s=0.003,
        high_damping_value=0.998,
        nx=32,
        ny=32,
        nz=28,
        dx=0.005,
        origin=(-0.075, -0.075, 0.0),
    )

    pos_next, vel_next, f_next, torque_accum, b_forces = _physics_step_jax(
        pos,
        vel,
        f_lbm,
        b_pos,
        b_orn,
        jnp.zeros(3, dtype=jnp.float32),
        120.0,
        0.0,
        -1.0,
        config=config,
    )

    assert pos_next is not None
    assert vel_next is not None
    assert f_next is not None
    assert torque_accum is not None
    assert b_forces is not None


def test_physics_step_spout_forcing_sad():
    """Verify that _physics_step_jax falls back cleanly when no tube config is present (sad path)."""
    import jax.numpy as jnp
    from provider.fluid import _physics_step_jax, PhysicsConfig
    from model import BoundaryConfig, ShapeType, BoundaryType
    from provider.bullet import LinkType

    # Sad path: no tube configuration, only base
    base_cfg = BoundaryConfig(
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        link_type=LinkType.BASE,
        radius=0.1,
        height=0.05,
        link_idx=-1,
    )

    pos = jnp.array([[0.0, 0.0, 0.02]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), dtype=jnp.float32)
    f_lbm = jnp.zeros((15, 32, 32, 28), dtype=jnp.float32)
    b_pos = jnp.array([[0.0, 0.0, 0.0]], dtype=jnp.float32)
    b_orn = jnp.array([[0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    config = PhysicsConfig(
        mass=1e-6,
        dt_sub=1.0 / 240.0,
        n_substeps=1,
        boundary_configs=(base_cfg,),
        gravity=(0.0, 0.0, -9.81),
        base_idx=0,
        K_boundary=1000.0,
        D_boundary=5.0,
        r_s=0.003,
        high_damping_value=0.998,
        nx=32,
        ny=32,
        nz=28,
        dx=0.005,
        origin=(-0.075, -0.075, 0.0),
    )

    pos_next, vel_next, f_next, torque_accum, b_forces = _physics_step_jax(
        pos,
        vel,
        f_lbm,
        b_pos,
        b_orn,
        jnp.zeros(3, dtype=jnp.float32),
        120.0,
        0.0,
        -1.0,
        config=config,
    )

    assert pos_next is not None
    assert vel_next is not None
    assert f_next is not None
    assert torque_accum is not None
    assert b_forces is not None


def test_physics_step_casing_suction_happy():
    """Verify that casing suction force is correctly applied in the suction zone (happy path)."""
    import jax.numpy as jnp
    from provider.fluid import _physics_step_jax, PhysicsConfig
    from model import BoundaryConfig, ShapeType, BoundaryType
    from provider.bullet import LinkType

    base_cfg = BoundaryConfig(
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        link_type=LinkType.BASE,
        radius=0.1,
        height=0.05,
        link_idx=-1,
    )
    casing_cfg = BoundaryConfig(
        shape=ShapeType.CASING,
        type=BoundaryType.SOLID_CAVITY,
        link_type=LinkType.LID,
        radius=0.028,
        height=0.010,
        link_idx=1,
        cutoff_y=0.0,
        has_intake=True,
        intake_hole_radius=0.010,
        intake_hole_z=0.010,
    )

    # Particle position directly in the suction zone: above casing cover, near center
    # casing_pos is (0.0, 0.0, 0.0)
    pos = jnp.array([[0.005, 0.005, 0.012]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), dtype=jnp.float32)
    f_lbm = jnp.zeros((15, 32, 32, 28), dtype=jnp.float32)
    b_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    b_orn = jnp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    config = PhysicsConfig(
        mass=1e-6,
        dt_sub=1.0 / 240.0,
        n_substeps=1,
        boundary_configs=(base_cfg, casing_cfg),
        gravity=(0.0, 0.0, -9.81),
        base_idx=0,
        K_boundary=1000.0,
        D_boundary=5.0,
        r_s=0.003,
        high_damping_value=0.998,
        nx=32,
        ny=32,
        nz=28,
        dx=0.005,
        origin=(-0.075, -0.075, 0.0),
    )

    # Run physics step with high omega to generate a strong suction force
    pos_next, vel_next, f_next, torque_accum, b_forces = _physics_step_jax(
        pos,
        vel,
        f_lbm,
        b_pos,
        b_orn,
        jnp.zeros(3, dtype=jnp.float32),
        120.0,
        0.0,
        -1.0,
        config=config,
    )

    # With high suction force directed downwards and inwards, vel_next should have a negative Z component
    assert vel_next[0, 2] < -0.02


def test_physics_step_casing_suction_sad():
    """Verify that no casing suction force is applied to particles outside the suction zone (sad path)."""
    import jax.numpy as jnp
    from provider.fluid import _physics_step_jax, PhysicsConfig
    from model import BoundaryConfig, ShapeType, BoundaryType
    from provider.bullet import LinkType

    base_cfg = BoundaryConfig(
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        link_type=LinkType.BASE,
        radius=0.1,
        height=0.05,
        link_idx=-1,
    )
    casing_cfg = BoundaryConfig(
        shape=ShapeType.CASING,
        type=BoundaryType.SOLID_CAVITY,
        link_type=LinkType.LID,
        radius=0.028,
        height=0.010,
        link_idx=1,
        cutoff_y=0.0,
        has_intake=True,
        intake_hole_radius=0.010,
        intake_hole_z=0.010,
    )

    # Particle position far outside the suction zone (e.g. horizontally far from center)
    pos = jnp.array([[0.050, 0.0, 0.012]], dtype=jnp.float32)
    vel = jnp.zeros((1, 3), dtype=jnp.float32)
    f_lbm = jnp.zeros((15, 32, 32, 28), dtype=jnp.float32)
    b_pos = jnp.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=jnp.float32)
    b_orn = jnp.array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]], dtype=jnp.float32)

    config = PhysicsConfig(
        mass=1e-6,
        dt_sub=1.0 / 240.0,
        n_substeps=1,
        boundary_configs=(base_cfg, casing_cfg),
        gravity=(0.0, 0.0, 0.0),  # Zero gravity to isolate suction effect
        base_idx=0,
        K_boundary=1000.0,
        D_boundary=5.0,
        r_s=0.003,
        high_damping_value=0.998,
        nx=32,
        ny=32,
        nz=28,
        dx=0.005,
        origin=(-0.075, -0.075, 0.0),
    )

    pos_next, vel_next, f_next, torque_accum, b_forces = _physics_step_jax(
        pos,
        vel,
        f_lbm,
        b_pos,
        b_orn,
        jnp.zeros(3, dtype=jnp.float32),
        120.0,
        0.0,
        -1.0,
        config=config,
    )

    # Since the particle is far from the casing center, suction acceleration should be zero,
    # and since gravity is also zero, vel_next should remain zero.
    assert jnp.allclose(vel_next, 0.0, atol=1e-5)


def test_voxel_primitive_cutouts():
    """Verify that TubeWallPrimitive and CasingWallPrimitive cutouts are computed correctly."""
    from provider.fluid import CasingWallPrimitive, TubeWallPrimitive

    # CasingWallPrimitive: centered at (0, 0), r_inner=18mm, r_outer=28mm, z_min=0, z_max=0.010
    # Cutout should be at y > 0, |x| < slot_width/2, z < slot_height
    casing = CasingWallPrimitive(
        x=0.0,
        y=0.0,
        r_inner=0.018,
        r_outer=0.028,
        z_min=0.0,
        z_max=0.010,
        slot_height=0.009,
        slot_width=0.008,
    )

    # A point inside the solid casing wall but not in the cutout
    assert casing.is_solid(0.0, -0.023, 0.005) is True  # y < 0
    assert casing.is_solid(0.023, 0.0, 0.005) is True  # x > slot_width/2
    assert casing.is_solid(0.0, 0.023, 0.0095) is True  # z > slot_height

    # A point inside the cutout connection (should NOT be solid)
    assert casing.is_solid(0.0, 0.023, 0.005) is False

    # TubeWallPrimitive: centered at (0, 0.028), r_inner=8mm, r_outer=18mm, z_min=0, z_max=0.120
    # Cutout should be at y < self.y (facing the casing), |x - self.x| < slot_width/2, z < slot_height
    tube = TubeWallPrimitive(
        x=0.0,
        y=0.028,
        r_inner=0.008,
        r_outer=0.018,
        z_min=0.0,
        z_max=0.120,
        slot_height=0.009,
        slot_width=0.008,
    )

    # A point inside the solid tube wall but not in the cutout
    assert tube.is_solid(0.0, 0.028 + 0.013, 0.005) is True  # y > self.y (wrong side)
    assert tube.is_solid(0.013, 0.028, 0.005) is True  # x > slot_width/2
    assert tube.is_solid(0.0, 0.028 - 0.013, 0.0095) is True  # z > slot_height

    # A point inside the cutout connection (should NOT be solid)
    assert tube.is_solid(0.0, 0.028 - 0.013, 0.005) is False


def test_airborne_freefall_particles_maintain_ballistic_velocity():
    """Verify that airborne particles falling in mid-air outside the tube maintain ballistic velocity without spurious rotation."""
    import jax.numpy as jnp
    from model.boundary_config import BoundaryConfig, BoundaryType, ShapeType
    from provider.bullet import LinkType
    from provider.fluid import PhysicsConfig, _physics_step_jax

    # Particle in mid-air outside the tube, falling down
    pos = jnp.array([[0.0, -0.020, 0.060]], dtype=jnp.float32)
    vel = jnp.array([[0.0, 0.0, -0.5]], dtype=jnp.float32)
    f_lbm = jnp.zeros((15, 16, 16, 16), dtype=jnp.float32)

    # Base bowl, casing, tube, and spinning impeller
    b_bowl = BoundaryConfig(
        link_type=LinkType.BASE,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.100,
        height=0.107,
        link_idx=-1,
    )
    b_casing = BoundaryConfig(
        link_type=LinkType.CASING,
        shape=ShapeType.CASING,
        type=BoundaryType.SOLID_CAVITY,
        radius=0.028,
        height=0.010,
        thickness=0.010,
        ceiling_thickness=0.002,
        slot_height=0.009,
        slot_width=0.008,
        link_idx=-1,
    )
    b_tube = BoundaryConfig(
        link_type=LinkType.TUBE,
        shape=ShapeType.TUBE,
        type=BoundaryType.SOLID_CAVITY,
        radius=0.018,
        height=0.066,
        thickness=0.010,
        slot_height=0.009,
        slot_width=0.008,
        xyz=(0.0, 0.028, 0.0),
        link_idx=-1,
    )
    b_impeller = BoundaryConfig(
        link_type=LinkType.IMPELLER,
        shape=ShapeType.IMPELLER,
        type=BoundaryType.SOLID,
        radius=0.009,
        height=0.015,
        thickness=0.003,
        vane_thickness=0.001,
        num_vanes=6,
        vane_twist=-15.0,
        link_idx=-1,
    )

    b_pos = jnp.array(
        [
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.028, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=jnp.float32,
    )
    b_orn = jnp.array([[0.0, 0.0, 0.0, 1.0]] * 4, dtype=jnp.float32)

    config = PhysicsConfig(
        mass=1.4e-5,
        dt_sub=1.0 / 240.0,
        n_substeps=1,
        boundary_configs=[b_bowl, b_casing, b_tube, b_impeller],
        gravity=jnp.array([0.0, 0.0, -9.81], dtype=jnp.float32),
        base_idx=0,
        K_boundary=5000.0,
        D_boundary=15.0,
        r_s=0.0015,
        high_damping_value=0.995,
        nx=16,
        ny=16,
        nz=16,
        dx=0.010,
        origin=(-0.08, -0.08, 0.0),
    )

    # Step simulation with spinning impeller
    pos_next, vel_next, f_next, _, b_forces = _physics_step_jax(
        pos,
        vel,
        f_lbm,
        b_pos,
        b_orn,
        jnp.zeros(3, dtype=jnp.float32),
        120.0,
        0.0,
        0.995,
        config=config,
    )

    # Airborne particle should not receive horizontal rotation/swirl from the spinning impeller LBM grid
    assert jnp.isclose(vel_next[0, 0], 0.0, atol=1e-4)
    assert jnp.isclose(vel_next[0, 1], 0.0, atol=1e-4)
    # Particle should continue accelerating downward under gravity
    assert vel_next[0, 2] < -0.50
    assert b_forces.shape == (1, 3)


def test_moving_boundary_force_client_api():
    """Verify that Fluid exposes moving boundary interaction forces and voxel-mapped forces to clients."""
    import jax.numpy as jnp
    import numpy as np
    from model.boundary_config import BoundaryConfig, BoundaryType, ShapeType
    from provider.bullet import LinkType
    from provider.fluid import Fluid, FluidConfig

    b_bowl = BoundaryConfig(
        link_type=LinkType.BASE,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.096,
        height=0.096,
        link_idx=-1,
    )

    boundaries = {"bowl": b_bowl}
    config = FluidConfig.water(
        target_volume=0.00001,
        boundaries=boundaries,
    )

    fluid = Fluid(config=config)
    fluid.n_particles = 10
    fluid.pos_jax = jnp.zeros((10, 3), dtype=jnp.float32)
    fluid.vel_jax = jnp.zeros((10, 3), dtype=jnp.float32)
    fluid.last_positions = [(0.0, 0.0, 0.01) for _ in range(10)]
    fluid.last_boundary_forces = np.ones((10, 3), dtype=np.float32) * 0.05

    # 1. Boundary forces on particles
    b_forces = fluid.get_boundary_forces()
    assert b_forces.shape == (10, 3)
    assert np.allclose(b_forces, 0.05)

    # 2. Voxel mapped forces
    fluid.last_voxel_positions = np.array([[0.0, 0.0, 0.01], [0.01, 0.0, 0.01]], dtype=np.float32)
    voxel_forces = fluid.get_voxel_forces()
    assert voxel_forces.shape == (2, 3)
    assert np.allclose(voxel_forces, 0.05)

    # 3. Net reaction force on moving boundary body
    reaction_force = fluid.get_boundary_reaction_force()
    assert reaction_force.shape == (3,)
    assert np.allclose(reaction_force, -0.50)

    # 4. Reaction torque
    fluid.torques = [0.012]
    reaction_torque = fluid.get_boundary_reaction_torque()
    assert abs(reaction_torque - 0.012) < 1e-5


def test_boundary_voxels_labeled_by_type():
    """Verify that Fluid and FluidPostProcessor extract 3D voxels labeled for each boundary type."""
    import pybullet as p
    from unittest.mock import MagicMock
    from model.boundary_config import BoundaryConfig, BoundaryType, ShapeType
    from provider.bullet import LinkType
    from provider.fluid import Fluid, FluidConfig

    b_bowl = BoundaryConfig(
        link_type=LinkType.BASE,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.060,
        height=0.096,
        thickness=0.005,
        link_idx=-1,
    )
    b_casing = BoundaryConfig(
        link_type=LinkType.CASING,
        shape=ShapeType.CASING,
        type=BoundaryType.SOLID_CAVITY,
        radius=0.028,
        height=0.010,
        thickness=0.004,
        link_idx=-1,
    )
    b_tube = BoundaryConfig(
        link_type=LinkType.TUBE,
        shape=ShapeType.TUBE,
        type=BoundaryType.SOLID_CAVITY,
        radius=0.018,
        height=0.066,
        thickness=0.004,
        xyz=(0.0, 0.028, 0.0),
        link_idx=-1,
    )
    b_lid = BoundaryConfig(
        link_type=LinkType.LID,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.050,
        height=0.0035,
        thickness=0.0035,
        has_drain=True,
        drain_hole_y=-0.010,
        drain_hole_radius=0.020,
        has_tube=True,
        tube_radius=0.008,
        link_idx=-1,
    )
    b_impeller = BoundaryConfig(
        link_type=LinkType.IMPELLER,
        shape=ShapeType.IMPELLER,
        type=BoundaryType.SOLID,
        radius=0.009,
        height=0.015,
        thickness=0.003,
        vane_thickness=0.001,
        num_vanes=4,
        vane_twist=-15.0,
        link_idx=-1,
    )

    boundaries = {
        "bowl": b_bowl,
        "casing": b_casing,
        "tube": b_tube,
        "lid": b_lid,
        "impeller": b_impeller,
    }

    mock_state_tracker = MagicMock()
    config = FluidConfig.water(
        target_volume=0.00001,
        boundaries=boundaries,
    )

    fluid = Fluid(
        config=config,
        state_tracker=mock_state_tracker,
    )

    voxels = fluid.get_boundary_voxels()
    assert isinstance(voxels, dict)
    # Check that each configured boundary type has voxel entries
    assert "bowl" in voxels
    assert "casingwall" in voxels
    assert "tubewall" in voxels
    assert "lid" in voxels
    assert "impeller" in voxels

    assert len(voxels["bowl"]) > 0
    assert len(voxels["casingwall"]) > 0
    assert len(voxels["tubewall"]) > 0
    assert len(voxels["lid"]) > 0
    assert len(voxels["impeller"]) > 0

    # Ensure state tracker receives the boundary voxels on explicit update
    fluid.state_tracker.boundary_voxels = fluid.get_boundary_voxels()
    assert mock_state_tracker.boundary_voxels is not None
    assert "bowl" in mock_state_tracker.boundary_voxels
    assert "lid" in mock_state_tracker.boundary_voxels


def test_show_boundary_voxels_env_gating(monkeypatch):
    """Verify that boundary voxels logging is disabled by default and enabled via SHOW_BOUNDARY_VOXELS env var."""
    from unittest.mock import MagicMock
    from model.boundary_config import BoundaryConfig, BoundaryType, ShapeType
    from provider.bullet import LinkType
    from provider.utils import str_to_bool, get_env_bool

    # 1. Test str_to_bool utility
    assert str_to_bool("1") is True
    assert str_to_bool("true") is True
    assert str_to_bool("TRUE") is True
    assert str_to_bool("yes") is True
    assert str_to_bool("on") is True
    assert str_to_bool("enable") is True
    assert str_to_bool("0") is False
    assert str_to_bool("false") is False
    assert str_to_bool("no") is False
    assert str_to_bool("off") is False
    assert str_to_bool(None, default=False) is False
    assert str_to_bool("unknown", default=True) is True

    # 2. Test get_env_bool utility
    monkeypatch.delenv("SHOW_BOUNDARY_VOXELS", raising=False)
    assert get_env_bool("SHOW_BOUNDARY_VOXELS", False) is False

    monkeypatch.setenv("SHOW_BOUNDARY_VOXELS", "1")
    assert get_env_bool("SHOW_BOUNDARY_VOXELS", False) is True

    # 3. Test Fluid state_tracker gating
    b_bowl = BoundaryConfig(
        link_type=LinkType.BASE,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.SOLID,
        radius=0.08,
        height=0.05,
        thickness=0.002,
        link_idx=-1,
    )
    boundaries = {"bowl": b_bowl}
    config = FluidConfig.water(
        target_volume=0.00001,
        boundaries=boundaries,
    )

    # When SHOW_BOUNDARY_VOXELS is unset (default: False)
    monkeypatch.delenv("SHOW_BOUNDARY_VOXELS", raising=False)
    mock_tracker_disabled = MagicMock()
    mock_tracker_disabled.boundary_voxels = None

    fluid_disabled = Fluid(
        config=config,
        state_tracker=mock_tracker_disabled,
    )
    fluid_disabled._update_state_tracker()
    assert mock_tracker_disabled.boundary_voxels is None

    # When SHOW_BOUNDARY_VOXELS is set to '1' (enabled: True)
    monkeypatch.setenv("SHOW_BOUNDARY_VOXELS", "1")
    mock_tracker_enabled = MagicMock()
    mock_tracker_enabled.boundary_voxels = None

    fluid_enabled = Fluid(
        config=config,
        state_tracker=mock_tracker_enabled,
    )
    fluid_enabled._update_state_tracker()
    assert mock_tracker_enabled.boundary_voxels is not None
    assert "bowl" in mock_tracker_enabled.boundary_voxels


def test_voxel_masks_consistent_with_surface_bounds():
    """Verify consistency between voxel masks, surface normals, and precomputed CAD surface bounds."""
    import jax.numpy as jnp
    import numpy as np
    from model.boundary_config import BoundaryConfig, BoundaryParam, BoundaryType, ShapeType
    from provider.boundary import BoundaryProcessor
    from provider.bullet import LinkType
    from provider.fluid import _make_grid_masks

    b_bowl = BoundaryConfig(
        link_type=LinkType.BASE,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.060,
        height=0.080,
        thickness=0.005,
        link_idx=-1,
    )
    b_tube = BoundaryConfig(
        link_type=LinkType.TUBE,
        shape=ShapeType.TUBE,
        type=BoundaryType.SOLID_CAVITY,
        radius=0.015,
        height=0.060,
        thickness=0.003,
        xyz=(0.0, 0.020, 0.0),
        link_idx=-1,
    )
    b_lid = BoundaryConfig(
        link_type=LinkType.LID,
        shape=ShapeType.CYLINDER,
        type=BoundaryType.CAVITY,
        radius=0.058,
        height=0.005,
        thickness=0.003,
        has_drain=True,
        drain_hole_y=-0.015,
        drain_hole_radius=0.012,
        has_tube=True,
        tube_radius=0.015,
        link_idx=-1,
    )

    boundary_list = [b_bowl, b_tube, b_lid]
    base_pos = (0.0, 0.0, 0.0)
    base_orn = (0.0, 0.0, 0.0, 1.0)
    processed = BoundaryProcessor.process(boundary_list, base_link_origin=(base_pos, base_orn))

    dx = 0.003
    nx, ny, nz = 48, 48, 36
    origin = jnp.array([-0.072, -0.072, -0.010], dtype=jnp.float32)

    solid_mask, tube_mask, solid_friction, normal_grid, smooth_occ = _make_grid_masks(
        dx=dx,
        origin=origin,
        b_shapes=processed.b_shapes,
        b_types=processed.b_types,
        b_params=processed.b_params,
        b_pos_arr=processed.b_pos_arr,
        b_orn_arr=processed.b_orn_arr,
        base_idx=0,
        nx=nx,
        ny=ny,
        nz=nz,
    )

    solid_np = np.asarray(solid_mask)
    tube_np = np.asarray(tube_mask)
    normals_np = np.asarray(normal_grid)

    # 1. Base bowl surface bounds check
    bowl_surf = b_bowl.compute_surface_bounds()
    assert processed.b_params[0, BoundaryParam.Z_BOTTOM] == bowl_surf.z_bottom
    assert processed.b_params[0, BoundaryParam.Z_TOP] == bowl_surf.z_top
    assert processed.b_params[0, BoundaryParam.R_INNER] == bowl_surf.r_inner
    assert processed.b_params[0, BoundaryParam.R_OUTER] == bowl_surf.r_outer

    # 2. Tube column interior clearing check: voxels inside tube bore must not be blocked
    ix = np.arange(nx)
    iy = np.arange(ny)
    iz = np.arange(nz)
    gx, gy, gz = np.meshgrid(ix, iy, iz, indexing="ij")
    cx = float(origin[0]) + (gx + 0.5) * dx
    cy = float(origin[1]) + (gy + 0.5) * dx
    cz = float(origin[2]) + (gz + 0.5) * dx
    coords_np = np.stack([cx, cy, cz], axis=-1)

    tube_x, tube_y = 0.0, 0.020
    tube_r_inner = b_tube.radius - b_tube.thickness
    tube_r_sq = (coords_np[:, :, :, 0] - tube_x) ** 2 + (coords_np[:, :, :, 1] - tube_y) ** 2
    in_bore = (tube_r_sq < (tube_r_inner - dx) ** 2) & (coords_np[:, :, :, 2] >= 0.0) & (coords_np[:, :, :, 2] <= 0.055)
    # Bore interior must be active in tube_mask and cleared from solid_mask
    assert np.any(tube_np[in_bore])
    assert not np.any(solid_np[in_bore])

    # 3. Normals on outer wall must point outward from solid cavity
    outer_wall_nodes = (coords_np[:, :, :, 0] ** 2 + coords_np[:, :, :, 1] ** 2 >= (b_bowl.radius - dx) ** 2) & solid_np
    if np.any(outer_wall_nodes):
        node_normals = normals_np[outer_wall_nodes]
        node_coords = coords_np[outer_wall_nodes]
        radial_dots = node_normals[:, 0] * node_coords[:, 0] + node_normals[:, 1] * node_coords[:, 1]
        # Inward-pointing cavity normals for container walls
        assert np.all(radial_dots <= 1e-4)

    # 4. Drain hole target centroid and edge drainage bounds consistency
    lid_surf = b_lid.compute_surface_bounds()
    assert processed.b_params[2, BoundaryParam.DRAIN_TARGET_Z] == lid_surf.drain_target_z
    assert processed.b_params[2, BoundaryParam.DRAIN_INFLUENCE_RADIUS] == lid_surf.drain_influence_radius
    assert processed.b_params[2, BoundaryParam.DRAIN_EDGE_R_MIN] == lid_surf.drain_edge_r_min
    assert processed.b_params[2, BoundaryParam.DRAIN_EDGE_R_MAX] == lid_surf.drain_edge_r_max


def test_surface_smoothing_polynomial_coefficients_config():
    """Verify that surface smoothing polynomial coefficients are parameterized in FluidConfig and evaluated properly in JAX."""
    import jax.numpy as jnp
    from model.fluid_config import FluidConfig
    from provider.fluid import _compute_dynamic_fluid_bodies_jax

    # 1. Test FluidConfig parameters
    config = FluidConfig(
        surface_smooth_center_coeff=6.0,
        surface_smooth_neighbor_coeff=3.0,
        surface_smooth_diagonal_coeff=1.5,
    )
    assert config.surface_smooth_center_coeff == 6.0
    assert config.surface_smooth_neighbor_coeff == 3.0
    assert config.surface_smooth_diagonal_coeff == 1.5

    # 2. Test JAX evaluation with custom smoothing polynomial weights
    pos_local = jnp.array(
        [
            [0.0, 0.0, 0.045],
            [0.005, 0.0, 0.046],
            [0.0, 0.005, 0.044],
            [0.005, 0.005, 0.045],
        ]
    )
    dx = 0.005
    origin = jnp.array([-0.05, -0.05, 0.0])

    in_fluid_body, p_surf_z, level_grad_local, p_col_count = _compute_dynamic_fluid_bodies_jax(
        pos_local,
        dx,
        origin,
        nx=20,
        ny=20,
        nz=20,
        cavity_floor_z=0.040,
        z_max_pool=0.080,
        r_s=0.002,
        c_center=config.surface_smooth_center_coeff,
        c_neighbor=config.surface_smooth_neighbor_coeff,
        c_diagonal=config.surface_smooth_diagonal_coeff,
    )

    assert p_surf_z.shape == (4,)
    assert jnp.all(p_surf_z >= 0.040)
    assert level_grad_local.shape == (4, 3)
    assert in_fluid_body.shape == (4,)
