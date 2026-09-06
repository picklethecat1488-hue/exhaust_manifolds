"""Unit tests for dynamic fluid body primitives (move, split, merge) and shape recomputation."""

import numpy as np
from model.fluid_body import (
    FluidBody,
    FluidBodyTracker,
    FluidBodyType,
    FluidStage,
    FluidCADContext,
    CADFeature,
    CADFeatureType,
)


def test_fluid_body_shape_recomputation():
    """Test recomputing geometric bounds, centroid, volume, and velocity for a fluid body."""
    positions = np.array(
        [
            [0.0, 0.0, 0.040],
            [0.010, 0.0, 0.040],
            [0.0, 0.010, 0.040],
            [0.010, 0.010, 0.050],
        ],
        dtype=np.float32,
    )
    velocities = np.array(
        [
            [0.0, 0.0, 0.1],
            [0.1, 0.0, 0.1],
            [0.0, 0.1, 0.1],
            [0.1, 0.1, 0.2],
        ],
        dtype=np.float32,
    )
    r_s = 0.0025

    body = FluidBody(
        body_id=1,
        body_type=FluidBodyType.POOL,
        particle_indices=np.array([0, 1, 2, 3]),
    )
    body.recompute_shape(positions, velocities, r_s)

    assert body.particle_count == 4
    assert np.isclose(body.centroid[0], 0.005)
    assert np.isclose(body.centroid[1], 0.005)
    assert np.isclose(body.centroid[2], 0.0425)
    assert body.bounds_min[2] <= 0.040
    assert body.bounds_max[2] >= 0.050
    assert body.volume > 0.0


def test_fluid_body_move_primitive():
    """Test translating a fluid body by a 3D displacement vector."""
    body = FluidBody(
        body_id=1,
        body_type=FluidBodyType.POOL,
        particle_indices=np.array([0, 1]),
        centroid=(0.0, 0.0, 0.050),
        bounds_min=(-0.010, -0.010, 0.040),
        bounds_max=(0.010, 0.010, 0.060),
    )

    body.move((0.005, -0.002, 0.010))

    assert np.isclose(body.centroid[0], 0.005)
    assert np.isclose(body.centroid[1], -0.002)
    assert np.isclose(body.centroid[2], 0.060)
    assert np.isclose(body.bounds_min[0], -0.005)
    assert np.isclose(body.bounds_max[2], 0.070)


def test_fluid_body_split_primitive():
    """Test splitting a fluid body into distinct child bodies."""
    positions = np.array(
        [
            # Cluster A (Pool)
            [0.0, 0.0, 0.040],
            [0.005, 0.0, 0.040],
            # Cluster B (Stream)
            [0.0, 0.028, 0.080],
            [0.0, 0.028, 0.090],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((4, 3), dtype=np.float32)
    r_s = 0.0025

    parent = FluidBody(
        body_id=10,
        body_type=FluidBodyType.POOL,
        particle_indices=np.array([0, 1, 2, 3]),
    )

    counter = [11]

    def next_id():
        nid = counter[0]
        counter[0] += 1
        return nid

    children = parent.split(
        [np.array([0, 1]), np.array([2, 3])],
        positions,
        velocities,
        r_s,
        next_id_fn=next_id,
    )

    assert len(children) == 2
    assert children[0].body_id == 11
    assert children[0].particle_count == 2
    assert np.isclose(children[0].centroid[2], 0.040)

    assert children[1].body_id == 12
    assert children[1].particle_count == 2
    assert np.isclose(children[1].centroid[2], 0.085)


def test_fluid_body_merge_primitive():
    """Test merging two fluid bodies into a unified physical body."""
    positions = np.array(
        [
            [0.0, 0.0, 0.040],
            [0.010, 0.0, 0.040],
            [0.020, 0.0, 0.040],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((3, 3), dtype=np.float32)
    r_s = 0.0025

    body1 = FluidBody(
        body_id=1,
        body_type=FluidBodyType.POOL,
        particle_indices=np.array([0, 1]),
    )
    body1.recompute_shape(positions, velocities, r_s)

    body2 = FluidBody(
        body_id=2,
        body_type=FluidBodyType.CLUSTER,
        particle_indices=np.array([2]),
    )
    body2.recompute_shape(positions, velocities, r_s)

    merged = body1.merge(body2, positions, velocities, r_s)

    assert merged.body_id == 1
    assert merged.particle_count == 3
    assert np.array_equal(merged.particle_indices, np.array([0, 1, 2]))
    assert np.isclose(merged.centroid[0], 0.010)


def test_fluid_body_tracker():
    """Test dynamic classification and updating of fluid bodies across the 5-stage fountain cascade."""
    positions = np.array(
        [
            # 1. Bowl Reservoir Pool particles (in front reservoir)
            [0.0, -0.050, 0.045],
            [0.010, -0.050, 0.045],
            [-0.010, -0.050, 0.045],
            # 2. Internal Stream particle (in tube at Y=0.028)
            [0.0, 0.028, 0.060],
            [0.002, 0.028, 0.065],
            # 3. Top Platform Water Sheet (Z=0.106 on platform around Y=0.028)
            [0.010, 0.028, 0.106],
            [-0.010, 0.028, 0.106],
            # 4. Upper Waterfall particle (cascading off platform lip)
            [0.028, 0.028, 0.100],
            # 5. Lid Drinking Shelf Pool (on lid shelf at Y=0.040, outside platform)
            [0.040, 0.040, 0.100],
            [0.0, -0.020, 0.099],  # at drain aperture!
            # 6. Lower Waterfall particle (falling through front cutout at Y=-0.020)
            [0.0, -0.020, 0.070],
            [0.005, -0.020, 0.075],
            # 7. Free splash cluster in air
            [0.060, 0.030, 0.080],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((13, 3), dtype=np.float32)

    cad_context = FluidCADContext(
        features=(
            CADFeature("Tube", x=0.0, y=0.028, z=0.041, r=0.010),
            CADFeature("Terrace", x=0.0, y=0.028, z=0.108, r=0.030),
            CADFeature("Drain", x=0.0, y=-0.020, z=0.098, r=0.0154),
            CADFeature("Pocket", x=0.0, y=0.0, z=0.098, r=0.080),
            CADFeature("Bowl", x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    tracker = FluidBodyTracker(r_s=0.0025)
    bodies = tracker.update_bodies(
        positions,
        velocities,
        cad_context=cad_context,
    )

    types = {b.body_type for b in bodies}
    assert FluidBodyType.POOL in types
    assert FluidBodyType.STREAM in types
    assert FluidBodyType.SHEET in types
    assert FluidBodyType.WATERFALL in types
    assert FluidBodyType.CLUSTER in types

    # Check semantic cascade stages
    stages = {b.stage for b in bodies}
    assert FluidStage.DELIVERY_STREAM in stages
    assert FluidStage.TOP_SHEET in stages
    assert FluidStage.LIP_WATERFALL in stages
    assert FluidStage.LID_POOL in stages
    assert FluidStage.DRAIN_WATERFALL in stages
    assert FluidStage.BOWL_POOL in stages
    assert FluidStage.SPLASH_CLUSTER in stages


def test_fluid_body_to_mesh_and_cad_solid():
    """Verify watertight mesh and CAD solid generation for various fluid body types."""
    from model.fluid_body import (
        generate_cylinder_mesh,
        generate_sphere_mesh,
        generate_heightfield_cylinder_mesh,
        generate_waterfall_mesh,
        generate_lip_waterfall_mesh,
        generate_box_mesh,
    )

    # 1. Cylinder mesh generator
    verts, faces = generate_cylinder_mesh(radius=0.050, z_min=0.040, z_max=0.080, center=(0.0, 0.0), n_segments=16)
    assert len(verts) == 16 * 2 + 2
    assert len(faces) == 16 * 4  # 16 quad sides (32 tris) + 16 bottom tris + 16 top tris = 64 tris
    assert np.all(verts[:, 2] >= 0.040)
    assert np.all(verts[:, 2] <= 0.080)

    # 2. Waterfall mesh generator
    wf_verts, wf_faces = generate_waterfall_mesh(None, z_top=0.105, z_bot=0.048, cutout_xy=(0.0, -0.020))
    assert len(wf_verts) > 0
    assert len(wf_faces) > 0
    wf_edges = {}
    for face in wf_faces:
        for i in range(3):
            e_canon = tuple(sorted((face[i], face[(i + 1) % 3])))
            wf_edges[e_canon] = wf_edges.get(e_canon, 0) + 1
    assert all(count == 2 for count in wf_edges.values())

    # 2b. Lip Waterfall mesh generator
    lip_verts, lip_faces = generate_lip_waterfall_mesh(None, center_xy=(0.0, 0.028), lip_radius=0.030)
    assert len(lip_verts) > 0
    assert len(lip_faces) > 0
    lip_edges = {}
    for face in lip_faces:
        for i in range(3):
            e_canon = tuple(sorted((face[i], face[(i + 1) % 3])))
            lip_edges[e_canon] = lip_edges.get(e_canon, 0) + 1
    assert all(count == 2 for count in lip_edges.values())

    # 3. Sphere mesh generator
    sp_verts, sp_faces = generate_sphere_mesh(center=(0.0, 0.0, 0.050), radius=0.010, n_lat=8, n_lon=16)
    assert len(sp_verts) > 0
    assert len(sp_faces) > 0
    # Check watertightness: every edge appears exactly twice
    sp_edges = {}
    for face in sp_faces:
        for i in range(3):
            e_canon = tuple(sorted((face[i], face[(i + 1) % 3])))
            sp_edges[e_canon] = sp_edges.get(e_canon, 0) + 1
    assert all(count == 2 for count in sp_edges.values())

    # 4. Heightfield cylinder mesh
    surf_pts = np.array(
        [
            [0.0, 0.0, 0.050],
            [0.005, 0.005, 0.048],
            [-0.005, 0.005, 0.049],
            [0.0, -0.005, 0.047],
        ],
        dtype=np.float32,
    )
    hf_verts, hf_faces = generate_heightfield_cylinder_mesh(
        radius=0.020,
        z_floor=0.0,
        surface_positions=surf_pts,
        default_z_top=0.050,
        center=(0.0, 0.0),
        n_rings=4,
        n_spokes=16,
    )
    assert hf_verts.shape[0] > 0
    assert hf_faces.shape[0] > 0

    # 4. Waterfall surface mesh (lower drain cutout)
    wf_verts, wf_faces = generate_waterfall_mesh(
        positions=surf_pts,
        z_top=0.050,
        z_bot=0.0,
        cutout_xy=(0.0, -0.020),
        n_segments=16,
    )
    assert wf_verts.shape[0] > 0
    assert wf_faces.shape[0] > 0

    # 5. Lip waterfall surface mesh (upper terrace 360-degree curtain)
    lip_verts, lip_faces = generate_lip_waterfall_mesh(
        positions=surf_pts,
        center_xy=(0.0, 0.028),
        lip_radius=0.030,
        z_top=0.108,
        z_bot=0.098,
        n_segments=16,
    )
    assert lip_verts.shape[0] > 0
    assert lip_faces.shape[0] > 0

    # 6. Pool FluidBody to_mesh and to_cad_solid
    pool = FluidBody(
        body_id=1,
        body_type=FluidBodyType.POOL,
        stage=FluidStage.BOWL_POOL,
        feature_type=CADFeatureType.BOWL,
        tier=2,
        particle_indices=np.array([0, 1, 2]),
        bounds_min=(-0.050, -0.050, 0.0),
        bounds_max=(0.050, 0.050, 0.020),
        centroid=(0.0, 0.0, 0.010),
        volume=0.0001,
        surface_positions=surf_pts,
    )
    p_verts, p_faces = pool.to_mesh()
    assert len(p_verts) > 0
    assert len(p_faces) > 0
    assert pool.to_cad_solid().volume > 0.0

    # 7. Stream FluidBody to_mesh and to_cad_solid
    stream = FluidBody(
        body_id=2,
        body_type=FluidBodyType.STREAM,
        stage=FluidStage.DELIVERY_STREAM,
        feature_type=CADFeatureType.TUBE,
        tier=0,
        particle_indices=np.array([0, 1]),
        bounds_min=(-0.005, 0.020, 0.040),
        bounds_max=(0.005, 0.035, 0.100),
        centroid=(0.0, 0.028, 0.070),
        volume=0.00001,
    )
    s_verts, s_faces = stream.to_mesh()
    assert len(s_verts) > 0
    assert len(s_faces) > 0
    assert stream.to_cad_solid().volume > 0.0

    # 8. Waterfall FluidBody to_mesh and to_cad_solid
    waterfall = FluidBody(
        body_id=3,
        body_type=FluidBodyType.WATERFALL,
        stage=FluidStage.LIP_WATERFALL,
        feature_type=CADFeatureType.TERRACE,
        tier=0,
        particle_indices=np.array([0, 1, 2]),
        bounds_min=(-0.030, 0.0, 0.098),
        bounds_max=(0.030, 0.060, 0.108),
        centroid=(0.0, 0.028, 0.103),
        volume=0.00002,
        surface_positions=surf_pts,
    )
    w_verts, w_faces = waterfall.to_mesh()
    assert len(w_verts) > 0
    assert len(w_faces) > 0
    assert waterfall.to_cad_solid().volume > 0.0

    # 9. Cluster FluidBody to_mesh and to_cad_solid
    cluster = FluidBody(
        body_id=4,
        body_type=FluidBodyType.CLUSTER,
        stage=FluidStage.SPLASH_CLUSTER,
        particle_indices=np.array([0, 1, 2, 3]),
        bounds_min=(-0.010, -0.010, 0.050),
        bounds_max=(0.010, 0.010, 0.060),
        centroid=(0.0, 0.0, 0.055),
        volume=0.000005,
        surface_positions=surf_pts,
    )
    c_verts, c_faces = cluster.to_mesh()
    assert len(c_verts) > 0
    assert len(c_faces) > 0
    assert cluster.to_cad_solid().volume > 0.0


def test_fluid_cad_context_tuple_representation():
    """Verify that FluidCADContext represents CAD geometry as an ordered sequence of CADFeatures with CADFeatureType."""
    ctx = FluidCADContext(
        features=(
            CADFeature(CADFeatureType.TUBE, x=0.0, y=0.028, z=0.041, r=0.010),
            CADFeature(CADFeatureType.TERRACE, x=0.0, y=0.028, z=0.108, r=0.030),
            CADFeature(CADFeatureType.DRAIN, x=0.0, y=-0.020, z=0.098, r=0.055),
            CADFeature(CADFeatureType.POCKET, x=0.0, y=0.0, z=0.098, r=0.080),
            CADFeature(CADFeatureType.BOWL, x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    # 1. Verify list/tuple of CADFeatures with CADFeatureType and (X, Y, Z, R) coordinates
    assert len(ctx.features) == 5

    names = [f.name for f in ctx.features]
    assert names == ["Tube", "Terrace", "Drain", "Pocket", "Bowl"]

    types = [f.feature_type for f in ctx.features]
    assert types == [
        CADFeatureType.TUBE,
        CADFeatureType.TERRACE,
        CADFeatureType.DRAIN,
        CADFeatureType.POCKET,
        CADFeatureType.BOWL,
    ]

    for feat in ctx.features:
        assert isinstance(feat.feature_type, CADFeatureType)
        assert isinstance(feat.name, str)
        assert len(feat.coords) == 4
        x, y, z, r = feat.coords
        assert isinstance(x, float)
        assert isinstance(y, float)
        assert isinstance(z, float)
        assert isinstance(r, float)

    # 2. Verify relative ordering index lookups and CADFeatureType/name lookups
    assert ctx.features[0].feature_type == CADFeatureType.TUBE
    assert ctx.get(CADFeatureType.TUBE) == ctx.features[0]
    assert ctx.get(CADFeatureType.TERRACE) == ctx.features[1]
    assert ctx.get(CADFeatureType.DRAIN) == ctx.features[2]
    assert ctx.get(CADFeatureType.POCKET) == ctx.features[3]
    assert ctx.get(CADFeatureType.BOWL) == ctx.features[4]

    # String lookup backwards compatibility
    assert ctx.get("Tube") == ctx.features[0]
    assert ctx.get("Terrace") == ctx.features[1]

    # 3. Verify semantic CAD feature accessors and derived properties
    tube = ctx.get(CADFeatureType.TUBE)
    assert tube is not None
    assert tube.y == 0.028
    assert tube.r == 0.010

    terrace = ctx.get(CADFeatureType.TERRACE)
    assert terrace is not None
    assert terrace.z == 0.108
    assert terrace.r == 0.030

    drain = ctx.get(CADFeatureType.DRAIN)
    assert drain is not None
    assert drain.y == -0.020
    assert drain.r == 0.055

    pocket = ctx.get(CADFeatureType.POCKET)
    assert pocket is not None
    assert pocket.r == 0.080

    bowl = ctx.get(CADFeatureType.BOWL)
    assert bowl is not None
    assert bowl.z == 0.041

    assert len(ctx.tubes) == 1
    assert len(ctx.terraces) == 1
    assert len(ctx.drains) == 1
    assert len(ctx.pockets) == 1
    assert len(ctx.bowls) == 1

    assert ctx.z_floor == 0.041
    assert ctx.z_lid == 0.098


def test_cascade_tier_continuity_and_intersections():
    """Verify that all cascade tiers physically and geometrically intersect without floating gaps."""
    ctx = FluidCADContext(
        features=(
            CADFeature(CADFeatureType.TUBE, x=0.0, y=0.028, z=0.041, r=0.010),
            CADFeature(CADFeatureType.TERRACE, x=0.0, y=0.028, z=0.108, r=0.030),
            CADFeature(CADFeatureType.DRAIN, x=0.0, y=-0.020, z=0.098, r=0.0154),
            CADFeature(CADFeatureType.POCKET, x=0.0, y=0.0, z=0.098, r=0.080),
            CADFeature(CADFeatureType.BOWL, x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    positions = np.array(
        [
            [0.0, 0.028, 0.060],  # Stream
            [0.002, 0.028, 0.065],
            [0.010, 0.028, 0.109],  # Top sheet
            [-0.010, 0.028, 0.109],
            [0.025, 0.028, 0.102],  # Lip waterfall
            [0.040, 0.0, 0.099],  # Lid pool
            [0.0, -0.020, 0.099],  # at drain aperture
            [0.0, -0.020, 0.085],  # Drain waterfall (upper)
            [0.0, -0.020, 0.070],  # Drain waterfall (mid)
            [0.002, -0.020, 0.055],  # Drain waterfall (lower, entering pool)
            [0.050, 0.050, 0.045],  # Bowl pool
            [-0.050, 0.050, 0.045],
        ],
        dtype=np.float32,
    )
    velocities = np.zeros((12, 3), dtype=np.float32)

    tracker = FluidBodyTracker(r_s=0.0025)
    bodies = tracker.update_bodies(positions, velocities, cad_context=ctx)
    body_map = {b.stage: b for b in bodies}

    assert FluidStage.DELIVERY_STREAM in body_map
    assert FluidStage.TOP_SHEET in body_map
    assert FluidStage.LIP_WATERFALL in body_map
    assert FluidStage.LID_POOL in body_map
    assert FluidStage.DRAIN_WATERFALL in body_map
    assert FluidStage.BOWL_POOL in body_map

    stream_verts, _ = body_map[FluidStage.DELIVERY_STREAM].to_mesh()
    sheet_verts, _ = body_map[FluidStage.TOP_SHEET].to_mesh()
    lip_verts, _ = body_map[FluidStage.LIP_WATERFALL].to_mesh()
    lid_verts, _ = body_map[FluidStage.LID_POOL].to_mesh()
    drain_verts, _ = body_map[FluidStage.DRAIN_WATERFALL].to_mesh()
    bowl_verts, _ = body_map[FluidStage.BOWL_POOL].to_mesh()

    # 1. Delivery Stream -> Top Sheet intersection
    assert np.max(stream_verts[:, 2]) >= np.min(sheet_verts[:, 2])

    # 2. Top Sheet -> Lip Waterfall intersection (top of curtain touches terrace sheet)
    assert np.max(lip_verts[:, 2]) >= np.min(sheet_verts[:, 2])

    # 3. Lip Waterfall -> Lid Pool intersection (curtain lands on lid shelf)
    assert np.min(lip_verts[:, 2]) <= np.max(lid_verts[:, 2])
    assert np.min(lip_verts[:, 2]) <= ctx.z_lid

    # 4. Lid Pool -> Drain Waterfall intersection (waterfall emerges directly out of lid pool)
    assert np.max(drain_verts[:, 2]) >= np.min(lid_verts[:, 2])
    assert np.max(drain_verts[:, 2]) >= ctx.z_lid

    # 5. Drain Waterfall -> Bowl Pool intersection (waterfall plunges into reservoir pool)
    assert np.min(drain_verts[:, 2]) <= np.max(bowl_verts[:, 2])


def test_lid_pool_cad_primitive_bounding_and_stability():
    """Verify that lid pool fluid body is strictly bounded by CAD pocket primitive without out-of-bounds geometry."""
    pocket_r = 0.080
    z_lid = 0.098
    terrace_z = 0.108
    drain_y = -0.020
    drain_r = 0.055
    terrace_y = 0.028
    terrace_r = 0.030

    ctx = FluidCADContext(
        features=(
            CADFeature(CADFeatureType.TUBE, x=0.0, y=terrace_y, z=0.041, r=0.010),
            CADFeature(CADFeatureType.TERRACE, x=0.0, y=terrace_y, z=terrace_z, r=terrace_r),
            CADFeature(CADFeatureType.DRAIN, x=0.0, y=drain_y, z=z_lid, r=0.015),
            CADFeature(CADFeatureType.CUTOUT, x=0.0, y=drain_y, z=z_lid, r=drain_r),
            CADFeature(CADFeatureType.POCKET, x=0.0, y=0.0, z=z_lid, r=pocket_r),
            CADFeature(CADFeatureType.BOWL, x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    # Frame 1: particles clustered on right lid shelf (X = 0.060, Y = 0.010, outside cutout and terrace)
    pos_f1 = np.array(
        [
            [0.060, 0.010, 0.100],
            [0.065, 0.015, 0.101],
            [0.055, 0.005, 0.099],
        ],
        dtype=np.float32,
    )
    vel_f1 = np.zeros((3, 3), dtype=np.float32)

    # Frame 2: particles shifted to left lid shelf (X = -0.060, Y = 0.010, outside cutout and terrace)
    pos_f2 = np.array(
        [
            [-0.060, 0.010, 0.100],
            [-0.065, 0.015, 0.101],
            [-0.055, 0.005, 0.099],
        ],
        dtype=np.float32,
    )
    vel_f2 = np.zeros((3, 3), dtype=np.float32)

    tracker = FluidBodyTracker(r_s=0.0025)

    bodies_f1 = tracker.update_bodies(pos_f1, vel_f1, cad_context=ctx)
    pool_f1 = [b for b in bodies_f1 if b.stage == FluidStage.LID_POOL][0]

    bodies_f2 = tracker.update_bodies(pos_f2, vel_f2, cad_context=ctx)
    pool_f2 = [b for b in bodies_f2 if b.stage == FluidStage.LID_POOL][0]

    # 1. Centroid XY is rock-solidly anchored to CAD pocket center (0, 0)
    assert np.isclose(pool_f1.centroid[0], 0.0)
    assert np.isclose(pool_f1.centroid[1], 0.0)
    assert np.isclose(pool_f2.centroid[0], 0.0)
    assert np.isclose(pool_f2.centroid[1], 0.0)

    # 2. Bounding box is clamped to CAD pocket primitive
    assert pool_f1.bounds_min[0] >= -pocket_r
    assert pool_f1.bounds_min[1] >= -pocket_r
    assert pool_f1.bounds_max[0] <= pocket_r
    assert pool_f1.bounds_max[1] <= pocket_r
    assert pool_f1.bounds_min[2] >= z_lid

    # 3. Mesh vertices never exceed CAD pocket radius (no extension outside into thin air)
    verts_f1, _ = pool_f1.to_mesh()
    d_xy_f1 = np.sqrt(verts_f1[:, 0] ** 2 + verts_f1[:, 1] ** 2)
    assert np.all(d_xy_f1 <= pocket_r + 1e-4)
    assert np.all(verts_f1[:, 2] >= z_lid - 1e-4)
    assert np.all(verts_f1[:, 2] <= terrace_z + 0.010)

    verts_f2, _ = pool_f2.to_mesh()
    d_xy_f2 = np.sqrt(verts_f2[:, 0] ** 2 + verts_f2[:, 1] ** 2)
    assert np.all(d_xy_f2 <= pocket_r + 1e-4)

    # 4. Mesh vertices must NOT cover the drain cutout hole or terrace platform
    d_drain_f1 = np.sqrt(verts_f1[:, 0] ** 2 + (verts_f1[:, 1] - drain_y) ** 2)
    assert not np.any(d_drain_f1 < drain_r - 0.001)

    d_terrace_f1 = np.sqrt(verts_f1[:, 0] ** 2 + (verts_f1[:, 1] - terrace_y) ** 2)
    assert not np.any(d_terrace_f1 < terrace_r - 0.001)

    # 5. CAD solid volume is strictly contained within CAD boundaries and has positive volume
    solid_f1 = pool_f1.to_cad_solid()
    assert solid_f1.volume > 0.0
    bbox_f1 = solid_f1.bounding_box()
    assert bbox_f1.min.X >= -pocket_r - 1e-4
    assert bbox_f1.max.X <= pocket_r + 1e-4
    assert bbox_f1.min.Y >= -pocket_r - 1e-4
    assert bbox_f1.max.Y <= pocket_r + 1e-4


def test_frame_0_initial_state_no_waterfall():
    """Verify that at frame 0 with fluid at rest in the reservoir bowl, no waterfalls or lid pools are visible."""
    ctx = FluidCADContext(
        features=(
            CADFeature(CADFeatureType.TUBE, x=0.0, y=0.028, z=0.041, r=0.010),
            CADFeature(CADFeatureType.TERRACE, x=0.0, y=0.028, z=0.108, r=0.030),
            CADFeature(CADFeatureType.DRAIN, x=0.0, y=-0.020, z=0.098, r=0.055),
            CADFeature(CADFeatureType.POCKET, x=0.0, y=0.0, z=0.098, r=0.080),
            CADFeature(CADFeatureType.BOWL, x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    # Frame 0: fluid particles resting in the bottom reservoir bowl (including under drain and near wall)
    pos_frame_0 = np.array(
        [
            [0.0, 0.0, 0.045],
            [0.020, 0.020, 0.050],
            [-0.020, 0.020, 0.050],
            [0.0, -0.020, 0.050],  # resting under drain cutout, but submerged in pool!
            [0.0, -0.050, 0.048],
            [0.075, 0.0, 0.052],  # near outer wall, submerged in pool!
            [-0.075, 0.0, 0.052],
        ],
        dtype=np.float32,
    )
    vel_frame_0 = np.zeros((7, 3), dtype=np.float32)

    tracker = FluidBodyTracker(r_s=0.0025)
    bodies = tracker.update_bodies(pos_frame_0, vel_frame_0, cad_context=ctx)

    stages = {b.stage for b in bodies}

    # In Frame 0:
    # 1. BOWL_POOL must be present and contain all submerged particles
    assert FluidStage.BOWL_POOL in stages
    bowl_body = [b for b in bodies if b.stage == FluidStage.BOWL_POOL][0]
    assert bowl_body.particle_count == 7

    # 2. ZERO waterfalls or elevated sheets/pools must exist in frame 0
    assert FluidStage.DRAIN_WATERFALL not in stages
    assert FluidStage.LIP_WATERFALL not in stages
    assert FluidStage.TOP_SHEET not in stages
    assert FluidStage.LID_POOL not in stages
    assert FluidStage.SPLASH_CLUSTER not in stages


def test_direct_top_sheet_to_drain_cascade():
    """Verify that water plunging directly from the top terrace sheet to the drain cutout activates drain waterfall."""
    ctx = FluidCADContext(
        features=(
            CADFeature(CADFeatureType.TUBE, label="Tube", x=0.0, y=0.028, z=0.041, r=0.010),
            CADFeature(CADFeatureType.TERRACE, label="Terrace", x=0.0, y=0.028, z=0.108, r=0.030),
            CADFeature(CADFeatureType.DRAIN, label="Drain_Center", x=0.0, y=-0.020, z=0.098, r=0.0154),
            CADFeature(CADFeatureType.POCKET, label="Pocket", x=0.0, y=0.0, z=0.098, r=0.080),
            CADFeature(CADFeatureType.BOWL, label="Bowl", x=0.0, y=0.0, z=0.041, r=0.090),
        )
    )

    # Water spilling directly off the front of the top terrace into the drain column (without resting in lid pool)
    pos = np.array(
        [
            [0.0, 0.028, 0.108],  # Top sheet
            [0.0, 0.020, 0.108],  # Top sheet near front edge
            [0.0, -0.015, 0.105],  # Plunging directly from terrace level above drain aperture
            [0.0, -0.020, 0.085],  # Falling in drain column
            [0.0, -0.020, 0.065],  # Falling in drain column
            [0.0, 0.0, 0.045],  # Resting in reservoir bowl pool
            [0.02, 0.02, 0.045],
        ],
        dtype=np.float32,
    )
    vel = np.zeros((7, 3), dtype=np.float32)

    tracker = FluidBodyTracker(r_s=0.0025)
    bodies = tracker.update_bodies(pos, vel, cad_context=ctx)
    body_map = {b.stage: b for b in bodies}

    assert FluidStage.TOP_SHEET in body_map
    assert FluidStage.DRAIN_WATERFALL in body_map
    assert FluidStage.BOWL_POOL in body_map

    drain_wf = body_map[FluidStage.DRAIN_WATERFALL]
    assert drain_wf.body_id == 1
    assert drain_wf.display_name == "drain_waterfall_drain_center"

    # Verify that the generated mesh spans all the way up towards the top terrace height
    verts, faces = drain_wf.to_mesh()
    assert np.max(verts[:, 2]) >= 0.105
    assert np.min(verts[:, 2]) <= 0.065


def test_fluid_cad_context_from_boundaries_and_urdf_metadata():
    """Verify that FluidCADContext is dynamically constructed from URDF boundary metadata."""
    from model.boundary_config import BoundaryConfig, ShapeType, BoundaryType, LinkType

    boundaries = [
        BoundaryConfig(
            link_idx=0,
            link_type=LinkType.BASE,
            shape=ShapeType.CYLINDER,
            type=BoundaryType.CAVITY,
            xyz=(0.0, 0.0, 0.041),
            radius=0.090,
            height=0.060,
        ),
        BoundaryConfig(
            link_idx=1,
            link_type=LinkType.TUBE,
            shape=ShapeType.TUBE,
            type=BoundaryType.SOLID,
            xyz=(0.0, 0.028, 0.041),
            radius=0.010,
            has_tube=True,
            tube_pos=(0.0, 0.028, 0.041),
            tube_radius=0.008,
        ),
        BoundaryConfig(
            link_idx=2,
            link_type=LinkType.LID,
            shape=ShapeType.CYLINDER,
            type=BoundaryType.CAVITY,
            xyz=(0.0, 0.0, 0.098),
            radius=0.080,
            height=0.015,
            has_intake=True,
            intake_pos=(0.0, 0.028, 0.010),
            intake_radius=0.030,
            has_drain=True,
            drain_pos=(0.0, -0.020, 0.0),
            drain_radius=0.055,
        ),
    ]

    ctx = FluidCADContext.from_boundaries(boundaries)

    assert len(ctx.bowls) == 1
    assert ctx.bowls[0].z == 0.041
    assert ctx.bowls[0].r == 0.090

    assert len(ctx.tubes) == 1
    assert ctx.tubes[0].y == 0.028
    assert ctx.tubes[0].r == 0.008

    assert len(ctx.terraces) == 1
    assert ctx.terraces[0].y == 0.028
    assert ctx.terraces[0].z == 0.108
    assert ctx.terraces[0].r == 0.030

    assert len(ctx.pockets) == 1
    assert ctx.pockets[0].z == 0.098
    assert ctx.pockets[0].r == 0.080

    assert len(ctx.cutouts) == 1
    assert ctx.cutouts[0].y == -0.020
    assert ctx.cutouts[0].r == 0.055

    assert len(ctx.drains) == 3
    drain_labels = {d.label for d in ctx.drains}
    assert drain_labels == {"Drain_Center", "Drain_Left", "Drain_Right"}
