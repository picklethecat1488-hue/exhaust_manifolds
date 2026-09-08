"""Pure JAX vectorized coordinate transformations and quaternion kinematics."""

import jax
import jax.numpy as jnp


@jax.jit
def _q_mult(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
    """Multiply two quaternions q1 and q2 in (x, y, z, w) format."""
    x1, y1, z1, w1 = q1[0], q1[1], q1[2], q1[3]
    x2, y2, z2, w2 = q2[0], q2[1], q2[2], q2[3]
    return jnp.array(
        [
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ],
        dtype=jnp.float32,
    )


@jax.jit
def _q_rotate(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate 3D vector v by quaternion q in (x, y, z, w) format."""
    q_xyz = q[..., :3]
    w = q[..., 3:4]
    cross1 = jnp.cross(q_xyz, v) + w * v
    return v + 2.0 * jnp.cross(q_xyz, cross1)


@jax.jit
def invert_orientation(orn: jnp.ndarray) -> jnp.ndarray:
    """Calculate the inverse (conjugate) of an orientation quaternion in (x, y, z, w) format."""
    return jnp.concatenate([-orn[..., :3], orn[..., 3:4]], axis=-1)


@jax.jit
def world_to_base_orientation(child_world_orn: jnp.ndarray, base_world_orn: jnp.ndarray) -> jnp.ndarray:
    """Calculate the relative orientation of a child frame with respect to the base frame (base_orn^-1 * child_orn)."""
    base_inv = invert_orientation(base_world_orn)
    rel_q = _q_mult(base_inv, child_world_orn)
    norm = jnp.linalg.norm(rel_q)
    return jnp.where(norm > 1e-6, rel_q / norm, rel_q)


# ==============================================================================
# Coordinate Space Position Transformations (World, Base Link, Local Link, Voxel Grid)
# ==============================================================================


@jax.jit
def world_to_base_frame(
    pos_world: jnp.ndarray,
    base_pos: jnp.ndarray,
    base_orn_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from World Space to Base Link Space."""
    return _q_rotate(base_orn_inv, pos_world - base_pos)


@jax.jit
def base_to_world_frame(
    pos_base: jnp.ndarray,
    base_pos: jnp.ndarray,
    base_orn: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from Base Link Space to World Space."""
    return _q_rotate(base_orn, pos_base) + base_pos


@jax.jit
def base_to_local_frame(
    pos_base: jnp.ndarray,
    b_pos_loc: jnp.ndarray,
    b_orn_loc_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from Base Link Space to specific Local Boundary Frame."""
    return _q_rotate(b_orn_loc_inv, pos_base - b_pos_loc)


@jax.jit
def local_to_base_frame(
    pos_local: jnp.ndarray,
    b_pos_loc: jnp.ndarray,
    b_orn_loc: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from Local Boundary Frame to Base Link Space."""
    return _q_rotate(b_orn_loc, pos_local) + b_pos_loc


@jax.jit
def world_to_local_frame(
    pos_world: jnp.ndarray,
    link_pos_world: jnp.ndarray,
    link_orn_world_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from World Space to Link Local Frame."""
    return _q_rotate(link_orn_world_inv, pos_world - link_pos_world)


@jax.jit
def local_to_world_frame(
    pos_local: jnp.ndarray,
    link_pos_world: jnp.ndarray,
    link_orn_world: jnp.ndarray,
) -> jnp.ndarray:
    """Transform 3D position vector from Link Local Frame to World Space."""
    return _q_rotate(link_orn_world, pos_local) + link_pos_world


@jax.jit
def base_to_voxel_coord(
    pos_base: jnp.ndarray,
    origin_base: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Convert Base Link metric coordinates (x, y, z) into fractional voxel lattice coordinates."""
    return (pos_base - origin_base) / dx


@jax.jit
def voxel_to_base_coord(
    coord_vox: jnp.ndarray,
    origin_base: jnp.ndarray,
    dx: float,
) -> jnp.ndarray:
    """Convert fractional voxel lattice coordinates into Base Link metric coordinates (x, y, z)."""
    return origin_base + coord_vox * dx


# ==============================================================================
# Coordinate Space Directional Vector Transformations (Velocities, Forces, Normals)
# ==============================================================================


@jax.jit
def world_to_base_vector(
    vec_world: jnp.ndarray,
    base_orn_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector (velocity, force, normal) from World Space to Base Link Space."""
    return _q_rotate(base_orn_inv, vec_world)


@jax.jit
def base_to_world_vector(
    vec_base: jnp.ndarray,
    base_orn: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector (velocity, force, normal) from Base Link Space to World Space."""
    return _q_rotate(base_orn, vec_base)


@jax.jit
def world_to_local_vector(
    vec_world: jnp.ndarray,
    link_orn_world_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector from World Space to Link Local Space."""
    return _q_rotate(link_orn_world_inv, vec_world)


@jax.jit
def local_to_world_vector(
    vec_local: jnp.ndarray,
    link_orn_world: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector from Link Local Space to World Space."""
    return _q_rotate(link_orn_world, vec_local)


@jax.jit
def local_to_base_vector(
    vec_local: jnp.ndarray,
    b_orn_loc: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector from Local Boundary Space to Base Link Space."""
    return _q_rotate(b_orn_loc, vec_local)


@jax.jit
def base_to_local_vector(
    vec_base: jnp.ndarray,
    b_orn_loc_inv: jnp.ndarray,
) -> jnp.ndarray:
    """Transform directional 3D vector from Base Link Space to Local Boundary Space."""
    return _q_rotate(b_orn_loc_inv, vec_base)


# ==============================================================================
# Coordinate System Conversions (Cartesian, Cylindrical/Polar, Spherical)
# ==============================================================================


@jax.jit
def cartesian_to_cylindrical(
    pos: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert 3D Cartesian coordinates (x, y, z) to Cylindrical (r, theta, z).

    Returns:
        tuple containing:
            - r: Radial distance in the xy-plane (meters)
            - theta: Azimuthal angle in radians [-pi, pi]
            - z: Height along the z-axis (meters)
    """
    r = jnp.sqrt(pos[..., 0] ** 2 + pos[..., 1] ** 2)
    theta = jnp.arctan2(pos[..., 1], pos[..., 0])
    z = pos[..., 2]
    return r, theta, z


@jax.jit
def cylindrical_to_cartesian(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    z: jnp.ndarray,
) -> jnp.ndarray:
    """Convert Cylindrical coordinates (r, theta, z) to 3D Cartesian (x, y, z)."""
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.stack([x, y, z], axis=-1)


@jax.jit
def cartesian_to_polar_2d(
    pos_xy: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Convert 2D Cartesian coordinates (x, y) to Polar (r, theta)."""
    r = jnp.sqrt(pos_xy[..., 0] ** 2 + pos_xy[..., 1] ** 2)
    theta = jnp.arctan2(pos_xy[..., 1], pos_xy[..., 0])
    return r, theta


@jax.jit
def polar_to_cartesian_2d(
    r: jnp.ndarray,
    theta: jnp.ndarray,
) -> jnp.ndarray:
    """Convert 2D Polar coordinates (r, theta) to Cartesian (x, y)."""
    x = r * jnp.cos(theta)
    y = r * jnp.sin(theta)
    return jnp.stack([x, y], axis=-1)


@jax.jit
def cartesian_to_spherical(
    pos: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Convert 3D Cartesian coordinates (x, y, z) to Spherical (r, theta, phi).

    Returns:
        tuple containing:
            - r: Radial distance from the origin (meters)
            - theta: Azimuthal angle in the xy-plane in radians [-pi, pi]
            - phi: Polar angle (inclination from the +z axis) in radians [0, pi]
    """
    r = jnp.sqrt(jnp.sum(pos**2, axis=-1) + 1e-8)
    theta = jnp.arctan2(pos[..., 1], pos[..., 0])
    phi = jnp.arccos(jnp.clip(pos[..., 2] / r, -1.0, 1.0))
    return r, theta, phi


@jax.jit
def spherical_to_cartesian(
    r: jnp.ndarray,
    theta: jnp.ndarray,
    phi: jnp.ndarray,
) -> jnp.ndarray:
    """Convert Spherical coordinates (r, theta, phi) to 3D Cartesian (x, y, z)."""
    x = r * jnp.sin(phi) * jnp.cos(theta)
    y = r * jnp.sin(phi) * jnp.sin(theta)
    z = r * jnp.cos(phi)
    return jnp.stack([x, y, z], axis=-1)


# ==============================================================================
# Surface Geometry & Hole Cutout Projection (Analytical Normals)
# ==============================================================================


@jax.jit
def point_in_surface_hole(
    pos_local: jnp.ndarray,
    hole_pos: jnp.ndarray,
    hole_normal: jnp.ndarray,
    hole_radius: float,
    normal_tol: float = 0.010,
) -> jnp.ndarray:
    """Determine whether 3D local points project within a circular hole cutout defined by center position and surface normal.

    Args:
        pos_local: 3D point coordinates in boundary local frame, shape (..., 3).
        hole_pos: 3D center position of the hole cutout in boundary local frame, shape (3,).
        hole_normal: 3D surface normal vector pointing out from the cut surface, shape (3,).
        hole_radius: Radius of the circular hole cutout in meters.
        normal_tol: Longitudinal tolerance along the surface normal axis in meters.

    Returns:
        Boolean array indicating whether each point falls within the hole cutout cylinder.
    """
    rel = pos_local - hole_pos
    norm_sq = jnp.sum(hole_normal**2)
    unit_norm = jnp.where(norm_sq > 1e-6, hole_normal / jnp.sqrt(norm_sq), jnp.array([0.0, 0.0, 1.0]))

    d_parallel = jnp.sum(rel * unit_norm, axis=-1)
    v_perp = rel - d_parallel[..., None] * unit_norm
    d_perp = jnp.sqrt(jnp.sum(v_perp**2, axis=-1) + 1e-12)

    return (d_perp < hole_radius) & (jnp.abs(d_parallel) <= normal_tol) & (hole_radius > 0.0)


@jax.jit
def compute_surface_normal(
    shape_type: int,
    pos_local: jnp.ndarray,
    radius: float,
    height: float,
) -> jnp.ndarray:
    """Compute outward-pointing unit surface normal vector for canonical shapes at local coordinates.

    Args:
        shape_type: ShapeType integer code (e.g. CYLINDER, TUBE, CANOPY, CASING, BOX, PLANE).
        pos_local: 3D coordinates in local boundary frame, shape (..., 3).
        radius: Radius parameter of the geometric shape in meters.
        height: Height parameter of the geometric shape in meters.

    Returns:
        Unit normal vector array, shape (..., 3).
    """
    x = pos_local[..., 0]
    y = pos_local[..., 1]
    z = pos_local[..., 2]
    r_xy = jnp.sqrt(x**2 + y**2 + 1e-12)

    # Cylinder / Tube normals: caps vs sidewall
    top_cap_norm = jnp.array([0.0, 0.0, 1.0])
    bot_cap_norm = jnp.array([0.0, 0.0, -1.0])
    radial_norm = jnp.stack([x / r_xy, y / r_xy, jnp.zeros_like(z)], axis=-1)

    dist_top = jnp.abs(z - height)
    dist_bot = jnp.abs(z)
    dist_side = jnp.abs(r_xy - radius)

    is_top = (dist_top <= dist_bot) & (dist_top <= dist_side)
    is_bot = (dist_bot < dist_top) & (dist_bot <= dist_side)
    cylinder_norm = jnp.where(
        is_top[..., None],
        top_cap_norm,
        jnp.where(is_bot[..., None], bot_cap_norm, radial_norm),
    )

    # Canopy normal: radial from origin
    r_3d = jnp.sqrt(x**2 + y**2 + z**2 + 1e-12)
    canopy_norm = jnp.stack([x / r_3d, y / r_3d, z / r_3d], axis=-1)

    # Default / Plane / Box (normal along Z)
    default_norm = jnp.array([0.0, 0.0, 1.0])

    return jnp.where(
        shape_type == 1,  # CANOPY
        canopy_norm,
        jnp.where(
            (shape_type == 0) | (shape_type == 2) | (shape_type == 6),  # CYLINDER, TUBE, CASING
            cylinder_norm,
            default_norm,
        ),
    )


@jax.jit
def match_intake_drain_ports(
    b_pos_arr: jnp.ndarray,
    b_orn_arr: jnp.ndarray,
    b_params: jnp.ndarray,
    distance_tol: float = 0.020,
    normal_alignment_threshold: float = -0.3,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Find matching fluid intake and drain port pairs across all boundaries in world space.

    A drain port on boundary J connects to an intake port on boundary I when:
    1. Both ports are active (has_drain[j] and has_intake[i]).
    2. Their world-space centroids are within distance_tol (||p_intake_world - p_drain_world|| < distance_tol).
    3. Their surface normals oppose each other (n_intake_world · n_drain_world < normal_alignment_threshold).

    Args:
        b_pos_arr: Boundary world position origins, shape (N, 3).
        b_orn_arr: Boundary world quaternions, shape (N, 4).
        b_params: Boundary parameters tensor, shape (N, P).
        distance_tol: Maximum allowable distance between mating port centroids in meters.
        normal_alignment_threshold: Cosine similarity threshold for opposing port normals.

    Returns:
        matches_mask: Boolean adjacency matrix (N, N) where matches_mask[i, j] is True if
                      intake on boundary I mates with drain on boundary J.
        match_distances: Float matrix (N, N) with world distances between port pairs.
    """
    has_intake = b_params[:, 37] > 0.5  # HAS_INTAKE
    intake_pos_local = b_params[:, 38:41]  # INTAKE_POS (x, y, z)
    intake_norm_local = b_params[:, 41:44]  # INTAKE_NORMAL (nx, ny, nz)

    has_drain = b_params[:, 12] > 0.5  # HAS_DRAIN
    drain_pos_local = b_params[:, 45:48]  # DRAIN_POS (x, y, z)
    drain_norm_local = b_params[:, 48:51]  # DRAIN_NORMAL (nx, ny, nz)

    # Transform intake ports to world space
    intake_pos_world = local_to_world_frame(intake_pos_local, b_pos_arr, b_orn_arr)
    intake_norm_world = local_to_world_vector(intake_norm_local, b_orn_arr)

    # Transform drain ports to world space
    drain_pos_world = local_to_world_frame(drain_pos_local, b_pos_arr, b_orn_arr)
    drain_norm_world = local_to_world_vector(drain_norm_local, b_orn_arr)

    # Pairwise distance matrix between intake i and drain j: shape (N, N)
    diff = intake_pos_world[:, None, :] - drain_pos_world[None, :, :]
    dist_matrix = jnp.sqrt(jnp.sum(diff**2, axis=-1) + 1e-12)

    # Pairwise normal dot product: shape (N, N)
    normal_dots = jnp.sum(intake_norm_world[:, None, :] * drain_norm_world[None, :, :], axis=-1)

    # Matched condition: active intake, active drain, close distance, opposing normals
    active_pair = has_intake[:, None] & has_drain[None, :]
    close_dist = dist_matrix < distance_tol
    opposing_norm = normal_dots < normal_alignment_threshold

    matches_mask = active_pair & close_dist & opposing_norm
    return matches_mask, dist_matrix
