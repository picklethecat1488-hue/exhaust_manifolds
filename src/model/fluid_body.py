"""Dynamic fluid body primitives and volume tracking models."""

from enum import Enum
import math
from typing import Any, NamedTuple, Optional, Sequence, Union
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


def generate_cylinder_mesh(
    radius: float,
    z_min: float,
    z_max: float,
    center: tuple[float, float] = (0.0, 0.0),
    n_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight 3D cylinder triangle mesh with outward normals."""
    cx, cy = center
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    # Bottom ring vertices (0 .. n_segments - 1)
    v_bottom = np.column_stack([cx + radius * cos_t, cy + radius * sin_t, np.full(n_segments, z_min)])
    # Top ring vertices (n_segments .. 2*n_segments - 1)
    v_top = np.column_stack([cx + radius * cos_t, cy + radius * sin_t, np.full(n_segments, z_max)])
    # Bottom center (2*n_segments)
    v_bottom_center = np.array([[cx, cy, z_min]])
    # Top center (2*n_segments + 1)
    v_top_center = np.array([[cx, cy, z_max]])

    vertices = np.vstack([v_bottom, v_top, v_bottom_center, v_top_center]).astype(np.float32)

    idx_bot_c = 2 * n_segments
    idx_top_c = 2 * n_segments + 1

    faces = []
    for i in range(n_segments):
        next_i = (i + 1) % n_segments
        # Side quad (2 triangles)
        faces.append([i, n_segments + i, n_segments + next_i])
        faces.append([i, n_segments + next_i, next_i])
        # Bottom cap (viewed from bottom)
        faces.append([idx_bot_c, next_i, i])
        # Top cap (viewed from top)
        faces.append([idx_top_c, n_segments + i, n_segments + next_i])

    return vertices, np.array(faces, dtype=np.uint32)


def generate_sphere_mesh(
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
    radius: float = 0.010,
    n_lat: int = 12,
    n_lon: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight 3D UV sphere triangle mesh with outward normals."""
    cx, cy, cz = center
    radius = max(radius, 1e-4)

    # North pole vertex (index 0)
    v_north = [cx, cy, cz + radius]
    vertices = [v_north]

    # Intermediate rings (latitudes from +pi/2 down to -pi/2 excluding poles)
    latitudes = np.linspace(np.pi / 2.0, -np.pi / 2.0, n_lat)[1:-1]
    longitudes = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)

    for lat in latitudes:
        r_ring = radius * np.cos(lat)
        z = cz + radius * np.sin(lat)
        for lon in longitudes:
            x = cx + r_ring * np.cos(lon)
            y = cy + r_ring * np.sin(lon)
            vertices.append([x, y, z])

    # South pole vertex (index len(vertices))
    south_idx = len(vertices)
    vertices.append([cx, cy, cz - radius])
    vertices_arr = np.array(vertices, dtype=np.float32)

    faces = []
    # North pole cap triangles (connected to index 0)
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        faces.append([0, 1 + j, 1 + next_j])

    # Intermediate quads
    n_rings = len(latitudes)
    for i in range(n_rings - 1):
        r1 = 1 + i * n_lon
        r2 = 1 + (i + 1) * n_lon
        for j in range(n_lon):
            next_j = (j + 1) % n_lon
            faces.append([r1 + j, r2 + j, r2 + next_j])
            faces.append([r1 + j, r2 + next_j, r1 + next_j])

    # South pole cap triangles
    last_ring_start = 1 + (n_rings - 1) * n_lon
    for j in range(n_lon):
        next_j = (j + 1) % n_lon
        faces.append([south_idx, last_ring_start + next_j, last_ring_start + j])

    faces_arr = np.array(faces, dtype=np.uint32)
    return vertices_arr, faces_arr


def generate_heightfield_cylinder_mesh(
    radius: float,
    z_floor: float,
    surface_positions: Optional[np.ndarray] = None,
    default_z_top: float = 0.078,
    center: tuple[float, float] = (0.0, 0.0),
    n_rings: int = 6,
    n_spokes: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight 3D cylinder triangle mesh with a dynamic top surface heightfield."""
    cx, cy = center
    spoke_angles = np.linspace(0.0, 2.0 * np.pi, n_spokes, endpoint=False)
    cos_s = np.cos(spoke_angles)
    sin_s = np.sin(spoke_angles)

    # 1. Build 2D surface height grid from surface particles strictly within containing radius
    nx, ny = 32, 32
    x_min, x_max = cx - radius, cx + radius
    y_min, y_max = cy - radius, cy + radius
    dx = max(1e-4, (x_max - x_min) / nx)
    dy = max(1e-4, (y_max - y_min) / ny)
    grid_z = np.full((nx, ny), default_z_top, dtype=np.float32)

    if surface_positions is not None and len(surface_positions) > 0:
        d_center_sq = (surface_positions[:, 0] - cx) ** 2 + (surface_positions[:, 1] - cy) ** 2
        in_bounds = (
            (d_center_sq <= (radius + 1e-4) ** 2)
            & (surface_positions[:, 2] >= z_floor)
            & (surface_positions[:, 2] <= default_z_top + 0.010)
        )
        valid_pos = surface_positions[in_bounds]
        if len(valid_pos) > 0:
            ix = np.clip(np.floor((valid_pos[:, 0] - x_min) / dx).astype(int), 0, nx - 1)
            iy = np.clip(np.floor((valid_pos[:, 1] - y_min) / dy).astype(int), 0, ny - 1)
            np.maximum.at(grid_z, (ix, iy), valid_pos[:, 2])

    def sample_z(x_arr: np.ndarray, y_arr: np.ndarray) -> np.ndarray:
        gx = np.clip(np.floor((x_arr - x_min) / dx).astype(int), 0, nx - 1)
        gy = np.clip(np.floor((y_arr - y_min) / dy).astype(int), 0, ny - 1)
        return grid_z[gx, gy]

    # Top center vertex (index 0)
    z_c = float(sample_z(np.array([cx]), np.array([cy]))[0])
    v_top_center = [cx, cy, z_c]

    # Top ring vertices (indices 1 .. n_rings * n_spokes)
    top_vertices = [v_top_center]
    r_steps = np.linspace(radius / n_rings, radius, n_rings)
    for r in r_steps:
        x_ring = cx + r * cos_s
        y_ring = cy + r * sin_s
        z_ring = sample_z(x_ring, y_ring)
        for s in range(n_spokes):
            top_vertices.append([float(x_ring[s]), float(y_ring[s]), float(z_ring[s])])

    # Bottom ring vertices (outer ring at z_floor)
    bot_ring_start = len(top_vertices)
    for s in range(n_spokes):
        x = float(cx + radius * cos_s[s])
        y = float(cy + radius * sin_s[s])
        top_vertices.append([x, y, z_floor])

    # Bottom center vertex (last index)
    bot_center_idx = len(top_vertices)
    top_vertices.append([cx, cy, z_floor])

    vertices = np.array(top_vertices, dtype=np.float32)

    faces = []
    # Inner ring triangles (connected to top center 0)
    for s in range(n_spokes):
        next_s = (s + 1) % n_spokes
        v1 = 1 + s
        v2 = 1 + next_s
        faces.append([0, v1, v2])

    # Intermediate ring quads (ring r to ring r+1)
    for r in range(n_rings - 1):
        r1_start = 1 + r * n_spokes
        r2_start = 1 + (r + 1) * n_spokes
        for s in range(n_spokes):
            next_s = (s + 1) % n_spokes
            p0 = r1_start + s
            p1 = r1_start + next_s
            p2 = r2_start + next_s
            p3 = r2_start + s
            faces.append([p0, p1, p2])
            faces.append([p0, p2, p3])

    # Side wall quads (connecting top outer ring to bottom ring)
    top_outer_start = 1 + (n_rings - 1) * n_spokes
    for s in range(n_spokes):
        next_s = (s + 1) % n_spokes
        t0 = top_outer_start + s
        t1 = top_outer_start + next_s
        b0 = bot_ring_start + s
        b1 = bot_ring_start + next_s
        faces.append([t0, b0, b1])
        faces.append([t0, b1, t1])

    # Bottom cap fan
    for s in range(n_spokes):
        next_s = (s + 1) % n_spokes
        b0 = bot_ring_start + s
        b1 = bot_ring_start + next_s
        faces.append([bot_center_idx, b1, b0])

    faces_arr = np.array(faces, dtype=np.uint32)
    return vertices, faces_arr


def generate_box_mesh(
    bounds_min: tuple[float, float, float],
    bounds_max: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight 3D axis-aligned box triangle mesh."""
    x0, y0, z0 = bounds_min
    x1, y1, z1 = bounds_max
    x1 = max(x1, x0 + 1e-4)
    y1 = max(y1, y0 + 1e-4)
    z1 = max(z1, z0 + 1e-4)

    vertices = np.array(
        [
            [x0, y0, z0],
            [x1, y0, z0],
            [x1, y1, z0],
            [x0, y1, z0],
            [x0, y0, z1],
            [x1, y0, z1],
            [x1, y1, z1],
            [x0, y1, z1],
        ],
        dtype=np.float32,
    )

    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [2, 3, 7],
            [2, 7, 6],
            [0, 4, 7],
            [0, 7, 3],
            [1, 2, 6],
            [1, 6, 5],
        ],
        dtype=np.uint32,
    )
    return vertices, faces


def generate_manifold_mesh_around_particles(
    positions: np.ndarray,
    r_s: float = 0.0025,
) -> tuple[np.ndarray, np.ndarray]:
    """Construct a watertight 3D manifold triangle mesh wrapped tightly around particle coordinates."""
    if positions is None or len(positions) == 0:
        return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint32)

    vol = len(positions) * (4.0 / 3.0) * math.pi * (r_s**3)
    equiv_radius = max(r_s, (3.0 * vol / (4.0 * math.pi)) ** (1.0 / 3.0))

    if len(positions) < 4:
        centroid = tuple(float(x) for x in np.mean(positions, axis=0))
        return generate_sphere_mesh(center=centroid, radius=equiv_radius)

    try:
        from scipy.spatial import ConvexHull

        # Expand points with thickness (+/- r_s * 0.5) to guarantee 3D volume for coplanar sets
        pts_expanded = np.vstack(
            [
                positions + np.array([0.0, 0.0, r_s * 0.5]),
                positions - np.array([0.0, 0.0, r_s * 0.5]),
            ]
        )
        hull = ConvexHull(pts_expanded)
        unique_indices = np.unique(hull.simplices)
        index_map = {orig: new for new, orig in enumerate(unique_indices)}
        vertices = pts_expanded[unique_indices].astype(np.float32)
        faces = np.vectorize(index_map.get)(hull.simplices).astype(np.uint32)
        return vertices, faces
    except Exception:
        centroid = tuple(float(x) for x in np.mean(positions, axis=0))
        max_extent = min(float(np.max(np.linalg.norm(positions - centroid, axis=1))) + r_s, equiv_radius * 1.5)
        safe_radius = max(equiv_radius, max_extent)
        return generate_sphere_mesh(center=centroid, radius=safe_radius)


def generate_lip_waterfall_mesh(
    positions: Optional[np.ndarray],
    center_xy: tuple[float, float] = (0.0, 0.0),
    lip_radius: float = 0.030,
    z_top: float = 0.113,
    z_bot: float = 0.106,
    thickness: float = 0.003,
    n_segments: int = 32,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight annular curtain waterfall mesh cascading off an elevated platform lip."""
    cx, cy = center_xy
    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    r_out_top = lip_radius
    r_in_top = max(0.005, lip_radius - thickness)
    r_out_bot = lip_radius + 0.002
    r_in_bot = max(0.005, lip_radius - thickness + 0.002)

    vertices = []
    for seg in range(n_segments):
        vertices.append([cx + r_out_top * cos_t[seg], cy + r_out_top * sin_t[seg], z_top])
    for seg in range(n_segments):
        vertices.append([cx + r_in_top * cos_t[seg], cy + r_in_top * sin_t[seg], z_top])
    for seg in range(n_segments):
        vertices.append([cx + r_out_bot * cos_t[seg], cy + r_out_bot * sin_t[seg], z_bot])
    for seg in range(n_segments):
        vertices.append([cx + r_in_bot * cos_t[seg], cy + r_in_bot * sin_t[seg], z_bot])

    vertices_arr = np.array(vertices, dtype=np.float32)
    faces = []

    # Top annular cap
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([seg, next_seg, n_segments + next_seg])
        faces.append([seg, n_segments + next_seg, n_segments + seg])

    # Outer cylindrical curtain
    r2_offset = 2 * n_segments
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([seg, r2_offset + seg, r2_offset + next_seg])
        faces.append([seg, r2_offset + next_seg, next_seg])

    # Inner cylindrical curtain
    r3_offset = 3 * n_segments
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([n_segments + seg, n_segments + next_seg, r3_offset + next_seg])
        faces.append([n_segments + seg, r3_offset + next_seg, r3_offset + seg])

    # Bottom annular cap
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([r2_offset + seg, r3_offset + seg, r3_offset + next_seg])
        faces.append([r2_offset + seg, r3_offset + next_seg, r2_offset + next_seg])

    faces_arr = np.array(faces, dtype=np.uint32)
    return vertices_arr, faces_arr


def generate_waterfall_mesh(
    positions: Optional[np.ndarray],
    z_top: float = 0.105,
    z_bot: float = 0.048,
    cutout_xy: tuple[float, float] = (0.0, 0.0),
    nominal_radius: float = 0.012,
    n_slices: int = 16,
    n_segments: int = 24,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate a watertight, smooth curved waterfall column mesh along the true flow spine from lip to pool."""
    cx, cy = cutout_xy
    z_slices = np.linspace(z_bot, z_top, n_slices)

    spine_x = []
    spine_y = []
    radii = []
    safe_nominal_r = min(0.012, max(0.006, nominal_radius))
    dz = (z_top - z_bot) / max(1, n_slices)
    for z in z_slices:
        if positions is not None and len(positions) > 0:
            mask = np.abs(positions[:, 2] - z) <= dz
            if np.any(mask):
                pts_slice = positions[mask]
                mean_xy = np.mean(pts_slice[:, :2], axis=0)
                sx = float(0.2 * cx + 0.8 * mean_xy[0])
                sy = float(0.2 * cy + 0.8 * mean_xy[1])
                spread = np.percentile(np.linalg.norm(pts_slice[:, :2] - [sx, sy], axis=1), 90)
                rad = max(0.005, min(0.016, float(spread) + 0.002))
            else:
                sx, sy = cx, cy
                rad = safe_nominal_r
        else:
            sx, sy = cx, cy
            rad = safe_nominal_r
        spine_x.append(sx)
        spine_y.append(sy)
        radii.append(rad)

    theta = np.linspace(0.0, 2.0 * np.pi, n_segments, endpoint=False)
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)

    vertices = []
    for i in range(n_slices):
        z = z_slices[i]
        sx = spine_x[i]
        sy = spine_y[i]
        r = radii[i]
        for seg in range(n_segments):
            vertices.append([sx + r * cos_t[seg], sy + r * sin_t[seg], z])

    bot_center_idx = len(vertices)
    vertices.append([spine_x[0], spine_y[0], z_bot])
    top_center_idx = len(vertices)
    vertices.append([spine_x[-1], spine_y[-1], z_top])

    vertices_arr = np.array(vertices, dtype=np.float32)

    faces = []
    for i in range(n_slices - 1):
        r1 = i * n_segments
        r2 = (i + 1) * n_segments
        for seg in range(n_segments):
            next_seg = (seg + 1) % n_segments
            faces.append([r1 + seg, r2 + seg, r2 + next_seg])
            faces.append([r1 + seg, r2 + next_seg, r1 + next_seg])

    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([bot_center_idx, next_seg, seg])

    top_ring_start = (n_slices - 1) * n_segments
    for seg in range(n_segments):
        next_seg = (seg + 1) % n_segments
        faces.append([top_center_idx, top_ring_start + seg, top_ring_start + next_seg])

    faces_arr = np.array(faces, dtype=np.uint32)
    return vertices_arr, faces_arr


class FluidBodyType(str, Enum):
    """Semantic classification of dynamic fluid bodies."""

    POOL = "pool"
    STREAM = "stream"
    SHEET = "sheet"
    WATERFALL = "waterfall"
    CLUSTER = "cluster"


class FluidStage(str, Enum):
    """Semantic cascade stage identifying the physical role and location of a fluid body."""

    DELIVERY_STREAM = "delivery_stream"
    TOP_SHEET = "top_sheet"
    LIP_WATERFALL = "lip_waterfall"
    LID_POOL = "lid_pool"
    DRAIN_WATERFALL = "drain_waterfall"
    BOWL_POOL = "bowl_pool"
    SPLASH_CLUSTER = "splash_cluster"


class CADFeatureType(str, Enum):
    """Semantic classification of CAD geometry features."""

    TUBE = "Tube"
    TERRACE = "Terrace"
    DRAIN = "Drain"
    CUTOUT = "Cutout"
    POCKET = "Pocket"
    BOWL = "Bowl"


class CADFeature(NamedTuple):
    """Spatial 4D coordinate tuple (X, Y, Z, R) with semantic CADFeatureType and optional label."""

    feature_type: CADFeatureType = CADFeatureType.TUBE
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0
    r: float = 0.0
    label: Optional[str] = None

    @property
    def name(self) -> str:
        """Return string name/label of the CAD feature."""
        if self.label:
            return self.label
        return self.feature_type.value if isinstance(self.feature_type, CADFeatureType) else str(self.feature_type)

    @property
    def coords(self) -> tuple[float, float, float, float]:
        """Return 4D geometric coordinates (X, Y, Z, R)."""
        return (self.x, self.y, self.z, self.r)


class FluidCADContext(NamedTuple):
    """Encapsulates CAD geometry as an ordered sequence of CADFeature instances."""

    features: Sequence[CADFeature] = ()

    def get(self, feature_type: Union[CADFeatureType, str]) -> Optional[CADFeature]:
        """Find a CADFeature by CADFeatureType or string name/label."""
        target = feature_type.value.lower() if isinstance(feature_type, CADFeatureType) else str(feature_type).lower()
        for feat in self.features:
            feat_name = (
                feat.feature_type.value.lower()
                if isinstance(feat.feature_type, CADFeatureType)
                else str(feat.feature_type).lower()
            )
            feat_label = feat.label.lower() if feat.label is not None else ""
            if target in (feat_name, feat_label):
                return feat
        return None

    def get_all(self, feature_type: Union[CADFeatureType, str]) -> list[CADFeature]:
        """Find all CADFeatures matching the given CADFeatureType or string name/label."""
        target = feature_type.value.lower() if isinstance(feature_type, CADFeatureType) else str(feature_type).lower()
        matches = []
        for feat in self.features:
            feat_name = (
                feat.feature_type.value.lower()
                if isinstance(feat.feature_type, CADFeatureType)
                else str(feat.feature_type).lower()
            )
            feat_label = feat.label.lower() if feat.label is not None else ""
            if target in (feat_name, feat_label):
                matches.append(feat)
        return matches

    @property
    def terraces(self) -> list[CADFeature]:
        """Get all terrace platform features."""
        return self.get_all(CADFeatureType.TERRACE)

    @property
    def drains(self) -> list[CADFeature]:
        """Get all drain cutout features."""
        return self.get_all(CADFeatureType.DRAIN)

    @property
    def cutouts(self) -> list[CADFeature]:
        """Get all cutout aperture features."""
        return self.get_all(CADFeatureType.CUTOUT)

    @property
    def pockets(self) -> list[CADFeature]:
        """Get all pocket/shelf features."""
        return self.get_all(CADFeatureType.POCKET)

    @property
    def tubes(self) -> list[CADFeature]:
        """Get all delivery tube features."""
        return self.get_all(CADFeatureType.TUBE)

    @property
    def bowls(self) -> list[CADFeature]:
        """Get all reservoir bowl features."""
        return self.get_all(CADFeatureType.BOWL)

    @property
    def z_floor(self) -> float:
        """Get floor elevation from bowl or tube."""
        bowl = self.get(CADFeatureType.BOWL)
        if bowl is not None and bowl.z != 0.0:
            return bowl.z
        tube = self.get(CADFeatureType.TUBE)
        return tube.z if tube is not None else 0.0

    @property
    def z_lid(self) -> float:
        """Get drinking lid shelf elevation."""
        pocket = self.get(CADFeatureType.POCKET)
        return pocket.z if pocket is not None else 0.0

    @staticmethod
    def _create_drain_features(
        drain_x: float,
        drain_y: float,
        z_lid: float,
        drain_r: float,
        tube_r: float = 0.0,
    ) -> tuple[list[CADFeature], list[CADFeature]]:
        """Construct dynamic drain spillway stream features and cutout aperture features from geometric parameters."""
        drain_features: list[CADFeature] = []
        cutout_features: list[CADFeature] = []

        if drain_r <= 0.0:
            drain_features.append(
                CADFeature(CADFeatureType.DRAIN, label="Drain", x=drain_x, y=drain_y, z=z_lid, r=drain_r)
            )
            return drain_features, cutout_features

        cutout_features.append(
            CADFeature(CADFeatureType.CUTOUT, label="Lid_Cutout", x=drain_x, y=drain_y, z=z_lid, r=drain_r)
        )

        # Stream radius dynamically bounded by tube flow channel or cutout aperture
        stream_r = min(drain_r, tube_r) if tube_r > 0.0 else drain_r
        stream_r = max(1e-4, stream_r)
        aperture_ratio = drain_r / stream_r

        if aperture_ratio > 1.5:
            # Wide aperture: generate multi-stream spillways spaced dynamically along the cutout arc
            n_streams = min(3, max(3, round(aperture_ratio)))
            max_angle = math.pi / 4.0
            for i in range(n_streams):
                norm = (i / (n_streams - 1) - 0.5) * 2.0
                angle = norm * max_angle
                x = drain_x + drain_r * math.sin(angle) * 0.707
                y = drain_y + drain_r * (1.0 - math.cos(angle)) * 0.50
                if math.isclose(norm, 0.0, abs_tol=1e-4):
                    label = "Drain_Center"
                elif norm < 0.0:
                    label = "Drain_Left" if n_streams == 3 else f"Drain_Left_{abs(i - n_streams // 2)}"
                else:
                    label = "Drain_Right" if n_streams == 3 else f"Drain_Right_{i - n_streams // 2}"
                drain_features.append(CADFeature(CADFeatureType.DRAIN, label=label, x=x, y=y, z=z_lid, r=stream_r))
        else:
            drain_features.append(
                CADFeature(CADFeatureType.DRAIN, label="Drain_Center", x=drain_x, y=drain_y, z=z_lid, r=drain_r)
            )

        return drain_features, cutout_features

    @classmethod
    def from_processed_boundaries(cls, pb: Any) -> "FluidCADContext":
        """Construct FluidCADContext dynamically from URDF boundary metadata and processed boundaries."""
        lid_b = getattr(pb, "lid", None)
        tube_b = getattr(pb, "tube_wall", None)
        base_b = getattr(pb, "base", None)
        z_offset = getattr(pb, "cavity_z_offset", 0.0)
        base_h = getattr(pb, "base_height", 0.0)
        z_lid = lid_b.z_floor if lid_b is not None else (z_offset + base_h)
        tube_y = tube_b.pos[1] if tube_b is not None else (lid_b.tube_y if lid_b is not None else 0.0)
        tube_r = tube_b.r_inner if tube_b is not None else (lid_b.tube_r if lid_b is not None else 0.0)
        terrace_r = lid_b.terrace_r if lid_b is not None else 0.0
        terrace_z = lid_b.terrace_z_max if lid_b is not None else 0.0
        drain_x = lid_b.pos[0] if lid_b is not None else 0.0
        drain_y = lid_b.drain_y if lid_b is not None else 0.0
        drain_r = lid_b.drain_r if lid_b is not None else 0.0
        pocket_r = lid_b.r_pocket if lid_b is not None else 0.0
        bowl_r = base_b.radius if base_b is not None else 0.0

        drain_features, cutout_features = cls._create_drain_features(
            drain_x=drain_x,
            drain_y=drain_y,
            z_lid=z_lid,
            drain_r=drain_r,
            tube_r=tube_r,
        )

        return cls(
            features=(
                CADFeature(CADFeatureType.TUBE, x=0.0, y=tube_y, z=z_offset, r=tube_r),
                CADFeature(CADFeatureType.TERRACE, x=0.0, y=tube_y, z=terrace_z, r=terrace_r),
                *drain_features,
                *cutout_features,
                CADFeature(CADFeatureType.POCKET, x=0.0, y=0.0, z=z_lid, r=pocket_r),
                CADFeature(CADFeatureType.BOWL, x=0.0, y=0.0, z=z_offset, r=bowl_r),
            )
        )

    @classmethod
    def from_boundaries(cls, boundaries: Sequence[Any]) -> "FluidCADContext":
        """Construct FluidCADContext directly from a sequence of URDFBoundary or BoundaryConfig models."""
        from model.boundary_config import BoundaryType, LinkType, ShapeType

        features: list[CADFeature] = []
        z_floor = 0.0
        tube_r = 0.0

        for b in boundaries:
            link_type = getattr(b, "link_type", None)
            shape = getattr(b, "shape", None)
            b_type = getattr(b, "type", None)
            xyz = getattr(b, "xyz", (0.0, 0.0, 0.0))
            radius = getattr(b, "radius", 0.0) or 0.0

            # Base container
            is_base = link_type in (LinkType.BASE, "base", "bowl") or (
                shape in (ShapeType.CYLINDER, "cylinder")
                and b_type in (BoundaryType.CAVITY, "cavity")
                and link_type not in (LinkType.LID, "lid")
            )
            if is_base:
                z_floor = float(xyz[2])
                features.append(
                    CADFeature(
                        CADFeatureType.BOWL, label="Bowl", x=float(xyz[0]), y=float(xyz[1]), z=z_floor, r=float(radius)
                    )
                )

            # Delivery tube
            is_tube = (
                getattr(b, "has_tube", False)
                or link_type in (LinkType.TUBE, "tube")
                or shape in (ShapeType.TUBE, "tube")
            )
            if is_tube:
                t_pos = getattr(b, "tube_pos", xyz)
                tube_r = float(getattr(b, "tube_radius", radius))
                features.append(
                    CADFeature(
                        CADFeatureType.TUBE,
                        label="Tube",
                        x=float(t_pos[0]),
                        y=float(t_pos[1]),
                        z=z_floor,
                        r=tube_r,
                    )
                )

            # Terrace platform
            is_terrace = getattr(b, "has_intake", False) or link_type in (LinkType.OUTLET, "terrace")
            if is_terrace:
                intake_pos = getattr(b, "intake_pos", xyz)
                intake_r = getattr(b, "intake_radius", radius)
                features.append(
                    CADFeature(
                        CADFeatureType.TERRACE,
                        label="Terrace",
                        x=float(intake_pos[0]),
                        y=float(intake_pos[1]),
                        z=float(xyz[2] + intake_pos[2]),
                        r=float(intake_r),
                    )
                )

            # Lid shelf and drain
            is_lid = link_type in (LinkType.LID, "lid")
            if is_lid:
                z_lid = float(xyz[2])
                features.append(
                    CADFeature(
                        CADFeatureType.POCKET,
                        label="Pocket",
                        x=float(xyz[0]),
                        y=float(xyz[1]),
                        z=z_lid,
                        r=float(radius),
                    )
                )

                if getattr(b, "has_drain", False) and getattr(b, "drain_radius", 0.0) > 0.0:
                    d_pos = getattr(b, "drain_pos", (0.0, 0.0, 0.0))
                    d_rad = float(getattr(b, "drain_radius", 0.0))
                    d_feats, c_feats = cls._create_drain_features(
                        drain_x=float(d_pos[0]),
                        drain_y=float(d_pos[1]),
                        z_lid=z_lid,
                        drain_r=d_rad,
                        tube_r=tube_r,
                    )
                    features.extend(c_feats)
                    features.extend(d_feats)

        return cls(features=tuple(features))


class FluidBody(BaseModel):
    """Represents a dynamic, contiguous 3D fluid body undergoing motion, deformation, splitting, or merging."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    body_id: int = Field(default=0, description="Unique tracking identifier for the dynamic fluid body.")
    body_type: FluidBodyType = Field(
        default=FluidBodyType.POOL, description="Semantic classification of the fluid body."
    )
    stage: FluidStage = Field(default=FluidStage.BOWL_POOL, description="Semantic cascade stage classification.")
    feature_type: Optional[CADFeatureType] = Field(
        default=None, description="Associated CAD feature type providing geometric origin/bounds."
    )
    cad_feature: Optional[CADFeature] = Field(
        default=None, description="Associated CAD feature primitive providing dynamic origin/bounds."
    )
    tier: int = Field(
        default=0, description="Cascade tier index (0 = top terrace, 1 = mid terrace/lid, 2 = reservoir bowl)."
    )
    cad_context: Optional[FluidCADContext] = Field(
        default=None, description="CAD geometry boundaries context for dynamic physical alignment."
    )
    particle_indices: np.ndarray = Field(description="Array of global particle indices belonging to this fluid body.")
    centroid: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Current 3D centroid coordinates."
    )
    bounds_min: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Axis-aligned bounding box minimum."
    )
    bounds_max: tuple[float, float, float] = Field(
        default=(0.0, 0.0, 0.0), description="Axis-aligned bounding box maximum."
    )
    velocity: tuple[float, float, float] = Field(default=(0.0, 0.0, 0.0), description="Mean 3D velocity vector (m/s).")
    volume: float = Field(default=0.0, description="Physical fluid volume of the body in cubic meters.")
    particle_count: int = Field(default=0, description="Total number of particles in this fluid body.")
    surface_positions: Optional[np.ndarray] = Field(
        default=None, description="Local or surface particle positions (M, 3) for dynamic surface heightfield sampling."
    )

    @property
    def display_name(self) -> str:
        """Return a unique, semantic identifier for this fluid body."""
        if self.cad_feature is not None and self.cad_feature.label:
            return f"{self.stage.value}_{self.cad_feature.label.lower()}"
        return f"{self.stage.value}_{self.body_id}"

    def to_mesh(self, n_segments: int = 32) -> tuple[np.ndarray, np.ndarray]:
        """Generate watertight 3D triangle mesh vertices and face indices for this fluid body."""
        cx, cy, _ = self.centroid
        z_min = self.bounds_min[2]
        z_max = self.bounds_max[2]
        ctx = self.cad_context

        feat = self.cad_feature
        if feat is None and ctx is not None and self.feature_type is not None:
            feat = ctx.get(self.feature_type)

        match self.body_type:
            case FluidBodyType.POOL:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = max(0.010, (rx + ry) / 2.0)
                if self.feature_type == CADFeatureType.POCKET or self.tier == 1 or self.stage == FluidStage.LID_POOL:
                    solid = self.to_cad_solid()
                    if hasattr(solid, "tessellate"):
                        verts, triangles = solid.tessellate(0.0005)
                        if len(verts) > 0 and len(triangles) > 0:
                            verts_arr = np.array([[v.X, v.Y, v.Z] for v in verts], dtype=np.float32)
                            faces_arr = np.array(triangles, dtype=np.uint32)
                            return verts_arr, faces_arr

                    center = (feat.x, feat.y) if feat is not None else (0.0, 0.0)
                    z_floor_val = ctx.z_lid if ctx is not None and ctx.z_lid > 0.0 else z_min
                    z_top_val = min(
                        feat.z if feat is not None and feat.z > 0.0 else (z_floor_val + 0.010),
                        max(z_max, z_floor_val + 0.003),
                    )
                    radius = feat.r if feat is not None and feat.r > 0.0 else radius
                    return generate_heightfield_cylinder_mesh(
                        radius=radius,
                        z_floor=z_floor_val,
                        surface_positions=self.surface_positions,
                        default_z_top=z_top_val,
                        center=center,
                        n_rings=6,
                        n_spokes=n_segments,
                    )
                else:
                    center = (feat.x, feat.y) if feat is not None else (0.0, 0.0)
                    z_floor_val = ctx.z_floor if ctx is not None and ctx.z_floor > 0.0 else z_min
                    z_top_val = min(
                        ctx.z_lid - 0.005 if ctx is not None and ctx.z_lid > 0.0 else z_max,
                        max(z_max, z_floor_val + 0.015),
                    )
                    radius = feat.r if feat is not None and feat.r > 0.0 else radius

                    return generate_heightfield_cylinder_mesh(
                        radius=radius,
                        z_floor=z_floor_val,
                        surface_positions=self.surface_positions,
                        default_z_top=z_top_val,
                        center=center,
                        n_rings=6,
                        n_spokes=n_segments,
                    )

            case FluidBodyType.STREAM:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = feat.r if feat is not None and feat.r > 0.0 else max(0.003, (rx + ry) / 2.0)
                center = (feat.x, feat.y) if feat is not None else (0.0, 0.0)
                z_bot_val = ctx.z_floor if ctx is not None and ctx.z_floor > 0.0 else z_min
                terrace = ctx.get(CADFeatureType.TERRACE) if ctx is not None else None
                z_top_val = terrace.z if terrace is not None and terrace.z > 0.0 else z_max
                return generate_cylinder_mesh(radius, z_bot_val, z_top_val, center=center, n_segments=n_segments)

            case FluidBodyType.WATERFALL:
                if (
                    self.feature_type == CADFeatureType.TERRACE
                    or self.tier == 0
                    or self.stage == FluidStage.LIP_WATERFALL
                ):
                    rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                    ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                    lip_radius = feat.r if feat is not None and feat.r > 0.0 else max(0.015, (rx + ry) / 2.0)
                    center_xy = (feat.x, feat.y) if feat is not None else (0.0, 0.0)
                    z_top_val = feat.z if feat is not None and feat.z > 0.0 else z_max
                    z_bot_val = ctx.z_lid if ctx is not None and ctx.z_lid > 0.0 else z_min
                    return generate_lip_waterfall_mesh(
                        self.surface_positions,
                        center_xy=center_xy,
                        lip_radius=lip_radius,
                        z_top=z_top_val,
                        z_bot=z_bot_val,
                        n_segments=n_segments,
                    )

                # Lower Drain Waterfall: Plunges from lid pool cutout down into reservoir bowl pool
                if self.surface_positions is not None and len(self.surface_positions) > 0:
                    mean_stream_xy = (
                        float(np.mean(self.surface_positions[:, 0])),
                        float(np.mean(self.surface_positions[:, 1])),
                    )
                else:
                    mean_stream_xy = (feat.x, feat.y) if feat is not None else (0.0, 0.0)

                stream_r = min(0.012, max(0.006, feat.r if feat else 0.010))
                z_top_val = (ctx.z_lid + 0.003) if ctx is not None and ctx.z_lid > 0.0 else z_max
                if self.surface_positions is not None and len(self.surface_positions) > 0:
                    max_stream_z = float(np.max(self.surface_positions[:, 2]))
                    z_top_val = max(z_top_val, min(max_stream_z + 0.002, z_max))

                z_bot_val = (ctx.z_floor + 0.015) if ctx is not None and ctx.z_floor > 0.0 else z_min
                if self.surface_positions is not None and len(self.surface_positions) > 0:
                    z_bot_val = max(z_bot_val, float(np.min(self.surface_positions[:, 2])) - 0.003)
                return generate_waterfall_mesh(
                    self.surface_positions,
                    z_top=z_top_val,
                    z_bot=z_bot_val,
                    cutout_xy=mean_stream_xy,
                    nominal_radius=stream_r,
                    n_segments=n_segments,
                )

            case FluidBodyType.SHEET:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = feat.r if feat is not None and feat.r > 0.0 else max(0.010, (rx + ry) / 2.0)
                center = (feat.x, feat.y) if feat is not None else (0.0, 0.0)
                z_floor_val = (
                    (ctx.z_lid + feat.z) / 2.0 if ctx is not None and feat is not None and feat.z > 0.0 else z_min
                )
                z_top_val = min(feat.z + 0.003, max(z_max, feat.z)) if feat is not None and feat.z > 0.0 else z_max
                return generate_heightfield_cylinder_mesh(
                    radius=radius,
                    z_floor=z_floor_val,
                    surface_positions=self.surface_positions,
                    default_z_top=z_top_val,
                    center=center,
                    n_rings=6,
                    n_spokes=n_segments,
                )

            case _:
                if self.surface_positions is not None and len(self.surface_positions) >= 4:
                    return generate_manifold_mesh_around_particles(self.surface_positions)
                equiv_radius = max(0.002, (3.0 * max(1e-9, self.volume) / (4.0 * math.pi)) ** (1.0 / 3.0))
                equiv_radius = min(equiv_radius, max(0.004, (self.bounds_max[2] - self.bounds_min[2]) / 2.0))
                return generate_sphere_mesh(center=self.centroid, radius=equiv_radius, n_lat=10, n_lon=20)

    def to_cad_solid(self) -> Any:
        """Build and return a watertight build123d Solid representation conforming to CAD design principles."""
        from build123d import Align, BuildPart, Cylinder, Location, Locations, Mode, Sphere

        cx, cy, _ = self.centroid
        z_min = self.bounds_min[2]
        z_max = self.bounds_max[2]
        ctx = self.cad_context

        feat = self.cad_feature
        if feat is None and ctx is not None and self.feature_type is not None:
            feat = ctx.get(self.feature_type)

        match self.body_type:
            case FluidBodyType.POOL:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = max(0.010, (rx + ry) / 2.0)
                if self.feature_type == CADFeatureType.POCKET or self.tier == 1 or self.stage == FluidStage.LID_POOL:
                    pos = (
                        feat.x if feat is not None else 0.0,
                        feat.y if feat is not None else 0.0,
                        ctx.z_lid if ctx is not None and ctx.z_lid > 0.0 else z_min,
                    )
                    radius = feat.r if feat is not None and feat.r > 0.0 else radius
                    h = max(0.002, min(0.008, z_max - pos[2]))

                    with BuildPart() as bp:
                        Cylinder(radius=radius, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.ADD)
                        if ctx is not None:
                            for terrace in ctx.terraces:
                                if terrace.r > 0.0:
                                    with Locations((terrace.x - pos[0], terrace.y - pos[1], 0.0)):
                                        Cylinder(
                                            radius=terrace.r,
                                            height=h * 3.0,
                                            align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                            mode=Mode.SUBTRACT,
                                        )
                            for cutout in ctx.cutouts:
                                if cutout.r > 0.0:
                                    with Locations((cutout.x - pos[0], cutout.y - pos[1], 0.0)):
                                        Cylinder(
                                            radius=cutout.r,
                                            height=h * 3.0,
                                            align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                            mode=Mode.SUBTRACT,
                                        )
                            if len(ctx.cutouts) == 0:
                                for drain in ctx.drains:
                                    if drain.r > 0.020:
                                        with Locations((drain.x - pos[0], drain.y - pos[1], 0.0)):
                                            Cylinder(
                                                radius=drain.r,
                                                height=h * 3.0,
                                                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                                mode=Mode.SUBTRACT,
                                            )
                    solid = bp.part
                    return solid.locate(Location(pos))
                else:
                    pos = (
                        feat.x if feat is not None else 0.0,
                        feat.y if feat is not None else 0.0,
                        ctx.z_floor if ctx is not None and ctx.z_floor > 0.0 else z_min,
                    )
                    radius = feat.r if feat is not None and feat.r > 0.0 else radius
                    h = max(0.015, min((ctx.z_lid if ctx and ctx.z_lid > 0.0 else z_max) - pos[2], z_max - pos[2]))
                    c = Cylinder(radius=radius, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    return c.locate(Location(pos))

            case FluidBodyType.STREAM:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = feat.r if feat is not None and feat.r > 0.0 else max(0.003, (rx + ry) / 2.0)
                pos = (
                    feat.x if feat is not None else 0.0,
                    feat.y if feat is not None else 0.0,
                    ctx.z_floor if ctx is not None and ctx.z_floor > 0.0 else z_min,
                )
                terrace = ctx.get(CADFeatureType.TERRACE) if ctx is not None else None
                z_top = terrace.z if terrace is not None and terrace.z > 0.0 else z_max
                h = max(
                    0.010,
                    z_top - pos[2],
                )
                c = Cylinder(radius=radius, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                return c.locate(Location(pos))

            case FluidBodyType.WATERFALL:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                if (
                    self.feature_type == CADFeatureType.TERRACE
                    or self.tier == 0
                    or self.stage == FluidStage.LIP_WATERFALL
                ):
                    pos = (
                        feat.x if feat is not None else 0.0,
                        feat.y if feat is not None else 0.0,
                        ctx.z_lid if ctx is not None and ctx.z_lid > 0.0 else z_min,
                    )
                    radius = feat.r if feat is not None and feat.r > 0.0 else max(0.008, (rx + ry) / 2.0)
                    h = max(
                        0.005,
                        (feat.z if feat is not None and feat.z > 0.0 else z_max) - pos[2],
                    )
                    c = Cylinder(radius=radius, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    return c.locate(Location(pos))
                else:
                    if self.surface_positions is not None and len(self.surface_positions) > 0:
                        stream_xy = (
                            float(np.mean(self.surface_positions[:, 0])),
                            float(np.mean(self.surface_positions[:, 1])),
                        )
                    else:
                        stream_xy = (feat.x, feat.y) if feat is not None else (0.0, 0.0)

                    pos = (
                        stream_xy[0],
                        stream_xy[1],
                        ctx.z_floor if ctx is not None and ctx.z_floor > 0.0 else z_min,
                    )
                    stream_r = min(0.012, max(0.006, feat.r if feat else 0.010))
                    z_top_val = (ctx.z_lid + 0.003) if ctx is not None and ctx.z_lid > 0.0 else z_max
                    if self.surface_positions is not None and len(self.surface_positions) > 0:
                        max_stream_z = float(np.max(self.surface_positions[:, 2]))
                        z_top_val = max(z_top_val, min(max_stream_z + 0.002, z_max))

                    h = max(
                        0.010,
                        z_top_val - pos[2],
                    )
                    c = Cylinder(radius=stream_r, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    return c.locate(Location(pos))

            case FluidBodyType.SHEET:
                rx = (self.bounds_max[0] - self.bounds_min[0]) / 2.0
                ry = (self.bounds_max[1] - self.bounds_min[1]) / 2.0
                radius = feat.r if feat is not None and feat.r > 0.0 else max(0.010, (rx + ry) / 2.0)
                pos = (
                    feat.x if feat is not None else 0.0,
                    feat.y if feat is not None else 0.0,
                    (ctx.z_lid + feat.z) / 2.0 if ctx is not None and feat is not None and feat.z > 0.0 else z_min,
                )
                h = max(
                    0.002,
                    ((feat.z + 0.003) if feat is not None and feat.z > 0.0 else z_max) - pos[2],
                )
                c = Cylinder(radius=radius, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                return c.locate(Location(pos))

            case _:
                equiv_radius = max(0.002, (3.0 * max(1e-9, self.volume) / (4.0 * math.pi)) ** (1.0 / 3.0))
                s = Sphere(radius=equiv_radius)
                return s.locate(Location(self.centroid))

    def move(self, displacement: tuple[float, float, float] | np.ndarray) -> None:
        """Translate the fluid body bounding geometry and centroid by a 3D displacement vector.

        Args:
            displacement: 3D vector representing translational shift (dx, dy, dz).
        """
        disp = np.asarray(displacement, dtype=np.float32)
        self.centroid = (
            float(self.centroid[0] + disp[0]),
            float(self.centroid[1] + disp[1]),
            float(self.centroid[2] + disp[2]),
        )
        self.bounds_min = (
            float(self.bounds_min[0] + disp[0]),
            float(self.bounds_min[1] + disp[1]),
            float(self.bounds_min[2] + disp[2]),
        )
        self.bounds_max = (
            float(self.bounds_max[0] + disp[0]),
            float(self.bounds_max[1] + disp[1]),
            float(self.bounds_max[2] + disp[2]),
        )

    def recompute_shape(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        r_s: float,
    ) -> None:
        """Continuously recompute the physical geometric bounds, centroid, and volume of the moving fluid body.

        Args:
            positions: Global particle positions array of shape (N, 3).
            velocities: Global particle velocities array of shape (N, 3).
            r_s: Particle radius in meters.
        """
        if len(self.particle_indices) == 0:
            self.particle_count = 0
            self.volume = 0.0
            self.surface_positions = None
            return

        body_pos = positions[self.particle_indices]
        body_vel = velocities[self.particle_indices]

        self.particle_count = len(self.particle_indices)
        vol_particle = (4.0 / 3.0) * math.pi * (r_s**3)
        self.volume = self.particle_count * vol_particle

        mean_c = np.mean(body_pos, axis=0)
        min_b = np.min(body_pos, axis=0) - r_s
        max_b = np.max(body_pos, axis=0) + r_s
        mean_v = np.mean(body_vel, axis=0)
        self.velocity = (float(mean_v[0]), float(mean_v[1]), float(mean_v[2]))

        ctx = self.cad_context
        feat = self.cad_feature
        if feat is None and ctx is not None and self.feature_type is not None:
            feat = ctx.get(self.feature_type)

        if feat is not None:
            match self.stage:
                case FluidStage.LID_POOL:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r,
                        feat.y - feat.r,
                        max(float(min_b[2]), ctx.z_lid if ctx is not None else float(min_b[2])),
                    )
                    terrace = ctx.get(CADFeatureType.TERRACE) if ctx is not None else None
                    z_t_max = terrace.z if terrace is not None and terrace.z > 0.0 else float(max_b[2])
                    self.bounds_max = (
                        feat.x + feat.r,
                        feat.y + feat.r,
                        min(float(max_b[2]), z_t_max),
                    )
                case FluidStage.BOWL_POOL:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r,
                        feat.y - feat.r,
                        max(float(min_b[2]), ctx.z_floor if ctx is not None else float(min_b[2])),
                    )
                    self.bounds_max = (
                        feat.x + feat.r,
                        feat.y + feat.r,
                        min(float(max_b[2]), ctx.z_lid if ctx is not None and ctx.z_lid > 0.0 else float(max_b[2])),
                    )
                case FluidStage.TOP_SHEET:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r,
                        feat.y - feat.r,
                        float(min_b[2]),
                    )
                    self.bounds_max = (
                        feat.x + feat.r,
                        feat.y + feat.r,
                        float(max_b[2]),
                    )
                case FluidStage.DELIVERY_STREAM:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r,
                        feat.y - feat.r,
                        max(float(min_b[2]), ctx.z_floor if ctx is not None else float(min_b[2])),
                    )
                    terrace = ctx.get(CADFeatureType.TERRACE) if ctx is not None else None
                    z_t_max = terrace.z if terrace is not None and terrace.z > 0.0 else float(max_b[2])
                    self.bounds_max = (
                        feat.x + feat.r,
                        feat.y + feat.r,
                        min(float(max_b[2]), z_t_max),
                    )
                case FluidStage.LIP_WATERFALL:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r - 0.003,
                        feat.y - feat.r - 0.003,
                        max(float(min_b[2]), ctx.z_lid if ctx is not None else float(min_b[2])),
                    )
                    self.bounds_max = (
                        feat.x + feat.r + 0.003,
                        feat.y + feat.r + 0.003,
                        min(float(max_b[2]), feat.z if feat.z > 0.0 else float(max_b[2])),
                    )
                case FluidStage.DRAIN_WATERFALL:
                    self.centroid = (feat.x, feat.y, float(mean_c[2]))
                    self.bounds_min = (
                        feat.x - feat.r,
                        feat.y - feat.r,
                        max(float(min_b[2]), ctx.z_floor if ctx is not None else float(min_b[2])),
                    )
                    terrace = ctx.get(CADFeatureType.TERRACE) if ctx is not None else None
                    z_t_max = (
                        terrace.z
                        if terrace is not None and terrace.z > 0.0
                        else ((ctx.z_lid + 0.003) if ctx is not None and ctx.z_lid > 0.0 else float(max_b[2]))
                    )
                    self.bounds_max = (
                        feat.x + feat.r,
                        feat.y + feat.r,
                        min(float(max_b[2]), z_t_max),
                    )
                case _:
                    self.centroid = (float(mean_c[0]), float(mean_c[1]), float(mean_c[2]))
                    self.bounds_min = (float(min_b[0]), float(min_b[1]), float(min_b[2]))
                    self.bounds_max = (float(max_b[0]), float(max_b[1]), float(max_b[2]))
        else:
            self.centroid = (float(mean_c[0]), float(mean_c[1]), float(mean_c[2]))
            self.bounds_min = (float(min_b[0]), float(min_b[1]), float(min_b[2]))
            self.bounds_max = (float(max_b[0]), float(max_b[1]), float(max_b[2]))

        if self.body_type == FluidBodyType.POOL and len(body_pos) > 0:
            z_thresh = np.percentile(body_pos[:, 2], 75.0)
            top_mask = body_pos[:, 2] >= z_thresh
            self.surface_positions = body_pos[top_mask]
        else:
            self.surface_positions = body_pos

    def split(
        self,
        clusters: list[np.ndarray],
        positions: np.ndarray,
        velocities: np.ndarray,
        r_s: float,
        next_id_fn: Any,
    ) -> list["FluidBody"]:
        """Split this fluid body into multiple independent child bodies based on disconnected cluster indices.

        Args:
            clusters: List of index arrays, each containing particle indices for a disconnected cluster.
            positions: Global particle positions array of shape (N, 3).
            velocities: Global particle velocities array of shape (N, 3).
            r_s: Particle radius in meters.
            next_id_fn: Callable returning a new unique integer body ID.

        Returns:
            List of newly created child FluidBody instances.
        """
        if len(clusters) <= 1:
            self.recompute_shape(positions, velocities, r_s)
            return [self]

        child_bodies: list[FluidBody] = []
        for cluster_indices in clusters:
            if len(cluster_indices) == 0:
                continue
            child = FluidBody(
                body_id=next_id_fn(),
                body_type=self.body_type,
                particle_indices=cluster_indices,
            )
            child.recompute_shape(positions, velocities, r_s)
            child_bodies.append(child)
        return child_bodies

    def merge(
        self,
        other: "FluidBody",
        positions: np.ndarray,
        velocities: np.ndarray,
        r_s: float,
    ) -> "FluidBody":
        """Merge another fluid body into this fluid body, forming a single unified fluid volume.

        Args:
            other: The other FluidBody instance to merge with.
            positions: Global particle positions array of shape (N, 3).
            velocities: Global particle velocities array of shape (N, 3).
            r_s: Particle radius in meters.

        Returns:
            Self, updated with merged particle indices and recomputed physical shape.
        """
        merged_indices = np.union1d(self.particle_indices, other.particle_indices)
        self.particle_indices = merged_indices
        self.recompute_shape(positions, velocities, r_s)
        return self


def cluster_particles(positions: np.ndarray, max_dist: float = 0.020) -> list[np.ndarray]:
    """Group 3D particles into spatially connected clusters within max_dist threshold."""
    if positions is None or len(positions) == 0:
        return []
    if len(positions) == 1:
        return [np.array([0])]

    from scipy.spatial import KDTree

    tree = KDTree(positions)
    pairs = tree.query_pairs(max_dist)

    parent = list(range(len(positions)))

    def find(i: int) -> int:
        path = []
        while parent[i] != i:
            path.append(i)
            i = parent[i]
        for node in path:
            parent[node] = i
        return i

    def union(i: int, j: int) -> None:
        root_i, root_j = find(i), find(j)
        if root_i != root_j:
            parent[root_i] = root_j

    for i, j in pairs:
        union(i, j)

    groups: dict[int, list[int]] = {}
    for idx in range(len(positions)):
        root = find(idx)
        groups.setdefault(root, []).append(idx)

    return [np.array(indices) for indices in groups.values()]


class FluidBodyTracker:
    """Manages the dynamic lifecycle of fluid bodies, performing move, split, and merge transitions."""

    def __init__(self, r_s: float = 0.0025) -> None:
        """Initialize tracker with particle radius.

        Args:
            r_s: Particle radius in meters.
        """
        self.r_s = r_s
        self._next_id = 1
        self.bodies: dict[int, FluidBody] = {}

    def _get_next_id(self) -> int:
        """Generate unique integer body ID."""
        nid = self._next_id
        self._next_id += 1
        return nid

    def update_bodies(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        cad_context: Optional[FluidCADContext] = None,
    ) -> list[FluidBody]:
        """Classify and recompute dynamic fluid bodies across the 5-stage multi-tier architecture.

        Args:
            positions: Global particle positions array of shape (N, 3).
            velocities: Global particle velocities array of shape (N, 3).
            cad_context: Dynamic CAD geometry boundaries context model.

        Returns:
            List of active FluidBody instances with recomputed physical shapes.
        """
        if len(positions) == 0:
            self.bodies.clear()
            return []

        active = positions[:, 2] < 100.0
        active_indices = np.flatnonzero(active)

        if len(active_indices) == 0:
            self.bodies.clear()
            return []

        ctx = cad_context if cad_context is not None else FluidCADContext()
        z_floor = ctx.z_floor
        z_lid = ctx.z_lid

        tube = ctx.get(CADFeatureType.TUBE)
        tube_y = tube.y if tube is not None else 0.0
        tube_r = tube.r if tube is not None else 0.0

        bowl = ctx.get(CADFeatureType.BOWL)

        pos_act = positions[active_indices]
        d_tube_xy = np.sqrt(pos_act[:, 0] ** 2 + (pos_act[:, 1] - tube_y) ** 2)

        # 1. Delivery Stream rising inside the delivery tube
        is_stream = (d_tube_xy <= tube_r + self.r_s * 0.5) & (pos_act[:, 2] >= z_floor) & (pos_act[:, 2] <= z_lid)

        # 2. Basin particles below lid
        in_basin = (pos_act[:, 2] < z_lid - self.r_s) & (~is_stream)

        # 3. Robust bowl reservoir free-surface elevation computed from undisturbed bed
        bed_mask = in_basin & (d_tube_xy > tube_r + self.r_s * 2.0)
        for drain in ctx.drains:
            d_d_xy = np.sqrt((pos_act[:, 0] - drain.x) ** 2 + (pos_act[:, 1] - drain.y) ** 2)
            bed_mask = bed_mask & (d_d_xy > max(0.020, drain.r + self.r_s * 2.0))

        bed_indices = np.flatnonzero(bed_mask)
        z_pool_max_allowed = z_lid - 0.015

        if len(bed_indices) > 0:
            bed_z = pos_act[bed_indices, 2]
            z_pool_surf = min(float(np.percentile(bed_z, 75.0) + self.r_s), z_pool_max_allowed)
            z_pool_surf = max(z_pool_surf, z_floor + self.r_s)
        else:
            basin_indices = np.flatnonzero(in_basin)
            if len(basin_indices) > 0:
                basin_z = pos_act[basin_indices, 2]
                z_pool_surf = min(float(np.percentile(basin_z, 75.0) + self.r_s), z_pool_max_allowed)
                z_pool_surf = max(z_pool_surf, z_floor + self.r_s)
            else:
                z_pool_surf = float(z_floor)

        body_specs: list[
            tuple[FluidBodyType, FluidStage, CADFeatureType, int, int, np.ndarray, Optional[CADFeature]]
        ] = []

        # 1. Delivery Stream bodies (per tube)
        for tb_idx, tube_feat in enumerate(ctx.tubes):
            b_id = tb_idx + 1
            body_specs.append(
                (FluidBodyType.STREAM, FluidStage.DELIVERY_STREAM, CADFeatureType.TUBE, 0, b_id, is_stream, tube_feat)
            )

        # 4. Terraces: Top Sheet and Lip Waterfall per terrace
        all_platform_mask = np.zeros(len(pos_act), dtype=bool)
        for t_idx, terrace in enumerate(ctx.terraces):
            d_t_xy = np.sqrt((pos_act[:, 0] - terrace.x) ** 2 + (pos_act[:, 1] - terrace.y) ** 2)
            is_terrace_zone = d_t_xy <= terrace.r + self.r_s
            is_lip_zone = d_t_xy <= terrace.r + max(0.006, self.r_s * 2.0)
            all_platform_mask |= is_lip_zone

            z_t_max = terrace.z if terrace.z > 0.0 else z_lid
            z_plat_mid = (z_lid + z_t_max) / 2.0

            is_top_sheet = is_terrace_zone & (pos_act[:, 2] >= z_plat_mid) & (~is_stream)
            has_top_sheet = np.count_nonzero(is_top_sheet) >= 2

            # Lip waterfall: active only when top sheet is active and spilling over lip / ridge
            is_lip_wf = (
                has_top_sheet
                & is_lip_zone
                & (pos_act[:, 2] < z_plat_mid)
                & (pos_act[:, 2] >= z_lid - self.r_s)
                & (~is_stream)
            )

            b_id = t_idx + 1
            body_specs.append(
                (
                    FluidBodyType.SHEET,
                    FluidStage.TOP_SHEET,
                    CADFeatureType.TERRACE,
                    0,
                    b_id,
                    is_top_sheet,
                    terrace,
                )
            )
            body_specs.append(
                (
                    FluidBodyType.WATERFALL,
                    FluidStage.LIP_WATERFALL,
                    CADFeatureType.TERRACE,
                    0,
                    b_id,
                    is_lip_wf,
                    terrace,
                )
            )

        # 5. Lid Pockets / Shelf Pools per pocket
        all_cutout_mask = np.zeros(len(pos_act), dtype=bool)
        for cutout in ctx.cutouts:
            d_c_xy = np.sqrt((pos_act[:, 0] - cutout.x) ** 2 + (pos_act[:, 1] - cutout.y) ** 2)
            all_cutout_mask |= d_c_xy <= cutout.r

        all_drain_column_mask = np.zeros(len(pos_act), dtype=bool)
        for drain in ctx.drains:
            d_d_xy = np.sqrt((pos_act[:, 0] - drain.x) ** 2 + (pos_act[:, 1] - drain.y) ** 2)
            drain_rad = max(0.015, drain.r + self.r_s * 2.0)
            all_drain_column_mask |= d_d_xy <= drain_rad

        for p_idx, pocket in enumerate(ctx.pockets):
            d_p_xy = np.sqrt((pos_act[:, 0] - pocket.x) ** 2 + (pos_act[:, 1] - pocket.y) ** 2)
            is_lid_pool = (
                (~all_platform_mask)
                & (~all_cutout_mask)
                & (~all_drain_column_mask)
                & (pos_act[:, 2] >= z_lid - self.r_s)
                & (d_p_xy <= pocket.r + self.r_s)
            )
            b_id = p_idx + 1
            body_specs.append(
                (FluidBodyType.POOL, FluidStage.LID_POOL, CADFeatureType.POCKET, 1, b_id, is_lid_pool, pocket)
            )

        # 6. Drains: Waterfall per drain with upstream adjacent inflow activation check
        z_terrace_max = max([t.z for t in ctx.terraces], default=z_lid)
        for d_idx, drain in enumerate(ctx.drains):
            d_d_xy = np.sqrt((pos_act[:, 0] - drain.x) ** 2 + (pos_act[:, 1] - drain.y) ** 2)
            drain_rad = max(0.015, drain.r + self.r_s * 2.0)
            in_drain_zone = d_d_xy <= drain_rad

            # Check upstream inflow from both routes:
            # Route A: Fluid at lid shelf reaching drain aperture
            lid_inflow_fluid = (pos_act[:, 2] >= z_lid - self.r_s) & (d_d_xy <= drain.r + self.r_s * 2.0)
            # Route B: Fluid plunging directly from top terrace sheet into drain aperture
            top_sheet_inflow_fluid = (pos_act[:, 2] >= z_plat_mid) & (d_d_xy <= drain_rad + max(0.010, self.r_s * 3.0))

            has_drain_inflow = (np.count_nonzero(lid_inflow_fluid) >= 2) or (
                np.count_nonzero(top_sheet_inflow_fluid) >= 2
            )

            # Falling particles in the air column between top sheet / lid and pool surface
            falling_in_col = (
                (pos_act[:, 2] > z_pool_surf)
                & (pos_act[:, 2] <= z_terrace_max + self.r_s)
                & in_drain_zone
                & (~is_stream)
            )
            has_falling_col = np.count_nonzero(falling_in_col) >= 2

            # Activate drain waterfall when upstream lid pool / top sheet is feeding the drain or particles are actively falling
            is_drain_wf = (has_drain_inflow or has_falling_col) & falling_in_col

            b_id = d_idx + 1
            body_specs.append(
                (
                    FluidBodyType.WATERFALL,
                    FluidStage.DRAIN_WATERFALL,
                    CADFeatureType.DRAIN,
                    1,
                    b_id,
                    is_drain_wf,
                    drain,
                )
            )

        # 7. Reservoir Bowl Pools (per bowl)
        for b_idx, bowl_feat in enumerate(ctx.bowls):
            is_bowl_pool = in_basin & (pos_act[:, 2] <= z_pool_surf)
            b_id = b_idx + 1
            body_specs.append(
                (FluidBodyType.POOL, FluidStage.BOWL_POOL, CADFeatureType.BOWL, 2, b_id, is_bowl_pool, bowl_feat)
            )

        # 8. Splash Clusters (airborne droplets not part of any active waterfall)
        is_cluster = in_basin & (pos_act[:, 2] > z_pool_surf) & (~all_drain_column_mask)

        active_bodies: list[FluidBody] = []
        for b_type, stage, feat_type, tier, b_id, mask, feat_inst in body_specs:
            indices = active_indices[mask]
            if len(indices) == 0:
                continue

            body = FluidBody(
                body_id=b_id,
                body_type=b_type,
                stage=stage,
                feature_type=feat_type,
                cad_feature=feat_inst,
                tier=tier,
                particle_indices=indices,
                cad_context=ctx,
            )
            body.recompute_shape(positions, velocities, self.r_s)
            self.bodies[b_id if b_type != FluidBodyType.POOL else (b_id + 100)] = body
            active_bodies.append(body)

        # Clusters for free splash droplets
        cluster_indices = active_indices[is_cluster]
        if len(cluster_indices) > 0:
            cluster_subsets = cluster_particles(positions[cluster_indices], max_dist=0.020)
            if len(cluster_subsets) > 10:
                cluster_subsets = sorted(cluster_subsets, key=len, reverse=True)[:10]
            for idx, c_subset in enumerate(cluster_subsets):
                c_indices = cluster_indices[c_subset]
                child = FluidBody(
                    body_id=idx + 1,
                    body_type=FluidBodyType.CLUSTER,
                    stage=FluidStage.SPLASH_CLUSTER,
                    particle_indices=c_indices,
                )
                child.recompute_shape(positions, velocities, self.r_s)
                active_bodies.append(child)

        return active_bodies
