"""Simulation hooks and flow metrics for the cat fountain project."""

import numpy as np
import pybullet as p
from provider.bullet import _is_real_physics_client, get_bullet_link_names
from typing import Any, Callable, cast, Optional
from provider import Bullet, LinkType, Fluid, Simulate, URDFShape, rerun_is_enabled
from model import FluidConfig, BoundaryConfig, ResolvedBoundaries


def compute_flow_metrics(provider: Any, step_idx: Optional[int] = None) -> dict[str, int]:
    """Compute flow, sheet, drainage, and reservoir volume metrics from current fluid state."""
    metrics = {
        "flow_spout": 0,
        "flow_tube": 0,
        "flow_lid_sheet": 0,
        "drainage_waterfall": 0,
        "drainage_cutout": 0,
        "pool_volume": 0,
    }
    if getattr(provider, "water_sim", None) is None:
        return metrics

    positions = getattr(provider.water_sim, "last_positions", None)
    if positions is None or len(positions) == 0:
        return metrics

    pos_pts = np.asarray(positions)
    active_mask = pos_pts[:, 2] < 100.0
    if not np.any(active_mask):
        return metrics

    pos_active = pos_pts[active_mask]
    xs, ys, zs = pos_active[:, 0], pos_active[:, 1], pos_active[:, 2]

    tube_x = 0.0
    tube_y = 0.028
    tube_r_inner = (provider.settings.tube_radius - provider.settings.tube_thickness) * 0.001
    floor_z = provider.settings.floor_z * 0.001
    tube_top_z = (provider.settings.floor_z + provider.settings.tube_height) * 0.001
    cutout_y = provider.settings.lid_cutout_y * 0.001
    cutout_r = provider.settings.lid_cutout_radius * 0.001
    bowl_r = (provider.settings.bowl_radius - provider.settings.bowl_thickness) * 0.001
    bowl_h = provider.settings.bowl_height * 0.001
    step_d = provider.settings.lid_step_depth * 0.001
    lid_mount_z = bowl_h - step_d
    lid_pocket_floor_z = lid_mount_z + provider.settings.lid_pocket_z_offset * 0.001
    lid_platform_z = lid_pocket_floor_z + 0.00175

    dist_tube = np.sqrt((xs - tube_x) ** 2 + (ys - tube_y) ** 2)
    dist_cutout = np.sqrt(xs**2 + (ys - cutout_y) ** 2)
    r_xy = np.sqrt(xs**2 + ys**2)
    tube_r_outer = provider.settings.tube_radius * 0.001

    # 1. Flow in vertical delivery tube (strictly inside bore from floor to spout exit)
    in_tube_mask = (dist_tube <= tube_r_inner) & (zs >= floor_z) & (zs < tube_top_z - 0.002)
    in_tube_cnt = int(np.sum(in_tube_mask))

    # 2. Flow emerging at spout dome
    at_spout_mask = (dist_tube <= tube_r_outer + 0.010) & (zs >= tube_top_z - 0.002) & (zs <= tube_top_z + 0.012)
    at_spout_cnt = int(np.sum(at_spout_mask))

    # 3. Flow on lid drinking shelf / tray (outside tube bore, inside lid rim, on top surface)
    on_cutout_hole = (dist_cutout <= cutout_r) & (dist_tube > 0.030)
    lid_sheet_mask = (
        (zs >= lid_pocket_floor_z - 0.002)
        & (zs <= lid_platform_z + 0.006)
        & (dist_tube > tube_r_inner)
        & (r_xy <= 0.082)
        & (~on_cutout_hole)
    )
    lid_sheet_cnt = int(np.sum(lid_sheet_mask))

    # 6. Reservoir pool volume (entire base container fluid layer below falling air gap, excluding tube bore)
    pool_mask = (zs >= floor_z - 0.003) & (zs < lid_mount_z - 0.015) & (r_xy <= bowl_r) & (~in_tube_mask)
    pool_cnt = int(np.sum(pool_mask))

    # 7. Reservoir water depth (height of fluid surface above bowl floor in meters)
    if pool_cnt > 0:
        pool_zs = zs[pool_mask]
        pool_surface_z = float(np.percentile(pool_zs, 95))
        water_depth = max(0.0, pool_surface_z - floor_z)
        surf_layer = pool_zs[pool_zs >= np.percentile(pool_zs, 80)]
        pool_surface_variance = float(np.var(surf_layer)) if len(surf_layer) > 0 else 0.0
        pool_wave_amplitude = float(np.max(pool_zs) - np.median(pool_zs)) if len(pool_zs) > 0 else 0.0
    else:
        pool_surface_z = floor_z
        water_depth = 0.0
        pool_surface_variance = 0.0
        pool_wave_amplitude = 0.0

    # 4. Drainage: Perimeter waterfall cascading into bowl (R >= 75mm, falling or rolling off lid rim)
    waterfall_mask = (r_xy >= 0.075) & (zs >= pool_surface_z + 0.005) & (zs <= lid_platform_z + 0.004)
    waterfall_cnt = int(np.sum(waterfall_mask))

    # 5. Drainage: Front cutout drain returning to bowl (inside cutout hole, falling through air gap above pool)
    drain_mask = (
        (dist_cutout <= cutout_r + 0.005)
        & (dist_tube > 0.030)
        & (zs >= pool_surface_z + 0.003)
        & (zs < lid_pocket_floor_z)
    )
    drain_cnt = int(np.sum(drain_mask))

    # 8. Ejected / airborne explosion particles breaching the top spout dome or outer perimeter
    dome_top_z = lid_mount_z + 0.0156
    ejected_mask = (zs > dome_top_z + 0.005) | ((r_xy > bowl_r + 0.005) & (zs > lid_mount_z))
    ejected_cnt = int(np.sum(ejected_mask))

    velocities = getattr(provider.water_sim, "last_velocities", None)
    if velocities is not None:
        vel_pts = np.asarray(velocities)
        if vel_pts.ndim == 2 and len(vel_pts) == len(active_mask):
            vz = vel_pts[active_mask, 2]
            v_all = vel_pts[active_mask]
            v_mags = np.linalg.norm(v_all, axis=-1)
            max_speed = float(np.max(v_mags)) if len(v_mags) > 0 else 0.0
            max_vz = float(np.max(vz)) if len(vz) > 0 else 0.0
            sheet_speed = float(np.mean(v_mags[lid_sheet_mask])) if lid_sheet_cnt > 0 else 0.0
            drain_down_vz = float(np.mean(-vz[drain_mask])) if drain_cnt > 0 else 0.0
            drain_stagnant_cnt = int(np.sum(drain_mask & (np.abs(vz) < 0.05)))
        else:
            max_speed = 0.0
            max_vz = 0.0
            sheet_speed = 0.0
            drain_down_vz = 0.0
            drain_stagnant_cnt = 0
    else:
        max_speed = 0.0
        max_vz = 0.0
        sheet_speed = 0.0
        drain_down_vz = 0.0
        drain_stagnant_cnt = 0

    metrics = {
        "flow_spout": at_spout_cnt,
        "flow_tube": in_tube_cnt,
        "flow_lid_sheet": lid_sheet_cnt,
        "drainage_waterfall": waterfall_cnt,
        "drainage_cutout": drain_cnt,
        "pool_volume": pool_cnt,
        "water_depth": round(water_depth, 5),
        "pool_surface_variance": round(pool_surface_variance, 7),
        "pool_wave_amplitude": round(pool_wave_amplitude, 5),
        "ejected_particles": ejected_cnt,
        "max_vertical_vel": round(max_vz, 4),
        "max_particle_speed": round(max_speed, 4),
        "mean_sheet_speed": round(sheet_speed, 4),
        "drain_downward_vel": round(drain_down_vz, 4),
        "drain_stagnant_particles": drain_stagnant_cnt,
    }
    provider.last_metrics = metrics
    if not hasattr(provider, "metrics_history") or provider.metrics_history is None:
        provider.metrics_history = []
    provider.metrics_history.append(metrics)

    if step_idx is not None and rerun_is_enabled():
        import rerun as rr

        rr.set_time("step", sequence=step_idx)
        rr.log("metrics/flow_spout", rr.Scalars(float(at_spout_cnt)))
        rr.log("metrics/flow_tube", rr.Scalars(float(in_tube_cnt)))
        rr.log("metrics/flow_lid_sheet", rr.Scalars(float(lid_sheet_cnt)))
        rr.log("metrics/drainage_waterfall", rr.Scalars(float(waterfall_cnt)))
        rr.log("metrics/drainage_cutout", rr.Scalars(float(drain_cnt)))
        rr.log("metrics/pool_volume", rr.Scalars(float(pool_cnt)))
        rr.log("metrics/water_depth", rr.Scalars(float(water_depth)))
        rr.log("metrics/ejected_particles", rr.Scalars(float(ejected_cnt)))
        rr.log("metrics/max_vertical_vel", rr.Scalars(float(max_vz)))
        rr.log("metrics/max_particle_speed", rr.Scalars(float(max_speed)))
        rr.log("metrics/mean_sheet_speed", rr.Scalars(float(sheet_speed)))
        rr.log("metrics/drain_downward_vel", rr.Scalars(float(drain_down_vz)))
        rr.log("metrics/drain_stagnant_particles", rr.Scalars(float(drain_stagnant_cnt)))

    return metrics


def get_simulate_hooks_impl(self: Any, sim_name: str) -> dict[Simulate, Callable[..., Any]]:
    """Return simulation hooks for the cat fountain."""
    self.water_sim = None
    self.last_metrics = {}
    self.metrics_history = []
    self.compute_flow_metrics = lambda step_idx=None: compute_flow_metrics(self, step_idx)

    def setup_simulation(body_id, client, name, boundaries, state_tracker=None):
        self.last_metrics = {}
        if _is_real_physics_client(client):
            p.setGravity(0.0, 0.0, -9.81, physicsClientId=client)

        link_names = get_bullet_link_names(body_id, client)
        resolved = ResolvedBoundaries.from_link_names(boundaries=boundaries, link_names=link_names)
        link_indices = resolved.link_indices
        resolved_boundaries = resolved.boundaries

        self.water_sim_damping = 0.995
        self.water_sim = Fluid(
            config=FluidConfig.water(
                sim_name=name,
                boundaries=resolved_boundaries,
                recycle_fluid=True,  # Enable fluid recycling
                gravity=(0.0, 0.0, -9.81),
                r_s=0.0015,
                target_volume=self.settings.target_volume,
                stiffness=1000.0,
                damping=0.995,
                viscosity=0.02,
                slot_height=self.settings.slot_height * 0.001,
                fallen_threshold_liters=0.001,
                damping_boundary=self.settings.damping_boundary,
                stiffness_boundary=self.settings.stiffness_boundary,
                nx=64,
                ny=64,
                nz=40,
                dx=0.0035,
                origin=(-0.112, -0.112, 0.0),
            ),
            provider=self,
            body_id=body_id,
            physics_client=client,
            state_tracker=state_tracker,
            link_indices=link_indices,
        )

        bullet_sim = Bullet(self.room, {}, "", "", 0, None, None)
        bullet_sim.reset_camera(client, view_from="top rear")

    def step_simulation(body_id, client, step_idx, name):
        assert self.water_sim is not None
        target_omega = 15.0
        max_force = 10.0
        vane_obj = cast(URDFShape, self.room["impeller"][0])
        target_omega = float(getattr(vane_obj, "urdf_motor_target", 15.0))
        max_force = float(getattr(vane_obj, "urdf_motor_force", 10.0))
        motor_power = getattr(self.settings, "motor_power", None)
        self.water_sim.update(
            body_id,
            client,
            target_omega=target_omega,
            max_force=max_force,
            motor_power=motor_power,
            damping=getattr(self, "water_sim_damping", 0.995),
        )

        compute_flow_metrics(self, step_idx=step_idx)

        if (
            not self.water_sim.recycle_fluid
            and len(self.water_sim.fallen_out_water_ids) * self.water_sim.vol_s * 1000.0
            >= self.water_sim.fallen_threshold_liters
        ):
            return f"{self.water_sim.fallen_threshold_liters}L of water fell out of bowl at step {step_idx}"
        return None

    return {
        Simulate.SETUP: setup_simulation,
        Simulate.STEP: step_simulation,
    }
