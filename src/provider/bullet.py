"""PyBullet simulation engine lifecycle manager."""

import os
from enum import IntEnum
import shutil
import tempfile
import pybullet as p
import socket
import xml.etree.ElementTree as ET
from typing import Any, Optional, Callable, cast
import rerun as rr
import queue
import threading
from provider.types import CollisionGroup, CollisionMask, URDFShape, URDFCollisionType, Simulate, LinkType


def _is_real_physics_client(physics_client: Any) -> bool:
    """Check if the given physics client ID is connected to a real physics server."""
    if not isinstance(physics_client, int) or physics_client < 0:
        return False
    try:
        info = p.getConnectionInfo(physicsClientId=physics_client)
        return bool(info.get("isConnected", False))
    except Exception:
        return False


def get_bullet_link_names(body_id: Optional[int], physics_client: Optional[int]) -> list[str]:
    """Extract joint/link names from a PyBullet body ID.

    Args:
        body_id: PyBullet multi-body ID.
        physics_client: PyBullet physics client ID.

    Returns:
        List of link name strings indexed by joint index.
    """
    if body_id is None or physics_client is None or not _is_real_physics_client(physics_client):
        return []
    return [
        p.getJointInfo(body_id, i, physicsClientId=physics_client)[12].decode("utf-8")
        for i in range(p.getNumJoints(body_id, physicsClientId=physics_client))
    ]


class BulletStateTracker:
    """Helper class to track and query PyBullet body and particle states efficiently."""

    def __init__(self, body_id: int, physics_client: int, label_to_link_idx: dict[str, int]):
        """Initialize the Tracker."""
        self.body_id = body_id
        self.physics_client = physics_client
        self.label_to_link_idx = label_to_link_idx
        self.particle_body_ids: list[int] = []
        self.particle_colors: list[list[float]] = []
        self.particle_radii: list[float] = []
        self.transforms: dict[str, tuple[list[float], list[float]]] = {}
        self.particle_positions: list[list[float]] = []
        self.boundary_voxels: Optional[dict[str, Any]] = None
        self.fluid_bodies: Optional[list[Any]] = None
        self.water_meshes: Optional[dict[str, tuple[np.ndarray, np.ndarray]]] = None
        self._last_checked_num_bodies = 0
        self.has_fluid_simulator = False

        # Cache inverse inertia transform for the base link to avoid slow dynamics IPC calls
        try:
            dynamics = p.getDynamicsInfo(self.body_id, -1, physicsClientId=self.physics_client)
            local_inertia_pos = dynamics[3]
            local_inertia_orn = dynamics[4]
            self._inv_inertia = p.invertTransform(local_inertia_pos, local_inertia_orn)
        except Exception:
            self._inv_inertia = None

    def _discover_new_particles(self) -> None:
        """Scan for newly created bodies since the last check and add them to particles."""
        is_real = _is_real_physics_client(self.physics_client)
        if not is_real:
            return

        num_bodies = p.getNumBodies(physicsClientId=self.physics_client)
        if num_bodies <= self._last_checked_num_bodies:
            return

        for i in range(self._last_checked_num_bodies, num_bodies):
            if i == self.body_id:
                continue
            dynamics = p.getDynamicsInfo(i, -1, physicsClientId=self.physics_client)
            mass = dynamics[0]
            if mass > 0.0:
                self.particle_body_ids.append(i)
                visual_data = p.getVisualShapeData(i, physicsClientId=self.physics_client)
                color = [0.5, 0.8, 1.0, 0.7]  # Default fallback color
                if visual_data:
                    color = list(visual_data[0][7])
                self.particle_colors.append(color)

                shape_data = p.getCollisionShapeData(i, -1, physicsClientId=self.physics_client)
                radius = 0.003  # Default fallback
                if shape_data:
                    radius = shape_data[0][3][0]
                self.particle_radii.append(radius)
        self._last_checked_num_bodies = num_bodies

    def update_state(self) -> None:
        """Query and update internal state properties from PyBullet."""
        if not self.has_fluid_simulator:
            self._discover_new_particles()

        self.transforms = {}
        for label, idx in self.label_to_link_idx.items():
            if idx == -1:
                base_pos, base_orn = p.getBasePositionAndOrientation(self.body_id, physicsClientId=self.physics_client)
                if self._inv_inertia is not None:
                    pos, orn = p.multiplyTransforms(base_pos, base_orn, self._inv_inertia[0], self._inv_inertia[1])
                else:
                    pos, orn = base_pos, base_orn
            else:
                state = p.getLinkState(self.body_id, idx, physicsClientId=self.physics_client)
                pos, orn = state[4], state[5]
            self.transforms[label] = (pos, orn)

        if not self.has_fluid_simulator:
            self.particle_positions = []
            for i in self.particle_body_ids:
                pos, _ = p.getBasePositionAndOrientation(i, physicsClientId=self.physics_client)
                self.particle_positions.append(pos)


class Bullet:
    """Manages PyBullet physics engine lifecycle, environment, loading URDF, and stepping simulation."""

    def __init__(
        self,
        room: Any,
        provider_hooks: dict[Simulate, Callable[..., Any]],
        proj_name: str,
        sim_target: str,
        steps: int,
        manager: Any,
        logger: Any,
        build_dir: str = "build",
        save_rrd: Optional[str] = None,
        rerun_port: Optional[int] = None,
        spawn_viewer: bool = True,
        stage_window_size: Optional[int] = None,
    ):
        """Initialize the simulator."""
        self.room = room
        self.provider_hooks = provider_hooks
        self.proj_name = proj_name
        self.sim_target = sim_target
        self.steps = steps
        self.manager = manager
        self.logger = logger
        self.build_dir = build_dir
        self.save_rrd = save_rrd
        self.rerun_port = rerun_port
        self.spawn_viewer = spawn_viewer
        self.stage_window_size = stage_window_size

    def _parse_urdf_meshes(self, build_proj_dir: str) -> dict[str, str]:
        """Parse URDF files to map link names to their OBJ filenames."""
        link_to_obj = {}
        if not os.path.exists(build_proj_dir):
            return link_to_obj
        try:
            urdf_files = [f for f in os.listdir(build_proj_dir) if f.endswith(".urdf")]
        except OSError:
            return link_to_obj

        for urdf_file in urdf_files:
            urdf_path = os.path.join(build_proj_dir, urdf_file)
            try:
                tree = ET.parse(urdf_path)
            except (OSError, ET.ParseError):
                continue
            root = tree.getroot()
            for link in root.findall(".//link"):
                link_name = link.attrib.get("name")
                if not link_name:
                    continue
                mesh = link.find(".//visual/geometry/mesh")
                if mesh is None:
                    mesh = link.find(".//collision/geometry/mesh")
                if mesh is not None:
                    filename = mesh.attrib.get("filename", "")
                    if filename.startswith("package://"):
                        obj_name = os.path.basename(filename[len("package://") :])
                        link_to_obj[link_name] = obj_name
        return link_to_obj

    def _copy_project_assets(self, build_proj_dir: str, proj_dir: str) -> None:
        """Copy required OBJ files referenced in the URDF to the temporary directory."""
        build_urdf_dir = build_proj_dir.replace("/obj/", "/urdf/").replace("/obj", "/urdf")
        if not os.path.exists(build_urdf_dir):
            build_urdf_dir = build_proj_dir
        link_to_obj = self._parse_urdf_meshes(build_urdf_dir)
        if not link_to_obj:
            # Fallback for unit tests where no compiled URDF exists on disk
            for geom, _ in self.room.values():
                u_geom = cast(URDFShape, geom)
                label = getattr(u_geom, "urdf_label", None)
                if label:
                    obj_filename = getattr(u_geom, "urdf_obj_filename", f"{label}.obj")
                    real_obj_path = os.path.join(build_proj_dir, obj_filename)
                    temp_obj_path = os.path.join(proj_dir, obj_filename)
                    if os.path.exists(real_obj_path):
                        shutil.copy(real_obj_path, temp_obj_path)
                    else:
                        raise FileNotFoundError(f"Required OBJ file not found for simulation: {real_obj_path}")
            return

        copied = set()
        for obj_filename in link_to_obj.values():
            if obj_filename in copied:
                continue
            real_obj_path = os.path.join(build_proj_dir, obj_filename)
            temp_obj_path = os.path.join(proj_dir, obj_filename)
            if os.path.exists(real_obj_path):
                shutil.copy(real_obj_path, temp_obj_path)
                copied.add(obj_filename)
            else:
                raise FileNotFoundError(f"Required OBJ file not found for simulation: {real_obj_path}")

    def _init_simulation_objects(
        self,
        physics_client: int,
        body_id: int,
        proj_dir: str,
        urdf_path: str,
    ) -> dict[str, int]:
        """Configure motor controls, log static assets, and setup concave collisions."""
        rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
        num_joints = p.getNumJoints(body_id, physicsClientId=physics_client)
        is_real = _is_real_physics_client(physics_client)

        if is_real:
            p.setCollisionFilterGroupMask(
                body_id, -1, CollisionGroup.CONTAINER, CollisionMask.ALL, physicsClientId=physics_client
            )
            for i in range(num_joints):
                p.setCollisionFilterGroupMask(
                    body_id, i, CollisionGroup.CONTAINER, CollisionMask.ALL, physicsClientId=physics_client
                )

        joint_name_to_index = {}
        for i in range(num_joints):
            info = p.getJointInfo(body_id, i, physicsClientId=physics_client)
            joint_name = info[1].decode("utf-8")
            joint_name_to_index[joint_name] = i

        label_to_link_idx = {}
        for geom, _ in self.room.values():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", None)
            if label:
                parent_label = getattr(u_geom, "urdf_parent", None)
                if not parent_label:
                    label_to_link_idx[label] = -1
                else:
                    joint_name = f"{parent_label}_to_{label}"
                    if joint_name in joint_name_to_index:
                        label_to_link_idx[label] = joint_name_to_index[joint_name]

        # Parse URDF XML to find concave collision links
        concave_links = set()
        if urdf_path and os.path.exists(urdf_path):
            try:
                tree = ET.parse(urdf_path)
                root = tree.getroot()
                for link_node in root.findall(".//link"):
                    link_name = link_node.attrib.get("name")
                    col_type_node = link_node.find(".//collision/collision_type")
                    if col_type_node is not None and col_type_node.text == "concave":
                        concave_links.add(link_name)
            except Exception:
                pass

        # Configure concave trimeshes for links marked concave
        for link_name in concave_links:
            if link_name in label_to_link_idx:
                link_idx = label_to_link_idx[link_name]
                p.setCollisionFilterGroupMask(body_id, link_idx, 0, 0, physicsClientId=physics_client)

                shapes = p.getCollisionShapeData(body_id, link_idx, physicsClientId=physics_client)
                for shape in shapes:
                    geom_type = shape[2]
                    if geom_type == p.GEOM_MESH:
                        mesh_scale = shape[3]

                        local_mesh_path = os.path.join(proj_dir, f"{link_name}.obj")
                        if not os.path.exists(local_mesh_path):
                            shape_filename = shape[4].decode("utf-8")
                            base_filename = os.path.basename(shape_filename)
                            local_mesh_path = os.path.join(proj_dir, base_filename)

                        if os.path.exists(local_mesh_path):
                            if link_idx == -1:
                                link_pos, link_orn = p.getBasePositionAndOrientation(
                                    body_id, physicsClientId=physics_client
                                )
                            else:
                                state = p.getLinkState(body_id, link_idx, physicsClientId=physics_client)
                                link_pos = state[0]
                                link_orn = state[1]

                            local_pos = shape[5]
                            local_orn = shape[6]
                            world_pos, world_orn = p.multiplyTransforms(link_pos, link_orn, local_pos, local_orn)

                            col_id = p.createCollisionShape(
                                shapeType=p.GEOM_MESH,
                                fileName=local_mesh_path,
                                flags=p.GEOM_FORCE_CONCAVE_TRIMESH,
                                meshScale=mesh_scale,
                                physicsClientId=physics_client,
                            )
                            static_body_id = p.createMultiBody(
                                baseMass=0.0,
                                baseCollisionShapeIndex=col_id,
                                basePosition=world_pos,
                                baseOrientation=world_orn,
                                physicsClientId=physics_client,
                            )
                            if is_real:
                                p.setCollisionFilterGroupMask(
                                    static_body_id,
                                    -1,
                                    CollisionGroup.CONTAINER,
                                    CollisionMask.ALL,
                                    physicsClientId=physics_client,
                                )
                            p.setCollisionFilterPair(body_id, static_body_id, -1, -1, 0, physicsClientId=physics_client)
                            for l_idx in range(num_joints):
                                p.setCollisionFilterPair(
                                    body_id, static_body_id, l_idx, -1, 0, physicsClientId=physics_client
                                )

        for geom, _ in self.room.values():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", None)
            parent_label = getattr(u_geom, "urdf_parent", None)
            if label and parent_label:
                joint_name = f"{parent_label}_to_{label}"
                if joint_name in joint_name_to_index:
                    idx = joint_name_to_index[joint_name]
                    motor_type = getattr(u_geom, "urdf_motor_type", None)
                    if motor_type:
                        target = getattr(u_geom, "urdf_motor_target", 0.0)
                        force = getattr(u_geom, "urdf_motor_force", 10.0)
                        if motor_type == "velocity":
                            p.setJointMotorControl2(
                                bodyUniqueId=body_id,
                                jointIndex=idx,
                                controlMode=p.VELOCITY_CONTROL,
                                targetVelocity=target,
                                force=force,
                                physicsClientId=physics_client,
                            )
                        elif motor_type == "torque":
                            p.setJointMotorControl2(
                                bodyUniqueId=body_id,
                                jointIndex=idx,
                                controlMode=p.TORQUE_CONTROL,
                                force=target,
                                physicsClientId=physics_client,
                            )
                    else:
                        p.setJointMotorControl2(
                            bodyUniqueId=body_id,
                            jointIndex=idx,
                            controlMode=p.VELOCITY_CONTROL,
                            force=0,
                            physicsClientId=physics_client,
                        )

        # Apply exact RGBA colors (including alpha transparency) from the room to PyBullet visual shapes
        if is_real:
            for name, (geom, rgba) in self.room.items():
                u_geom = cast(URDFShape, geom)
                label = getattr(u_geom, "urdf_label", None)
                if label and label in label_to_link_idx:
                    link_idx = label_to_link_idx[label]
                    p.changeVisualShape(body_id, link_idx, rgbaColor=rgba, physicsClientId=physics_client)

        urdf_dir = os.path.dirname(urdf_path) if urdf_path else proj_dir
        link_to_obj = self._parse_urdf_meshes(urdf_dir)
        for geom, rgba in self.room.values():
            u_geom = cast(URDFShape, geom)
            label = getattr(u_geom, "urdf_label", None)
            if label:
                obj_filename = link_to_obj.get(label, getattr(u_geom, "urdf_obj_filename", f"{label}.obj"))
                temp_obj_path = os.path.join(proj_dir, obj_filename)
                if os.path.exists(temp_obj_path):
                    rgba_255 = [int(round(c * 255.0)) for c in rgba]
                    rr.log(
                        f"world/{label}",
                        rr.Asset3D(path=temp_obj_path, albedo_factor=rgba_255),
                        static=True,
                    )

        return label_to_link_idx

    def reset_camera(self, physics_client: int, view_from: str = "iso") -> None:
        """Reset the PyBullet visualizer camera based on a view string."""
        bb = self.room.compound.bounding_box()
        center_m = [
            bb.center().X * 0.001,
            bb.center().Y * 0.001,
            bb.center().Z * 0.001,
        ]
        max_dim = max(
            (bb.max.X - bb.min.X) * 0.001,
            (bb.max.Y - bb.min.Y) * 0.001,
            (bb.max.Z - bb.min.Z) * 0.001,
        )
        camera_distance = max(max_dim * 2.0, 0.3)

        mapping = {
            "iso": (45.0, -30.0),
            "top": (0.0, -89.0),
            "bottom": (0.0, 89.0),
            "front": (0.0, 0.0),
            "rear": (180.0, 0.0),
            "left": (270.0, 0.0),
            "right": (90.0, 0.0),
        }

        view_from_lower = view_from.lower()
        yaw, pitch = mapping.get(view_from_lower, (45.0, -30.0))
        parts = view_from_lower.replace(",", " ").split()
        if len(parts) > 1:
            yaws = []
            pitches = []
            for part in parts:
                if part in mapping:
                    yaws.append(mapping[part][0])
                    pitches.append(mapping[part][1])
            if yaws and pitches:
                yaw = sum(yaws) / len(yaws)
                pitch = sum(pitches) / len(pitches)

        p.resetDebugVisualizerCamera(
            cameraDistance=camera_distance,
            cameraYaw=yaw,
            cameraPitch=pitch,
            cameraTargetPosition=center_m,
            physicsClientId=physics_client,
        )

    def _init_rerun(self) -> None:
        """Initialize Rerun connection, connecting to an existing viewer or spawning a new one."""
        rr.init(self.proj_name or "pybullet_simulation")
        if not self.spawn_viewer:
            return

        def is_port_in_use(port: int) -> bool:
            try:
                with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                    s.settimeout(0.1)
                    return s.connect_ex(("127.0.0.1", port)) == 0
            except Exception:
                return False

        target_port = self.rerun_port if self.rerun_port is not None else 9876
        if is_port_in_use(target_port):
            rr.connect_grpc(f"rerun+http://127.0.0.1:{target_port}/proxy")
        else:
            rr.spawn(port=target_port)

    def run(self) -> None:
        """Execute the PyBullet simulation run loop."""
        temp_dir = tempfile.mkdtemp()
        log_queue = None
        log_thread = None
        try:
            # Create the simulation project assets
            proj_dir = os.path.join(temp_dir, self.proj_name)
            os.makedirs(proj_dir, exist_ok=True)
            build_proj_dir = os.path.join(self.build_dir, "obj", self.proj_name)
            if not os.path.exists(build_proj_dir):
                build_proj_dir = os.path.join(self.build_dir, self.proj_name)
            self.room.translate_joints()
            self._copy_project_assets(build_proj_dir, proj_dir)

            # Determine URDF filename using Lister
            from list import Lister

            lister = Lister(self.manager, self.logger)
            urdf_rel_path = lister.get_urdf_output(self.sim_target)
            real_urdf_path = os.path.join(self.build_dir, urdf_rel_path)
            temp_urdf_filename = os.path.basename(urdf_rel_path)
            urdf_path = os.path.join(temp_dir, temp_urdf_filename)

            if os.path.exists(real_urdf_path):
                shutil.copy(real_urdf_path, urdf_path)
            else:
                raise FileNotFoundError(f"Required URDF file not found for simulation: {real_urdf_path}")

            physics_client = p.connect(p.DIRECT)
            self._init_rerun()

            if self.save_rrd:
                rr.save(self.save_rrd)

            try:
                is_real = _is_real_physics_client(physics_client)
                if is_real:
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0, physicsClientId=physics_client)

                p.setGravity(*self.room.gravity, physicsClientId=physics_client)
                if is_real:
                    p.setPhysicsEngineParameter(numSubSteps=2, physicsClientId=physics_client)
                body_id = p.loadURDF(urdf_path, useFixedBase=True, physicsClientId=physics_client)
                if body_id < 0:
                    raise RuntimeError("PyBullet failed to load the URDF.")

                label_to_link_idx = self._init_simulation_objects(physics_client, body_id, proj_dir, urdf_path)
                state_tracker = BulletStateTracker(body_id, physics_client, label_to_link_idx)

                # Parse boundaries metadata
                boundaries_metadata = {}
                for geom, _ in self.room.values():
                    u_geom = cast(URDFShape, geom)
                    label = getattr(u_geom, "urdf_label", None)
                    if label:
                        geom_boundaries = getattr(u_geom, "urdf_boundaries", None)
                        if geom_boundaries:
                            boundaries_metadata[label] = geom_boundaries

                # Setup Hooks
                setup_hook = self.provider_hooks.get(Simulate.SETUP, None)
                if setup_hook:
                    import inspect

                    sig = inspect.signature(setup_hook)
                    if "state_tracker" in sig.parameters:
                        setup_hook(
                            body_id, physics_client, self.sim_target, boundaries_metadata, state_tracker=state_tracker
                        )
                    else:
                        setup_hook(body_id, physics_client, self.sim_target, boundaries_metadata)

                if is_real:
                    p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1, physicsClientId=physics_client)

                log_queue = None
                log_thread = None
                is_logging_enabled = self.spawn_viewer or (self.save_rrd is not None)

                if is_logging_enabled:
                    q = queue.Queue(maxsize=128)
                    log_queue = q

                    def logging_worker():
                        while True:
                            item = q.get()
                            if item is None:
                                q.task_done()
                                break
                            (
                                transforms,
                                particle_positions,
                                particle_colors,
                                particle_radii,
                                boundary_voxels,
                                water_meshes,
                                step_idx,
                            ) = item
                            self.room._log_rerun(
                                transforms,
                                particle_positions,
                                particle_colors,
                                particle_radii=particle_radii,
                                boundary_voxels=boundary_voxels,
                                water_meshes=water_meshes,
                                step_idx=step_idx,
                            )
                            q.task_done()

                    t = threading.Thread(target=logging_worker, daemon=True)
                    log_thread = t
                    t.start()

                for step_idx in range(self.steps):
                    step_hook = self.provider_hooks.get(Simulate.STEP, None)
                    terminated = False
                    if step_hook:
                        res = step_hook(body_id, physics_client, step_idx, self.sim_target)
                        if isinstance(res, str):
                            self.logger.print(f"Simulation terminated: {res}", symbol="🛑")
                            terminated = True

                    if is_logging_enabled:
                        state_tracker.update_state()

                    if not terminated:
                        p.stepSimulation(physicsClientId=physics_client)

                    if is_logging_enabled and log_queue is not None:
                        try:
                            log_queue.put_nowait(
                                (
                                    state_tracker.transforms,
                                    state_tracker.particle_positions,
                                    state_tracker.particle_colors,
                                    state_tracker.particle_radii,
                                    state_tracker.boundary_voxels,
                                    state_tracker.water_meshes,
                                    step_idx,
                                )
                            )
                        except queue.Full:
                            pass

                    # Check for staging frame window checkpoint
                    if self.stage_window_size and self.save_rrd and ((step_idx + 1) % self.stage_window_size == 0):
                        if log_queue is not None:
                            log_queue.join()
                        base_dir = os.path.dirname(self.save_rrd) or "."
                        base_name = os.path.splitext(os.path.basename(self.save_rrd))[0]
                        import re

                        prefix = re.sub(r"_\d+$", "", base_name)
                        staged_name = f"{prefix}_{step_idx + 1}.rrd"
                        staged_path = os.path.join(base_dir, staged_name)
                        if os.path.exists(self.save_rrd) and os.path.abspath(self.save_rrd) != os.path.abspath(
                            staged_path
                        ):
                            shutil.copyfile(self.save_rrd, staged_path)
                            self.logger.print(
                                f"Staged checkpoint saved: {staged_path} ({step_idx + 1} frames)", symbol="💾"
                            )

                    if terminated:
                        break

                if is_logging_enabled and log_queue is not None and log_thread is not None:
                    log_queue.put(None)
                    log_thread.join()

            except KeyboardInterrupt:
                self.logger.print("Simulation stopped.", symbol="💥")
            finally:
                try:
                    p.disconnect(physicsClientId=physics_client)
                except Exception:
                    pass
                try:
                    rr.disconnect()
                except Exception:
                    pass

        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)
