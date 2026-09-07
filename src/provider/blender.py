"""High-fidelity Blender rendering backend for CAD components and fluid simulations."""

import glob
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional, Sequence, Union

import jinja2
import numpy as np

from .types import ColorType
from .utils import get_rgba_color


@dataclass
class CameraViewport:
    """Camera position, orientation, and framing parameters for rendering."""

    location: tuple[float, float, float]
    target: tuple[float, float, float]
    focal_length_mm: float = 50.0
    lens_type: str = "PERSP"

    @classmethod
    def from_view_string(
        cls,
        view_from: str = "iso",
        center: tuple[float, float, float] = (0.0, 0.0, 0.0),
        bounding_radius: float = 0.15,
    ) -> "CameraViewport":
        """Compute camera location and target from a view string and scene bounding sphere."""
        cx, cy, cz = center
        dist = max(bounding_radius * 2.5, 0.25)

        # Parse coordinate tuple format like "0.2,-0.3,0.25" or "pos=(...),target=(...)"
        coord_match = re.match(
            r"^\s*\(?\s*([-\d.]+)\s*,\s*([-\d.]+)\s*,\s*([-\d.]+)\s*\)?\s*$",
            view_from.strip(),
        )
        if coord_match:
            gx, gy, gz = (
                float(coord_match.group(1)),
                float(coord_match.group(2)),
                float(coord_match.group(3)),
            )
            return cls(location=(gx, gy, gz), target=center)

        mapping = {
            "iso": (45.0, 30.0),
            "isometric": (45.0, 30.0),
            "front": (0.0, 5.0),
            "rear": (180.0, 5.0),
            "back": (180.0, 5.0),
            "left": (-90.0, 5.0),
            "right": (90.0, 5.0),
            "top": (0.0, 89.0),
            "bottom": (0.0, -89.0),
            "front-left": (-45.0, 20.0),
            "front-right": (45.0, 20.0),
            "rear-left": (-135.0, 20.0),
            "rear-right": (135.0, 20.0),
        }

        view_lower = view_from.lower().replace("_", "-")
        yaw_deg, pitch_deg = mapping.get(view_lower, (45.0, 30.0))

        # Check multi-word compound views like "front left" or "front,left"
        parts = view_lower.replace(",", " ").split()
        if len(parts) > 1:
            yaws = []
            pitches = []
            for part in parts:
                if part in mapping:
                    yaws.append(mapping[part][0])
                    pitches.append(mapping[part][1])
            if yaws and pitches:
                yaw_deg = sum(yaws) / len(yaws)
                pitch_deg = sum(pitches) / len(pitches)

        yaw_rad = math.radians(yaw_deg)
        pitch_rad = math.radians(pitch_deg)

        # Spherical coordinates relative to target center
        # Pitch: 0 = horizontal (facing from South / -Y), +90 = top (+Z)
        r_xy = dist * math.cos(pitch_rad)
        cam_x = cx + r_xy * math.sin(yaw_rad)
        cam_y = cy - r_xy * math.cos(yaw_rad)
        cam_z = cz + dist * math.sin(pitch_rad)

        return cls(location=(cam_x, cam_y, cam_z), target=center)


@dataclass
class RenderConfig:
    """Configuration settings for Blender rendering."""

    resolution: tuple[int, int] = (2560, 1440)
    fps: int = 60
    samples: int = 32
    engine: str = "BLENDER_EEVEE"
    view_from: str = "iso"
    shadow_catcher: bool = True
    background_color: tuple[float, float, float, float] = (0.06, 0.07, 0.09, 1.0)
    output_mp4: Optional[str] = None
    crf: int = 18
    blender_executable: str = field(default_factory=lambda: BlenderRenderer.find_blender_binary())


class BlenderRenderer:
    """Orchestrates headless Blender rendering for CAD geometry and dynamic fluid simulations."""

    @staticmethod
    def find_blender_binary() -> str:
        """Find the path to the Blender executable across Linux, macOS, CI runners, and Windows."""
        # 1. Environment variable overrides (common in CI/Docker/custom installs)
        for env_var in ("BLENDER_BIN", "BLENDER_EXECUTABLE", "BLENDER_PATH", "BLENDER"):
            if env_var in os.environ and os.environ[env_var].strip():
                candidate = os.environ[env_var].strip()
                if os.path.exists(candidate) and os.path.isfile(candidate):
                    return candidate
                # Check if it's an application bundle or directory containing blender
                if os.path.isdir(candidate):
                    mac_bundle_exec = os.path.join(candidate, "Contents", "MacOS", "Blender")
                    if os.path.exists(mac_bundle_exec):
                        return mac_bundle_exec
                    for exe_name in ("blender", "blender.exe"):
                        nested = os.path.join(candidate, exe_name)
                        if os.path.exists(nested):
                            return nested
                which_cand = shutil.which(candidate)
                if which_cand:
                    return which_cand

        # 2. Standard PATH lookup
        for name in ("blender", "blender.exe", "blender-launcher", "blender-launcher.exe"):
            which_blender = shutil.which(name)
            if which_blender:
                return which_blender

        # 3. Python environment / Conda / Virtualenv prefix
        conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        if conda_prefix:
            conda_candidates = [
                os.path.join(conda_prefix, "bin", "blender"),
                os.path.join(conda_prefix, "blender.exe"),
                os.path.join(conda_prefix, "Scripts", "blender.exe"),
                os.path.join(conda_prefix, "Library", "bin", "blender.exe"),
            ]
            for p in conda_candidates:
                if os.path.exists(p) and os.path.isfile(p):
                    return p

        # 4. CI / Container / Runner Tool Cache (GitHub Actions, GitLab CI, etc.)
        ci_patterns: list[str] = []
        runner_tool_cache = os.environ.get("RUNNER_TOOL_CACHE")
        if runner_tool_cache and os.path.isdir(runner_tool_cache):
            ci_patterns.extend(
                [
                    os.path.join(runner_tool_cache, "blender", "**", "blender"),
                    os.path.join(runner_tool_cache, "blender", "**", "blender.exe"),
                    os.path.join(runner_tool_cache, "Blender", "**", "blender"),
                    os.path.join(runner_tool_cache, "Blender", "**", "blender.exe"),
                ]
            )

        # 5. Linux standard & container paths
        linux_paths = [
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/snap/bin/blender",
            "/var/lib/snapd/snap/bin/blender",
            "/var/lib/flatpak/exports/bin/org.blender.Blender",
            os.path.expanduser("~/.local/bin/blender"),
            os.path.expanduser("~/bin/blender"),
            "/opt/blender/blender",
        ]
        linux_patterns = [
            "/opt/blender*/blender",
            "/opt/blender*/**/blender",
            "/usr/local/blender*/blender",
            "/usr/local/blender*/**/blender",
            os.path.expanduser("~/.local/share/blender*/blender"),
            os.path.expanduser("~/blender*/blender"),
        ]

        # 6. macOS standard application and Homebrew paths
        mac_paths = [
            "/opt/homebrew/bin/blender",
            "/usr/local/bin/blender",
            "/Applications/Blender.app/Contents/MacOS/Blender",
            "/Applications/Blender.app/Contents/MacOS/blender",
            os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/Blender"),
            os.path.expanduser("~/Applications/Blender.app/Contents/MacOS/blender"),
        ]
        mac_patterns = [
            "/Applications/Blender*.app/Contents/MacOS/Blender*",
            os.path.expanduser("~/Applications/Blender*.app/Contents/MacOS/Blender*"),
        ]

        # 7. Windows standard installation paths
        win_paths = [
            r"C:\Program Files\Blender Foundation\blender.exe",
            r"C:\tools\blender\blender.exe",
            r"C:\ProgramData\chocolatey\bin\blender.exe",
        ]
        prog_files = os.environ.get("ProgramFiles", r"C:\Program Files")
        prog_files_x86 = os.environ.get("ProgramFiles(x86)", r"C:\Program Files (x86)")
        local_app_data = os.environ.get("LOCALAPPDATA", "")
        win_patterns = [
            os.path.join(prog_files, "Blender Foundation", "Blender*", "blender.exe"),
            os.path.join(prog_files_x86, "Blender Foundation", "Blender*", "blender.exe"),
        ]
        if local_app_data:
            win_patterns.append(
                os.path.join(local_app_data, "Programs", "Blender Foundation", "Blender*", "blender.exe")
            )

        # Check explicit static paths first
        for p in (*linux_paths, *mac_paths, *win_paths):
            if os.path.exists(p) and os.path.isfile(p):
                return p

        # Check glob patterns (sorted in reverse so newer versions take precedence)
        for pattern in (*ci_patterns, *linux_patterns, *mac_patterns, *win_patterns):
            matches = glob.glob(pattern, recursive=True)
            if matches:
                valid_files = [m for m in matches if os.path.isfile(m)]
                if valid_files:
                    valid_files.sort(reverse=True)
                    return valid_files[0]

        return "blender"

    @staticmethod
    def find_ffmpeg_binary() -> str:
        """Find the path to the FFmpeg executable across Linux, macOS, CI runners, and Windows."""
        # 1. Environment variable overrides
        for env_var in ("FFMPEG_BIN", "FFMPEG_EXECUTABLE", "FFMPEG_PATH", "FFMPEG"):
            if env_var in os.environ and os.environ[env_var].strip():
                candidate = os.environ[env_var].strip()
                if os.path.exists(candidate) and os.path.isfile(candidate):
                    return candidate
                which_cand = shutil.which(candidate)
                if which_cand:
                    return which_cand

        # 2. Standard PATH lookup
        for name in ("ffmpeg", "ffmpeg.exe"):
            which_ffmpeg = shutil.which(name)
            if which_ffmpeg:
                return which_ffmpeg

        # 3. Python environment / Conda / Virtualenv prefix
        conda_prefix = os.environ.get("CONDA_PREFIX") or sys.prefix
        if conda_prefix:
            conda_candidates = [
                os.path.join(conda_prefix, "bin", "ffmpeg"),
                os.path.join(conda_prefix, "ffmpeg.exe"),
                os.path.join(conda_prefix, "Scripts", "ffmpeg.exe"),
                os.path.join(conda_prefix, "Library", "bin", "ffmpeg.exe"),
            ]
            for p in conda_candidates:
                if os.path.exists(p) and os.path.isfile(p):
                    return p

        # 4. Standard platform paths
        platform_paths = [
            "/usr/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/snap/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",
            os.path.expanduser("~/.local/bin/ffmpeg"),
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
            r"C:\tools\ffmpeg\bin\ffmpeg.exe",
        ]
        for p in platform_paths:
            if os.path.exists(p) and os.path.isfile(p):
                return p

        return "ffmpeg"

    @classmethod
    def is_available(cls) -> bool:
        """Check if Blender is installed and runnable on this system."""
        try:
            bin_path = cls.find_blender_binary()
            res = subprocess.run(
                [bin_path, "-b", "--version"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=10,
            )
            return res.returncode == 0
        except Exception:
            return False

    @classmethod
    def render_still(
        cls,
        room: Any,
        output_image: str,
        config: Optional[RenderConfig] = None,
    ) -> str:
        """Render a single high-resolution beauty frame of the room geometry."""
        config = config or RenderConfig()
        output_path = os.path.abspath(output_image)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_data_path = os.path.join(tmp_dir, "scene_data.json")
            script_path = os.path.join(tmp_dir, "render_script.py")

            # 1. Export static room meshes to OBJ files
            cls._export_room_to_dir(room, tmp_dir, scene_data_path, config)

            # 2. Write Blender execution script
            cls._write_blender_script(
                script_path=script_path,
                scene_data_path=scene_data_path,
                output_path=output_path,
                is_animation=False,
                config=config,
            )

            # 3. Run Blender headless
            cmd = [config.blender_executable, "-b", "-P", script_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Blender render failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}")

        return output_path

    @classmethod
    def render_turntable_to_mp4(
        cls,
        room: Any,
        output_mp4: str,
        duration_sec: float = 4.0,
        config: Optional[RenderConfig] = None,
    ) -> str:
        """Render a 360-degree turntable camera orbit animation of static room geometry to MP4."""
        config = config or RenderConfig()
        output_path = os.path.abspath(output_mp4)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        total_frames = int(max(1, round(duration_sec * config.fps)))

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_data_path = os.path.join(tmp_dir, "scene_data.json")
            script_path = os.path.join(tmp_dir, "render_script.py")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # 1. Export static room meshes
            cls._export_room_to_dir(room, tmp_dir, scene_data_path, config)
            with open(scene_data_path, "r", encoding="utf-8") as f:
                manifest = json.load(f)
            manifest["turntable"] = True
            with open(scene_data_path, "w", encoding="utf-8") as f:
                json.dump(manifest, f, indent=2)

            # 2. Write Blender script
            cls._write_blender_script(
                script_path=script_path,
                scene_data_path=scene_data_path,
                output_path=os.path.join(frames_dir, "frame_####.png"),
                is_animation=True,
                config=config,
                total_frames=total_frames,
            )

            # 3. Run Blender headless
            cmd = [config.blender_executable, "-b", "-P", script_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Blender render failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}")

            # 4. Encode to MP4 via ffmpeg
            cls._encode_frames_to_mp4(
                frames_dir=frames_dir,
                output_mp4=output_path,
                fps=config.fps,
                crf=config.crf,
            )

        return output_path

    @classmethod
    def render_simulation_to_mp4(
        cls,
        room: Any,
        output_mp4: str,
        sim_steps: int = 1000,
        config: Optional[RenderConfig] = None,
        fluid_bodies_per_frame: Optional[list[list[Any]]] = None,
        water_meshes_per_frame: Optional[list[dict[str, tuple[np.ndarray, np.ndarray]]]] = None,
        rigid_transforms_per_frame: Optional[list[dict[str, tuple[list[float], list[float]]]]] = None,
    ) -> str:
        """Render a full simulation animation sequence to an H.264 MP4 video."""
        config = config or RenderConfig()
        output_path = os.path.abspath(output_mp4)
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with tempfile.TemporaryDirectory() as tmp_dir:
            scene_data_path = os.path.join(tmp_dir, "scene_data.json")
            script_path = os.path.join(tmp_dir, "render_script.py")
            frames_dir = os.path.join(tmp_dir, "frames")
            os.makedirs(frames_dir, exist_ok=True)

            # 1. Export static room meshes and animation frames
            cls._export_simulation_sequence_to_dir(
                room=room,
                target_dir=tmp_dir,
                scene_data_path=scene_data_path,
                config=config,
                sim_steps=sim_steps,
                fluid_bodies_per_frame=fluid_bodies_per_frame,
                water_meshes_per_frame=water_meshes_per_frame,
                rigid_transforms_per_frame=rigid_transforms_per_frame,
            )

            # 2. Write Blender execution script
            cls._write_blender_script(
                script_path=script_path,
                scene_data_path=scene_data_path,
                output_path=os.path.join(frames_dir, "frame_####.png"),
                is_animation=True,
                config=config,
                total_frames=sim_steps,
            )

            # 3. Run Blender headless to render frame sequence
            cmd = [config.blender_executable, "-b", "-P", script_path]
            res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            if res.returncode != 0:
                raise RuntimeError(f"Blender render failed (exit code {res.returncode}):\n{res.stderr}\n{res.stdout}")

            # 4. Compile frame sequence into H.264 MP4 via ffmpeg
            cls._encode_frames_to_mp4(
                frames_dir=frames_dir,
                output_mp4=output_path,
                fps=config.fps,
                crf=config.crf,
            )

        return output_path

    @classmethod
    def _export_room_to_dir(
        cls,
        room: Any,
        target_dir: str,
        scene_data_path: str,
        config: RenderConfig,
    ) -> None:
        """Export room geometry items into OBJ files and create the scene metadata manifest."""
        import trimesh
        from build123d import Compound, Shape

        items_meta = []
        all_verts = []

        for name, (geom, rgba) in room.items():
            # Handle build123d shapes and trimesh geometry
            shape = getattr(geom, "part", geom)
            if isinstance(shape, (Shape, Compound)) or hasattr(shape, "solids"):
                obj_name = f"{name}.obj"
                obj_path = os.path.join(target_dir, obj_name)
                stl_path = os.path.join(target_dir, f"{name}.stl")
                from build123d import export_stl

                export_stl(shape, stl_path)
                tm = trimesh.load(stl_path)
                tm.apply_scale(0.001)  # Convert mm to meters
                tm.export(obj_path)
                if hasattr(tm, "vertices") and len(tm.vertices) > 0:
                    all_verts.append(np.asarray(tm.vertices))

                is_water = "water" in name.lower() or "fluid" in name.lower()
                is_trans = rgba[3] < 0.90 or "clear" in name.lower() or "acrylic" in name.lower()
                items_meta.append(
                    {
                        "name": name,
                        "file": obj_path,
                        "rgba": list(rgba),
                        "is_water": is_water,
                        "is_transparent": is_trans,
                        "scale": 1.0,  # Scaled to meters in OBJ
                    }
                )
            elif hasattr(geom, "vertices") and hasattr(geom, "faces"):
                obj_name = f"{name}.obj"
                obj_path = os.path.join(target_dir, obj_name)
                geom.export(obj_path)
                if len(geom.vertices) > 0:
                    all_verts.append(np.asarray(geom.vertices))
                is_water = "water" in name.lower() or "fluid" in name.lower()
                is_trans = rgba[3] < 0.90 or "clear" in name.lower() or "acrylic" in name.lower()
                items_meta.append(
                    {
                        "name": name,
                        "file": obj_path,
                        "rgba": list(rgba),
                        "is_water": is_water,
                        "is_transparent": is_trans,
                        "scale": 1.0,
                    }
                )

        # Compute scene bounding box for camera placement
        if all_verts:
            cat_verts = np.vstack(all_verts)
            c_min = np.min(cat_verts, axis=0)
            c_max = np.max(cat_verts, axis=0)
            center = tuple(((c_min + c_max) / 2.0).tolist())
            b_rad = float(np.linalg.norm(c_max - c_min) / 2.0)
        else:
            center = (0.0, 0.0, 0.06)
            b_rad = 0.12

        viewport = CameraViewport.from_view_string(
            view_from=config.view_from,
            center=center,
            bounding_radius=b_rad,
        )

        manifest = {
            "items": items_meta,
            "center": list(center),
            "camera": {
                "location": list(viewport.location),
                "target": list(viewport.target),
                "focal_length": viewport.focal_length_mm,
            },
            "frames": [],
        }

        with open(scene_data_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def _export_simulation_sequence_to_dir(
        cls,
        room: Any,
        target_dir: str,
        scene_data_path: str,
        config: RenderConfig,
        sim_steps: int = 1000,
        fluid_bodies_per_frame: Optional[list[list[Any]]] = None,
        water_meshes_per_frame: Optional[list[dict[str, tuple[np.ndarray, np.ndarray]]]] = None,
        rigid_transforms_per_frame: Optional[list[dict[str, tuple[list[float], list[float]]]]] = None,
    ) -> None:
        """Export full simulation frame meshes and joint transforms to directory."""
        import trimesh

        cls._export_room_to_dir(room, target_dir, scene_data_path, config)
        with open(scene_data_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        frames_meta = []
        for step_idx in range(sim_steps):
            frame_items = []
            # Export fluid bodies if present
            if fluid_bodies_per_frame and step_idx < len(fluid_bodies_per_frame):
                bodies = fluid_bodies_per_frame[step_idx]
                for b_idx, body in enumerate(bodies):
                    verts, faces = body.to_mesh()
                    if len(verts) > 0 and len(faces) > 0:
                        b_name = f"water_{body.display_name}_{step_idx:05d}_{b_idx}.obj"
                        b_path = os.path.join(target_dir, b_name)
                        tm = trimesh.Trimesh(vertices=verts, faces=faces)
                        tm.export(b_path)
                        frame_items.append(
                            {
                                "name": body.display_name,
                                "file": b_path,
                                "rgba": [0.2, 0.65, 0.95, 0.40],
                                "is_water": True,
                                "is_transparent": True,
                                "scale": 1.0,
                            }
                        )
            # Export water meshes if present
            elif water_meshes_per_frame and step_idx < len(water_meshes_per_frame):
                meshes_dict = water_meshes_per_frame[step_idx]
                for m_idx, (m_name, (verts, faces)) in enumerate(meshes_dict.items()):
                    if len(verts) > 0 and len(faces) > 0:
                        b_name = f"water_{m_name}_{step_idx:05d}_{m_idx}.obj"
                        b_path = os.path.join(target_dir, b_name)
                        tm = trimesh.Trimesh(vertices=verts, faces=faces)
                        tm.export(b_path)
                        frame_items.append(
                            {
                                "name": m_name,
                                "file": b_path,
                                "rgba": [0.2, 0.65, 0.95, 0.40],
                                "is_water": True,
                                "is_transparent": True,
                                "scale": 1.0,
                            }
                        )

            rigid_transforms = (
                rigid_transforms_per_frame[step_idx]
                if rigid_transforms_per_frame and step_idx < len(rigid_transforms_per_frame)
                else {}
            )
            frames_meta.append(
                {
                    "step": step_idx,
                    "water_items": frame_items,
                    "transforms": rigid_transforms,
                }
            )

        manifest["frames"] = frames_meta
        with open(scene_data_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

    @classmethod
    def _write_blender_script(
        cls,
        script_path: str,
        scene_data_path: str,
        output_path: str,
        is_animation: bool,
        config: RenderConfig,
        total_frames: int = 1,
    ) -> None:
        """Generate the Python script executed inside headless Blender via Jinja2 template."""
        templates_dir = os.path.join(os.path.dirname(__file__), "templates")
        env = jinja2.Environment(
            loader=jinja2.FileSystemLoader(templates_dir),
            trim_blocks=True,
            lstrip_blocks=True,
            autoescape=False,
        )
        template = env.get_template("render_blender.py.j2")
        rendered_script = template.render(
            scene_data_path=scene_data_path,
            output_path=output_path,
            is_animation=is_animation,
            total_frames=max(1, total_frames),
            engine=config.engine.upper(),
            resolution_x=config.resolution[0],
            resolution_y=config.resolution[1],
            fps=config.fps,
            samples=config.samples,
            background_color=config.background_color,
        )
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(rendered_script.strip())

    @classmethod
    def _encode_frames_to_mp4(
        cls,
        frames_dir: str,
        output_mp4: str,
        fps: int = 60,
        crf: int = 18,
    ) -> None:
        """Compile a rendered frame sequence into a broadcast-ready H.264 MP4."""
        ffmpeg_bin = cls.find_ffmpeg_binary()
        if not shutil.which(ffmpeg_bin) and not os.path.exists(ffmpeg_bin):
            raise FileNotFoundError(f"ffmpeg binary not found at '{ffmpeg_bin}'. Required for MP4 compilation.")

        # Check if frames were generated
        png_files = [f for f in os.listdir(frames_dir) if f.endswith(".png")]
        if not png_files:
            raise RuntimeError(f"No rendered PNG frames found in '{frames_dir}' to encode into MP4.")

        input_pattern = os.path.join(frames_dir, "frame_%05d.png")
        cmd = [
            ffmpeg_bin,
            "-y",
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",
            "-preset",
            "slow",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            output_mp4,
        ]
        res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if res.returncode != 0:
            raise RuntimeError(f"FFmpeg MP4 encoding failed:\n{res.stderr}")
