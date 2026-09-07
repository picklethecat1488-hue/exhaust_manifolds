"""Unit tests for Blender rendering backend and camera viewports."""

import os
from unittest.mock import MagicMock, patch
import numpy as np
import pytest

from provider.blender import BlenderRenderer, CameraViewport, RenderConfig
from provider.room import Room


class TestBlenderRenderer:
    """Test suite for BlenderRenderer, CameraViewport, and RenderConfig."""

    def test_camera_viewport_from_named_views(self):
        """Verify CameraViewport calculation for standard named views."""
        center = (0.0, 0.0, 0.05)
        b_rad = 0.10

        vp_iso = CameraViewport.from_view_string("iso", center=center, bounding_radius=b_rad)
        assert vp_iso.target == center
        assert vp_iso.location[2] > center[2]

        vp_front = CameraViewport.from_view_string("front", center=center, bounding_radius=b_rad)
        assert vp_front.target == center
        # Front faces from South (-Y)
        assert vp_front.location[1] < center[1]

        vp_top = CameraViewport.from_view_string("top", center=center, bounding_radius=b_rad)
        assert vp_top.target == center
        assert vp_top.location[2] > center[2]

        vp_left = CameraViewport.from_view_string("left", center=center, bounding_radius=b_rad)
        assert vp_left.target == center
        assert vp_left.location[0] < center[0]

    def test_camera_viewport_from_coordinate_string(self):
        """Verify CameraViewport parses raw coordinate tuples."""
        center = (0.0, 0.0, 0.0)
        vp_coord = CameraViewport.from_view_string("(0.25, -0.35, 0.40)", center=center)
        assert vp_coord.target == center
        assert vp_coord.location == (0.25, -0.35, 0.40)

        vp_coord_unwrapped = CameraViewport.from_view_string("0.1, -0.2, 0.3", center=center)
        assert vp_coord_unwrapped.location == (0.1, -0.2, 0.3)

    def test_find_blender_binary(self):
        """Verify find_blender_binary finds an executable or respects BLENDER_BIN env var."""
        with patch.dict(os.environ, {"BLENDER_BIN": "/custom/path/to/blender"}, clear=False):
            with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=True):
                assert BlenderRenderer.find_blender_binary() == "/custom/path/to/blender"

        with patch.dict(os.environ, {"BLENDER_PATH": "/opt/custom/blender"}, clear=False):
            with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=True):
                assert BlenderRenderer.find_blender_binary() == "/opt/custom/blender"

    def test_find_blender_binary_ci_and_linux_paths(self):
        """Verify find_blender_binary resolves standard Linux/CI paths and glob patterns."""
        with patch.dict(os.environ, {}, clear=True):
            with patch("shutil.which", return_value=None):
                with patch("glob.glob") as mock_glob:
                    mock_glob.side_effect = lambda pat, **kwargs: (
                        ["/opt/blender-5.1.0/blender"] if "/opt/blender*" in pat else []
                    )
                    with patch("os.path.exists", return_value=False), patch("os.path.isfile", return_value=True):
                        assert BlenderRenderer.find_blender_binary() == "/opt/blender-5.1.0/blender"

    def test_find_ffmpeg_binary(self):
        """Verify find_ffmpeg_binary discovers ffmpeg across env vars, conda, and standard paths."""
        with patch.dict(os.environ, {"FFMPEG_BIN": "/custom/ffmpeg"}, clear=False):
            with patch("os.path.exists", return_value=True), patch("os.path.isfile", return_value=True):
                assert BlenderRenderer.find_ffmpeg_binary() == "/custom/ffmpeg"

        with patch.dict(os.environ, {}, clear=True):
            with patch("shutil.which", return_value="/usr/bin/ffmpeg"):
                assert BlenderRenderer.find_ffmpeg_binary() == "/usr/bin/ffmpeg"

    @patch("subprocess.run")
    def test_is_available(self, mock_run):
        """Verify is_available returns True on successful version check."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Blender 5.1.2\n")
        assert BlenderRenderer.is_available() is True

        mock_run.return_value = MagicMock(returncode=1, stderr="error")
        assert BlenderRenderer.is_available() is False

    @patch("subprocess.run")
    def test_render_still_execution(self, mock_run):
        """Verify render_still exports scene data and executes Blender script."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Render success", stderr="")
        room = Room()
        from build123d import Box

        room.add("cube", Box(10, 10, 10), color=(1.0, 0.0, 0.0, 1.0))

        cfg = RenderConfig(view_from="iso", samples=16)
        out_path = BlenderRenderer.render_still(room, "build/test_still.png", config=cfg)
        assert os.path.basename(out_path) == "test_still.png"
        assert mock_run.called

    @patch("provider.blender.BlenderRenderer._encode_frames_to_mp4")
    @patch("subprocess.run")
    def test_render_turntable_execution(self, mock_run, mock_encode):
        """Verify render_turntable_to_mp4 exports room, sets turntable flag, runs blender, and encodes MP4."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Turntable success", stderr="")
        room = Room()
        from build123d import Box

        room.add("cube", Box(10, 10, 10), color=(0.2, 0.6, 0.9, 0.8))

        cfg = RenderConfig(fps=30, samples=16, view_from="iso")
        out_path = BlenderRenderer.render_turntable_to_mp4(
            room, "build/test_turntable.mp4", duration_sec=1.0, config=cfg
        )
        assert os.path.basename(out_path) == "test_turntable.mp4"
        assert mock_run.called
        assert mock_encode.called

    @patch("provider.blender.BlenderRenderer._encode_frames_to_mp4")
    @patch("subprocess.run")
    def test_render_simulation_to_mp4_execution(self, mock_run, mock_encode):
        """Verify render_simulation_to_mp4 exports frames and rigid transforms."""
        mock_run.return_value = MagicMock(returncode=0, stdout="Sim render success", stderr="")
        room = Room()
        from build123d import Box

        room.add("casing", Box(20, 20, 20), color=(0.8, 0.8, 0.8, 1.0))

        water_meshes = [
            {"pool": (np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0]]), np.array([[0, 1, 2]]))}
        ]
        transforms = [{"casing": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0])}]

        cfg = RenderConfig(fps=30, samples=16)
        out_path = BlenderRenderer.render_simulation_to_mp4(
            room=room,
            output_mp4="build/test_sim.mp4",
            sim_steps=1,
            config=cfg,
            water_meshes_per_frame=water_meshes,
            rigid_transforms_per_frame=transforms,
        )
        assert os.path.basename(out_path) == "test_sim.mp4"
        assert mock_run.called
        assert mock_encode.called

    def test_write_blender_script_jinja2_template(self, tmp_path):
        """Verify _write_blender_script correctly loads Jinja2 template and produces valid Python code."""
        script_file = tmp_path / "test_render_script.py"
        scene_file = tmp_path / "scene_data.json"
        output_image = tmp_path / "frame.png"

        cfg = RenderConfig(resolution=(2560, 1440), fps=60, samples=32, engine="BLENDER_EEVEE")
        BlenderRenderer._write_blender_script(
            script_path=str(script_file),
            scene_data_path=str(scene_file),
            output_path=str(output_image),
            is_animation=False,
            config=cfg,
            total_frames=1,
        )

        assert script_file.exists()
        code = script_file.read_text(encoding="utf-8")
        assert "import bpy" in code
        assert "BLENDER_EEVEE" in code
        assert "2560" in code
        assert "1440" in code
        # Check Python compilation of generated script
        compile(code, str(script_file), "exec")
