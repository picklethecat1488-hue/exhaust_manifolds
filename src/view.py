"""View manifold geometry using ocp_vscode."""

import argparse
import importlib
import fnmatch
import os
import sys
import shutil
import tempfile
import time
import pybullet as p
from daemon import DaemonClient
from model import AppConfig
from pathlib import Path
from typing import Sequence, Optional, List, Any, cast, Iterable, Union
from build123d import *  # type: ignore
from target_parser import TargetParser
from provider import ProviderManager, Section, TargetList, Room, Simulate, Mode, Provider, URDFShape
from pydantic import validate_call
from shell import Logger
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="ocp_vscode.*")
warnings.filterwarnings("ignore", message=".*collapse value from viewer.*")

from ocp_vscode import set_port, Collapse, Camera, show as ocp_show  # type: ignore
from build import Builder
from list import Lister

SPINNER_TEXT = "Visualizing..."


def show(*args, **kwargs):
    """Bypass ocp_vscode visualization during headless runs."""
    if "--no-gui" in sys.argv:
        return
    return ocp_show(*args, **kwargs)


class ProviderResolver:
    """Utility to resolve a single Provider from a targets provider reference."""

    @staticmethod
    def resolve(resolved_provider: Any, target: str) -> Optional[Provider]:
        """Resolve a single Provider from ProviderRouter or Provider."""
        if isinstance(resolved_provider, Provider):
            return resolved_provider
        if hasattr(resolved_provider, "_mock_return_value"):
            return cast(Any, resolved_provider)

        p_name = TargetParser.get_project_name(target)
        for prov in getattr(resolved_provider, "providers", []):
            if prov.name == p_name:
                return prov
        return None


class Viewer:
    """Builds and displays geometry rooms for visualization."""

    VISUAL_ACTIONS = [Section.VIEW, Section.PART, Section.DIAGRAM]

    def __init__(self, manager: ProviderManager, logger: Logger):
        """Initialize the viewer."""
        self.manager = manager
        self.logger = logger
        self.target_parser = TargetParser(manager.router)

    def get_summary(self, names: Sequence[str]) -> str:
        """Return a truncated summary string of the names being shown."""
        if len(names) > 8:
            return f"{', '.join(names[:8])} ... ({len(names)} items)"
        return ", ".join(names)

    def _get_view_items(self, targets: TargetList) -> List[tuple[Any, str, tuple[float, float, float], float]]:
        """Collect items from a VIEW room."""
        items = []
        results = self.manager.router.run(targets)
        for room_name, room in results:
            for item_name, (geom, rgba) in room.items():
                items.append((geom, f"{room_name}_{item_name}", rgba[:3], rgba[3]))
        return items

    def _get_part_items(self, targets: Any) -> List[tuple[Any, str, tuple[float, float, float], float]]:
        """Collect geometry from a PART build."""
        items = []
        results = self.manager.router.run(targets)

        # Determine subassemblies used from the TargetList to correctly label items
        subs = targets.subassemblies if targets.subassemblies else [None]

        for name, geom in results:
            res_list = geom if isinstance(geom, list) else [geom]
            for i, item in enumerate(res_list):
                sub = subs[i] if i < len(subs) else None
                display_name = f"{name}_{sub}" if sub else name
                rgba = self.manager.router.get_color(name, sub)
                items.append((item, display_name, rgba[:3], rgba[3]))
        return items

    def _get_diagram_items(
        self, targets: TargetList
    ) -> List[tuple[Any, str, Optional[tuple[float, float, float]], float]]:
        """Collect a compound from a DIAGRAM build."""
        items = []
        results = self.manager.router.run(targets)
        for p_name, room in results:
            items.append((room.compound, f"{p_name}_diagram", None, 1.0))
        return items

    @validate_call(config={"arbitrary_types_allowed": True})
    def show_view(
        self,
        input_targets: Sequence[str],
        build_dir: str = "build",
        no_build: bool = False,
        sim_steps: int = 2000,
        save_rrd: Optional[str] = None,
        save_mp4: Optional[str] = None,
        view_from: str = "iso",
        fps: int = 60,
        resolution: Union[str, tuple[int, int]] = (2560, 1440),
        samples: int = 32,
        rerun_port: Optional[int] = None,
        no_gui: bool = False,
        stage_window_size: Optional[int] = None,
    ):
        """Build and show the requested geometry in ocp_vscode."""
        display_items = []
        is_simulate = False
        provider: Optional[Provider] = None
        sim_target = None

        if isinstance(resolution, str):
            res_parts = resolution.lower().split("x")
            if len(res_parts) == 2:
                try:
                    res_tuple = (int(res_parts[0]), int(res_parts[1]))
                except ValueError:
                    res_tuple = (2560, 1440)
            else:
                res_tuple = (2560, 1440)
        else:
            res_tuple = tuple(resolution)

        for target in input_targets:
            for action in self.VISUAL_ACTIONS:
                try:
                    targets = self.target_parser.resolve(target, action)
                    if not targets:
                        continue
                    if Mode.SIMULATE in getattr(targets, "modes", []):
                        is_simulate = True
                        provider = ProviderResolver.resolve(targets.provider, target)
                        sim_target = list(targets)[0] if targets else TargetParser.get_base_target(target)
                    match action:
                        case Section.VIEW:
                            display_items.extend(self._get_view_items(targets))
                            break
                        case Section.PART:
                            display_items.extend(self._get_part_items(targets))
                            break
                        case Section.DIAGRAM:
                            display_items.extend(self._get_diagram_items(targets))
                            break
                except ValueError:
                    continue

        if not display_items:
            raise ValueError("No geometry generated for the specified targets.")

        room = Room(config=self.manager.config, is_simulate=is_simulate)
        for obj, name, color, alpha in display_items:
            # Assembly names cannot contain slashes as they are path delimiters
            base_name = name.replace("/", "_")
            safe_name = base_name
            counter = 1
            while safe_name in room:
                safe_name = f"{base_name}_{counter}"
                counter += 1
            room.add(safe_name, obj, color=color, alpha=alpha)

        if room.is_simulate and provider:
            proj_name = "default"
            for target in input_targets:
                proj_name = TargetParser.get_project_name(target)
                if proj_name != "default":
                    break

            if not no_build:
                # Compile OBJs and URDFs prior to simulating to ensure they are up to date
                base_targets = [f"{TargetParser.get_project_name(t)}/*" for t in input_targets]
                builder = Builder(self.manager, self.logger)
                builder._load_manifest(build_dir)
                builder.generate_parts(build_dir, names=base_targets, force_update=False)
                builder.generate_urdfs(build_dir, names=base_targets, force_update=False)
                builder._save_manifest(build_dir)

            room.simulate(
                provider_hooks=provider.get_simulate_hooks(sim_target or "default"),
                proj_name=proj_name,
                sim_target=sim_target or "default",
                steps=sim_steps,
                manager=self.manager,
                logger=self.logger,
                build_dir=build_dir,
                save_rrd=save_rrd,
                save_mp4=save_mp4,
                view_from=view_from,
                fps=fps,
                resolution=res_tuple,
                samples=samples,
                rerun_port=rerun_port,
                spawn_viewer=not no_gui,
                stage_window_size=stage_window_size,
            )
        else:
            summary = self.get_summary(list(room.keys()))
            self.logger.print(f"Showing {summary}", symbol="👁️ ")
            if save_mp4:
                self.logger.print(f"Rendering turntable MP4 via Blender: {save_mp4}", symbol="🎬")
                from provider.blender import BlenderRenderer, RenderConfig

                render_cfg = RenderConfig(
                    resolution=res_tuple,
                    fps=fps,
                    samples=samples,
                    view_from=view_from,
                    output_mp4=save_mp4,
                )
                BlenderRenderer.render_turntable_to_mp4(
                    room=room,
                    output_mp4=save_mp4,
                    config=render_cfg,
                )
                self.logger.print(f"Exported H.264 MP4 video to {save_mp4}", symbol="✨")
            if not no_gui:
                show(room.compound, names=["View"], collapse=Collapse.LEAVES, reset_camera=Camera.RESET)


def get_args():
    """Get parsed arguments for the viewer."""
    parser = argparse.ArgumentParser(description="View Utility.")
    parser.add_argument(
        "targets", nargs="*", help="The targets to visualize (e.g. tube/driver, tube/wire, tube/driver_left)."
    )
    parser.add_argument("-l", "--list", action="store_true", help="List available visual targets")
    parser.add_argument("--no-build", action="store_true", help="Skip compiling parts and URDFs prior to simulating")
    parser.add_argument(
        "--build-dir", default="build", help="Directory where compiled parts and URDFs are stored (default: 'build')"
    )
    parser.add_argument(
        "-s",
        "--sim-steps",
        type=int,
        default=20000,
        required=False,
        help="Maximum number of steps to take before stopping the simulation.",
    )
    parser.add_argument(
        "--save-rrd",
        help="Path to save the rerun (.rrd) recording file.",
    )
    parser.add_argument(
        "--save-mp4",
        "--export-mp4",
        dest="save_mp4",
        default=None,
        help="Path to export H.264 MP4 video using Blender (e.g. 'recordings/video.mp4').",
    )
    parser.add_argument(
        "--view-from",
        default="iso",
        help="Camera viewpoint for visualization or rendering (e.g. 'iso', 'front', 'rear', 'left', 'right', 'top', 'bottom', or '(x,y,z)').",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=60,
        help="Frames per second for MP4 video export (default: 60).",
    )
    parser.add_argument(
        "--resolution",
        default="2560x1440",
        help="Render resolution formatted as WxH (default: '2560x1440').",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=32,
        help="Blender Cycles/EEVEE render samples per frame (default: 32).",
    )
    parser.add_argument(
        "--stage-window",
        "--stage-window-size",
        "--staging-frame-window-size",
        type=int,
        default=None,
        dest="stage_window_size",
        help="Number of simulation frames per staging checkpoint window (e.g. 1000).",
    )
    parser.add_argument(
        "-p",
        "--port",
        type=int,
        help="Port number for the visualization viewer (ocp_vscode or Rerun viewer).",
    )
    parser.add_argument(
        "--no-gui",
        action="store_true",
        help="Prevent spawning local visualization windows (ocp_vscode or Rerun viewer).",
    )
    args = parser.parse_args()

    if not args.list and not args.targets:
        parser.error("the following arguments are required: target (or use --list)")

    return args


def main():
    """Build and show the requested geometry in ocp_vscode."""
    from provider.utils import initialize_jax_environment

    initialize_jax_environment()

    args = get_args()

    # Allow overriding the OCP Viewer port
    if args.port:
        set_port(args.port)

    logger = Logger(text="Visualizing...")

    config = AppConfig()
    manager = ProviderManager(config, logger=logger)
    viewer = Viewer(manager, logger)
    try:
        if args.list:
            logger.text = "Listing targets..."
            Lister(manager, logger).list_targets(Viewer.VISUAL_ACTIONS)
        else:
            viewer.show_view(
                cast(Sequence[str], args.targets),
                build_dir=args.build_dir,
                no_build=args.no_build,
                sim_steps=args.sim_steps,
                save_rrd=args.save_rrd,
                save_mp4=args.save_mp4,
                view_from=args.view_from,
                fps=args.fps,
                resolution=args.resolution,
                samples=args.samples,
                rerun_port=args.port,
                no_gui=args.no_gui,
                stage_window_size=args.stage_window_size,
            )
    finally:
        logger.done()


if __name__ == "__main__":
    DaemonClient().run("view", sys.argv[1:])
