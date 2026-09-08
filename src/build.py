"""Orchestrate geometry generation and export for discovered projects."""

import argparse
import io
import hashlib
import yaml
import os
import sys
from typing import cast
from pathlib import Path
from daemon import DaemonClient
from model import AppConfig
from build123d import *  # type: ignore
from build123d import export_stl, export_brep, Shape  # type: ignore
from target_parser import TargetParser
from typing import Optional, Any, Sequence, Callable
from pydantic import validate_call
from provider import ProviderManager, Section, Mode, SUBASSEMBLIES, Room, URDFShape
import zipfile
from shell import Logger
from concurrent.futures import ThreadPoolExecutor
import threading
from list import Lister

SPINNER_TEXT = "Building..."


class Builder:
    """Coordinates build actions and file exports using project providers."""

    def __init__(self, manager: ProviderManager, logger: Optional[Logger] = None):
        """Initialize builder dependencies and measurements."""
        self.manager = manager
        self.config = manager.config
        self.logger = logger or Logger(enabled=False)
        self.target_parser = TargetParser(manager.router)
        self.lister = Lister(manager, self.logger)
        self.build_manifest: dict[str, dict[str, str]] = {"brep": {}, "file": {}}
        self.lock = threading.Lock()
        self.executor = ThreadPoolExecutor()
        template_path = Path(__file__).parent / "urdf_template.yaml"
        with open(template_path, "r") as f:
            templates = yaml.safe_load(f)
        self.robot_template = templates["robot_template"].strip()
        self.link_template = templates["link_template"].strip()
        self.joint_template = templates["joint_template"].strip()
        self.axis_limit_template = templates["axis_limit_template"].strip()

    def _get_summary(self, names: Sequence[str]) -> str:
        """Return a truncated summary string of the target names."""
        count = len(names)
        if count > 8:
            return f"{', '.join(names[:8])} ... ({count} items)"
        return ", ".join(names)

    def _load_manifest(self, out_dir: str):
        """Load an existing build manifest from the output directory if not already loaded."""
        if getattr(self, "manifest_out_dir", None) == str(out_dir):
            return
        manifest_path = Path(out_dir) / "build_manifest.yaml"
        if manifest_path.exists():
            try:
                with open(manifest_path, "r") as f:
                    data = yaml.safe_load(f)
                    # Migrate old flat manifest format to nested format
                    if "brep" not in data and "sha1" not in data and "stl" not in data and "file" not in data:
                        self.build_manifest = {"brep": data, "file": {}}
                    else:
                        # Migrate legacy keys to 'file' if present in nested format
                        if "sha1" in data:
                            data["file"] = data.pop("sha1")
                        if "stl" in data:
                            data["file"] = data.pop("stl")
                        self.build_manifest = data
            except (yaml.YAMLError, OSError):
                pass
        self.manifest_out_dir = str(out_dir)

    def _save_manifest(self, out_dir: str):
        """Write the current build manifest to a YAML file in the output directory."""
        manifest_path = Path(out_dir) / "build_manifest.yaml"
        with open(manifest_path, "w") as f:
            yaml.dump(self.build_manifest, f, sort_keys=False)

    def _get_part_hash(self, part: Part) -> str:
        """Calculate a stable hash for a build123d Part using its BREP representation."""
        # Use BREP for hashing because it is faster to generate and
        # provides a more stable geometric identity than a mesh.
        with io.BytesIO() as brep_stream:
            export_brep(part, brep_stream)
            return hashlib.sha1(brep_stream.getvalue()).hexdigest()

    def _get_diagram_hash(self, room: Room, options: Any) -> str:
        """Calculate a hash for the diagram based on its room contents and options."""
        hasher = hashlib.sha1()

        # 1. Hash the geometry using BREP representation
        try:
            with io.BytesIO() as brep_stream:
                export_brep(room.compound, brep_stream)
                hasher.update(brep_stream.getvalue())
        except Exception:
            # Fallback to key/item count representation if export_brep fails
            hasher.update(str(sorted(room.keys())).encode("utf-8"))

        # 2. Hash the labels
        sorted_labels = sorted(getattr(room, "_labels", []), key=lambda x: x[0])
        labels_str = str(
            [
                (name, text, (loc.X, loc.Y, loc.Z) if hasattr(loc, "X") else tuple(loc), repr(opts))
                for name, text, loc, opts in sorted_labels
            ]
        )
        hasher.update(labels_str.encode("utf-8"))

        # 3. Hash options representation
        if options:
            opts_dict = (
                options.model_dump(mode="json") if hasattr(options, "model_dump") else getattr(options, "__dict__", {})
            )
            hasher.update(str(sorted(opts_dict.items())).encode("utf-8"))

        return hasher.hexdigest()

    def _get_urdf_hash(self, room: Room, p_name: str) -> str:
        """Calculate a hash for the URDF output."""
        with io.StringIO() as urdf_stream:
            room.export_urdf(urdf_stream, p_name)
            return hashlib.sha1(urdf_stream.getvalue().encode("utf-8")).hexdigest()

    def _get_file_hash(self, path: Path) -> str:
        """Calculate the SHA1 hash of a file on disk."""
        return hashlib.sha1(path.read_bytes()).hexdigest()

    def _export_obj(self, shape: Shape, file_path: str, tolerance: float = 0.1, scale: float = 1.0) -> bool:
        """Export build123d shape to OBJ format."""
        from OCP.BRepTools import BRepTools
        from OCP.BRepMesh import BRepMesh_IncrementalMesh
        from OCP.BRep import BRep_Tool
        from OCP.TopLoc import TopLoc_Location

        if shape.wrapped is not None:
            BRepTools.Clean_s(shape.wrapped)
            BRepMesh_IncrementalMesh(shape.wrapped, tolerance, False, 0.1, True)

        vertices: list[Vector] = []
        triangles: list[tuple[int, int, int]] = []
        offset = 0

        for face in shape.faces():
            loc = TopLoc_Location()
            poly = BRep_Tool.Triangulation_s(face.wrapped, loc)
            if poly is None:
                continue

            trsf = loc.Transformation()
            face_nodes = [
                Vector(
                    float(poly.Node(i).Transformed(trsf).X()),
                    float(poly.Node(i).Transformed(trsf).Y()),
                    float(poly.Node(i).Transformed(trsf).Z()),
                )
                for i in range(1, poly.NbNodes() + 1)
            ]
            vertices.extend(face_nodes)

            for i in range(1, poly.NbTriangles() + 1):
                tri = poly.Triangle(i)
                n1, n2, n3 = tri.Get()
                triangles.append((n1 - 1 + offset, n2 - 1 + offset, n3 - 1 + offset))

            offset += poly.NbNodes()

        with open(file_path, "w") as f:
            f.write("# Exported by build.py\n")
            f.write(f"# Vertices: {len(vertices)}, Triangles: {len(triangles)}\n")

            # Write scaled vertices
            for v in vertices:
                f.write(f"v {v.X * scale:.6f} {v.Y * scale:.6f} {v.Z * scale:.6f}\n")

            # OBJ syntax links normals directly to face formatting: f v1//vn1 v2//vn2 v3//vn3
            for t in triangles:
                # Get the 3 vertices for the triangle face
                v0 = vertices[t[0]]
                v1 = vertices[t[1]]
                v2 = vertices[t[2]]

                # Cross product to find face perpendicular normal vector
                edge1 = Vector(v1.X - v0.X, v1.Y - v0.Y, v1.Z - v0.Z)
                edge2 = Vector(v2.X - v0.X, v2.Y - v0.Y, v2.Z - v0.Z)
                normal = edge1.cross(edge2)

                if normal.length > 1e-6:
                    normal = normal.normalized()

                f.write(f"vn {normal.X:.6f} {normal.Y:.6f} {normal.Z:.6f}\n")

            # Write faces referencing 1-based indexing
            for i, t in enumerate(triangles):
                norm_idx = i + 1  # 1-based indexing for normals
                v1 = t[0] + 1
                v2 = t[1] + 1
                v3 = t[2] + 1
                f.write(f"f {v1}//{norm_idx} {v2}//{norm_idx} {v3}//{norm_idx}\n")

        return True

    def _export_stl_cleaned(
        self,
        shape: Shape,
        file_path: str,
        tolerance: float = 0.001,
        angular_tolerance: float = 0.03,
    ) -> bool:
        """Clean cached OpenCASCADE mesh and export to STL with high resolution settings."""
        from OCP.BRepTools import BRepTools

        if shape.wrapped is not None:
            BRepTools.Clean_s(shape.wrapped)
        return export_stl(shape, file_path, tolerance=tolerance, angular_tolerance=angular_tolerance)

    def _export_single_part_formats(
        self,
        part: Shape,
        out_dir: str,
        export_types: Sequence[str],
        part_outputs: Sequence[str],
        current_hash: str,
        force_update: bool,
    ) -> None:
        """Export all requested formats (STL, OBJ) for a single part sequentially."""
        for export_type in export_types:
            if export_type == "obj":
                obj_file_name = next(p for p in part_outputs if p.endswith(".obj"))
                obj_path = Path(out_dir) / obj_file_name
                obj_path.parent.mkdir(parents=True, exist_ok=True)
                self._export_if_changed(
                    obj_path,
                    obj_file_name,
                    current_hash,
                    lambda p=obj_path: self._export_obj(part, str(p), scale=1.0),
                    force_update,
                )
            elif export_type == "stl":
                mesh_file_name = next(p for p in part_outputs if p.endswith(".stl"))
                path_obj = Path(out_dir) / mesh_file_name
                path_obj.parent.mkdir(parents=True, exist_ok=True)
                path_str = str(path_obj)
                self._export_if_changed(
                    path_obj,
                    mesh_file_name,
                    current_hash,
                    lambda ps=path_str: self._export_stl_cleaned(part, ps),
                    force_update,
                )

    def _export_if_changed(
        self,
        path: Path,
        manifest_key: str,
        current_hash: str,
        export_fn: Callable[[], Any],
        force_update: bool = False,
    ):
        """Register hash in manifest and export content only if it has changed."""
        # Return early if the hash matches the manifest and the file exists.
        with self.lock:
            brep_manifest = self.build_manifest.setdefault("brep", {})
            file_manifest = self.build_manifest.setdefault("file", {})

            if not force_update and brep_manifest.get(manifest_key) == current_hash and path.exists():
                if manifest_key not in file_manifest:
                    file_manifest[manifest_key] = self._get_file_hash(path)
                return

        # Perform the actual export (heavy meshing/writing) outside the lock
        export_fn()

        # Update the manifest and print to log inside the lock
        with self.lock:
            brep_manifest = self.build_manifest.setdefault("brep", {})
            file_manifest = self.build_manifest.setdefault("file", {})
            brep_manifest[manifest_key] = current_hash
            file_manifest[manifest_key] = self._get_file_hash(path)
            self.logger.print(f"Saved {path}", symbol="📄")

    def _resolve_subassemblies(self, targets: Any, base_subs: Any) -> Sequence[str]:
        """Determine which subassemblies should be built for a target."""
        if base_subs:
            return base_subs
        else:
            all_subs = set()
            for target in targets:
                manifest = self.manager.router.manifest.get(target, {})
                action_cfg = manifest.get(Section.PART, {})
                target_subs = action_cfg.get(SUBASSEMBLIES, [])
                all_subs.update(target_subs)
            return sorted(list(all_subs))

    @validate_call(config={"arbitrary_types_allowed": True})
    def _export_parts(self, out_dir: str, batch_results: Any, sub: Optional[str] = None, force_update: bool = False):
        """Export parts from a batch run."""
        futures = []
        for name, results in batch_results:
            # Results is either a single geometry or a list of geometries
            res_list = results if isinstance(results, list) else [results]
            for geom in res_list:
                p_name, _ = TargetParser.split_target(name)

                # Create provider-specific subdirectory
                target_dir = Path(out_dir) / p_name
                target_dir.mkdir(parents=True, exist_ok=True)

                # Resolve export types from manifest
                export_types = self.manager.router.get_export_types(name, sub)

                if geom.part:
                    current_hash = self._get_part_hash(geom.part)
                    part_outputs = self.lister.get_part_outputs(name, sub)

                    futures.append(
                        self.executor.submit(
                            self._export_single_part_formats,
                            geom.part,
                            out_dir,
                            export_types,
                            part_outputs,
                            current_hash,
                            force_update,
                        )
                    )

        # Wait for all submitted exports to complete
        for fut in futures:
            fut.result()

    @validate_call(config={"arbitrary_types_allowed": True})
    def generate_parts(self, out_dir, names: list[str] | None = None, force_update: Optional[bool] = None):
        """Export STL files for generated parts."""
        if force_update is None:
            force_update = bool(names)
        if names:
            target_lists = []
            for name in names:
                # Only resolve targets that are intended for the PART action
                if self.target_parser.parse(name, Section.PART):
                    target_lists.append(self.target_parser.resolve(name, Section.PART))
        else:
            target_lists = [self.manager.router.targets.supporting(Section.PART).for_modes([Mode.PRINT])]

        if not target_lists:
            return

        for base_targets in target_lists:
            self.logger.print(
                f"Compiling {Section.PART}s: {self._get_summary(list(base_targets))}",
                symbol="🛠️ ",
            )
            has_base_targets: set[str] = set(base_targets)

            # Run targets which have subassemblies, then run any remaining base targets.
            for sub in self._resolve_subassemblies(base_targets, base_targets.subassemblies):
                run_targets = base_targets.for_subassemblies([sub])
                batch_results = self.manager.router.run(run_targets)
                self._export_parts(out_dir, batch_results, sub=sub, force_update=force_update)
                for t in run_targets:
                    has_base_targets.discard(t)

            if has_base_targets:
                batch_results = self.manager.router.run(base_targets.for_targets(has_base_targets))
                self._export_parts(out_dir, batch_results, force_update=force_update)

    @validate_call(config={"arbitrary_types_allowed": True})
    def generate_diagram(self, out_dir, names: list[str] | None = None, force_update: Optional[bool] = None):
        """Export an exploded diagram for the parts."""
        if force_update is None:
            force_update = bool(names)
        if names:
            target_lists = []
            for name in names:
                # Only resolve targets that are intended for the DIAGRAM action
                if self.target_parser.parse(name, Section.DIAGRAM):
                    target_lists.append(self.target_parser.resolve(name, Section.DIAGRAM))
        else:
            target_lists = [self.manager.router.targets.supporting(Section.DIAGRAM).for_modes([Mode.DEFAULT])]

        if not target_lists:
            return

        futures = []
        for base_targets in target_lists:
            self.logger.print(
                f"Compiling {Section.DIAGRAM}s: {self._get_summary(list(base_targets))}",
                symbol="🛠️ ",
            )
            for target in base_targets:
                single_target_list = base_targets.for_targets([target])
                results = self.manager.router.run(single_target_list)

                for p_name, room in results or []:
                    diagram_file = self.lister.get_diagram_output(target)
                    path_obj = Path(out_dir) / diagram_file
                    path_obj.parent.mkdir(parents=True, exist_ok=True)
                    path_str = str(path_obj)

                    provider = next((p for p in self.manager.router.providers if p.name == p_name), None)
                    options = getattr(provider.settings, "diagram_options", None) if provider else None

                    current_hash = self._get_diagram_hash(room, options)
                    futures.append(
                        self.executor.submit(
                            self._export_if_changed,
                            path_obj,
                            diagram_file,
                            current_hash,
                            lambda r=room, ps=path_str, o=options: r.export_diagram(ps, o),
                            force_update,
                        )
                    )

        # Wait for all submitted diagram exports to complete
        for fut in futures:
            fut.result()

    @validate_call(config={"arbitrary_types_allowed": True})
    def generate_urdfs(self, out_dir, names: list[str] | None = None, force_update: Optional[bool] = None):
        """Export combined URDF/OBJ assets from views that support simulate mode."""
        if force_update is None:
            force_update = bool(names)
        if names:
            target_lists = []
            for name in names:
                if self.target_parser.parse(name, Section.VIEW):
                    target_lists.append(self.target_parser.resolve(name, Section.VIEW))
        else:
            target_lists = [self.manager.router.targets.supporting(Section.VIEW).for_modes([Mode.SIMULATE])]

        if not target_lists:
            return

        futures = []
        for base_targets in target_lists:
            self.logger.print(
                f"Compiling URDFs: {self._get_summary(list(base_targets))}",
                symbol="🤖",
            )

            simulate_targets = base_targets.for_modes([Mode.SIMULATE])
            if not simulate_targets:
                continue

            results = self.manager.router.run(simulate_targets)
            for fq_target, room in results or []:
                proj_name = TargetParser.get_project_name(fq_target)

                urdf_file = self.lister.get_urdf_output(fq_target)
                urdf_path = Path(out_dir) / urdf_file
                urdf_path.parent.mkdir(parents=True, exist_ok=True)
                path_str = str(urdf_path)

                # Validate OBJ links
                for geom, _ in room.values():
                    label = getattr(geom, "urdf_label", None)
                    if not label:
                        continue
                    local_shape = geom.location.inverse() * geom

                    # Try progressive fallback on label to locate correct part outputs
                    obj_file_name = None
                    parts = label.split("_")
                    for i in range(len(parts), 0, -1):
                        candidate_label = "_".join(parts[:i])
                        candidate_fq = (
                            f"{proj_name}/{candidate_label}" if "/" not in candidate_label else candidate_label
                        )
                        part_outputs = self.lister.get_part_outputs(candidate_fq, None)
                        found = next((p for p in part_outputs if p.endswith(".obj")), None)
                        if found:
                            obj_file_name = found
                            break

                    if not obj_file_name:
                        raise ValueError(f"OBJ file for link '{label}' could not be resolved.")

                    obj_path = Path(out_dir) / obj_file_name
                    if not obj_path.exists():
                        raise ValueError(f"OBJ file for link '{label}' does not exist: {obj_path}. ")

                    # Save resolved filename on geom so export_urdf can use it
                    u_geom = cast(URDFShape, geom)
                    u_geom.urdf_obj_filename = os.path.basename(obj_file_name)
                    current_hash = self._get_part_hash(local_shape)

                    with self.lock:
                        brep_manifest = self.build_manifest.setdefault("brep", {})
                        manifest_hash = brep_manifest.get(obj_file_name)
                    if manifest_hash != current_hash:
                        raise ValueError(f"OBJ file for link '{label}' is out of date. ")

                current_hash = self._get_urdf_hash(room, proj_name)
                futures.append(
                    self.executor.submit(
                        self._export_if_changed,
                        urdf_path,
                        urdf_file,
                        current_hash,
                        lambda r=room, ps=path_str, pn=proj_name: r.export_urdf(ps, pn),
                        force_update,
                    )
                )

        for fut in futures:
            fut.result()

    def generate_all(self, out_dir, names: list[str] | None = None, zip_name="build.zip"):
        """Generate diagrams, parts, and package them."""

        def zip_build(zip_file_str, outputs):
            """Write generated files into a zip archive."""
            with zipfile.ZipFile(zip_file_str, "w", zipfile.ZIP_DEFLATED) as zipf:
                for output in outputs:
                    file_path = Path(out_dir) / output
                    if file_path.exists():
                        zipf.write(str(file_path), output)

        def needs_zip(zip_path, outputs) -> bool:
            """Check if the zip archive needs to be rebuilt."""
            if not zip_path.exists():
                return True

            zip_mtime = zip_path.stat().st_mtime
            for output in outputs:
                file_path = Path(out_dir) / output
                if file_path.exists() and file_path.stat().st_mtime > zip_mtime:
                    return True

            try:
                with zipfile.ZipFile(zip_path, "r") as zipf:
                    namelist = zipf.namelist()
                    for output in outputs:
                        file_path = Path(out_dir) / output
                        if file_path.exists() and output not in namelist:
                            return True
            except Exception:
                return True

            return False

        # Export the diagram and files
        if not self.manager.router.providers:
            raise ValueError("No projects discovered. Nothing to build.")

        self.generate_parts(out_dir=out_dir, names=names)
        self.generate_diagram(out_dir=out_dir, names=names)
        self.generate_urdfs(out_dir=out_dir, names=names)

        # Compress the build
        zip_path = Path(out_dir) / zip_name
        zip_file_str = str(zip_path)
        outputs = self.lister.get_outputs(names)

        if needs_zip(zip_path, outputs):
            zip_build(zip_file_str, outputs)
            self.logger.print(f"Done writing {zip_file_str}", symbol="📦")
        else:
            self.logger.print(f"{zip_name} is already up-to-date", symbol="📦")


def get_args():
    """Get parsed arguments for the program."""
    parser = argparse.ArgumentParser(description="Build Utility.")
    parser.add_argument("-e", "--env", required=False, default=None, help="Output environment to file and exit.")

    parser.add_argument("-out", "--outdir", default="build", help="Target directory for outputs")

    parser.add_argument(
        "targets",
        nargs="*",
        help="Specific targets to build. Usage: build.py part1 part2. If omitted, all targets are built.",
    )

    args = parser.parse_args()
    return args


def main(logger, args):
    """Initialize the build environment and perform build actions."""
    # Generate optional arguments
    # Create the output directory
    path = Path(args.outdir)
    path.mkdir(parents=True, exist_ok=True)

    config = AppConfig()
    manager = ProviderManager(config, logger=logger)
    builder = Builder(manager, logger)
    try:
        if not args.env is None:
            builder.config.dump_env(args.env)
            logger.print(f"Saved environment to {args.env}", symbol="⚙️ ")
        else:
            builder._load_manifest(args.outdir)
            targets = args.targets if args.targets else None
            builder.generate_all(out_dir=args.outdir, names=targets)
            builder._save_manifest(args.outdir)

    finally:
        logger.done()


if __name__ == "__main__":
    """Program entry point."""
    DaemonClient().run("build", sys.argv[1:])
