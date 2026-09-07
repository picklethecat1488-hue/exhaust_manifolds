"""Utility functions for build providers."""

import os
from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypeVar, Callable, overload, Union, Optional, cast
import yaml
from .types import Mode, Section, ColorType, MODES, SUBASSEMBLIES, COLOR, MATERIAL, EXPORT

T = TypeVar("T", bound=type)


@overload
def discover_provider(cls: T) -> T: ...


@overload
def discover_provider(*, enabled: bool = True) -> Callable[[T], T]: ...


def discover_provider(cls: T | None = None, *, enabled: bool = True) -> Any:
    """Mark a Provider subclass for automatic discovery by ProviderManager."""

    def decorator(target: T) -> T:
        setattr(target, "_discover_provider", enabled)
        return target

    if cls is None:
        return decorator
    return decorator(cls)


def _merge_manifests(dst: dict, src: dict):
    """Recursively merge src dictionary into dst dictionary."""
    for k, v in src.items():
        if k in dst and isinstance(dst[k], dict) and isinstance(v, dict):
            _merge_manifests(dst[k], v)
        else:
            dst[k] = v


def load_manifest(path: str) -> dict[str, dict[Any, Any]]:
    """
    Load and parse a manifest from a YAML file, resolving any imports.

    Example YAML format:
    ```yaml
    imports:
      - print_materials.yaml
    material:
      pla:
        density: 1.24
    part_a:
      wire:
        modes: [default]
        subassemblies: [left]
      part:
        modes: [default, bare]
        subassemblies: [left]
      color: [0.8, 0.8, 0.8, 1.0]
      material: pla
    ```
    """
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}

    manifest = {}

    # 1. Handle imports recursively
    import_paths = []
    if "import" in data:
        val = data.pop("import")
        if isinstance(val, str):
            import_paths.append(val)
    if "imports" in data:
        val = data.pop("imports")
        if isinstance(val, list):
            import_paths.extend(val)
        elif isinstance(val, str):
            import_paths.append(val)

    base_dir = os.path.dirname(os.path.abspath(path))
    for imp_path in import_paths:
        abs_imp_path = os.path.normpath(os.path.join(base_dir, imp_path))
        if os.path.exists(abs_imp_path):
            imported_manifest = load_manifest(abs_imp_path)
            _merge_manifests(manifest, imported_manifest)

    # 2. Parse current manifest target/section config
    current_manifest = {}
    for target, actions in data.items():
        if target == MATERIAL:
            # Merging material definitions section
            current_manifest[MATERIAL] = actions
            continue

        target_cfg = {}
        if isinstance(actions, dict):
            for key, val in actions.items():
                # Handle Color metadata
                if key == "color":
                    if isinstance(val, dict):
                        target_cfg[COLOR] = {
                            str(k): (tuple(float(x) for x in v) if isinstance(v, (list, tuple)) else ColorType(v))
                            for k, v in val.items()
                        }
                    elif isinstance(val, (list, tuple)):
                        target_cfg[COLOR] = tuple(float(x) for x in val)
                    else:
                        target_cfg[COLOR] = ColorType(val)
                    continue

                # Handle Material metadata
                if key == MATERIAL:
                    target_cfg[MATERIAL] = val
                    continue

                # Handle Export metadata
                if key == EXPORT:
                    target_cfg[EXPORT] = val
                    continue

                # Map string keys to Section Enums
                try:
                    section_key = Section(key)
                except ValueError:
                    target_cfg[key] = val
                    continue

                section_cfg = {}
                if isinstance(val, dict):
                    if MODES in val:
                        section_cfg[MODES] = [str(m) if section_key == Section.CONFIG else Mode(m) for m in val[MODES]]
                    if SUBASSEMBLIES in val:
                        section_cfg[SUBASSEMBLIES] = [str(s) for s in val[SUBASSEMBLIES]]
                target_cfg[section_key] = section_cfg
        else:
            target_cfg = actions

        current_manifest[target] = target_cfg

    _merge_manifests(manifest, current_manifest)
    return manifest


def get_rgba_color(
    color: Union[str, ColorType, tuple[float, float, float]],
    alpha: float,
    default_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
) -> tuple[float, float, float, float]:
    """Convert a color name (or ColorType enum) to an RGBA tuple."""
    if isinstance(color, (tuple, list)):
        return (*color, alpha)  # type: ignore

    color_map = {
        ColorType.RED: (1.0, 0.0, 0.0),
        ColorType.GREEN: (0.0, 1.0, 0.0),
        ColorType.BLUE: (0.0, 0.0, 1.0),
        ColorType.ORANGE: (1.0, 0.65, 0.0),
        ColorType.CYAN: (0.0, 1.0, 1.0),
        ColorType.YELLOW: (1.0, 1.0, 0.0),
        ColorType.MAGENTA: (1.0, 0.0, 1.0),
        ColorType.GREY: (0.5, 0.5, 0.5),
        ColorType.BLACK: (0.0, 0.0, 0.0),
        ColorType.PURPLE: (0.5, 0.0, 0.5),
    }
    name = str(color)
    rgb = color_map.get(cast(ColorType, name), default_rgb)
    return (*rgb, alpha)


def initialize_jax_environment(cache_dir: Optional[Union[str, os.PathLike]] = None) -> None:
    """Initialize and configure the JAX environment deterministically.

    Suppresses noisy C-level MPS startup banners, silences JAX logger propagation to stdout,
    configures persistent compilation caching, and disables verbose compile logs unless requested.
    """
    import sys
    import warnings
    import logging
    from pathlib import Path

    warnings.filterwarnings("ignore", category=UserWarning, message=".*jax-mps was built for jaxlib.*")
    warnings.filterwarnings("ignore", category=UserWarning, message=".*Platform 'mps' is experimental.*")

    # Disable aggressive CUDA device memory preallocation in JAX
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    os.environ.setdefault("XLA_PYTHON_CLIENT_MEM_FRACTION", "0.75")

    # Configure deterministic single-threaded XLA CPU execution to prevent native Eigen threadpool futex deadlocks
    os.environ.setdefault("XLA_FLAGS", "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1")

    # Unset experimental async dispatch on MPS backend to prevent deadlocks
    if os.environ.get("JAX_MPS_ASYNC_DISPATCH") == "1":
        os.environ["JAX_MPS_ASYNC_DISPATCH"] = "0"

    # Silence C-level MPS startup banners on stderr during initial JAX device probe
    _redirected = False
    _saved_stderr = None
    _stderr_fd = None
    try:
        if hasattr(sys.stderr, "fileno"):
            _stderr_fd = sys.stderr.fileno()
            _saved_stderr = os.dup(_stderr_fd)
            _devnull = os.open(os.devnull, os.O_WRONLY)
            os.dup2(_devnull, _stderr_fd)
            os.close(_devnull)
            _redirected = True
    except (OSError, ValueError, AttributeError):
        _redirected = False

    try:
        import jax

        _ = jax.devices()
    finally:
        if _redirected and _saved_stderr is not None and _stderr_fd is not None:
            os.dup2(_saved_stderr, _stderr_fd)
            os.close(_saved_stderr)

    # Enable JAX compilation caching globally to prevent JIT compile latency
    if cache_dir is None:
        cache_dir = Path(__file__).resolve().parents[2] / "build" / "jax_cache"
    jax.config.update("jax_compilation_cache_dir", str(cache_dir))

    if os.environ.get("JAX_LOG_COMPILES") == "1":
        jax.config.update("jax_log_compiles", True)
        jax.config.update("jax_explain_cache_misses", True)
    else:
        jax.config.update("jax_log_compiles", False)
        jax.config.update("jax_explain_cache_misses", False)

    # Silence JAX and fluid simulation loggers from console output by default
    from .types import DAEMON_LOGGERS

    for logger_name in DAEMON_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.INFO)
        logging.getLogger(logger_name).propagate = False


def merge_rrd_recordings(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    application_id: str = "simulation",
) -> Path:
    """Merge and concatenate multiple RRD recording files into a single unified recording.

    Args:
        input_paths: List or sequence of input .rrd file paths to combine.
        output_path: Destination path for the unified combined .rrd recording.
        application_id: Application ID string for the Rerun recording stream.

    Returns:
        Path to the generated combined .rrd file.
    """
    import rerun as rr

    out_p = Path(output_path)
    out_p.parent.mkdir(parents=True, exist_ok=True)

    rec = rr.RecordingStream(application_id=application_id)
    rec.save(str(out_p))

    for in_path in input_paths:
        p_in = Path(in_path)
        if not p_in.exists():
            continue
        reader = rr.bindings.RrdReaderInternal(str(p_in))
        chunks = list(reader.stream())
        if chunks:
            rr.bindings.send_chunks(chunks, recording=rec.to_native())
    rec.flush()
    return out_p


def rerun_is_enabled() -> bool:
    """Return True if rerun logging is enabled and active."""
    try:
        import rerun as rr

        return bool(hasattr(rr, "is_enabled") and rr.is_enabled())
    except ImportError:
        return False


def str_to_bool(val: Any, default: bool = False) -> bool:
    """Convert a string or arbitrary value to a boolean.

    Recognizes '1', 'true', 'yes', 'on', 'enable', 'enabled' (case-insensitive) as True,
    and '0', 'false', 'no', 'off', 'disable', 'disabled' as False.

    Args:
        val: Input value or string representation.
        default: Fallback boolean value if val is None or unrecognized.

    Returns:
        Boolean evaluation.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    s = str(val).strip().lower()
    if s in ("1", "true", "yes", "on", "enable", "enabled", "t", "y"):
        return True
    if s in ("0", "false", "no", "off", "disable", "disabled", "f", "n"):
        return False
    return default


def get_env_bool(name: str, default: bool = False) -> bool:
    """Read an environment variable and convert it to a boolean.

    Args:
        name: Name of the environment variable.
        default: Default boolean value if the environment variable is unset.

    Returns:
        Boolean representation of the environment variable.
    """
    val = os.getenv(name)
    if val is None:
        return default
    return str_to_bool(val, default=default)
