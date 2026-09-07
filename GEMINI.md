# Workspace Rules: Pre-Commit Validation

Before finalizing any task, committing changes, or proposing modifications to the codebase, you MUST run formatting, linting, compile check, and the tests to ensure the codebase remains healthy:

```bash
# 1. Activate the conda environment
conda activate cq

# 2. Check Python syntax/compilation errors
python -m compileall -q .

# 3. Check code formatting
ruff format --check .

# 4. Check linting rules
ruff check .

# 5. Run the fast test suite (excludes slow tests)
pytest
```

## Validation Guidelines
1. **Execution**: Prefer offloading test suites (`pytest`, `pytest -m "slow"`, `python src/smoke.py`) and pre-commit validation to the `anvil` cloud server (e.g., via `bin/anvil run "PYTHONPATH=src /home/ubuntu/miniforge3/envs/cq/bin/pytest"` or `ssh anvil`) to free up local CPU/GPU compute for rapid iteration, CAD generation, and interactive experiments. When running locally, activate the `cq` conda environment.
2. **Outcome Verification**: Confirm that all checks (format, lint, compile, and pytest) pass with exit code `0`.
3. **Resolution**: If any component fails (such as syntax error, ruff failure, or failing test), you must address the failure and re-run the check before concluding your work.
4. **Integration Smoke Tests**: The integration smoke tests (`python src/smoke.py`) are highly resource-intensive and should always be run on `anvil`.
5. **Process Management & Rerun Hygiene**: When restarting or re-running test suites (`pytest`, `pytest -m "slow"`, `bin/anvil pytest`, `python src/smoke.py`, etc.), you MUST explicitly terminate/kill any preceding running instances of that test or task before launching a new execution. Never allow multiple overlapping runs of the same test command.

---

## Software Architecture & Code Guidelines

### 1. Test Isolation & Marking
* Unit tests MUST be completely isolated from implementation code. Do NOT mix unit tests inside implementation files.
* Core framework tests go in [src/tests/](file:///Users/daparker/gh/hardware/src/tests/) and project-specific tests go in [src/projects/tests/](file:///Users/daparker/gh/hardware/src/projects/tests/).
* **Slow Tests**: 3D CAD boolean checks, PyBullet physics simulations, and JAX SPH fluid dynamics tests are highly resource-intensive. They must be decorated with `@pytest.mark.slow` (or have `slow` in their test markers) so they do not block the fast pre-commit check. Run them manually or in nightly validation with:
  ```bash
  pytest -m "slow"
  ```

### 2. Geometry Providers & Discoverability
* Custom geometry projects must be packages nested within [src/projects/](file:///Users/daparker/gh/hardware/src/projects/).
* The provider class must inherit from `Provider` and be decorated with `@discover_provider` (imported from [src/provider/utils.py](file:///Users/daparker/gh/hardware/src/provider/utils.py)).
* Always export the provider at the package level (`__init__.py`) and import it in [src/projects/\_\_init\_\_.py](file:///Users/daparker/gh/hardware/src/projects/__init__.py).
* **Project Manifest Integration**: All custom geometry parts, components, clips, or support structures that participate in assemblies or are needed for manufacturing MUST be explicitly registered in the project's `manifest.yaml` (nested under the project folder) to ensure correct build-chain discovery, STL/OBJ generation, and inclusion in final build artifacts.
* Builder methods should return shape/build geometries (e.g., `BuildPart`), while diagram/view actions should populate a `Room` object via `room.add(...)` or `room.add_label(...)`.

### 3. Configuration & Lazy Initialization
* Always use `@cached_property` for `default_config` and any sub-tools (Builders, Configurators) in your provider class. This guarantees correct orchestration timing, prevents staling configs, and minimizes expensive CAD allocations.
* **Geometry Parametrization & Measurement Comparands**: Always define base geometry parameters in the project's measurements.yaml and read them dynamically via config settings. Compute derived geometry coordinates, dimensions, and branch comparison thresholds dynamically relative to these settings (e.g., using clearances, wall thicknesses, aperture ratios, and flow channel dimensions) instead of hardcoding absolute values. **Do NOT hardcode absolute numeric literals for coordinates, radii, offsets, or comparison thresholds (e.g., `if drain_r >= 0.030:` or `stream_r = 0.015`) in python source code; they must be parameterized in measurements.yaml, derived from URDF boundary metadata, or queried from build123d CAD geometry.** This prevents geometry regressions (e.g. intersections, misaligned steps, or floating shells) when base dimensions are scaled or overridden. For CAD modeling, measurements must be strictly geometrical; operational and physical simulation parameters (such as `tube_velocity`, `tube_pressure_factor`, etc.) should be composed/derived dynamically from base geometric and material parameters rather than configured as independent arbitrary constants.
* Settings and configuration schemas must use Pydantic models (subclassing `BaseModel`) defined under [src/projects_config/](file:///Users/daparker/gh/hardware/src/projects_config/).
* General domain data models (such as SPH boundary configs, fluid properties, and simulator schemas) must be separated into the [src/model/](file:///Users/daparker/gh/hardware/src/model/) subfolder.
* Config overrides can be injected dynamically via environment variables patterned as `<PROJECT>__<SETTING>` (e.g., `EXHAUST_MANIFOLDS__WALL_THICKNESS`).
* **Data Model Integrity**: Prefer using strongly typed data models with well-defined properties and methods over runtime dynamic attribute parsing (e.g., avoiding loose `hasattr` or `getattr` checks on untyped objects where static type annotations should instead guarantee structure).
* **Error Handling & Exception Guardrails**: Use explicit bounds checking, deterministic version/feature branching, and parameter validation rather than generic `try/except` blocks. Do NOT use `try/except` structures in core computation or logic paths except to guard external I/O operations (such as filesystem access, networking, or database calls). **Do NOT place silent `try/except/pass` blocks in any Python source code or generated templates (`try: ... except Exception: pass`). All branches (e.g. API version differences such as `bpy.app.version >= (3, 2, 0)`) must be resolved deterministically using explicit condition checks. Any caught exceptions must be logged descriptively, re-raised, or allowed to propagate.**
* **Parameter Validation**: Prefer Pydantic parameter validation over manual validation checks in code. If dynamic runtime validation is necessary (e.g., in math or physics functions), raise a descriptive `ValueError` to indicate invalid parameters rather than silently failing or falling back.
* **Method Parameterization**: Prefer passing parameters and configuration models explicitly into methods and functions rather than having them read instance attributes or parent provider properties internally. This keeps computation blocks pure, modular, and easy to unit test.
* **Configuration Persistence**: For configuration actions, they should persist saved settings to the Pydantic environment file (`.env`) in addition to updating any source project data files (like `measurements.yaml`). This ensures they are immediately active in the build environment.
* **No Fallback Constants**: Do NOT place fallback constants directly in the codebase when parsing configs, settings, or CAD feature coordinates (e.g., using ternary fallbacks like `ctx.tube.y if ctx else 0.028` or `getattr` defaults like `0.004` or `0.90`). All configuration fields and CAD coordinates must be strongly typed and resolved dynamically via configuration models, metadata definitions, CAD context features, or joint state queries rather than having hardcoded fallback/default values defined in python source code. Fallback constants of `0`, `0.0`, `(0.0, 0.0)`, or `None` are strictly required to represent unconfigured properties or missing coordinates. This ensures configuration changes propagate cleanly and prevents silent regressions.
* **No Dead Code**: Unused code (such as dangling clauses, functions, or parameters that do nothing) and settings that do not affect or update anything must be removed from the repository. Maintain a clean, minimalist codebase to prevent confusion and bugs.
* **Parameter & Signature Hygiene**: Whenever modifying, refactoring, or simplifying functions, subroutines, or methods, any parameters that become unused (such as legacy flags, signs like `normal_sign: float`, unused tolerances, or obsolete scalars) MUST be immediately pruned from both the function signature and all caller invocations with each change. Do NOT leave unused parameters in signatures, accept dummy parameters, or pass dead constant arguments.

### 4. Physical Simulation & URDF Metadata
* For components participating in physics simulations (e.g., PyBullet, JAX fluids), attach URDF and simulation attributes to shape geometries.
* Use `URDFMetadata` blocks to wrap geometries, providing `label`, `material`, `density`, `collision_type` (`URDFCollisionType`), etc.
* Available fields include:
  - `urdf_label` (`str`): Unique label in URDF.
  - `urdf_material` (`str`): Material name (e.g., `"petg"`, `"acrylic"`).
  - `urdf_density` (`float`): Density in $\text{kg/m}^3$.
  - `urdf_collision_type` (`URDFCollisionType`): Convex, concave, compound, analytical, or none.
  - Kinematic joint constraints (`urdf_joint_type`, `urdf_joint_axis`, limits) and motor properties (`urdf_motor_type`, target, force).
* **Physics Parameters Definition**: All physical properties and simulation parameters—including magnetic coupling attraction forces, joint constraints, kinematics, and physical barriers—MUST be defined in the URDF metadata or settings schema rather than being hardcoded in python source code.
* **Temporary Debugging Constants**: Adding constant values in physics code is acceptable during active local debugging/iteration. However, before concluding a task, proposing changes, or running pre-commit checks, all such temporary constants MUST be replaced with dynamic queries referencing the boundary configuration model or URDF metadata.
* **Dynamic Physics via URDF & CAD Geometry**: The physics and simulation code (e.g., in [fluid.py](file:///Users/daparker/gh/hardware/src/provider/fluid.py), [boundary.py](file:///Users/daparker/gh/hardware/src/provider/boundary.py), [fluid_body.py](file:///Users/daparker/gh/hardware/src/model/fluid_body.py), and [bullet.py](file:///Users/daparker/gh/hardware/src/provider/bullet.py)) MUST construct CAD context features, fluid bodies, and physics constraints dynamically using values read from URDF metadata, `BoundaryConfig`, PyBullet joint information, or build123d CAD shapes. All threshold comparisons (such as single-stream vs. multi-spillway cascades or fluid zone bounds) must be derived relative to physical dimensions (e.g. delivery tube radius, aperture ratios, or container depths) rather than hardcoded numeric comparands. Extend the URDF metadata schema as needed to support new physical properties.
* **Coordination of CAD & URDF**: When modifying physical CAD geometries (such as heights, pockets, snouts, or slots), you MUST update the corresponding `URDFMetadata`, joints, and analytical `URDFBoundary` offsets (e.g. `xyz` translations) to ensure physical simulation models remain accurate and zero-intersection constraints are preserved. **The ultimate goal is that the assembled unit's simulated physical behavior matches real-world expectations derived directly from its CAD specifications (e.g., matching flow behavior, motor torque, and clearances to prevent collision tunneling or incorrect flow visualisations).**

### 5. SPH Fluid Simulation & Numerical Stability
* **Analytical Boundaries**: Prefer analytical boundaries (`URDFCollisionType.ANALYTICAL`) over concave meshes (`URDFCollisionType.CONCAVE`) for JAX SPH fluid simulation. This prevents boundary particle tunneling and accelerates collision resolution.
* **Cylinder Boundaries**: For cylinder cavity boundary configurations, treat height as infinite along the local Z axis where possible to avoid particle escape at high pressures.
* **Fluid Recycling**: Ensure `fluid.recycle_fluid = True` is used in steady-state flow loops, with boundary coordinates matching physical limits.
* **JAX-JIT Compilation**: Prefer using `jax.jit` and pure functions during physics computations in JAX to leverage static optimization, compilation speedups, and hardware acceleration.
* **Semantic Coordinate Transforms**: Direct matrix and raw quaternion operations (`q_inv`, `q_mult`, `q_rotate`) are strictly BANNED in JAX simulation and provider production code. All spatial transitions and frame changes MUST use semantic coordinate transformations (`world_to_base_frame`, `base_to_world_frame`, `base_to_local_frame`, `local_to_base_frame`, `base_to_voxel_coord`) and coordinate system conversions (`cartesian_to_cylindrical`, `cylindrical_to_cartesian`, `cartesian_to_spherical`) from `provider.transforms`. This guarantees mathematical consistency across coordinate frames (World, Base Link, Local Link, Voxel Grid) and prevents phantom collision boundaries or force misprojections.
* **Numeric Damping**: For long-running simulation validations, enforce stabilization velocity damping (e.g., `0.95`) to prevent numerical velocity buildup.
* **Physical Contact & Non-Floating Invariants**: Fluid particles residing in containers under gravity must make direct physical contact with the bottom floor ($\min(Z) \le Z_{\text{floor}} + 2 \cdot r_s + \text{margin}$) and spread to outer containment boundaries ($r \to R_{\text{wall}}$), forming a continuous fluid mass. Fluid tests must explicitly assert these contact invariants to prevent artificial mid-air hovering, floating shells, or disconnected particle clusters.
* **Test Failure Replication**: When unphysical behaviors (such as mid-air hovering, suction traps, or hollow shells) are observed during visual simulation inspection, test cases must be updated with assertions that actively reproduce the failure under flawed dynamics and only pass when the physical dynamics are verified.

### 6. Declarative Wiring & Routing Engine
* Declare footprint, physical dimensions, pinouts, and net connections in the project's `wiring.yaml` file.
* Keep orthogonal routing layout automated using pathfinding algorithms. Wire path crossover bumps must be computed automatically to avoid visual intersections.

### 7. Documentation & Lint Style
* Code documentation MUST be PEP-257 compliant and comprehensive. Write docstrings for all custom classes, methods, functions, and properties.
* Docstring correctness is checked automatically by ruff linting rules (group `D` configured in [pyproject.toml](file:///Users/daparker/gh/hardware/pyproject.toml)).
* **String Enums for Keys**: Prefer defining structured string enums (subclassing `str` and `Enum`) over passing raw string literals directly for dictionary keys, joint/link labels, or configuration modes. This prevents typos and improves code readability/refactoring.
* **Named Constant Formatting**: Constant values in production code must be assigned to module-level or class-level `ALL_CAPS` named constant variables rather than being embedded as inline magic literals.
* **Idiomatic Iteration & Enumeration**: Prefer looping over sequences and arrays directly or using `enumerate(...)` (e.g., `for idx, shape in enumerate(b_shapes):` or `for shape in b_shapes:`) rather than indexing by integer range bounds (such as `for k in range(b_shapes.shape[0]):` or `for i in range(len(items)):`).
* **Pattern Matching (`match` / `case`)**: Prefer Python `match / case` pattern matching syntax when comparing the same subject field or expression against multiple comparands, enum variants, or constant branches rather than chained `if / elif / elif / else` ladders.
* **Import Placement**: Imports should be done at the top of the file/listing, unless doing so would cause module load race conditions or circular dependencies (such as importing model classes inside provider packages).

### 8. Work Tracking & Task Management
* **Task List (`TODO.md`)**: Maintain and track planned tasks, active implementation steps, outstanding engineering checklist items, and completed work in a `TODO.md` file in the workspace root. Keep the checklist updated (`[ ]` -> `[x]`) as subtasks progress to provide clear visibility and alignment.

### 9. Code Generation & Jinja2 Templates
* **Jinja2 Templating Engine**: Always use Jinja2 (`jinja2`) to generate templated Python scripts, Blender headless scripts, URDF models, or simulation configurations rather than embedding large multi-line f-strings directly inside Python source files.
* **Dedicated Templates Directory**: All templated script files (`.py.j2`, `.yaml.j2`, `.urdf.j2`, `.sh.j2`) MUST be stored in a dedicated `templates/` folder nested within the respective package or module (e.g., `src/provider/templates/`).
* **Clean Rendering & Context Separation**: Render external Jinja2 templates via `jinja2.Environment(loader=jinja2.FileSystemLoader(...), trim_blocks=True, lstrip_blocks=True)` or package loaders, passing configuration parameters as explicit dictionaries or strongly typed models.
* **No Silent Exception Swallowing in Templates**: Templated scripts (`.py.j2`) MUST follow the same code quality standards as Python source files. Never generate `try/except/pass` fallback ladders inside Jinja2 templates; inspect environment capabilities, platform parameters, or application versions deterministically via template variables or standard version properties.

---

## Remote Cloud Server (`anvil`)

For resource-intensive workloads, parameter sweeps, fluid dynamics simulations, integration smoke tests (`python src/smoke.py`), validation experiments, and data collection, the `anvil` cloud server is available:

```ssh-config
Host anvil
  HostName 163.192.17.220
  User ubuntu
  IdentityFile "~/.ssh/FLINT'S KEY.pem"
  IdentitiesOnly yes
```

### Usage Guidelines:
1. **Remote Execution**: Use `bin/anvil run "<command>"` or SSH targeting `ubuntu@anvil` (or `ssh anvil`) to run full test suites (`pytest`), slow physics benchmarks (`pytest -m "slow"`), large JAX SPH simulation grids, parameter sweeps, and integration smoke tests (`python src/smoke.py`).
2. **Conda Environment & Binaries**: On `anvil`, execute commands within the `cq` conda environment using `conda run -n cq --no-capture-output <command>` (prefer relative executable names like `python`, `pytest`, `ruff` over absolute paths).
3. **Preceding Run Cancellation**: Before initiating a new remote execution or benchmark on `anvil`, ensure any active or stale background runs of the same command are cancelled or terminated to avoid cloud resource contention and duplicate processing.
