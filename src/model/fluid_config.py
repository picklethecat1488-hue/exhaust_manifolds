"""Fluid simulation configuration models."""

from typing import Any, Optional
from pydantic import BaseModel, Field, field_validator, model_validator


class FluidConfig(BaseModel):
    """Configuration options for SPH fluid solver."""

    @model_validator(mode="before")
    @classmethod
    def sync_radii(cls, data: Any) -> Any:
        """Sync particle_radius and r_s attributes during model initialization."""
        if isinstance(data, dict):
            if "particle_radius" in data and "r_s" not in data:
                data["r_s"] = data["particle_radius"]
            elif "r_s" in data and "particle_radius" not in data:
                data["particle_radius"] = data["r_s"]
        return data

    # --- SPH Particle & Fluid Properties ---
    r_s: float = Field(default=0.003, description="SPH search/spacing radius (meters).")
    particle_radius: float = Field(default=0.003, description="Physical radius of each SPH fluid particle (meters).")
    rest_density: float = Field(default=1000.0, description="Rest density of the fluid (kg/m^3).")
    viscosity: float = Field(default=0.5, description="Dynamic viscosity coefficient of the fluid solver.")
    stiffness: float = Field(default=2000.0, description="Stiffness parameter (k) for the equation of state.")
    k: Optional[float] = Field(default=None, description="Alternate stiffness parameter identifier.")
    gravity: tuple[float, float, float] = Field(
        default=(0.0, 0.0, -9.81), description="Gravity vector (x, y, z) in m/s^2."
    )

    # --- SPH Kernel & Numerical Solvers Constants ---
    smoothing_factor: float = Field(
        default=3.0, description="Kernel smoothing radius multiplier relative to particle radius."
    )
    sphere_vol_factor: float = Field(
        default=4.0 / 3.0, description="Volume coefficient multiplier for spherical particle volumes."
    )
    poly6_coeff_numerator: float = Field(
        default=315.0, description="Numerator coefficient for Poly6 density kernel estimation."
    )
    poly6_coeff_denominator: float = Field(
        default=64.0, description="Denominator coefficient for Poly6 density kernel estimation."
    )
    spiky_grad_coeff: float = Field(default=-45.0, description="Coefficient for Spiky gradient kernel calculation.")
    visc_lap_coeff: float = Field(default=45.0, description="Coefficient for Viscosity Laplacian kernel calculation.")
    pressure_avg_factor: float = Field(
        default=2.0, description="Averaging coefficient for pairwise SPH pressure calculation."
    )
    min_distance_threshold: float = Field(
        default=1e-6, description="Minimum distance threshold to prevent numerical divide-by-zero errors."
    )

    # --- Surface Smoothing Polynomial Kernel Coefficients ---
    surface_smooth_center_coeff: float = Field(
        default=4.0, description="Center weight for 2D separable surface smoothing polynomial kernel."
    )
    surface_smooth_neighbor_coeff: float = Field(
        default=2.0, description="Orthogonal neighbor weight for 2D separable surface smoothing polynomial kernel."
    )
    surface_smooth_diagonal_coeff: float = Field(
        default=1.0, description="Diagonal neighbor weight for 2D separable surface smoothing polynomial kernel."
    )

    # --- Boundary Collision Properties ---
    stiffness_boundary: float = Field(default=1000.0, description="Boundary collision penalty stiffness coefficient.")
    damping_boundary: float = Field(default=5.0, description="Boundary collision penalty damping coefficient.")

    # --- Simulation Domain & Volumes ---
    nx: int = Field(default=32, description="Number of grid cells along X axis.")
    ny: int = Field(default=32, description="Number of grid cells along Y axis.")
    nz: int = Field(default=28, description="Number of grid cells along Z axis.")
    dx: float = Field(default=0.005, description="Grid cell width (meters).")
    origin: tuple[float, float, float] = Field(
        default=(-0.075, -0.075, 0.0), description="Grid origin position (meters)."
    )
    target_volume: float = Field(default=0.0005, description="Target total fluid volume to spawn (m^3).")
    spawn_buffer: float = Field(
        default=0.002, description="Spawning buffer distance offset from container side walls (meters)."
    )
    volume_threshold_liters: float = Field(default=0.400, description="Container target volume threshold (liters).")
    fallen_threshold_liters: float = Field(
        default=0.050, description="Maximum allowed volume of fluid to leak or fall (liters)."
    )
    boundaries: dict[str, Any] = Field(
        default_factory=lambda: {
            "bowl": {
                "shape": "cylinder",
                "type": "cavity",
                "radius": 0.076,
                "height": 0.096,
                "xyz": [0.0, 0.0, 0.0],
                "rpy": [0.0, 0.0, 0.0],
                "link_type": -1,
                "link_idx": -1,
            }
        },
        description="Dictionary of physical boundary definitions.",
    )
    recycle_fluid: bool = Field(default=False, description="Enable recycling of fallen particles back into the bowl.")
    high_damping_value: float = Field(
        default=0.998, description="Damping value applied to particles outside the base (fraction)."
    )
    sim_name: str = Field(default="", description="User-defined simulation run name tag.")

    @field_validator("boundaries")
    @classmethod
    def validate_boundaries(cls, v: Any) -> dict[str, Any]:
        """Enforce that every fluid configuration model has a BASE LinkType boundary."""
        if not isinstance(v, dict) or not v:
            raise ValueError("Boundaries dictionary is required and cannot be empty.")

        from provider.bullet import LinkType
        from model.boundary_config import BoundaryConfig

        has_base = False
        for label, val in v.items():
            vals = val if isinstance(val, list) else [val]
            for item in vals:
                if isinstance(item, BoundaryConfig):
                    b_link_type = item.link_type
                elif isinstance(item, dict):
                    b_link_type = item.get("link_type")
                else:
                    b_link_type = getattr(item, "link_type", None)

                if b_link_type == LinkType.BASE or b_link_type == -1 or str(b_link_type).lower() in ("base", "-1"):
                    has_base = True
                    break
            if has_base:
                break

        if not has_base:
            raise ValueError("Every fluid configuration model must contain a BASE LinkType boundary.")

        return v

    @staticmethod
    def water(**kwargs: Any) -> "FluidConfig":
        """Create a FluidConfig for water with default values.

        Args:
            **kwargs: Overrides for the default fluid configuration parameters.

        Returns:
            FluidConfig: A configuration object for water.
        """
        params: dict[str, Any] = {
            "rest_density": 1000.0,
            "r_s": 0.003,
            "viscosity": 0.02,
            "stiffness": 100.0,
        }
        params.update(kwargs)
        return FluidConfig(**params)
