"""Cat fountain configuration and measurement logic."""

from typing import Any, Optional, Union, cast
from functools import cached_property
from pathlib import Path
import numpy as np
from pydantic import BaseModel, Field
from model import load_measurements, DiagramOptions, DiagramStyle


class CatFountainConfig(BaseModel):
    """Configuration settings for the cat fountain."""

    measurements_path: Optional[str] = Field(
        default=None,
        description="Optional override for the measurements YAML file path.",
    )

    diagram_options: DiagramOptions = Field(
        default_factory=lambda: DiagramOptions(line_weight=1, view_from="iso", style=DiagramStyle.HIDDEN),
        description="Diagram export options",
    )

    material: str = Field(
        default="petg",
        description="The material name to use for the cat fountain parts (e.g. petg, pla, abs).",
    )

    target_volume: float = Field(
        default=0.0010,
        description="Target total fluid volume to spawn (m^3).",
    )

    stiffness_boundary: float = Field(default=1000.0, description="SPH boundary penalty stiffness coefficient")
    damping_boundary: Optional[float] = Field(default=None, description="SPH boundary penalty damping coefficient")

    # =========================================================================
    # Internal & Helper Properties
    # =========================================================================

    @cached_property
    def _raw_data(self) -> dict[Union[int, str], Any]:
        """Load and normalize raw measurements from the YAML file."""
        if self.measurements_path is None:
            raise ValueError("measurements_path is not set.")

        raw = load_measurements(cast(str, self.measurements_path))
        return {int(k) if isinstance(k, int) or (isinstance(k, str) and k.isdigit()) else k: v for k, v in raw.items()}

    @property
    def manifest_path(self) -> Path:
        """Return the path to the manifest configuration file."""
        if self.measurements_path is not None:
            return Path(self.measurements_path).parent / "manifest.yaml"
        return Path(__file__).parent.parent / "projects" / "cat_fountain" / "manifest.yaml"

    @cached_property
    def _material_data(self) -> dict[str, Any]:
        """Load and return materials data from the manifest."""
        from provider.utils import load_manifest

        path = self.manifest_path
        if path.exists():
            manifest = load_manifest(str(path))
        else:
            manifest = {}
        return manifest.get("material", {})

    # =========================================================================
    # Bowl Parameters
    # =========================================================================

    @cached_property
    def bowl_radius(self) -> float:
        """Return the bowl radius."""
        return float(self._raw_data.get("bowl_radius", 80.0))

    @cached_property
    def bowl_height(self) -> float:
        """Return the bowl height."""
        return float(self._raw_data.get("bowl_height", 40.0))

    @cached_property
    def bowl_thickness(self) -> float:
        """Return the bowl thickness."""
        return float(self._raw_data.get("bowl_thickness", 4.0))

    @cached_property
    def floor_z(self) -> float:
        """Return the dry compartment floor Z coordinate in millimeters."""
        return float(self._raw_data.get("floor_z", 36.0))

    # =========================================================================
    # Tube Parameters
    # =========================================================================

    @cached_property
    def tube_radius(self) -> float:
        """Return the tube outer radius."""
        return float(self._raw_data.get("tube_radius", 8.0))

    @cached_property
    def tube_thickness(self) -> float:
        """Return the tube wall thickness."""
        return float(self._raw_data.get("tube_thickness", 2.0))

    @cached_property
    def tube_height(self) -> float:
        """Return the tube height."""
        return float(self._raw_data.get("tube_height", 100.0))

    @cached_property
    def slot_height(self) -> float:
        """Return the vertical tube intake slot height in millimeters."""
        return float(self._raw_data.get("slot_height", 15.0))

    # =========================================================================
    # Impeller Parameters
    # =========================================================================

    @cached_property
    def impeller_radius(self) -> float:
        """Return the impeller outer radius."""
        return float(self._raw_data.get("impeller_radius", 12.0))

    @cached_property
    def impeller_height(self) -> float:
        """Return the impeller height."""
        return float(self._raw_data.get("impeller_height", 15.0))

    @cached_property
    def impeller_shaft_radius(self) -> float:
        """Return the impeller shaft hole radius."""
        return float(self._raw_data.get("impeller_shaft_radius", 2.5))

    @cached_property
    def impeller_shaft_clearance(self) -> float:
        """Return the radial clearance between the guide post and impeller sleeve bore."""
        return float(self._raw_data.get("impeller_shaft_clearance", 0.30))

    @cached_property
    def impeller_base_height(self) -> float:
        """Return the height of the impeller base disk."""
        return float(self._raw_data.get("impeller_base_height", 6.0))

    @cached_property
    def impeller_blades(self) -> int:
        """Return the number of impeller blades."""
        return int(self._raw_data.get("impeller_blades", 6))

    @cached_property
    def vane_twist(self) -> float:
        """Return the total twist angle of the impeller blades in degrees."""
        return float(self._raw_data.get("vane_twist", -1080.0))

    @cached_property
    def motor_target(self) -> float:
        """Return the target motor angular velocity in rad/s in the simulation."""
        return float(self._raw_data.get("motor_target", 120.0))

    @cached_property
    def motor_power(self) -> float:
        """Return the maximum motor output power in Watts."""
        return float(self._raw_data.get("motor_power", 30.0))

    @cached_property
    def motor_force_sim(self) -> float:
        """Return the maximum joint torque limit allowed in the physics simulator (N*m)."""
        return float(self._raw_data.get("motor_force_sim", 10.0))

    # =========================================================================
    # Spout Parameters
    # =========================================================================

    @cached_property
    def spout_length(self) -> float:
        """Return the spout length extension."""
        return float(self._raw_data.get("spout_length", 30.0))

    @cached_property
    def spout_angle(self) -> float:
        """Return the angle of the spout outlet."""
        return float(self._raw_data.get("spout_angle", 45.0))

    # =========================================================================
    # Material Parameters
    # =========================================================================

    @cached_property
    def petg_density(self) -> float:
        """Return the density of PETG material dynamically from manifest configuration."""
        return float(self._material_data.get("petg", {}).get("density", 1.27))

    @cached_property
    def petg_boundary_friction(self) -> float:
        """Return the boundary friction of PETG material dynamically from manifest configuration."""
        return float(self._material_data.get("petg", {}).get("boundary_friction", 0.0))

    @cached_property
    def petg_contact_angle(self) -> float:
        """Return the contact angle of PETG material dynamically from manifest configuration."""
        return float(self._material_data.get("petg", {}).get("contact_angle", 0.0))

    @property
    def density(self) -> float:
        """Return the density of the configured material dynamically from manifest configuration."""
        return float(self._material_data.get(self.material, {}).get("density", 0.0))

    @property
    def boundary_friction(self) -> float:
        """Return the boundary friction of the configured material dynamically from manifest configuration."""
        return float(self._material_data.get(self.material, {}).get("boundary_friction", 0.0))

    @property
    def contact_angle(self) -> float:
        """Return the contact angle of the configured material dynamically from manifest configuration."""
        return float(self._material_data.get(self.material, {}).get("contact_angle", 75.0))

    # =========================================================================
    # Lid & Spout Deflection Boundary Parameters
    # =========================================================================

    @cached_property
    def lid_pocket_radius(self) -> float:
        """Return the lid pocket radius in millimeters."""
        return float(self._raw_data.get("lid_pocket_radius", 80.0))

    @cached_property
    def lid_pocket_cavity_height(self) -> float:
        """Return the lid pocket cavity height in millimeters."""
        return float(self._raw_data.get("lid_pocket_cavity_height", 10.0))

    @cached_property
    def lid_pocket_thickness(self) -> float:
        """Return the lid pocket wall thickness in millimeters."""
        return float(self._raw_data.get("lid_pocket_thickness", 4.0))

    @cached_property
    def lid_pocket_z_offset(self) -> float:
        """Return the lid pocket Z offset in millimeters."""
        return float(self._raw_data.get("lid_pocket_z_offset", 3.0))

    @cached_property
    def drain_hole_y(self) -> float:
        """Return the drain hole Y coordinate in millimeters."""
        return float(self._raw_data.get("drain_hole_y", 65.0))

    @cached_property
    def drain_hole_radius(self) -> float:
        """Return the drain hole radius in millimeters."""
        return float(self._raw_data.get("drain_hole_radius", 15.0))

    @cached_property
    def spout_deflection_radius(self) -> float:
        """Return the spout deflection cap radius in millimeters."""
        return float(self._raw_data.get("spout_deflection_radius", 13.0))

    @cached_property
    def spout_deflection_height(self) -> float:
        """Return the spout deflection cap cavity height in millimeters."""
        return float(self._raw_data.get("spout_deflection_height", 0.0))

    @cached_property
    def spout_deflection_thickness(self) -> float:
        """Return the spout deflection cap thickness in millimeters."""
        return float(self._raw_data.get("spout_deflection_thickness", 40.0))

    @cached_property
    def spout_deflection_z_offset(self) -> float:
        """Return the spout deflection cap Z offset in millimeters."""
        return float(self._raw_data.get("spout_deflection_z_offset", 16.0))

    @cached_property
    def lid_platform_slope_angle(self) -> float:
        """Return the forward slope angle of the lid platform in degrees."""
        return float(self._raw_data.get("lid_platform_slope_angle", 2.5))

    # =========================================================================
    # Simulation Parameters
    # =========================================================================

    @cached_property
    def spawn_spacing_factor(self) -> float:
        """Return the spawn spacing factor."""
        return float(self._raw_data.get("spawn_spacing_factor", 2.2))

    @cached_property
    def z_spawn_buffer(self) -> float:
        """Return the Z spawn buffer."""
        return float(self._raw_data.get("z_spawn_buffer", 0.001))

    @cached_property
    def max_spawn_height(self) -> float:
        """Return the max spawn height."""
        return float(self._raw_data.get("max_spawn_height", 0.120))

    @cached_property
    def impeller_clearance_radius(self) -> float:
        """Return the impeller clearance radius."""
        return float(self._raw_data.get("impeller_clearance_radius", 0.015))

    @cached_property
    def spout_min_height(self) -> float:
        """Return the spout min height."""
        return float(self._raw_data.get("spout_min_height", 0.095))

    @cached_property
    def spout_max_y(self) -> float:
        """Return the spout max Y limit."""
        return float(self._raw_data.get("spout_max_y", 0.030))

    # =========================================================================
    # Refinement Clearances & Dimensions
    # =========================================================================

    @cached_property
    def lid_clearance(self) -> float:
        """Return the lid fitment clearance."""
        return float(self._raw_data.get("lid_clearance", 0.15))

    @cached_property
    def lid_step_depth(self) -> float:
        """Return the lid step depth."""
        return float(self._raw_data.get("lid_step_depth", 5.0))

    @cached_property
    def lid_step_width(self) -> float:
        """Return the lid step width."""
        return float(self._raw_data.get("lid_step_width", 2.0))

    @cached_property
    def lid_cutout_radius(self) -> float:
        """Return the radius of the large circular cutout in the lid in millimeters."""
        return float(self._raw_data.get("lid_cutout_radius", 55.0))

    @cached_property
    def lid_cutout_y(self) -> float:
        """Return the Y position of the large circular cutout in the lid in millimeters."""
        return float(self._raw_data.get("lid_cutout_y", -20.0))

    @cached_property
    def lid_cutout_ridge_width(self) -> float:
        """Return the width of the small ridge around the lid opening in millimeters."""
        return float(self._raw_data.get("lid_cutout_ridge_width", 1.5))

    @cached_property
    def lid_cutout_ridge_height(self) -> float:
        """Return the height of the small ridge around the lid opening in millimeters."""
        return float(self._raw_data.get("lid_cutout_ridge_height", 2.0))

    @cached_property
    def tube_lid_clearance(self) -> float:
        """Return the clearance between the tube and the lid socket."""
        return float(self._raw_data.get("tube_lid_clearance", 0.1))

    @cached_property
    def bottom_cover_clearance(self) -> float:
        """Return the bottom cover clearance."""
        return float(self._raw_data.get("bottom_cover_clearance", 0.2))

    @cached_property
    def impeller_clearance(self) -> float:
        """Return the radial clearance between impeller and tube."""
        return float(self._raw_data.get("impeller_clearance", 0.1))

    @cached_property
    def bottom_cover_drain_radius(self) -> float:
        """Return the bottom cover central drainage hole radius."""
        return float(self._raw_data.get("bottom_cover_drain_radius", 8.0))

    @cached_property
    def rubber_feet_count(self) -> int:
        """Return the number of rubber feet depressions on the bottom cover."""
        return int(self._raw_data.get("rubber_feet_count", 4))

    @cached_property
    def rubber_feet_radius(self) -> float:
        """Return the radius of the rubber feet depressions in millimeters."""
        return float(self._raw_data.get("rubber_feet_radius", 5.5))

    @cached_property
    def rubber_feet_depth(self) -> float:
        """Return the depth of the rubber feet depressions in millimeters."""
        return float(self._raw_data.get("rubber_feet_depth", 1.0))

    @cached_property
    def rubber_feet_pitch_radius(self) -> float:
        """Return the radial pitch distance from center for rubber feet depressions."""
        return float(self._raw_data.get("rubber_feet_pitch_radius", 78.0))

    @cached_property
    def pcb_hole_radius(self) -> float:
        """Return the mounting hole radius for PCBs."""
        return float(self._raw_data.get("pcb_hole_radius", 1.0))

    @cached_property
    def pcb_boss_radius(self) -> float:
        """Return the outer boss radius for PCB mounting posts."""
        return float(self._raw_data.get("pcb_boss_radius", 2.2))

    # Fuel Gauge
    @cached_property
    def fuel_gauge_spacing_x(self) -> float:
        """Return X hole spacing for the Fuel Gauge."""
        return float(self._raw_data.get("fuel_gauge_spacing_x", 20.0))

    @cached_property
    def fuel_gauge_spacing_y(self) -> float:
        """Return Y hole spacing for the Fuel Gauge."""
        return float(self._raw_data.get("fuel_gauge_spacing_y", 16.0))

    @cached_property
    def fuel_gauge_standoff_height(self) -> float:
        """Return standoff height for the Fuel Gauge."""
        return float(self._raw_data.get("fuel_gauge_standoff_height", 4.0))

    # Proximity Sensor
    @cached_property
    def proximity_sensor_spacing_x(self) -> float:
        """Return X hole spacing for the proximity sensor."""
        return float(self._raw_data.get("proximity_sensor_spacing_x", 21.0))

    @cached_property
    def proximity_sensor_spacing_y(self) -> float:
        """Return Y hole spacing for the proximity sensor."""
        return float(self._raw_data.get("proximity_sensor_spacing_y", 13.5))

    @cached_property
    def proximity_sensor_standoff_height(self) -> float:
        """Return standoff height for the proximity sensor."""
        return float(self._raw_data.get("proximity_sensor_standoff_height", 4.0))

    @cached_property
    def proximity_sensor_pocket_width(self) -> float:
        """Return pocket width for the proximity sensor."""
        return float(self._raw_data.get("proximity_sensor_pocket_width", 12.0))

    @cached_property
    def proximity_sensor_pocket_height(self) -> float:
        """Return pocket height for the proximity sensor."""
        return float(self._raw_data.get("proximity_sensor_pocket_height", 14.0))

    # Raspberry Pi Pico
    @cached_property
    def pico_spacing_x(self) -> float:
        """Return X hole spacing for Raspberry Pi Pico W."""
        return float(self._raw_data.get("pico_spacing_x", 13.0))

    @cached_property
    def pico_spacing_y(self) -> float:
        """Return Y hole spacing for Raspberry Pi Pico W."""
        return float(self._raw_data.get("pico_spacing_y", 48.0))

    @cached_property
    def pico_standoff_height(self) -> float:
        """Return standoff height for Raspberry Pi Pico W."""
        return float(self._raw_data.get("pico_standoff_height", 4.0))

    # Charger
    @cached_property
    def charger_spacing_x(self) -> float:
        """Return X hole spacing for Charger."""
        return float(self._raw_data.get("charger_spacing_x", 14.0))

    @cached_property
    def charger_spacing_y(self) -> float:
        """Return Y hole spacing for Charger."""
        return float(self._raw_data.get("charger_spacing_y", 25.0))

    @cached_property
    def charger_standoff_height(self) -> float:
        """Return standoff height for Charger."""
        return float(self._raw_data.get("charger_standoff_height", 2.0))

    @cached_property
    def charger_port_width(self) -> float:
        """Return charger port cutout width."""
        return float(self._raw_data.get("charger_port_width", 16.0))

    @cached_property
    def charger_port_height(self) -> float:
        """Return charger port cutout height."""
        return float(self._raw_data.get("charger_port_height", 9.0))

    # Current Monitor
    @cached_property
    def current_monitor_spacing_x(self) -> float:
        """Return X hole spacing for current monitor."""
        return float(self._raw_data.get("current_monitor_spacing_x", 21.0))

    @cached_property
    def current_monitor_spacing_y(self) -> float:
        """Return Y hole spacing for current monitor."""
        return float(self._raw_data.get("current_monitor_spacing_y", 17.5))

    @cached_property
    def current_monitor_standoff_height(self) -> float:
        """Return standoff height for current monitor."""
        return float(self._raw_data.get("current_monitor_standoff_height", 4.0))

    # Motor Driver
    @cached_property
    def motor_driver_spacing_x(self) -> float:
        """Return X hole spacing for motor driver."""
        return float(self._raw_data.get("motor_driver_spacing_x", 20.0))

    @cached_property
    def motor_driver_spacing_y(self) -> float:
        """Return Y hole spacing for motor driver."""
        return float(self._raw_data.get("motor_driver_spacing_y", 15.0))

    @cached_property
    def motor_driver_standoff_height(self) -> float:
        """Return standoff height for motor driver."""
        return float(self._raw_data.get("motor_driver_standoff_height", 4.0))

    # Motor mount
    @cached_property
    def motor_spacing_x(self) -> float:
        """Return mounting screw spacing for N20 Motor."""
        return float(self._raw_data.get("motor_spacing_x", 17.0))

    @cached_property
    def motor_set_screw_angle(self) -> float:
        """Return the angle of the set screw relative to the Z axis."""
        return float(self._raw_data.get("motor_set_screw_angle", 45.0))

    @cached_property
    def motor_set_screw_x_offset(self) -> float:
        """Return the X start position of the set screw."""
        return float(self._raw_data.get("motor_set_screw_x_offset", 10.5))

    @cached_property
    def motor_set_screw_radius(self) -> float:
        """Return the radius of the set screw hole."""
        return float(self._raw_data.get("motor_set_screw_radius", 1.75))

    @cached_property
    def motor_set_screw_length(self) -> float:
        """Return the length of the set screw hole."""
        return float(self._raw_data.get("motor_set_screw_length", 24.0))

    # LED Button / hole
    @cached_property
    def led_hole_width(self) -> float:
        """Return the side width of the square LED hole."""
        return float(self._raw_data.get("led_hole_width", 5.0))

    @cached_property
    def led_plug_width(self) -> float:
        """Return the side width of the LED plug insert."""
        return float(self._raw_data.get("led_plug_width", 4.8))

    @cached_property
    def led_flange_width(self) -> float:
        """Return the side width of the LED cover flange."""
        return float(self._raw_data.get("led_flange_width", 7.0))

    @cached_property
    def led_flange_thickness(self) -> float:
        """Return the thickness of the LED cover flange."""
        return float(self._raw_data.get("led_flange_thickness", 1.0))

    @cached_property
    def led_plug_length(self) -> float:
        """Return the length/depth of the LED plug insert."""
        return float(self._raw_data.get("led_plug_length", 3.0))

    # Sensor Port Boss
    @cached_property
    def sensor_boss_x(self) -> float:
        """Return X size of sensor outer boss."""
        return float(self._raw_data.get("sensor_boss_x", 12.0))

    @cached_property
    def sensor_boss_y(self) -> float:
        """Return Y size of sensor outer boss."""
        return float(self._raw_data.get("sensor_boss_y", 14.0))

    @cached_property
    def sensor_boss_z(self) -> float:
        """Return Z size of sensor outer boss."""
        return float(self._raw_data.get("sensor_boss_z", 14.0))

    # Central Boss & Shaft Hole parameters
    @cached_property
    def boss_body_radius_offset(self) -> float:
        """Return the radius offset for the boss body."""
        return float(self._raw_data.get("boss_body_radius_offset", 3.5))

    @cached_property
    def boss_body_height(self) -> float:
        """Return the height of the boss body."""
        return float(self._raw_data.get("boss_body_height", 2.0))

    @cached_property
    def motor_boss_height(self) -> float:
        """Return the total height of the motor mounting boss."""
        return float(self._raw_data.get("motor_boss_height", 20.0))

    @cached_property
    def motor_collar_clearance_radius(self) -> float:
        """Return the motor collar clearance hole radius."""
        return float(self._raw_data.get("motor_collar_clearance_radius", 3.2))

    @cached_property
    def motor_collar_clearance_height(self) -> float:
        """Return the motor collar clearance hole height."""
        return float(self._raw_data.get("motor_collar_clearance_height", 2.0))

    @cached_property
    def shaft_hole_radius_offset(self) -> float:
        """Return the shaft hole radius offset."""
        return float(self._raw_data.get("shaft_hole_radius_offset", 0.1))

    @cached_property
    def shaft_hole_height(self) -> float:
        """Return the shaft hole height."""
        return float(self._raw_data.get("shaft_hole_height", 1.8))

    @cached_property
    def oring_groove_radius_offset(self) -> float:
        """Return the O-ring groove radius offset."""
        return float(self._raw_data.get("oring_groove_radius_offset", 1.3))

    @cached_property
    def oring_groove_height(self) -> float:
        """Return the O-ring groove height."""
        return float(self._raw_data.get("oring_groove_height", 2.2))

    # Impeller sleeve
    @cached_property
    def impeller_sleeve_height(self) -> float:
        """Return the impeller sleeve height."""
        return float(self._raw_data.get("impeller_sleeve_height", 4.0))

    # Snap-fit Lid
    @cached_property
    def snap_ridge_z(self) -> float:
        """Return the Z position offset for the snap ridge."""
        return float(self._raw_data.get("snap_ridge_z", 2.5))

    @cached_property
    def snap_ridge_major_offset(self) -> float:
        """Return the major radius offset for the snap ridge Torus."""
        return float(self._raw_data.get("snap_ridge_major_offset", -0.1))

    @cached_property
    def snap_ridge_minor(self) -> float:
        """Return the minor radius for the snap ridge Torus."""
        return float(self._raw_data.get("snap_ridge_minor", 0.5))

    @cached_property
    def snap_groove_minor(self) -> float:
        """Return the minor radius for the snap groove Torus."""
        return float(self._raw_data.get("snap_groove_minor", 0.6))

    # Drain Shroud & Trimmer
    @cached_property
    def drain_shroud_radius(self) -> float:
        """Return the radius of the drain shroud."""
        return float(self._raw_data.get("drain_shroud_radius", 18.0))

    @cached_property
    def drain_shroud_height(self) -> float:
        """Return the height of the drain shroud."""
        return float(self._raw_data.get("drain_shroud_height", 15.0))

    @cached_property
    def drain_shroud_z(self) -> float:
        """Return the Z offset position for the drain shroud."""
        return float(self._raw_data.get("drain_shroud_z", -15.0))

    @cached_property
    def trimmer_height(self) -> float:
        """Return the height of the trimmer cylinder."""
        return float(self._raw_data.get("trimmer_height", 30.0))

    @cached_property
    def retention_boss_height(self) -> float:
        """Return the height of the retention boss limiting vertical travel."""
        return float(self._raw_data.get("retention_boss_height", 1.0))

    # Magnetic Centrifugal Pump Redesign
    @cached_property
    def magnet_radius(self) -> float:
        """Return the radius of the N52 coupling magnets."""
        return float(self._raw_data.get("magnet_radius", 3.0))

    @cached_property
    def magnet_thickness(self) -> float:
        """Return the thickness of the N52 coupling magnets."""
        return float(self._raw_data.get("magnet_thickness", 3.0))

    @cached_property
    def magnet_clearance(self) -> float:
        """Return the pocket clearance for inserting the coupling magnets."""
        return float(self._raw_data.get("magnet_clearance", 0.1))

    @cached_property
    def magnet_count(self) -> int:
        """Return the number of coupling magnets in the drive and rotor rings."""
        return int(self._raw_data.get("magnet_count", 4))

    @cached_property
    def magnet_ring_radius(self) -> float:
        """Return the radial distance of the magnets from the center of rotation."""
        return float(self._raw_data.get("magnet_ring_radius", 12.0))

    @cached_property
    def pump_well_depth(self) -> float:
        """Return the depth of the central isolation pump well."""
        return float(self._raw_data.get("pump_well_depth", 10.0))

    @cached_property
    def pump_well_wall(self) -> float:
        """Return the wall thickness of the central isolation pump well."""
        return float(self._raw_data.get("pump_well_wall", 1.5))

    @cached_property
    def pump_inlet_radius(self) -> float:
        """Return the radius of the axial water inlet."""
        return float(self._raw_data.get("pump_inlet_radius", 6.0))

    @cached_property
    def pump_inlet_width(self) -> float:
        """Return the width of the pump cover side water intake slot."""
        return float(self._raw_data.get("pump_inlet_width", 12.0))

    @cached_property
    def pump_inlet_height(self) -> float:
        """Return the height of the pump cover side water intake slot."""
        return float(self._raw_data.get("pump_inlet_height", 4.0))

    @cached_property
    def pump_cover_height(self) -> float:
        """Return the height of the pump cover cap over the casing."""
        return float(self._raw_data.get("pump_cover_height", 4.0))

    @cached_property
    def pump_cover_sleeve_height(self) -> float:
        """Return the height of the pump cover tube sleeve."""
        return float(self._raw_data.get("pump_cover_sleeve_height", 8.0))

    @cached_property
    def pump_cover_tube_clearance(self) -> float:
        """Return the clearance for the pump cover sleeve sliding over the tube."""
        return float(self._raw_data.get("pump_cover_tube_clearance", 0.35))

    @cached_property
    def pump_cover_snap_clearance(self) -> float:
        """Return the radial clearance for the pump cover downward lip entering the casing recess."""
        return float(self._raw_data.get("pump_cover_snap_clearance", 0.45))

    @cached_property
    def pump_cover_lip_thickness(self) -> float:
        """Return the thickness of the pump cover downward lip."""
        return float(self._raw_data.get("pump_cover_lip_thickness", 1.5))

    @cached_property
    def pump_cover_internal_clearance(self) -> float:
        """Return the extra radial clearance inside the pump cover over the volute chamber."""
        return float(self._raw_data.get("pump_cover_internal_clearance", 0.40))

    @cached_property
    def pump_cover_clip_opening_width(self) -> float:
        """Return the width of the North-facing snap opening in the pump cover tube sleeve."""
        return float(self._raw_data.get("pump_cover_clip_opening_width", 14.8))

    @cached_property
    def pump_outlet_radius(self) -> float:
        """Return the inner radius of the tangential water outlet nozzle."""
        return float(self._raw_data.get("pump_outlet_radius", 4.0))

    @cached_property
    def pump_casing_clearance(self) -> float:
        """Return the tight clearance inside the scroll casing around the impeller."""
        return float(self._raw_data.get("pump_casing_clearance", 1.0))

    @cached_property
    def drive_hub_shaft_radius(self) -> float:
        """Return the motor shaft hole radius on the drive hub."""
        return float(self._raw_data.get("drive_hub_shaft_radius", 0.85))

    @cached_property
    def drive_hub_recess_radius(self) -> float:
        """Return the radius of the bowl's recess for the drive hub."""
        return float(self._raw_data.get("drive_hub_recess_radius", 17.0))

    @cached_property
    def drive_hub_recess_depth(self) -> float:
        """Return the depth of the bowl's recess for the drive hub."""
        return float(self._raw_data.get("drive_hub_recess_depth", 5.3))

    @cached_property
    def drive_hub_standoff_radius(self) -> float:
        """Return the outer radius of the drive hub standoff thrust ring."""
        return float(self._raw_data.get("drive_hub_standoff_radius", 2.5))

    @cached_property
    def drive_hub_standoff_height(self) -> float:
        """Return the height of the drive hub standoff thrust ring."""
        return float(self._raw_data.get("drive_hub_standoff_height", 0.8))

    @cached_property
    def bottom_cover_opening_width(self) -> float:
        """Return the width of the bottom cover side opening."""
        return float(self._raw_data.get("bottom_cover_opening_width", 20.0))

    @cached_property
    def motor_clip_width(self) -> float:
        """Return the width of the motor retaining clip."""
        return float(self._raw_data.get("motor_clip_width", 17.6))

    @cached_property
    def motor_clip_thickness(self) -> float:
        """Return the thickness of the motor retaining clip."""
        return float(self._raw_data.get("motor_clip_thickness", 1.8))

    @cached_property
    def motor_clip_length(self) -> float:
        """Return the length of the motor retaining clip."""
        return float(self._raw_data.get("motor_clip_length", 26.0))

    @cached_property
    def motor_clip_cutout_width(self) -> float:
        """Return the cutout width of the motor retaining clip U-fork."""
        return float(self._raw_data.get("motor_clip_cutout_width", 14.2))

    def __init__(self, **data: Any):
        """Initialize settings and calculate physical defaults."""
        super().__init__(**data)
        if self.damping_boundary is None:
            r_s = 0.0015
            spacing = 1.3 * r_s
            mass = 1000.0 * (spacing**3)
            dt_sub = 1.0 / 1200.0
            safety_factor = 225.0
            self.damping_boundary = float((safety_factor * mass) / dt_sub)
