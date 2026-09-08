"""Cat fountain geometry provider."""

from build123d import *  # type: ignore
import math
from model import (
    method_cache,
    TextArgs,
    ShapeType,
    BoundaryType,
    PinModel,
    LabelModel,
    FootprintModel,
    NetModel,
    Wiring,
    DiagramStyle,
)
from pathlib import Path
from provider import (
    Provider,
    Section,
    Mode as ProviderMode,
    discover_provider,
    Room,
    Simulate,
    URDFBoundary,
    URDFMetadata,
    URDFShape,
    URDFCollisionType,
    URDFBoundaryType,
    URDFJointType,
    URDFMotorType,
    IntakePort,
    DrainPort,
    TubePort,
    FlowSlot,
    SpoutDeflection,
    ImpellerVanes,
    MagneticCoupling,
    LinkType,
    WiringDiagram,
)
from projects_config import CatFountainConfig
from . import layouts
from typing import cast, Callable, Sequence, Any, Optional


@discover_provider
class CatFountainProvider(Provider):
    """Provider for cat fountain geometry."""

    # Slot cutter constants (treating length as an infinite cutting plane)
    SLOT_WIDTH = 2.0
    SLOT_LENGTH = 100.0

    water_sim: Optional[Any] = None

    @property
    def default_config(self) -> CatFountainConfig:
        """Return the default configuration for the cat fountain project."""
        if not hasattr(self, "_cached_default_config"):
            self._cached_default_config = CatFountainConfig(
                measurements_path=str(Path(__file__).parent / "measurements.yaml")
            )
        return self._cached_default_config

    @property
    def settings(self) -> CatFountainConfig:
        """Return the typed configuration settings."""
        return cast(CatFountainConfig, super().settings)

    @property
    def part(self) -> dict[str, Callable[..., BuildPart]]:
        """Map part names to their build handler methods."""
        return {
            "bowl": self.build_bowl,
            "impeller": self.build_impeller,
            "bottom_cover": self.build_bottom_cover,
            "lid": self.build_lid,
            "led_cover": self.build_led_cover,
            "drive_hub": self.build_drive_hub,
            "pump_cover": self.build_pump_cover,
            "motor_clip": self.build_motor_clip,
        }

    @property
    def diagram(self) -> dict[str, Callable[[Room, Sequence[str], ProviderMode], None]]:
        """Map diagram names to their build handler methods."""
        return {
            name: (self.build_wiring_diagram if "wiring" in name else self.build_diagram)
            for name in self.targets.supporting(Section.DIAGRAM)
        }

    @property
    def view(self) -> dict[str, Callable[[Room, ProviderMode], None]]:
        """Map room names to view functions."""
        return {
            "product": self.build_product,
        }

    @method_cache
    def build_bowl(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the cat fountain bowl with integrated 2L reservoir, integrated tube, motor compartment, and screw mounts."""
        r = self.settings.bowl_radius
        h = self.settings.bowl_height
        t = self.settings.bowl_thickness
        pin_r = self.settings.impeller_shaft_radius
        floor_z = self.settings.floor_z
        tube_y = 0.0

        # PCB settings
        hole_r = self.settings.pcb_hole_radius
        boss_r = self.settings.pcb_boss_radius

        with BuildPart() as bowl:
            # Outer bowl body
            Cylinder(radius=r, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Create the global outer cylinder for trimming standoffs (oversized to prevent coincident face mesh artifacts)
            outer_cylinder = Cylinder(
                radius=r + 1.0, height=h, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.PRIVATE
            )

            # Subtract inner water reservoir (enclosed storage tank area above floor_z)
            # Make a step at the top rim of the bowl inner wall for the lid seat
            with Locations((0, 0, floor_z)):
                reservoir_shape = Cylinder(
                    radius=r - t,
                    height=h - floor_z - self.settings.lid_step_depth,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
            # Step recess at the top rim
            with Locations((0, 0, h - self.settings.lid_step_depth)):
                Cylinder(
                    radius=r - t + self.settings.lid_step_width,
                    height=self.settings.lid_step_depth + 2.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
            # Annular groove for lid snap fit (scaling with lid radius)
            groove_major = r - t + self.settings.lid_step_width - self.settings.lid_clearance
            with Locations((0, 0, h - 2.5)):
                Torus(major_radius=groove_major, minor_radius=0.6, mode=Mode.SUBTRACT)

            # Subtract dry motor controller compartment under the floor
            with Locations((0, 0, 0)):
                Cylinder(
                    radius=r - t, height=floor_z - t, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT
                )

            # Subtract a matching snap-fit groove on the inner wall of the dry compartment
            # Located to align with the bottom cover's snap ring (Z = 1.1 on cover of height 4.0)
            with Locations((0, 0, 1.1)):
                Cylinder(
                    radius=r - t + 0.6,
                    height=1.2,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # --- INTEGRATED TUBE & CENTRIFUGAL VOLUTE CASING ---
            tube_x = 0.0
            tube_y = 28.0
            tube_r = self.settings.tube_radius
            tube_t = self.settings.tube_thickness
            tube_in_r = tube_r - tube_t

            # Dynamically calculate casing dimensions based on impeller settings
            hub_r = self.settings.impeller_radius + self.settings.magnet_radius + 1.0
            impeller_r = hub_r + 4.0
            chamber_r = impeller_r + self.settings.pump_casing_clearance
            casing_r = chamber_r + self.settings.pump_well_wall
            snap_r = chamber_r + 1.0

            # 1. Casing & Tube outer solid geometries
            with Locations((0, 0, floor_z)):
                # Volute casing outer cylinder
                Cylinder(radius=casing_r, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
            with Locations((tube_x, tube_y, floor_z)):
                # Integrated vertical tube outer cylinder
                Cylinder(radius=tube_r, height=self.settings.tube_height, align=(Align.CENTER, Align.CENTER, Align.MIN))
            # Solid connector block for the tangential flow nozzle (running along Y)
            with Locations((0, 14.0, floor_z)):
                Box(tube_r * 2.0, 28.0, 10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # 2. Subtract inner cavities
            with Locations((0, 0, floor_z)):
                # Volute impeller chamber
                Cylinder(
                    radius=chamber_r, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT
                )
                # Snap fit recess for pump cover at the top rim
                with Locations((0, 0, 8.5)):
                    Cylinder(
                        radius=snap_r, height=1.5, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT
                    )
            with Locations((tube_x, tube_y, floor_z + 2.0)):
                # Tube inner cavity (starting 2mm above floor to create a bottom pocket)
                Cylinder(
                    radius=tube_in_r,
                    height=self.settings.tube_height + 2.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
            # Rectangular flow channel between casing and tube (Z = floor_z + 1.0 to floor_z + 9.0)
            with Locations((0, 14.0, floor_z + 1.0)):
                Box(8.0, 28.0, 8.0, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT)

            # Central solid boss and guide post for the isolated impeller
            with Locations((0, 0, floor_z)):
                # Static guide post (radius 2.5mm, height 12.0mm)
                Cylinder(radius=pin_r + 1.5, height=2.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                Cylinder(radius=pin_r, height=12.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Motor mounting boss projecting down from the bowl floor (Z = floor_z)
            # This creates a solid 1.5mm barrier floor at the center of rotation
            with Locations((0, 0, floor_z)):
                # Outer boss body (increased radius to 21.0mm to provide full support under the impeller casing)
                boss_h = self.settings.motor_boss_height
                Cylinder(radius=21.0, height=boss_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

                # Local boss extension to house the speed sensor pocket (North side, Y = 21.0)
                with Locations((0, 21.0, 0)):
                    Box(8.0, 6.0, boss_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

                # 1. Recess for the drive hub
                # Formed as a U-shaped slot open to the South side (Y < 0) for horizontal slide-in assembly
                recess_r = self.settings.drive_hub_recess_radius
                recess_d = self.settings.drive_hub_recess_depth
                with Locations((0, 0, -self.settings.pump_well_wall)):
                    Cylinder(
                        radius=recess_r,
                        height=recess_d,
                        align=(Align.CENTER, Align.CENTER, Align.MAX),
                        mode=Mode.SUBTRACT,
                    )
                    # Subtract slot extension along Y < 0
                    with Locations((0, -recess_r, 0)):
                        Box(
                            recess_r * 2.0,
                            recess_r * 2.0,
                            recess_d,
                            align=(Align.CENTER, Align.CENTER, Align.MAX),
                            mode=Mode.SUBTRACT,
                        )

                # Speed sensor pocket (5.0mm wide, 2.2mm thick, open from Z = -3.0 down through bottom)
                with Locations((0, 18.5, -3.0)):
                    Box(5.0, 2.2, 20.0, align=(Align.CENTER, Align.CENTER, Align.MAX), mode=Mode.SUBTRACT)

                # 2. Pocket for the BetaFPV 1102 motor body (open from motor_pocket_z down through bottom)
                motor_pocket_z = -(self.settings.pump_well_wall + recess_d)
                motor_pocket_h = boss_h - (self.settings.pump_well_wall + recess_d) + 5.0
                with Locations((0, 0, motor_pocket_z)):
                    # Motor body pocket (radius 7.0mm for 13.8mm diameter, breaking through bottom)
                    Cylinder(
                        radius=7.0,
                        height=motor_pocket_h,
                        align=(Align.CENTER, Align.CENTER, Align.MAX),
                        mode=Mode.SUBTRACT,
                    )
                    # Motor front face alignment boss pocket (radius 2.75mm, height 1.6mm)
                    Cylinder(radius=2.75, height=1.6, align=(Align.CENTER, Align.CENTER, Align.MAX), mode=Mode.SUBTRACT)

                # 3. Horizontal slide-in slot for the motor retaining clip (Z = 23.5 to 25.5 mm, open to Y < 0)
                clip_slot_w = self.settings.motor_clip_width + 0.4
                clip_slot_h = self.settings.motor_clip_thickness + 0.2
                # Shifted slot center to Y = 5.0 and length to 41.0 to accommodate the clip's 5.0mm front projection
                with Locations((0, 5.0, -17.5)):
                    Box(clip_slot_w, 41.0, clip_slot_h, align=(Align.CENTER, Align.MAX, Align.MIN), mode=Mode.SUBTRACT)

            # Helper to create cylindrical standoff posts with blind holes
            def add_standoffs(
                locations: list[Location],
                boss_radius: float,
                standoff_height: float,
                hole_radius: float,
                hole_depth: float,
                boss_align: tuple[Align, Align, Align] = (Align.CENTER, Align.CENTER, Align.MAX),
                hole_align: tuple[Align, Align, Align] = (Align.CENTER, Align.CENTER, Align.MIN),
                hole_z_offset: float = 0.0,
            ):
                for loc in locations:
                    with Locations(loc):
                        Cylinder(radius=boss_radius, height=standoff_height, align=boss_align)
                        with Locations((0, 0, hole_z_offset)):
                            Cylinder(
                                radius=hole_radius,
                                height=hole_depth,
                                align=hole_align,
                                mode=Mode.SUBTRACT,
                            )

            # Helper function to generate a 3D-printed two-sided slide-in card guide
            def add_pcb_clip(
                center_x: float,
                center_y: float,
                board_w: float,  # Width along Y (20.3)
                board_l: float,  # Length along X (25.4)
                standoff_h: float,
                label: str,
            ):
                with Locations((center_x, center_y, floor_z - t)):
                    # 1. Base plate connecting the two guide tracks (recessed to save material and provide airflow)
                    Box(board_l - 4.0, board_w - 2.0, standoff_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

                    # 2. Left and Right guide tracks (at X = +/- board_l/2)
                    track_w = board_w
                    track_l = 3.0
                    track_h = standoff_h + 3.2

                    with Locations((-board_l / 2.0 - 0.75, 0, 0), (board_l / 2.0 + 0.75, 0, 0)):
                        Box(track_l, track_w, track_h, align=(Align.CENTER, Align.CENTER, Align.MAX))

                    # 3. Subtract the slot for the PCB inside the left/right tracks (1.9mm height)
                    with Locations((0, 0, -standoff_h)):
                        with Locations((-board_l / 2.0 - 0.15, 0, 0), (board_l / 2.0 + 0.15, 0, 0)):
                            Box(
                                2.0,
                                board_w + 0.3,
                                1.9,
                                align=(Align.CENTER, Align.CENTER, Align.MAX),
                                mode=Mode.SUBTRACT,
                            )

                # Add engraved label
                label_y_offset = -(board_w / 2.0 + 5.0)
                with BuildSketch() as label_sketch:
                    Text(label, font_size=4.5, align=(Align.CENTER, Align.CENTER))
                mirrored_sketch = label_sketch.sketch.mirror(Plane.YZ)
                ext_text = extrude(mirrored_sketch, amount=1.5, mode=Mode.PRIVATE)
                bowl.part -= Location((center_x, center_y + label_y_offset, floor_z - t)) * ext_text

            # Helper function to generate standoff posts and blind screw holes on the ceiling
            def add_pcb_mount(
                center_x: float,
                center_y: float,
                spacing_x: float,
                spacing_y: Optional[float],
                standoff_h: float,
                label: str,
            ):
                dxs = [-spacing_x / 2.0, spacing_x / 2.0]
                dys = [-spacing_y / 2.0, spacing_y / 2.0] if spacing_y else [0.0]

                locs = []
                for dx in dxs:
                    for dy in dys:
                        locs.append(Location((center_x + dx, center_y + dy, floor_z - t)))

                add_standoffs(
                    locations=locs,
                    boss_radius=boss_r,
                    standoff_height=standoff_h,
                    hole_radius=hole_r,
                    hole_depth=standoff_h + 2.5,
                    boss_align=(Align.CENTER, Align.CENTER, Align.MAX),
                    hole_align=(Align.CENTER, Align.CENTER, Align.MIN),
                    hole_z_offset=-standoff_h,
                )

                label_y_offset = (spacing_y / 2.0 + 5.0) if spacing_y is not None else 8.0
                if center_y + label_y_offset > r - t - 10.0:
                    label_y_offset = -label_y_offset
                with BuildSketch() as label_sketch:
                    Text(label, font_size=4.5, align=(Align.CENTER, Align.CENTER))
                # Mirror the sketch horizontally so it reads correctly from below (looking up)
                mirrored_sketch = label_sketch.sketch.mirror(Plane.YZ)
                ext_text = extrude(mirrored_sketch, amount=1.5, mode=Mode.PRIVATE)
                bowl.part -= Location((center_x, center_y + label_y_offset, floor_z - t)) * ext_text  # type: ignore

            # Mount all PCBs in the dry compartment (standard boards use standoffs, TMC6300 uses slide-in guide)
            pcb_mounts = [
                (
                    50.0,
                    -45.0,
                    self.settings.fuel_gauge_spacing_x,
                    self.settings.fuel_gauge_spacing_y,
                    self.settings.fuel_gauge_standoff_height,
                    "FUEL",
                    "standoff",
                ),
                (
                    -50.0,
                    0.0,
                    self.settings.pico_spacing_x,
                    self.settings.pico_spacing_y,
                    self.settings.pico_standoff_height,
                    "PICO W",
                    "standoff",
                ),
                (
                    0.0,
                    -79.0,
                    self.settings.charger_spacing_x,
                    self.settings.charger_spacing_y,
                    self.settings.charger_standoff_height,
                    "CHARGER",
                    "standoff",
                ),
                (
                    50.0,
                    -15.0,
                    self.settings.current_monitor_spacing_x,
                    self.settings.current_monitor_spacing_y,
                    self.settings.current_monitor_standoff_height,
                    "CURRENT",
                    "standoff",
                ),
                (
                    50.0,
                    15.0,
                    self.settings.motor_driver_spacing_x,
                    self.settings.motor_driver_spacing_y,
                    self.settings.motor_driver_standoff_height,
                    "TMC6300",
                    "clip",
                ),
            ]
            for cx, cy, sx, sy, sh, lbl, mtype in pcb_mounts:
                if mtype == "clip":
                    # For the clip bracket, sx (spacing_x) is board length, sy (spacing_y) is board width
                    add_pcb_clip(cx, cy, sy, sx, sh, lbl)
                else:
                    add_pcb_mount(cx, cy, sx, sy, sh, lbl)

            # Charging port hole in the outer wall of the dry compartment (back side, y = -r)
            # Centered on the USB port, which is located on the underside of the PCB
            usb_z = floor_z - t - self.settings.charger_standoff_height - 3.2
            with Locations((0, -r + t / 2.0, usb_z)):
                with BuildPart(mode=Mode.PRIVATE) as port_cutout:
                    Box(
                        self.settings.charger_port_width,
                        10.0,
                        self.settings.charger_port_height,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER),
                    )
                    fillet(port_cutout.edges().filter_by(Axis.Y), radius=2.0)
                add(port_cutout, mode=Mode.SUBTRACT)

            # Radial ventilation slits in the back wall around the charging port
            for angle in [252, 261, 279, 288]:
                with Locations(Rot(0, 0, angle)):
                    with Locations((r - t / 2.0, 0, 10.0)):
                        Box(t + 4.0, 2.5, 12.0, align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

            # Cutout for a square RGB LED on the front-right side (at angle 75, z = 8.0)
            with Locations(Rot(0, 0, 75.0)):
                with Locations((r - t / 2.0, 0, 8.0)):
                    Box(
                        10.0,
                        self.settings.led_hole_width,
                        self.settings.led_hole_width,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER),
                        mode=Mode.SUBTRACT,
                    )

            # Proximity sensor mounts / cutouts at East (0), North (90), West (180)
            boss_sx = self.settings.sensor_boss_x
            boss_sy = 28.0  # Increased from 14.0 to cover the 21.0mm sensor spacing and standoffs
            boss_sz = self.settings.sensor_boss_z
            tof_spacing_x = self.settings.proximity_sensor_spacing_x
            tof_spacing_y = self.settings.proximity_sensor_spacing_y
            tof_standoff = self.settings.proximity_sensor_standoff_height

            for s_angle in [0.0, 90.0, 180.0]:
                with Locations(Rot(0, 0, s_angle)):
                    with Locations(Location((r - t / 2.0 - 3.0, 0, 16.0), (0, -30, 0))):
                        # Outer flat sensor cover boss with rounded corners (extended vertically downwards)
                        with Locations(Location((11.0, 0, -2.0))):
                            with BuildPart(mode=Mode.PRIVATE) as boss_part:
                                Box(boss_sx, boss_sy, boss_sz, align=(Align.MAX, Align.CENTER, Align.CENTER))
                                fillet(boss_part.edges(), radius=2.0)
                            add(boss_part)

                        # Subtract the sensor pocket (extended vertically downwards)
                        with Locations(Location((11.0, 0, -2.0))):
                            Box(
                                boss_sx + 7.0,
                                self.settings.proximity_sensor_pocket_width,
                                self.settings.proximity_sensor_pocket_height,
                                align=(Align.MAX, Align.CENTER, Align.CENTER),
                                mode=Mode.SUBTRACT,
                            )

                        # Flat mounting standoff posts on the INSIDE of the bowl wall
                        # Build in a private block in local coordinates
                        with BuildPart(mode=Mode.PRIVATE) as standoffs_part:
                            # Bottom standoffs (dz = -tof_spacing_y / 2.0)
                            locs_bottom = []
                            for dy in [-tof_spacing_x / 2.0, tof_spacing_x / 2.0]:
                                dz = -tof_spacing_y / 2.0
                                locs_bottom.append(Location((-4.0, dy, dz), (0, 90, 0)))

                            add_standoffs(
                                locations=locs_bottom,
                                boss_radius=boss_r,
                                standoff_height=tof_standoff,
                                hole_radius=hole_r,
                                hole_depth=7.0,
                                boss_align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                hole_align=(Align.CENTER, Align.CENTER, Align.CENTER),
                            )

                            # Top standoffs (dz = tof_spacing_y / 2.0) extended to merge with the wall
                            locs_top = []
                            for dy in [-tof_spacing_x / 2.0, tof_spacing_x / 2.0]:
                                dz = tof_spacing_y / 2.0
                                locs_top.append(Location((-4.0, dy, dz), (0, 90, 0)) * Location((0, 0, 5.0)))

                            add_standoffs(
                                locations=locs_top,
                                boss_radius=boss_r,
                                standoff_height=14.0,  # 4.0 original + 10.0 extension towards wall
                                hole_radius=hole_r,
                                hole_depth=7.0,
                                boss_align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                hole_align=(Align.CENTER, Align.CENTER, Align.CENTER),
                                hole_z_offset=-5.0,  # Shift hole back to original centering
                            )

                        # Transform local standoffs to global coordinates using active locations
                        from build123d import LocationList

                        combined_loc = Location()
                        ctx = LocationList._get_context()
                        if ctx is not None:
                            for loc_list in ctx.locations:
                                combined_loc = combined_loc * loc_list

                        if standoffs_part.part is not None:
                            global_standoffs = combined_loc * standoffs_part.part

                            # Trim any protrusions extending past the outer cylinder
                            trimmed_standoffs = global_standoffs.intersect(outer_cylinder)

                            # Add trimmed standoffs directly to the bowl part (bypassing locations list)
                            if bowl.part is not None:
                                merged_shape = bowl.part + trimmed_standoffs
                                # Ensure final returned shape is a Part object, wrapping the internal Solid
                                bowl.part = Part(children=merged_shape.solids(), label=target)

            with URDFMetadata(
                label=target,
                material=self.get_material(target) or "utr8100",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.ANALYTICAL,
            ):
                with URDFBoundary.from_shape(
                    reservoir_shape,
                    link_type=LinkType.BASE,
                    shape=ShapeType.CYLINDER,
                    type=BoundaryType.CAVITY,
                    thickness=t * 0.001,
                ):
                    TubePort(location=Location((0.0, tube_y, 0.0)), radius=tube_in_r * 0.001)

                with Locations((0.0, 28.0, floor_z)):
                    tube_geom = Cylinder(
                        radius=self.settings.tube_radius,
                        height=self.settings.tube_height,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.PRIVATE,
                    )
                with URDFBoundary.from_shape(
                    tube_geom,
                    link_type=LinkType.TUBE,
                    shape=ShapeType.TUBE,
                    type=BoundaryType.SOLID_CAVITY,
                    thickness=self.settings.tube_thickness * 0.001,
                ):
                    IntakePort(location=Location((0.0, 0.0, 0.0), (0, -90, 0)), radius=tube_in_r * 0.001)
                    DrainPort(
                        location=Location((0.0, 0.0, self.settings.tube_height), (0, 0, 0)),
                        radius=tube_in_r * 0.001,
                    )
                    FlowSlot(height=9.0 * 0.001, width=8.0 * 0.001)
                    SpoutDeflection(
                        radius=(self.settings.spout_deflection_radius + 1.0) * 0.001,
                        height=(self.settings.spout_deflection_thickness + 9.0) * 0.001,
                    )

                casing_thick = (casing_r - chamber_r) * 0.001
                with Locations((0.0, 0.0, floor_z)):
                    casing_geom = Cylinder(
                        radius=casing_r,
                        height=10.0,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.PRIVATE,
                    )
                with URDFBoundary.from_shape(
                    casing_geom,
                    link_type=LinkType.CASING,
                    shape=ShapeType.CASING,
                    type=BoundaryType.SOLID_CAVITY,
                    thickness=casing_thick,
                    is_submerged=True,
                ):
                    IntakePort(
                        location=Location((0.0, -casing_r, (self.settings.pump_inlet_height / 2.0 - 1.5)), (0, -90, 0)),
                        radius=min(self.settings.pump_inlet_width, self.settings.pump_inlet_height) * 0.5 * 0.001,
                    )
                    DrainPort(location=Location((0.0, 28.0, 0.0), (0, 90, 0)), radius=tube_in_r * 0.001)
                    TubePort(location=Location((0.0, 28.0, 0.0), (0, 0, 0)))
                    FlowSlot(height=9.0 * 0.001, width=8.0 * 0.001)

        # Define joints
        motor_shaft_z = floor_z - 17.5 + self.settings.motor_clip_thickness + 9.3
        RigidJoint("impeller_post", bowl.part, Location((0, 0, floor_z + 2.0)))
        RigidJoint("motor_shaft", bowl.part, Location((0, 0, motor_shaft_z)))
        RigidJoint("pump_cover_seat", bowl.part, Location((0, 0, floor_z + 10.0)))
        RigidJoint("motor_clip_seat", bowl.part, Location((0, 0, floor_z - 17.5)))
        RigidJoint("lid_seat", bowl.part, Location((0, 0, h)))
        RigidJoint("cover_seat", bowl.part, Location((0, 0, 0)))
        RigidJoint(
            "sensor_port_east",
            bowl.part,
            Location((r - t / 2.0 - 3.0, 0, 16.0), (0, -30, 0)) * Location((11.2, 0, 0)),
        )
        RigidJoint(
            "sensor_port_north",
            bowl.part,
            Location(Rot(0, 0, 90.0)) * Location((r - t / 2.0 - 3.0, 0, 16.0), (0, -30, 0)) * Location((11.2, 0, 0)),
        )
        RigidJoint(
            "sensor_port_west",
            bowl.part,
            Location(Rot(0, 0, 180.0)) * Location((r - t / 2.0 - 3.0, 0, 16.0), (0, -30, 0)) * Location((11.2, 0, 0)),
        )
        RigidJoint(
            "led_port",
            bowl.part,
            Location(Rot(0, 0, 75.0))
            * Location((r - t / 2.0, 0, 8.0), (0, 90, 0))
            * Location((0, 0, self.settings.led_flange_thickness + 1.0)),
        )

        return bowl

    @method_cache
    def build_impeller(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the lightweight centrifugal impeller (magnetic-drive)."""
        hub_r = self.settings.impeller_radius + self.settings.magnet_radius + 1.0
        hub_h = self.settings.impeller_base_height
        impeller_r = hub_r + 4.0
        pin_r = self.settings.impeller_shaft_radius
        pin_clearance = self.settings.impeller_shaft_clearance
        sleeve_r = 4.5
        mr = self.settings.magnet_radius + self.settings.magnet_clearance
        mt = self.settings.magnet_thickness + self.settings.magnet_clearance
        ring_r = self.settings.magnet_ring_radius

        with BuildPart() as impeller:
            # Impeller Hub base (carrying the magnets)
            with Locations((0, 0, 0)):
                Cylinder(radius=hub_r, height=hub_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Central guide post sleeve (protrudes up to 10.0mm for stability on the post)
            with Locations((0, 0, 0)):
                Cylinder(radius=sleeve_r, height=10.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Subtract guide post central hole (with generous clearance for free rotation after resin coating)
            # Rests directly on top of the bowl's 2.0mm post flange shoulder, elevating the base disk 2.0mm off the floor
            Cylinder(
                radius=pin_r + pin_clearance,
                height=14.0,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.SUBTRACT,
            )

            # Subtract 4 magnet pockets on the bottom face of the base disk
            for i in range(self.settings.magnet_count):
                angle = i * (360.0 / self.settings.magnet_count)
                with Locations(Rot(0, 0, angle)):
                    with Locations((ring_r, 0, 0)):
                        Cylinder(
                            radius=mr, height=mt, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT
                        )

            # Add radial blades on the top face of the hub base (Z = hub_h)
            # Starting directly at the central hub sleeve (sleeve_r) with zero empty space
            blade_h = 4.0
            blade_len = hub_r - sleeve_r
            num_blades = self.settings.impeller_blades
            for i in range(num_blades):
                angle = i * (360.0 / num_blades)
                with Locations(Location((0, 0, hub_h)) * Rot(0, 0, angle)):
                    with Locations((sleeve_r, 0, 0)):
                        Box(blade_len, 1.2, blade_h, align=(Align.MIN, Align.CENTER, Align.MIN))

            with URDFMetadata(
                label=target,
                material=self.get_material(target) or "petg",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.ANALYTICAL,
                motor_type=URDFMotorType.VELOCITY,
                motor_target=self.settings.motor_target,
                motor_force=self.settings.motor_power / self.settings.motor_target,
                magnet_radius=self.settings.magnet_radius,
                magnet_thickness=self.settings.magnet_thickness,
                pump_well_wall=self.settings.pump_well_wall,
                magnet_count=self.settings.magnet_count,
                impeller_shaft_radius=self.settings.impeller_shaft_radius,
            ):
                with URDFBoundary.from_shape(
                    impeller.part,
                    link_type=LinkType.IMPELLER,
                    shape=ShapeType.IMPELLER,
                    type=BoundaryType.SOLID,
                    radius=impeller_r * 0.001,
                    thickness=pin_r * 0.001,
                ):
                    ImpellerVanes(count=num_blades, twist=self.settings.vane_twist, thickness=1.2 * 0.001)
                    MagneticCoupling(
                        radius=self.settings.magnet_radius,
                        thickness=self.settings.magnet_thickness,
                        count=self.settings.magnet_count,
                        well_wall=self.settings.pump_well_wall,
                        shaft_radius=self.settings.impeller_shaft_radius,
                    )

        RevoluteJoint(label="pin", to_part=impeller.part, axis=Axis((0, 0, 0), (0, 0, 1)), angular_range=(0, 360))

        return impeller

    @method_cache
    def build_bottom_cover(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the detachable bottom cover for the motor controller compartment."""
        r = self.settings.bowl_radius
        t = self.settings.bowl_thickness
        clearance = self.settings.bottom_cover_clearance
        cover_r = r - t - clearance

        with BuildPart() as cover:
            Cylinder(radius=cover_r, height=4.0, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Add a 0.5mm snap-fit annular ring around the perimeter
            with BuildPart(mode=Mode.PRIVATE) as snap_ring_part:
                Cylinder(radius=cover_r + 0.5, height=1.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                Cylinder(
                    radius=cover_r - 2.0, height=2.0, align=(Align.CENTER, Align.CENTER, Align.MIN), mode=Mode.SUBTRACT
                )
                fillet(snap_ring_part.edges().filter_by(GeomType.CIRCLE), radius=0.4)

            with Locations((0, 0, 1.2)):
                add(snap_ring_part.part)

            # Funnel-shaped top surface to drain water towards the central drain hole
            with Locations((0, 0, 1.5)):
                Cone(
                    bottom_radius=0.0,
                    top_radius=80.0,
                    height=2.5,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # Central drain hole drilled through the entire cover
            Cylinder(
                radius=self.settings.bottom_cover_drain_radius,
                height=10.0,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )

            # Peripheral wire/access opening notch on the edge
            opening_w = self.settings.bottom_cover_opening_width
            with Locations((0, -cover_r, 0.0)):
                Box(opening_w, 20.0, 10.0, align=(Align.CENTER, Align.CENTER, Align.CENTER), mode=Mode.SUBTRACT)

            # Symmetrically spaced depressions for adhesive rubber feet on the bottom face (Z = 0)
            feet_r = self.settings.rubber_feet_radius
            feet_d = self.settings.rubber_feet_depth
            pitch_r = self.settings.rubber_feet_pitch_radius
            feet_count = self.settings.rubber_feet_count
            if feet_count > 0 and feet_r > 0.0 and feet_d > 0.0:
                for i in range(feet_count):
                    angle_deg = (i * 360.0 / feet_count) + 45.0
                    rad = math.radians(angle_deg)
                    fx = pitch_r * math.cos(rad)
                    fy = pitch_r * math.sin(rad)
                    with Locations((fx, fy, 0.0)):
                        Cylinder(
                            radius=feet_r,
                            height=feet_d,
                            align=(Align.CENTER, Align.CENTER, Align.MIN),
                            mode=Mode.SUBTRACT,
                        )

            URDFMetadata(
                label=target,
                material=self.get_material(target) or "utr8100",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.CONVEX,
                parent="bowl",
                joint_type=URDFJointType.FIXED,
            )

        RigidJoint("mount", cover.part, Location((0, 0, 0)))

        return cover

    @method_cache
    def build_lid(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the top cover lid which acts as a drinking shelf and covers/stabilizes the vertical delivery tube."""
        r = self.settings.bowl_radius
        h = self.settings.bowl_height
        t = self.settings.bowl_thickness
        tube_y = 28.0

        lid_r = r - t + self.settings.lid_step_width - self.settings.lid_clearance
        lid_h = 8.0
        step_d = self.settings.lid_step_depth
        step_w = self.settings.lid_step_width
        clearance = self.settings.lid_clearance
        socket_r = self.settings.tube_radius + self.settings.tube_lid_clearance
        platform_r = self.settings.spout_deflection_radius

        with BuildPart() as lid:
            lid_disk = Cylinder(radius=lid_r, height=lid_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Add the snap-fit annular Torus ridge
            with Locations((0, 0, self.settings.snap_ridge_z)):
                Torus(
                    major_radius=lid_r + self.settings.snap_ridge_major_offset,
                    minor_radius=self.settings.snap_ridge_minor,
                )

            with BuildPart() as pocket_tool:
                Cylinder(radius=80.0, height=6.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                bottom_edge = pocket_tool.edges().filter_by(GeomType.CIRCLE).sort_by(Axis.Z)[0]
                fillet(bottom_edge, radius=1.5)
            with Locations((0.0, 0.0, 3.0)):
                add(pocket_tool, mode=Mode.SUBTRACT)

            # Subtract the large circular cutout, but preserve the circular platform at Y = 28.0
            cutout_r = self.settings.lid_cutout_radius
            cutout_y = self.settings.lid_cutout_y

            with BuildPart(mode=Mode.PRIVATE) as cutout_tool:
                # Large cutout cylinder
                Cylinder(radius=cutout_r, height=lid_h + 10.0, align=(Align.CENTER, Align.CENTER, Align.CENTER))
                # Add back the platform cylinder to preserve it
                with Locations((0, 28.0 - cutout_y, 0)):  # Relative to cutout center Y
                    Cylinder(
                        radius=platform_r,
                        height=lid_h + 20.0,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER),
                        mode=Mode.SUBTRACT,
                    )
            with Locations((0, cutout_y, 0)):
                add(cutout_tool, mode=Mode.SUBTRACT)

            # Create a small protective ridge around the opening (on the pocket floor, from Z = 3.0)
            ridge_w = self.settings.lid_cutout_ridge_width
            ridge_h = self.settings.lid_cutout_ridge_height
            with BuildSketch() as ridge_sketch:
                # Outer ridge: ring from cutout_r to cutout_r + ridge_w at (0, cutout_y)
                with Locations((0, cutout_y)):
                    Circle(radius=cutout_r + ridge_w)
                    Circle(radius=cutout_r, mode=Mode.SUBTRACT)
                # Subtract the platform area
                with Locations((0, 28.0)):
                    Circle(radius=platform_r, mode=Mode.SUBTRACT)

                # Inner ridge: ring from platform_r to platform_r + ridge_w at (0, 28.0) intersected with the cutout circle
                with BuildSketch(mode=Mode.PRIVATE) as inner_ridge:
                    with Locations((0, 28.0)):
                        Circle(radius=platform_r + ridge_w)
                        Circle(radius=platform_r, mode=Mode.SUBTRACT)
                    with Locations((0, cutout_y)):
                        Circle(radius=cutout_r, mode=Mode.INTERSECT)
                add(inner_ridge)

            with Locations((0, 0, 3.0)):
                extrude(ridge_sketch.sketch, amount=ridge_h)

            with Locations((0.0, tube_y, 3.0)):
                with Locations(Rot(-self.settings.lid_platform_slope_angle, 0, 0)):
                    Cylinder(radius=platform_r, height=3.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    with Locations((0, 0, 3.0)):
                        Cylinder(radius=platform_r, height=1.0, align=(Align.CENTER, Align.CENTER, Align.MIN))
                        Cylinder(
                            radius=platform_r - 2.0,
                            height=1.0,
                            align=(Align.CENTER, Align.CENTER, Align.MIN),
                            mode=Mode.SUBTRACT,
                        )
                    Cylinder(
                        radius=socket_r,
                        height=10.0,
                        align=(Align.CENTER, Align.CENTER, Align.CENTER),
                        mode=Mode.SUBTRACT,
                    )

            with Locations((0.0, tube_y, 0)):
                socket_r = self.settings.tube_radius + self.settings.tube_lid_clearance
                dome_out_r = socket_r + 1.5
                dome_in_r = self.settings.tube_radius - self.settings.tube_thickness + self.settings.tube_lid_clearance
                dome_slot_len = (dome_out_r + 1.0) * 2.0

                with BuildPart(mode=Mode.PRIVATE) as dome_tool:
                    with Locations((0, 0, 6.0)):
                        outer_dome = Sphere(radius=dome_out_r)
                        Sphere(radius=dome_in_r, mode=Mode.SUBTRACT)
                        for angle in [0, 45, 90, 135]:
                            with Locations(Rot(0, 0, angle)):
                                Box(
                                    self.SLOT_WIDTH,
                                    dome_slot_len,
                                    dome_in_r,
                                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                                    mode=Mode.SUBTRACT,
                                )
                        # Retention boss extending down from inner sphere ceiling to limit impeller vertical travel (added after cuts)
                        with Locations((0, 0, dome_in_r)):
                            Cylinder(
                                radius=3.0,
                                height=self.settings.retention_boss_height,
                                align=(Align.CENTER, Align.CENTER, Align.MAX),
                            )

                add(dome_tool.part)

                with Locations((0, 0, -10.0)):
                    Cylinder(
                        radius=socket_r,
                        height=16.0,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.SUBTRACT,
                    )

        # Define joints and fluid ports
        intake_z = self.settings.lid_pocket_thickness - 0.25
        RigidJoint("intake_port", lid.part, Location((0.0, tube_y, intake_z), (0, 180, 0)))
        RigidJoint("drain_port", lid.part, Location((0.0, cutout_y, 0.0), (0, 0, 0)))
        RigidJoint("tube_port", lid.part, Location((0.0, tube_y, intake_z), (0, 0, 0)))
        RigidJoint("mount", lid.part, Location((0, 0, step_d)))

        # Define CAD feature geometry primitives for analytical boundaries
        with Locations((0, 0, self.settings.lid_pocket_z_offset)):
            pocket_geom = Cylinder(
                radius=self.settings.lid_pocket_radius,
                height=self.settings.lid_pocket_cavity_height,
                align=(Align.CENTER, Align.CENTER, Align.MIN),
                mode=Mode.PRIVATE,
            )
        with Locations((0, tube_y, 6.0)):
            dome_geom = Sphere(radius=dome_out_r, mode=Mode.PRIVATE)

        with URDFMetadata(
            geometry=lid,
            label=target,
            material=self.get_material(target) or "utr8100",
            density=self.settings.density,
            boundary_friction=self.settings.boundary_friction,
            collision_type=URDFCollisionType.ANALYTICAL,
            parent="bowl",
            joint_type=URDFJointType.FIXED,
        ):
            with URDFBoundary.from_shape(
                pocket_geom,
                link_type=LinkType.LID,
                shape=ShapeType.CYLINDER,
                type=BoundaryType.CAVITY,
                thickness=self.settings.lid_pocket_thickness * 0.001,
            ):
                IntakePort(location=Location((0.0, tube_y, intake_z), (0, 180, 0)), radius=platform_r * 0.001)
                DrainPort(location=Location((0.0, cutout_y, 0.0), (0, 0, 0)), radius=cutout_r * 0.001)
                TubePort(
                    location=Location((0.0, tube_y, intake_z), (0, 0, 0)),
                    radius=(self.settings.tube_radius - self.settings.tube_thickness) * 0.001,
                )

            with URDFBoundary.from_shape(
                dome_geom,
                link_type=LinkType.LID,
                shape=ShapeType.CANOPY,
                type=BoundaryType.SOLID,
                thickness=(dome_out_r - dome_in_r) * 0.001,
            ):
                pass

        return lid

    @method_cache
    def build_led_cover(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build a translucent push-fit cover/diffuser for the RGB status LED."""
        flange_w = self.settings.led_flange_width
        flange_t = self.settings.led_flange_thickness
        plug_w = self.settings.led_plug_width
        plug_l = self.settings.led_plug_length

        with BuildPart() as cover:
            Box(flange_w, flange_w, flange_t, align=(Align.CENTER, Align.CENTER, Align.MIN))
            fillet(cover.edges().filter_by(Axis.Z), radius=1.0)

            Box(plug_w, plug_w, plug_l, align=(Align.CENTER, Align.CENTER, Align.MAX))

            URDFMetadata(
                label=target,
                material=self.get_material(target) or "utr8100",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.CONVEX,
                parent="bowl",
                joint_type=URDFJointType.FIXED,
            )

        RigidJoint("mount", cover.part, Location((0, 0, 0)))

        return cover

    @method_cache
    def build_drive_hub(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the dry-side magnet drive hub mounted on the motor D-shaft."""
        hub_r = self.settings.impeller_radius + self.settings.magnet_radius + 1.6
        hub_h = self.settings.magnet_thickness + self.settings.magnet_clearance + 0.8
        standoff_h = self.settings.drive_hub_standoff_height
        mr = self.settings.magnet_radius + self.settings.magnet_clearance
        mt = self.settings.magnet_thickness + self.settings.magnet_clearance
        ring_r = self.settings.magnet_ring_radius
        ring_inner_r = hub_r - 1.4

        with BuildPart() as hub:
            Cylinder(radius=hub_r, height=hub_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # Add integral outer perimeter standoff thrust ring on top face (Z = hub_h) for anti-wobble stability
            if standoff_h > 0.0:
                with Locations((0, 0, hub_h)):
                    Cylinder(radius=hub_r, height=standoff_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                    Cylinder(
                        radius=ring_inner_r,
                        height=standoff_h + 0.1,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.SUBTRACT,
                    )

            # Motor shaft hole with generous print-shrinkage clearance for smooth hand press-fit
            shaft_r = self.settings.drive_hub_shaft_radius
            total_h = hub_h + standoff_h
            with BuildSketch() as hole_sketch:
                Circle(radius=shaft_r)
            ext_hole = extrude(hole_sketch.sketch, amount=total_h + 2.0, mode=Mode.PRIVATE)
            hub.part -= Location((0, 0, -1.0)) * ext_hole

            # Magnet pockets on the top face (Z = hub_h)
            for i in range(self.settings.magnet_count):
                angle = i * (360.0 / self.settings.magnet_count)
                with Locations(Location((0, 0, hub_h)) * Rot(0, 0, angle)):
                    with Locations((ring_r, 0, 0)):
                        # Pockets cut downward into the cylinder
                        Cylinder(
                            radius=mr, height=mt, align=(Align.CENTER, Align.CENTER, Align.MAX), mode=Mode.SUBTRACT
                        )

            URDFMetadata(
                label=target,
                material=self.get_material(target) or "petg",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.CONVEX,
                motor_type=URDFMotorType.VELOCITY,
                motor_target=self.settings.motor_target,
                motor_force=self.settings.motor_power / self.settings.motor_target,
                parent="bowl",
                joint_type=URDFJointType.CONTINUOUS,
            )

        RevoluteJoint(label="motor", to_part=hub.part, axis=Axis((0, 0, 0), (0, 0, 1)))
        return hub

    @method_cache
    def build_pump_cover(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the revised wet-side pump cover that slides down the tube and mounts directly over the tube and casing."""
        tube_x = 0.0
        tube_y = 28.0
        tube_r = self.settings.tube_radius
        tube_clearance = self.settings.pump_cover_tube_clearance
        hub_r = self.settings.impeller_radius + self.settings.magnet_radius + 1.0
        impeller_r = hub_r + 4.0
        chamber_r = impeller_r + self.settings.pump_casing_clearance
        pump_well_wall = self.settings.pump_well_wall
        casing_r = chamber_r + pump_well_wall
        snap_r = chamber_r + 1.0
        snap_clearance = self.settings.pump_cover_snap_clearance
        internal_clearance = self.settings.pump_cover_internal_clearance
        cover_r = snap_r - snap_clearance  # Fits into casing snap recess with epoxy resin clearance
        cover_h = self.settings.pump_cover_height
        sleeve_h = self.settings.pump_cover_sleeve_height
        inlet_w = self.settings.pump_inlet_width
        inlet_h = self.settings.pump_inlet_height
        clip_opening_w = self.settings.pump_cover_clip_opening_width
        lip_thickness = self.settings.pump_cover_lip_thickness
        lip_in_r = cover_r - lip_thickness

        with BuildPart() as cover:
            # 1. Main casing cap over impeller volute (Z = 0 at casing top rim)
            with Locations((0, 0, 0)):
                Cylinder(radius=casing_r, height=cover_h, align=(Align.CENTER, Align.CENTER, Align.MIN))
                # Downward lip entering casing snap recess (Z = -1.5 to 0) with tapered entry chamfer
                with Locations((0, 0, -1.5)):
                    Cone(
                        bottom_radius=cover_r - 0.3,
                        top_radius=cover_r,
                        height=1.5,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                    )

            # 2. Tube sleeve sliding down the vertical delivery tube
            with Locations((tube_x, tube_y, 0)):
                Cylinder(radius=tube_r + 2.5, height=sleeve_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # 3. Connecting bridge between casing cap and tube sleeve
            with Locations((0, tube_y / 2.0, 0)):
                Box(tube_r * 2.0 + 5.0, tube_y, cover_h, align=(Align.CENTER, Align.CENTER, Align.MIN))

            # 4. Subtract vertical tube bore through sleeve with bottom entry lead-in cone
            with Locations((tube_x, tube_y, -2.0)):
                Cylinder(
                    radius=tube_r + tube_clearance,
                    height=sleeve_h + 4.0,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )
                with Locations((0, 0, 0)):
                    Cone(
                        bottom_radius=tube_r + tube_clearance + 0.6,
                        top_radius=tube_r + tube_clearance,
                        height=1.5,
                        align=(Align.CENTER, Align.CENTER, Align.MIN),
                        mode=Mode.SUBTRACT,
                    )

            # 5. Subtract North-facing C-clip snap opening through sleeve for easy tube installation
            if clip_opening_w > 0.0:
                with Locations((tube_x, tube_y, -2.0)):
                    Box(
                        clip_opening_w,
                        20.0,
                        sleeve_h + 4.0,
                        align=(Align.CENTER, Align.MIN, Align.MIN),
                        mode=Mode.SUBTRACT,
                    )

            # 6. Subtract internal impeller / guide post clearance cavity inside casing cap
            with Locations((0, 0, -1.5)):
                Cylinder(
                    radius=lip_in_r,
                    height=cover_h,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            # 7. Side water intake window on South face (opposite the tube: Y < 0)
            with Locations((0, -casing_r + 2.0, 0.0)):
                Box(
                    inlet_w,
                    10.0,
                    inlet_h,
                    align=(Align.CENTER, Align.CENTER, Align.MIN),
                    mode=Mode.SUBTRACT,
                )

            with URDFMetadata(
                label=target,
                material=self.get_material(target) or "utr8100",
                density=self.settings.density,
                boundary_friction=self.settings.boundary_friction,
                collision_type=URDFCollisionType.ANALYTICAL,
                parent="bowl",
                joint_type=URDFJointType.FIXED,
            ):
                with URDFBoundary.from_shape(
                    cover.part,
                    link_type=LinkType.PUMP_COVER,
                    shape=ShapeType.CYLINDER,
                    type=BoundaryType.CAVITY,
                    thickness=pump_well_wall * 0.001,
                    is_submerged=True,
                ):
                    IntakePort(
                        location=Location((0.0, -casing_r, (inlet_h / 2.0)), (0, -90, 0)),
                        radius=min(inlet_w, inlet_h) * 0.5 * 0.001,
                    )

        RigidJoint("mount", cover.part, Location((0, 0, 0)))
        return cover

    def build_diagram(self, room: Room, targets: Sequence[str], mode: ProviderMode) -> None:
        """Build an exploded assembly diagram for the cat fountain."""
        bowl_part = self.build_bowl("bowl").part
        impeller_part = self.build_impeller("impeller").part
        bottom_cover_part = self.build_bottom_cover("bottom_cover").part
        lid_part = self.build_lid("lid").part
        drive_hub_part = self.build_drive_hub("drive_hub").part
        pump_cover_part = self.build_pump_cover("pump_cover").part
        motor_clip_part = self.build_motor_clip("motor_clip").part

        assert (
            bowl_part is not None
            and impeller_part is not None
            and bottom_cover_part is not None
            and lid_part is not None
            and drive_hub_part is not None
            and pump_cover_part is not None
            and motor_clip_part is not None
        )

        # 2. Position them in their standard assembled configuration using joints
        bowl_part.joints["impeller_post"].connect_to(impeller_part.joints["pin"])
        bowl_part.joints["cover_seat"].connect_to(bottom_cover_part.joints["mount"])
        bowl_part.joints["lid_seat"].connect_to(lid_part.joints["mount"])
        bowl_part.joints["motor_shaft"].connect_to(drive_hub_part.joints["motor"])
        bowl_part.joints["pump_cover_seat"].connect_to(pump_cover_part.joints["mount"])
        bowl_part.joints["motor_clip_seat"].connect_to(motor_clip_part.joints["mount"])

        # Build and connect the LED cover
        led_cover = self.build_led_cover("led_cover").part
        assert led_cover is not None
        bowl_part.joints["led_port"].connect_to(led_cover.joints["mount"])

        # Explode outwards by translating Z axis for LED cover
        assert led_cover.location is not None
        led_cover.location = led_cover.location * Location((0, 0, 30.0))

        # 3. Explode the parts by translating their .location attributes
        assert impeller_part.location is not None
        assert bottom_cover_part.location is not None
        assert lid_part.location is not None
        assert drive_hub_part.location is not None
        assert pump_cover_part.location is not None
        assert motor_clip_part.location is not None

        impeller_part.location = Location((0, 0, 50)) * impeller_part.location
        bottom_cover_part.location = Location((0, 0, -50)) * bottom_cover_part.location
        lid_part.location = Location((0, 0, 80)) * lid_part.location
        drive_hub_part.location = Location((0, 0, -25)) * drive_hub_part.location
        pump_cover_part.location = Location((0, 0, 110)) * pump_cover_part.location
        motor_clip_part.location = Location((0, -35, 0)) * motor_clip_part.location

        # 4. Add the exploded parts to the room using manifest colors
        for name, part in [
            ("bowl", bowl_part),
            ("impeller", impeller_part),
            ("bottom_cover", bottom_cover_part),
            ("lid", lid_part),
            ("led_cover", led_cover),
            ("drive_hub", drive_hub_part),
            ("pump_cover", pump_cover_part),
            ("motor_clip", motor_clip_part),
        ]:
            rgba = self.get_color(name)
            room.add(name, part, color=rgba[:3], alpha=rgba[3])

        # 5. Add connector lines indicating assembly paths
        impeller_conn = Line(
            bowl_part.joints["impeller_post"].location.position, impeller_part.joints["pin"].location.position
        )
        room.add("impeller_connector", impeller_conn)

        pump_cover_conn = Line(
            bowl_part.joints["pump_cover_seat"].location.position, pump_cover_part.joints["mount"].location.position
        )
        room.add("pump_cover_connector", pump_cover_conn)

        drive_hub_conn = Line(
            bowl_part.joints["motor_shaft"].location.position, drive_hub_part.joints["motor"].location.position
        )
        room.add("drive_hub_connector", drive_hub_conn)

        motor_clip_conn = Line(
            bowl_part.joints["motor_clip_seat"].location.position, motor_clip_part.joints["mount"].location.position
        )
        room.add("motor_clip_connector", motor_clip_conn)

        # 6. Add labels for each part
        room.add_label("bowl_label", "BOWL", bowl_part.center() + Vector(-120, -20, 10), options=TextArgs(font_size=10))
        room.add_label(
            "impeller_label", "IMPELLER", impeller_part.center() + Vector(-50, -10, 10), options=TextArgs(font_size=10)
        )
        room.add_label(
            "cover_label", "COVER", bottom_cover_part.center() + Vector(-80, 0, -10), options=TextArgs(font_size=10)
        )
        room.add_label("lid_label", "LID", lid_part.center() + Vector(-50, -10, 20), options=TextArgs(font_size=10))
        room.add_label(
            "led_cover_label",
            "LED COVER",
            led_cover.center() + Vector(30, 0, 10),
            options=TextArgs(font_size=10),
        )
        room.add_label(
            "drive_hub_label",
            "DRIVE HUB",
            drive_hub_part.center() + Vector(-50, 0, -10),
            options=TextArgs(font_size=10),
        )
        room.add_label(
            "pump_cover_label",
            "PUMP COVER",
            pump_cover_part.center() + Vector(-50, -10, 10),
            options=TextArgs(font_size=10),
        )
        room.add_label(
            "motor_clip_label",
            "MOTOR CLIP",
            motor_clip_part.center() + Vector(-50, -10, 0),
            options=TextArgs(font_size=10),
        )

    def build_product(self, room: Room, mode: ProviderMode) -> None:
        """Place all parts of the cat fountain in the room for visualization/simulation."""
        bowl_part = self.build_bowl("bowl", mode=mode).part
        impeller_part = self.build_impeller("impeller", mode=mode).part
        bottom_cover_part = self.build_bottom_cover("bottom_cover", mode=mode).part
        lid_part = self.build_lid("lid", mode=mode).part
        drive_hub_part = self.build_drive_hub("drive_hub", mode=mode).part
        pump_cover_part = self.build_pump_cover("pump_cover", mode=mode).part
        motor_clip_part = self.build_motor_clip("motor_clip", mode=mode).part

        assert (
            bowl_part is not None
            and impeller_part is not None
            and bottom_cover_part is not None
            and lid_part is not None
            and drive_hub_part is not None
            and pump_cover_part is not None
            and motor_clip_part is not None
        )

        # 2. Position them in their standard assembled configuration using joints
        bowl_part.joints["impeller_post"].connect_to(impeller_part.joints["pin"])
        bowl_part.joints["motor_shaft"].connect_to(drive_hub_part.joints["motor"])
        bowl_part.joints["pump_cover_seat"].connect_to(pump_cover_part.joints["mount"])
        bowl_part.joints["cover_seat"].connect_to(bottom_cover_part.joints["mount"])
        bowl_part.joints["lid_seat"].connect_to(lid_part.joints["mount"])
        bowl_part.joints["motor_clip_seat"].connect_to(motor_clip_part.joints["mount"])

        # Build and connect the LED cover using joints
        led_cover = self.build_led_cover("led_cover", mode=mode).part
        assert led_cover is not None
        bowl_part.joints["led_port"].connect_to(led_cover.joints["mount"])

        # 3. Add the positioned parts directly to the room using manifest colors
        for name, part in [
            ("bowl", bowl_part),
            ("lid", lid_part),
            ("impeller", impeller_part),
            ("bottom_cover", bottom_cover_part),
            ("led_cover", led_cover),
            ("drive_hub", drive_hub_part),
            ("pump_cover", pump_cover_part),
            ("motor_clip", motor_clip_part),
        ]:
            rgba = self.get_color(name)
            room.add(name, part, color=rgba[:3], alpha=rgba[3])
        self.room = room

    def get_simulate_hooks_impl(self, sim_name: str) -> dict[Simulate, Callable[..., Any]]:
        """Return the simulation hooks for the cat fountain."""
        from .simulate_hooks import get_simulate_hooks_impl as impl

        return impl(self, sim_name)

    @property
    def config(self) -> dict[str, Callable[[str, Optional[str]], Any]]:
        """A mapping of Modes to configuration handler methods."""
        from .config import config_tune

        return {
            "tune": lambda target, sa: config_tune(self, target, sa),
        }

    def build_motor_clip(
        self, target: str, subassembly: str = "default", mode: ProviderMode = ProviderMode.DEFAULT
    ) -> BuildPart:
        """Build the slide-in motor retaining clip (fork) to secure the motor in place."""
        clip_w = self.settings.motor_clip_width
        clip_l = self.settings.motor_clip_length
        clip_h = self.settings.motor_clip_thickness
        fork_w = self.settings.motor_clip_cutout_width

        with BuildPart() as clip:
            # Main flat slide plate extending South from the center (0, 0)
            # Extends from Y = -clip_l to Y = 5.0 (so U-cutout centered at Y=0 is 5mm from front)
            with Locations((0, -clip_l, 0)):
                Box(clip_w, clip_l + 5.0, clip_h, align=(Align.CENTER, Align.MIN, Align.MIN))

            # Subtract U-cutout centered at (0, 0) of diameter fork_w (radius fork_w / 2.0)
            Cylinder(
                radius=fork_w / 2.0,
                height=clip_h + 10.0,
                align=(Align.CENTER, Align.CENTER, Align.CENTER),
                mode=Mode.SUBTRACT,
            )

            # Subtract the leading guide slot from Y = 0 to Y = 10.0 to break through the front (Y > 0)
            with Locations((0, 0, 0)):
                Box(fork_w, 20.0, clip_h + 10.0, align=(Align.CENTER, Align.MIN, Align.CENTER), mode=Mode.SUBTRACT)

            # Add a pull handle at the back (South end, Y = -clip_l)
            with Locations((0, -clip_l, 0)):
                Box(clip_w + 4.0, 3.0, clip_h + 3.0, align=(Align.CENTER, Align.MAX, Align.MIN))

            URDFMetadata(
                label=target,
                material=self.get_material(target) or "petg",
                density=self.settings.petg_density,
                boundary_friction=self.settings.petg_boundary_friction,
                collision_type=URDFCollisionType.CONVEX,
                parent="bowl",
                joint_type=URDFJointType.FIXED,
            )

        RigidJoint("mount", clip.part, Location((0, 0, 0)))
        return clip

    def build_wiring_diagram(self, room: Room, targets: Sequence[str], mode: ProviderMode) -> None:
        """Build a 2D colored wiring diagram for the cat water fountain."""
        import math

        self.settings.diagram_options.style = DiagramStyle.COLOR
        self.settings.diagram_options.view_from = "top"

        r = self.settings.bowl_radius
        t = self.settings.bowl_thickness

        # Draw the bowl outline in grey
        with BuildSketch() as bowl_outline:
            Circle(radius=r)
            Circle(radius=r - t, mode=Mode.SUBTRACT)
        room.add("bowl_outline", bowl_outline.sketch, color="grey")

        # Draw the motor compartment outline in grey
        with BuildSketch() as motor_comp:
            Circle(radius=15.0)
            Circle(radius=14.0, mode=Mode.SUBTRACT)
        room.add("motor_compartment", motor_comp.sketch, color="grey")

        # Get bowl part to query joints and load/resolve Wiring
        bowl_part = self.build_bowl("bowl").part
        yaml_path = Path(__file__).parent / "wiring.yaml"
        wiring = Wiring(yaml_path, bowl_part)

        # Build the diagram using the diagram class
        diagram = WiringDiagram(wiring)
        diagram.build(room)

    def compute_flow_metrics(self) -> dict[str, Any]:
        """Compute continuous flow and drainage metrics delegating to simulation hooks."""
        from .simulate_hooks import compute_flow_metrics

        return compute_flow_metrics(self)
