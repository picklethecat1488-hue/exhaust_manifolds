# Cat Fountain

This project designs a custom 3D-printable automatic pet cat water fountain. It features a bottom water bowl, a mechanical water spout, a vertical delivery tube, and a spout nozzle at the top to create a gentle flowing stream of water for cats. The complete design will incorporate an upper drinking level and enclosed 2L water storage tank, as well as a compartment for the motor controller.

The design utilizes precise build123d joint placement to ensure alignment of the impeller on the central shaft pin, and rigid mounting sockets for the water tube and spout nozzle.

![Diagram](cat_fountain_diagram.svg)

*Exploded assembly diagram.*

## Build Files

After running `build.py`, you should see these files in your build output organized by subdirectories under the build folder:

- **build/svg/cat_fountain/cat_fountain_diagram.svg** - An exploded assembly diagram of the cat fountain.
- **build/svg/cat_fountain/cat_fountain_wiring_diagram.svg** - A detailed 2D wiring diagram showing schematic connections and orthogonal routing.
- **build/stl/cat_fountain/bowl.stl** / **impeller.stl** / etc. - The 3D printable STL files (in standard millimeters scale).
- **build/obj/cat_fountain/bowl.obj** / **impeller.obj** / etc. - The 3D visual OBJ files (in standard meters scale).
- **build/urdf/cat_fountain/product.urdf** - The URDF definition file for visualization/simulation.

## Visualization & Simulation

To view the cat fountain assembly in the CAD viewer:
```bash
python src/view.py cat_fountain/product
```

To run the physics simulation of the cat fountain in PyBullet:
```bash
python src/view.py cat_fountain/product:view/simulate -s 1000
```

## System Block Diagram

```mermaid
graph TD
    %% Power Subsystem
    USBC["USB-C Charger Port"] -->|5V Power In| BQ["BQ25185 Charger & Boost"]
    BAT[("18650 Li-ion Battery")] <-->|Analog Cell Connection| MAX["MAX17048 Fuel Gauge"]
    MAX -->|Jumpered Cell Connection| BQ
    
    %% Power Distribution
    BQ -->|5V System Power| PICO["Raspberry Pi Pico W"]
    BQ -->|5V Motor Power| INA["INA219 Current Sensor"]
    INA -->|Monitored Power| DRV["TMC6300 Motor Driver"]
    
    %% I2C Bus Connections
    PICO -->|I2C Host Bus| I2CBus{I2C Bus: GP12/GP13}
    I2CBus <--> MAX
    I2CBus <--> INA
    I2CBus <--> TOF1["VL53L0X ToF (North)"]
    I2CBus <--> TOF2["VL53L0X ToF (East)"]
    I2CBus <--> TOF3["VL53L0X ToF (West)"]
    
    %% Control Lines
    PICO -->|GP4: XSHUT| TOF1
    PICO -->|GP5: XSHUT| TOF2
    PICO -->|GP6: XSHUT| TOF3
    TOF1 -->|GP7: INT| PICO
    TOF2 -->|GP9: INT| PICO
    TOF3 -->|GP10: INT| PICO
    MAX -->|GP14: ALRT| PICO
    PICO -->|GP18, GP19, GP20, GP21: Control| DRV
    PICO -->|GP22: DIAG| DRV
    PICO -->|GP26: SPEED_SENSE| SENS["AH3577 Speed Sensor"]
    PICO -->|GP11: PIO/WS2812| LED["RGB NeoPixel LED"]
    
    %% Output Connections
    DRV -->|Driven Power: U/V/W| MOTOR["BetaFPV 1102 BLDC Motor"]
```

## Bill of Materials (BOM)

To build the cat fountain with I2C communication across key monitoring subsystems (fuel gauge, current sensor, LED controller, proximity sensors), here are the recommended components to purchase. All selected parts are highly standard in the electronics maker ecosystem (Adafruit, SparkFun, TI, Raspberry Pi).

![Wiring Diagram](cat_fountain_wiring_diagram.svg)

*Wiring diagram showing component interconnections.*

| Component | Recommended Part | Description | I2C Address (Hex) | Key Features |
| :--- | :--- | :--- | :--- | :--- |
| **Microcontroller Board** | **Raspberry Pi Pico W** | Main controller running MicroPython/C++. Controls the I2C bus, reads sensors, and drives motor speed. | *Host Controller* | Dual ARM Cortex-M0+, built-in Wi-Fi/Bluetooth, two hardware I2C buses (I2C0, I2C1). |
| **IR Proximity Sensors (Qty: 3)** | **Adafruit VL53L0X Time-of-Flight (ToF)** | Long-range laser distance sensor used for cat proximity detection in North, East, and West directions. | `0x29` (default)<br>*Re-addressed to `0x30`, `0x31`, `0x32` at boot* | Measures precise distances up to 2m, unaffected by ambient light. Uses shutdown pin (XSHUT) for startup addressing. |
| **RGB LED Indicator** | **Standard RGB NeoPixel (WS2812B / SK6812)** | High-brightness status LED to indicate battery capacity and device state. | *N/A (Driven via PIO GP11)* | Addressable RGB LED driven directly from Raspberry Pi Pico W using a PIO state machine. |
| **USBC Charger & Boost** | **Adafruit BQ25185 Charger & Boost (6106)** | USB-C power management IC for charging the battery and boosting to 5V for the water pump motor. | *N/A (Standalone)* | Standalone linear charger. Uses jumpers/resistors to configure chemistry/current. S1 (FAULT) & S2 (CHG) are left disconnected. |
| **Battery Fuel Gauge** | **Adafruit MAX17048 LiPo Fuel Gauge** | Battery monitor board to track cell voltage and state of charge (percentage) over I2C. | `0x36` | Primary interface to access battery and charging status. The 18650 battery connects directly to this board, which is then jumpered to the charger. Includes configurable alert interrupt (ALRT) pin connected to GP14. |
| **Battery** | **Standard 18650 3.7V Li-ion Cell** | Main energy source (e.g. Samsung 30Q or Panasonic NCR18650B, 3000+ mAh). | *N/A (Analog)* | Rechargeable lithium-ion cell to fit the internal battery storage area. |
| **BLDC Motor Driver** | **SparkFun TMC6300 Breakout** | Low-voltage three-phase brushless motor driver breakout to drive the 15W BLDC motor. | *N/A (Driven by GP18, GP19, GP20, GP21, GP22)* | Supports 2V-11V range, up to 2A continuous, breadboard-compatible 25.4mm x 20.3mm footprint. |
| **BLDC Motor (15W)** | **BetaFPV 1102 (18000KV) Brushless Motor** | Ultra-lightweight 1S-compatible brushless motor to drive the dry-side magnet hub. | *N/A (Driven by TMC6300)* | 15W power class, 13.8mm diameter, 1.5mm shaft, lightweight 2.9g, mounts with M1.4 screws. |
| **Coupling Magnets (Qty: 6)** | **6mm x 3mm N52 Neodymium Discs** | Rare-earth magnets to couple the dry motor shaft to the wet impeller. | *N/A (Magnetic)* | Strong N52 grade, 3 magnets on the dry hub and 3 on the wet impeller with alternating polarities. |
| **I2C Current Sensor** | [Adafruit INA219 Current Sensor](https://www.adafruit.com/product/904) | High-side current and power monitor to measure motor current draw and calculate load torque. | `0x40` | Measures current up to 3.2A with 1% accuracy. Used for low water / refill detection. |
| **Speed Sensor** | **Digital Latching Hall Sensor (e.g., US1881 / AH3577)** | Measures impeller rotation speed contactlessly by detecting alternating magnet poles. | *N/A (Driven by GP26)* | Latching digital output, SOT-23 or TO-92 package, fits inside custom pocket in the motor boss. |

### Technical Integration Notes

1. **Handling Multiple Proximity Sensors**:
   Since all three `VL53L0X` sensors share the same default I2C address (`0x29`), you must connect the `XSHUT` (shutdown) pin of each sensor to a GPIO pin on the Pico. At boot, pull all `XSHUT` pins LOW to disable the sensors. Then, enable them one by one (pull `XSHUT` HIGH) and send an I2C command to assign a unique address (`0x30`, `0x31`, `0x32`) to the active sensor.
2. **I2C Bus Voltage & Pull-Ups**:
   The Raspberry Pi Pico operates at 3.3V logic. Ensure all boards are powered at 3.3V (or have 3.3V level-shifting built-in). Add `4.7kΩ` pull-up resistors to the `SDA` and `SCL` lines of each active I2C bus.
3. **Motor Speed & Torque Control**:
   * The TMC6300 brushless motor driver allows adjusting the motor speed via 3-phase PWM driven directly by Pico GPIO pins `GP19`, `GP20`, and `GP21`, with `GP18` acting as the active-high driver enable. This can be modulated depending on cat proximity to dynamically speed up the Archimedes screw pump when a cat approaches, and slow down or enter standby when idle.
   * **Water Level / Refill Detection**: The INA219 current sensor measures motor current draw over I2C. When the water level in the bowl runs low, the impeller spins in air rather than water, causing motor load torque and current draw to drop significantly. The RP2040 can monitor this current drop over I2C, trigger a "low water" alert, and pulse the RGB NeoPixel status LED to notify the user.
4. **Debug SWD & UART Pins**:
   * **SWD Debugging**: The separate 3-pin debug header at the bottom edge of the Pico W provides **SWCLK**, **GND**, and **SWDIO** for hardware debugging (e.g., using a Raspberry Pi Debug Probe or Picoprobe). These do not occupy standard GPIO pins.
   * **UART Console / Printf Debugging**: **GP0 (TX)** and **GP1 (RX)** are reserved as the default debug UART0 port, leaving them completely free from control or sensor connections.
5. **Passive Battery Temperature Monitoring**:
   * The Adafruit BQ25185 board operates as a standalone charger without I2C control, and its /CE (Charge Enable) pin is hard-wired to GND, meaning the Pico W cannot programmatically disable charging.
   * To ensure battery safety inside the sealed dry electronics compartment, the BQ25185's built-in hardware temperature protection should be used: cut the TH trace jumper on the back of the board and connect a 10kΩ NTC thermistor (attached to the battery cell) between the TH pad and GND. This allows the BQ25185 to automatically suspend charging in hardware if the battery exceeds safe temperature limits.
   * The Pico W can passively monitor the internal ambient temperature of the electronics compartment using the RP2040's built-in internal temperature sensor (on ADC channel 4). If the temperature exceeds 45°C, the Pico W should enter a low-power sleep mode and cut power to the motor driver.
6. **Fuel Gauge Alert Interrupt**:
   * The active-low `ALRT` (Alert) pin of the MAX17048 fuel gauge is wired to Pico GPIO `GP14`. The firmware configures this pin with an internal pull-up and attaches an interrupt service routine (ISR). This enables the MAX17048 to asynchronously wake the microcontroller or trigger an interrupt on low-battery (e.g. State of Charge falls below 10%) or battery voltage alerts, rather than requiring the Pico W to constantly wake up to poll the fuel gauge, optimizing overall system power efficiency.
7. **Battery and Charger Status Monitoring**:
   * S1 and S2 status connections from the BQ25185 charger to the Pico W are removed.
   * Instead, the MAX17048 fuel gauge serves as the primary interface to access battery and charging status.
   * The Pico W queries the MAX17048 over I2C to obtain battery state of charge (SoC) and cell voltage (VCELL). A rising voltage/SoC trend indicates charging is occurring, while a falling trend indicates discharging.
   * The battery connects directly to the MAX17048, which is then jumpered to the BQ25185 charger to allow charging and system power pathing.
   * **System Power (VSYS)**: The BQ25185 charger boosts/routes the power to its system power rail. A wire runs from the **5V SYS output terminal block** on the BQ25185 charger to the **VSYS input pin (pin 39)** on the Raspberry Pi Pico W.
8. **Wiring Gauge Recommendations & Safety**:
   * **Low-Power Control and Logic (Pico, VL53L0X, MAX17048, INA219, WS2812B)**: Use **30AWG wire wrap wire** since these signals carry very low current (< 50mA).
   * **High-Power Motor Lines (TMC6300 VCC/GND and Motor Phases U/V/W)**: Use **20AWG solid core wire**. Since the 15W brushless motor can draw up to 4.0A at peak load, the 20AWG wire (rated up to 11A) is required to prevent overheating, thermal degradation, or voltage drop.


### 3D-Printed Parts & Materials

The following parts are manufactured to form the physical structure and assembly of the cat fountain:

| Part Name | Qty | Recommended Material | Description / Use Case |
| :--- | :--- | :--- | :--- |
| **Bowl** | 1 | UTR-8100 Translucent (SLA) | The main water reservoir (2L capacity) with integrated offset tube, static guide post, and raised volute casing. |
| **Lid** | 1 | UTR-8100 Translucent (SLA) | Top cover acting as a drinking shelf, stabilizing the delivery tube, and containing a terrace shelf. |
| **Impeller** | 1 | PETG / ABS | The lightweight magnetic-drive centrifugal impeller with 6 radial blades. |
| **Bottom Cover** | 1 | UTR-8100 Translucent (SLA) | Bottom enclosure cover protecting the electronics compartment. |
| **LED Cover** | 1 | UTR-8100 Translucent (SLA) | Translucent push-fit diffuser plug for the status RGB LED. |
| **Drive Hub** | 1 | PETG | Dry-side hub carrying the driving coupling magnets, mounted to the motor shaft. |
| **Pump Cover** | 1 | PETG | Wet-side snap-fit cap with an axial intake hole that seals the impeller casing. |

> [!NOTE]
> **Manufacturing & Post-Processing Details**:
> 1. **Translucent Body**: All body components (Bowl, Lid, Bottom Cover, LED Cover) are SLA-printed using UTR-8100 translucent resin to ensure light transparency and visual monitoring of water levels.
> 2. **Food Safety Post-Processing**: SLA photopolymer resin is toxic to pets if untreated. All resin components in contact with water (Bowl, Lid) must be coated with a food-grade epoxy (e.g., Max CLR or similar FDA-compliant epoxy coating) as a post-processing step before use to completely seal the resin.
> 3. **Impeller and Drive Hub Post-Processing**: The impeller and drive hub are printed using PETG. High-precision post-processing (such as removing support residue from the 6 radial blades and ensuring clean magnet pockets) is necessary to keep the impeller dynamically balanced during rotation.
> 4. **Resin Coating & Tolerance Hand Filing**: Epoxy or polyurethane resin coating adds ~0.1–0.2 mm of surface film thickness. Because fine clearances (such as the impeller blade chamber clearances, guide post sleeve bore, and snap-fit retention rings/grooves on the bottom cover and pump cover) have tight mechanical tolerances, light hand-filing or wet-sanding of mating surfaces, snap ridges, and bearing sleeves may be required after applying resin coating to remove excess buildup and guarantee smooth rotation and reliable snap engagement.
> 5. **Drive Hub Standoff Ring & Shaft Fit**: The drive hub features an integrated annular standoff ring around the shaft hole on the bottom surface, acting as a built-in thrust washer that elevates the hub face 0.8 mm (exceeding typical 0.3 mm 3D-printing tolerances) to minimize rotational contact area and friction against the motor housing. The motor shaft bore is sized to 1.70 mm (0.85 mm radius) to accommodate 3D-printing hole shrinkage, enabling a smooth hand press-fit onto the 1.50 mm motor shaft and securing with a drop of medium CA glue or Loctite retaining compound.
> 6. **Impeller Post-Shoulder Elevation**: The impeller's central guide sleeve bore seats directly on the top surface of the bowl's 2.0 mm guide post flange shoulder. This elevates the entire wide 14 mm base disk 2.0 mm above the bowl floor (eliminating all surface friction against the floor), while the continuous radial blades extend smoothly from the central sleeve without empty radial gaps.
> 7. **Pump Cover Slide-On Mounting & Intake**: The pump cover features a vertical sleeve that slides down over the bowl's vertical delivery tube with conical lead-in chamfers on all entry rims for effortless tool-free alignment, drawing submerged water through a South-facing intake aperture into the eye of the impeller blades.
> 8. **Rubber Feet Depressions**: The underside of the bottom cover contains 4 symmetrically spaced circular recesses (pitch radius 78 mm, depth 1.0 mm, diameter 11.0 mm) to firmly seat self-adhesive anti-slip silicone/rubber bumper feet and prevent tabletop sliding.

### Fasteners & O-Rings

These fasteners and seal components are required to assemble the 3D-printed body parts and secure the mechanical and electrical sub-components:

| Component | Qty | Size/Spec | Use Case |
| :--- | :--- | :--- | :--- |
| **Bottom Cover Joint** | None | Snap-fit | The bottom cover features an integrated snap-fit ring around its perimeter to secure it to the bowl. |
| **Bottom Cover O-Ring** | 1 | Silicone O-Ring, 2mm cross-section, ~190mm ID / 194mm OD | Placed in the bottom cover groove to provide a water-tight seal for the dry compartment. |
| **Motor Screws** | 2 | M1.4 x 4mm (Machine Screws) | Secures the BetaFPV 1102 BLDC motor to the central boss face. |
| **Proximity Sensor Screws** | 6 | M2 x 6mm (Machine Screws) | Mounts the three proximity sensors to the internal bosses. |
| **Charger Board Screws** | 4 | M2 x 4mm or 5mm (Machine Screws) | Secures the Adafruit BQ25185 board to the dry compartment ceiling. |
| **Fuel Gauge Screws** | 2 | M2 x 4mm or 5mm (Machine Screws) | Secures the Adafruit MAX17048 board to the dry compartment ceiling. |
| **Pico W Screws** | 4 | M2 x 4mm or 5mm (Machine Screws) | Secures the Raspberry Pi Pico W to the dry compartment ceiling. |
| **Motor Driver Mount** | None | Integrated Slide-in Guide Clip | Mounts the SparkFun TMC6300 BLDC driver board without screws. |
| **Current Sensor Screws** | 2 | M2 x 4mm or 5mm (Machine Screws) | Secures the Adafruit INA219 board to the dry compartment ceiling. |

## Battery Life Estimation

To maximize portable operation on a single **18650 3.7V Li-ion battery (3000 mAh / 11.1 Wh)**, the system implements a strict low-power duty cycle. Below is the power consumption breakdown and the firmware settings required to achieve an estimated **7.0 days of battery life**.

### Power Consumption Breakdown

| Subsystem State | Components Active | Current Draw (at 3.7V) | Power Draw | Daily Duty Cycle |
| :--- | :--- | :--- | :--- | :--- |
| **Active Mode** (Cat detected, pump running) | Pico W (active), TMC6300 + BetaFPV 1102 BLDC Motor (typical load), INA219 (measuring), 3x VL53L0X (measuring), RGB LED (pulsing status) | **~670 mA** | 2480 mW | **2.08%** (15 events/day, 2 mins each) |
| **Sleep Mode** (Idle, monitoring proximity) | Pico W (light sleep), TMC6300 (disabled), INA219 (power-down), 3x VL53L0X (shutdown), RGB LED (off) | **~4.1 mA** | 15 mW | **97.92%** (wakes up 50ms every 5s to poll) |

### Calculations
* **Average Daily Current Draw**: 
  $$I_{\text{avg}} = (I_{\text{active}} \times 0.0208) + (I_{\text{sleep}} \times 0.9792) = (670\text{ mA} \times 0.0208) + (4.1\text{ mA} \times 0.9792) = 13.94\text{ mA} + 4.01\text{ mA} = 17.95\text{ mA}$$
* **Estimated Runtime**:
  $$\text{Runtime} = \frac{3000\text{ mAh}}{17.95\text{ mA}} \approx 167\text{ hours} \approx \mathbf{7.0\text{ days}}$$

### Required Settings for Optimal Battery Life

To achieve this estimate, the firmware and hardware must be configured with the following power-saving settings:

1. **Microcontroller Sleep Management**:
   * The Raspberry Pi Pico W must disable the onboard Wi-Fi/Bluetooth chip (`cyw43_arch_gpio_put` to set the power pin LOW) when offline.
   * Put the RP2040 microcontroller into **light sleep mode** using a hardware timer (RTC) to wake up every **5.0 seconds** to poll for cat presence.
2. **Proximity Sensor Hardware Shutdown**:
   * Pull the `XSHUT` (shutdown) pins of all three `VL53L0X` sensors **LOW** when the Pico enters sleep. This forces the sensors into a hardware standby drawing only **$5\mu\text{A}$** each, instead of leaving them active at $20\text{ mA}$ each.
   * On wakeup, enable only one sensor at a time (pull `XSHUT` HIGH), take a single-shot measurement, and immediately disable it again.
3. **Motor Driver & Current Sensor Standby State**:
   * When no cat is present, pull the TMC6300 driver enable pin (`GP18`) and phase inputs (`GP19`, `GP20`, `GP21`) **LOW** to put it in standby and cut off motor current draw (standby current < 1µA).
   * Put the INA219 current sensor into **power-down mode** over I2C to reduce its standby current to just 15µA.
   * Wake up the INA219 only when active pumping is triggered.
4. **Status LED Duty Cycling**:
   * The RGB NeoPixel status LED should remain **OFF** during sleep mode. For battery level indication, blink the LED briefly (e.g. 50ms pulse) once every 10 seconds rather than leaving it on continuously.
5. **Asynchronous Battery Monitoring**:
   * By utilizing the MAX17048 `ALRT` hardware interrupt wired to `GP14`, the Pico W avoids waking up periodically to poll the battery state of charge over I2C. The microcontroller can remain in a low-power state and receive battery alerts asynchronously.


