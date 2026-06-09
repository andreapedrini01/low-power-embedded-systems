# Arduino Nano 33 BLE Sense - Datasheet (ABX00031)

> Source: Arduino official user manual, SKU ABX00031, modified 28/05/2026.
> This file is a faithful conversion of `ABX00031-datasheet.pdf` for use as
> reference material. Section numbering matches the original PDF.

---

## Description

Arduino Nano 33 BLE Sense is a miniature module containing a NINA-B306 module,
based on the Nordic nRF52840, an Arm Cortex-M4F, a crypto chip able to securely
store certificates and pre-shared keys, and a 9-axis IMU. The module can be
mounted as a DIP component (with pin headers) or as an SMT component, soldering
it directly via the castellated pads.

**Target areas:** Maker, enhancements, IoT applications.

---

## Features

### NINA-B306 Module

**Processor**
- 64 MHz Arm Cortex-M4F (with FPU)
- 1 MB Flash + 256 KB RAM

**Bluetooth 5 multiprotocol radio**
- 2 Mbps
- CSA #2
- Advertising Extensions
- Long Range
- +8 dBm TX power
- -95 dBm sensitivity
- **4.8 mA in TX (0 dBm)**
- **4.6 mA in RX (1 Mbps)**
- Integrated balun with 50 Ω single-ended output

**IEEE 802.15.4 radio support**
- Thread
- Zigbee

**Peripherals**
- Full-speed 12 Mbps USB
- NFC-A tag
- Arm CryptoCell CC310 security subsystem
- QSPI/SPI/TWI/I²S/PDM/QDEC
- High-speed 32 MHz SPI
- Quad SPI interface 32 MHz
- EasyDMA for all digital interfaces
- 12-bit 200 ksps ADC
- 128-bit AES/ECB/CCM/AAR co-processor

### LSM9DS1 (9-axis IMU)
- 3 acceleration channels, 3 angular rate channels, 3 magnetic field channels
- ±2/±4/±8/±16 g linear acceleration full scale
- ±4/±8/±12/±16 gauss magnetic full scale
- ±245/±500/±2000 dps angular rate full scale
- 16-bit data output

### LPS22HB (Barometer and Temperature Sensor)
- 260 to 1260 hPa absolute pressure range with 24-bit precision
- High overpressure capability: 20× full-scale
- Embedded temperature compensation
- 16-bit temperature data output
- 1 Hz to 75 Hz output data rate
- Interrupts: Data Ready, FIFO flags, pressure thresholds

### HTS221 (Relative Humidity Sensor)
- 0–100% relative humidity range
- High rH sensitivity: 0.004% rH/LSB
- Humidity accuracy: ±3.5% rH, 20 to +80% rH
- Temperature accuracy: ±0.5 °C, 15 to +40 °C
- 16-bit humidity and temperature output data

### APDS-9960 (Proximity, Ambient Light, RGB and Gesture Sensor)
- Ambient light and RGB color sensing with UV and IR blocking filters
- Very high sensitivity (suited for operation behind dark glass)
- Proximity sensing with ambient light rejection
- Complex gesture sensing

### MP34DT05 (Digital Microphone)
- AOP = 122.5 dBSPL
- 64 dB signal-to-noise ratio
- Omnidirectional sensitivity
- −26 dBFS ± 3 dB sensitivity

### ATECC608A (Crypto Chip)
- Cryptographic co-processor with secure hardware-based key storage
- Protected storage for up to 16 keys, certificates or data
- ECDH: FIPS SP800-56A Elliptic Curve Diffie-Hellman
- NIST standard P256 elliptic curve support
- SHA-256 and HMAC hash including off-chip context save/restore
- AES-128 encrypt/decrypt, Galois field multiply for GCM

### MPM3610 DC-DC
- Regulates input voltage up to 21 V with a minimum of 65% efficiency at minimum load
- More than 85% efficiency at 12 V

---

## 1. The Board

The Nano 33 BLE Sense, like all Nano form factor boards, does not have a battery
charger but can be powered through USB or headers.

> NOTE: Nano 33 BLE Sense supports only 3.3 V I/Os and is NOT 5 V tolerant.
> Do not connect 5 V signals directly or the board will be damaged. The 5 V
> pin does NOT supply voltage; it is connected, through a jumper, to the USB
> power input.

### 1.1 Ratings

#### 1.1.1 Recommended Operating Conditions

| Symbol | Description                                  | Min            | Max            |
|--------|----------------------------------------------|----------------|----------------|
| —      | Conservative thermal limits for the board    | -40 °C (-40 °F)| 85 °C (185 °F) |

### 1.2 Power Consumption

| Symbol | Description                                  | Min | Typ | Max | Unit |
|--------|----------------------------------------------|-----|-----|-----|------|
| PBL    | Power consumption with busy loop             | —   | TBC | —   | mW   |
| PLP    | Power consumption in low-power mode          | —   | TBC | —   | mW   |
| PMAX   | Maximum power consumption                    | —   | —   | TBC | mW   |

> **TBC = To Be Confirmed.** The datasheet does not specify concrete values for
> board-level active, low-power or maximum consumption. The only consumption
> figures provided in the document are for the BLE radio (see Features above):
> 4.8 mA TX @ 0 dBm and 4.6 mA RX @ 1 Mbps.

---

## 2. Functional Overview

### 2.1 Board Topology

**Top side**

| Ref. | Description                                | Ref. | Description                       |
|------|--------------------------------------------|------|-----------------------------------|
| U1   | NINA-B306 Bluetooth Low Energy 5.0 Module  | U6   | MP2322GQH Step-Down Converter     |
| U2   | LSM9DS1TR IMU                              | PB1  | IT-1185AP1C-160G-GTR Push button  |
| U3   | MP34DT06JTR MEMS Microphone                | HS-1 | HTS221 Humidity Sensor            |
| U4   | ATECC608A Crypto Chip                      | DL1  | LED L                             |
| U5   | APDS-9660 Ambient Module                   | DL2  | LED Power                         |

**Bottom side**

| Ref. | Description     | Ref. | Description     |
|------|-----------------|------|-----------------|
| SJ1  | VUSB Jumper     | SJ2  | D7 Jumper       |
| SJ3  | 3V3 Jumper      | SJ4  | D8 Jumper       |

### 2.2 Processor

The main processor is an Arm Cortex-M4F running at up to 64 MHz. Most of its
pins are connected to the external headers; some are reserved for internal
communication with the wireless module and the on-board internal I²C
peripherals (IMU and Crypto).

> NOTE: Pins A4 and A5 have an internal pull-up and default to be used as an
> I²C bus, so usage as analog inputs is not recommended.

### 2.3 Crypto

The crypto chip provides a secure way to store secrets (such as certificates)
and accelerates secure protocols while never exposing secrets in plain text.

### 2.4 IMU

The 9-axis IMU can be used to measure board orientation (gravity vector or 3D
compass) or to measure shocks, vibration, acceleration and rotation speed.

### 2.5 Barometer and Temperature Sensor

Measures ambient pressure. The integrated temperature sensor can be used to
compensate the pressure measurement.

### 2.6 Relative Humidity and Temperature Sensor

Measures ambient relative humidity. Includes an integrated temperature sensor
that can be used to compensate the measurement.

### 2.7 Digital Proximity, Ambient Light, RGB and Gesture Sensor

#### 2.7.1 Gesture Detection
Four directional photodiodes sense reflected IR energy (sourced by the
integrated LED) to convert physical motion (velocity, direction, distance) to
digital information. Features automatic activation, ambient light subtraction,
cross-talk cancellation, dual 8-bit data converters, power-saving inter-conversion
delay, 32-dataset FIFO, and interrupt-driven I²C communication.

#### 2.7.2 Proximity Detection
Distance measurement via photodiode detection of reflected IR energy.
Detect/release events are interrupt-driven, triggering when proximity result
crosses upper or lower threshold settings. The proximity engine includes
offset adjustment registers and automatic ambient light subtraction.

#### 2.7.3 Color and ALS Detection
Provides red, green, blue and clear light intensity data. Each R, G, B, C
channel has UV and IR blocking filters and a dedicated 16-bit data converter
producing simultaneous output.

### 2.8 Digital Microphone

The MP34DT05 is an ultra-compact, low-power, omnidirectional, digital MEMS
microphone built with a capacitive sensing element and an IC interface.

### 2.9 Power Tree

The board can be powered via USB connector, VIN or VUSB pins on headers.

> NOTE: Since VUSB feeds VIN via a Schottky diode, and the DC-DC regulator's
> minimum specified input voltage is 4.5 V, the minimum supply voltage from
> USB has to be in the 4.8 V to 4.96 V range, depending on the current drawn.

---

## 3. Board Operation

### 3.1 Getting Started - IDE
Install the Arduino Desktop IDE. Connect the Nano 33 BLE Sense to the computer
with a Micro-B USB cable, which also provides power.

### 3.2 Getting Started - Arduino Cloud Editor
The Arduino Cloud Editor works out-of-the-box with a simple plugin.

### 3.3 Getting Started - Arduino Cloud
Arduino Cloud allows logging, graphing and analyzing sensor data, triggering
events, and automating home or business tasks.

### 3.4 Sample Sketches
Available in the "Examples" menu of the Arduino IDE or in the Documentation
section of the Arduino Pro website.

### 3.5 Online Resources
ProjectHub, Arduino Library Reference, and the Arduino Store.

### 3.6 Board Recovery
All Arduino boards have a built-in bootloader that allows flashing via USB. If
a sketch locks up the processor and the board is not reachable via USB,
bootloader mode can be entered by double-tapping the reset button right after
power up.

---

## 4. Connector Pinouts

### 4.1 USB

| Pin | Function | Type         | Description                                                 |
|-----|----------|--------------|-------------------------------------------------------------|
| 1   | VUSB     | Power        | Power Supply Input. If the board is powered via VUSB header, this is an Output (1) |
| 2   | D-       | Differential | USB differential data −                                     |
| 3   | D+       | Differential | USB differential data +                                     |
| 4   | ID       | Analog       | Selects Host/Device functionality                           |
| 5   | GND      | Power        | Power Ground                                                |

### 4.2 Headers

The board exposes two 15-pin connectors that can be assembled with pin headers
or soldered through castellated vias.

| Pin | Function   | Type      | Description                                                   |
|-----|------------|-----------|---------------------------------------------------------------|
| 1   | D13        | Digital   | GPIO                                                          |
| 2   | +3V3       | Power Out | Internally generated power output to external devices         |
| 3   | AREF       | Analog    | Analog reference; can be used as GPIO                         |
| 4   | A0/DAC0    | Analog    | ADC in / DAC out; can be used as GPIO                         |
| 5   | A1         | Analog    | ADC in; can be used as GPIO                                   |
| 6   | A2         | Analog    | ADC in; can be used as GPIO                                   |
| 7   | A3         | Analog    | ADC in; can be used as GPIO                                   |
| 8   | A4/SDA     | Analog    | ADC in; I²C SDA; can be used as GPIO (1)                      |
| 9   | A5/SCL     | Analog    | ADC in; I²C SCL; can be used as GPIO (1)                      |
| 10  | A6         | Analog    | ADC in; can be used as GPIO                                   |
| 11  | A7         | Analog    | ADC in; can be used as GPIO                                   |
| 12  | VUSB       | Power I/O | Normally NC; can be connected to VUSB by shorting a jumper    |
| 13  | RST        | Digital   | Active-low reset input (duplicate of pin 18)                  |
| 14  | GND        | Power     | Power Ground                                                  |
| 15  | VIN        | Power In  | VIN power input                                               |
| 16  | TX         | Digital   | USART TX; can be used as GPIO                                 |
| 17  | RX         | Digital   | USART RX; can be used as GPIO                                 |
| 18  | RST        | Digital   | Active-low reset input (duplicate of pin 13)                  |
| 19  | GND        | Power     | Power Ground                                                  |
| 20  | D2         | Digital   | GPIO                                                          |
| 21  | D3/PWM     | Digital   | GPIO; can be used as PWM                                      |
| 22  | D4         | Digital   | GPIO                                                          |
| 23  | D5/PWM     | Digital   | GPIO; can be used as PWM                                      |
| 24  | D6/PWM     | Digital   | GPIO; can be used as PWM                                      |
| 25  | D7         | Digital   | GPIO                                                          |
| 26  | D8         | Digital   | GPIO                                                          |
| 27  | D9/PWM     | Digital   | GPIO; can be used as PWM                                      |
| 28  | D10/PWM    | Digital   | GPIO; can be used as PWM                                      |
| 29  | D11/MOSI   | Digital   | SPI MOSI; can be used as GPIO                                 |
| 30  | D12/MISO   | Digital   | SPI MISO; can be used as GPIO                                 |

### 4.3 Debug

On the bottom side, under the communication module, debug signals are arranged
as 3×2 test pads with 100 mil pitch, with pin 4 removed.

| Pin | Function | Type      | Description                                |
|-----|----------|-----------|--------------------------------------------|
| 1   | +3V3     | Power Out | Internally generated voltage reference     |
| 2   | SWD      | Digital   | nRF52840 Single Wire Debug Data            |
| 3   | SWCLK    | Digital In| nRF52840 Single Wire Debug Clock           |
| 5   | GND      | Power     | Power Ground                               |
| 6   | RST      | Digital In| Active-low reset input                     |

---

## 5. Mechanical Information

### 5.1 Board Outline and Mounting Holes

The board measures are mixed between metric and imperial. Imperial measures
maintain the 100 mil pitch grid between pin rows so the board fits a
breadboard; the board length is metric.

---

## 6. Certifications

### 6.1 Declaration of Conformity CE DoC (EU)
Arduino declares the product is in conformity with the essential requirements
of the listed EU Directives, qualifying for free movement within the EU/EEA.

### 6.2 Declaration of Conformity to EU RoHS & REACH 211 01/19/2021

| Substance                                | Maximum limit (ppm) |
|------------------------------------------|---------------------|
| Lead (Pb)                                | 1000                |
| Cadmium (Cd)                             | 100                 |
| Mercury (Hg)                             | 1000                |
| Hexavalent Chromium (Cr⁶⁺)               | 1000                |
| Polybrominated Biphenyls (PBB)           | 1000                |
| Polybrominated Diphenyl Ethers (PBDE)    | 1000                |
| Bis(2-Ethylhexyl) phthalate (DEHP)       | 1000                |
| Benzyl butyl phthalate (BBP)             | 1000                |
| Dibutyl phthalate (DBP)                  | 1000                |
| Diisobutyl phthalate (DIBP)              | 1000                |

No exemptions are claimed. Products are also fully compliant with REACH (EC)
1907/2006.

### 6.3 Conflict Minerals Declaration
Arduino does not directly source or process conflict minerals (Tin, Tantalum,
Tungsten, Gold). Conflict minerals appear in products only as solder or alloy
components. Suppliers have been verified for compliance.

---

## 7. FCC Caution

This device complies with part 15 of the FCC Rules. Operation is subject to
the following two conditions:

1. This device may not cause harmful interference.
2. This device must accept any interference received, including interference
   that may cause undesired operation.

**FCC RF Radiation Exposure Statement**
- This transmitter must not be co-located or operating in conjunction with any
  other antenna or transmitter.
- Equipment complies with RF radiation exposure limits set forth for an
  uncontrolled environment.
- Equipment should be installed and operated with a minimum distance of 20 cm
  between the radiator and the body.

> Operating temperature of the EUT cannot exceed 85 °C and shall not be lower
> than -40 °C.

| Frequency band   | Maximum output power (ERP) |
|------------------|----------------------------|
| 863–870 MHz      | 5.47 dBm                   |

---

## 8. Company Information

- **Company name:** Arduino S.r.l.
- **Address:** Via Andrea Appiani 25, 20900 Monza, Italy

---

## 9. Reference Documentation

| Reference                                | Link                                                                                     |
|------------------------------------------|------------------------------------------------------------------------------------------|
| Arduino IDE (Desktop)                    | https://www.arduino.cc/en/software                                                       |
| Arduino Cloud Editor                     | https://create.arduino.cc/editor                                                         |
| Arduino Cloud Editor - Getting Started   | https://docs.arduino.cc/arduino-cloud/guides/editor/                                     |
| Arduino Project Hub                      | https://create.arduino.cc/projecthub?by=part&part_id=11332&sort=trending                 |
| Library Reference                        | https://www.arduino.cc/reference/en/                                                     |
| Forum                                    | http://forum.arduino.cc/                                                                 |
| NINA-B306 datasheet                      | https://content.u-blox.com/sites/default/files/NINA-B3_DataSheet_UBX-17052099.pdf        |
| ATECC608A datasheet                      | Microchip DS40001977B                                                                    |
| MPM3610 datasheet                        | https://www.monolithicpower.com/pub/media/document/MPM3610_r1.01.pdf                     |
| ATECC608A library                        | https://github.com/arduino-libraries/ArduinoECCX08                                       |
| LSM9DS1 library                          | https://github.com/adafruit/Adafruit_LSM9DS1                                             |
| LPS22HB library                          | https://github.com/stm32duino/LPS22HB                                                    |
| HTS221 library                           | https://github.com/stm32duino/HTS221                                                     |
| APDS9960 library                         | https://github.com/adafruit/Adafruit_APDS9960                                            |

---

## 10. Revision History

| Date       | Revision | Changes                                  |
|------------|----------|------------------------------------------|
| 25/04/2024 | 3        | Updated link to new Cloud Editor         |
| 03/08/2022 | 2        | Reference documentation links updates    |
| 27/04/2021 | 1        | General datasheet updates                |

---

## Summary Of Power-Relevant Facts (For Project Reference)

This is a derived summary of every power-related number actually present in the
datasheet. It is not part of the original document.

| Quantity                                  | Value          | Source section |
|-------------------------------------------|----------------|----------------|
| Operating voltage (logic)                 | 3.3 V          | 1, 2.9         |
| Minimum USB supply voltage (to feed VIN)  | 4.8 to 4.96 V  | 2.9            |
| Max VIN accepted by DC-DC                 | 21 V           | Features       |
| DC-DC efficiency at minimum load          | ≥ 65%          | Features       |
| DC-DC efficiency at 12 V                  | > 85%          | Features       |
| BLE radio current, TX @ 0 dBm             | 4.8 mA         | Features       |
| BLE radio current, RX @ 1 Mbps            | 4.6 mA         | Features       |
| Operating temperature range               | -40 to 85 °C   | 1.1.1          |
| Active power consumption (busy loop)      | **TBC**        | 1.2            |
| Low-power-mode consumption                | **TBC**        | 1.2            |
| Maximum power consumption                 | **TBC**        | 1.2            |

The datasheet does **not** specify board-level active or sleep current. Any
project that requires absolute energy figures must either (a) measure them
directly with a power meter, or (b) cite the values listed in the linked
NINA-B306 / nRF52840 datasheets.
