# EnergyAwarePath - Adaptive Multi-Exit Inference

**Authors:** Pedrini, Bellini  
**Hardware:** Arduino Nano 33 BLE Sense Lite (nRF52840)

## Overview

EnergyAwarePath implements adaptive multi-exit inference for energy-aware path
selection. The system adjusts inference complexity and idle behavior based on
the remaining energy budget, demonstrating both latency and energy savings.

### Features

- Multi-exit neural network with 3 inference levels
- Adaptive policy that selects exit + sleep duration based on budget
- Real sleep between checkpoints and during IMU sampling idle gaps
- On-device TinyML with int8 quantization
- IMU-based context awareness (accelerometer + gyroscope)
- Per-checkpoint and per-mission energy estimation
- Batch comparison mode for fair multi-policy evaluation

### Adaptive Behavior

At each checkpoint the system evaluates 3 path branches. Both inference cost
and post-checkpoint sleep adapt to the remaining budget:

| Budget       | Inference        | Post-checkpoint sleep |
|--------------|------------------|-----------------------|
| ≥ 0.6        | Full model       | 0 ms                  |
| 0.3 – 0.6    | Exit 1           | 500 ms (deep)         |
| < 0.3        | Linear formula   | 2000 ms (deep)        |

## Project Structure

```
EnergyAwarePath/
├── README.md
├── ADAPTIVE_EXTENSION.md
├── ENERGY_EXTENSION.md
├── EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb
└── EnergyAwarePath/
    ├── Energy-Aware-Path.ino
    ├── config.h
    ├── inference.cpp/h
    ├── planner.cpp/h
    ├── features.cpp/h
    ├── sensors.cpp/h
    ├── power.cpp/h
    ├── model.h
    └── model_exit1.h
```

## Setup

### Training

1. Open `EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb` in Colab
2. Run all cells
3. Download `model.h` and `model_exit1.h`

### Deployment

1. Copy model files to Arduino sketch folder
2. Install libraries: `Arduino_LSM9DS1`, `Harvard_TinyMLx`
3. Upload to Arduino Nano 33 BLE Sense Lite
4. Open Serial Monitor (115200 baud)

### Usage

```
n    - Execute next checkpoint
s    - Show summary
r    - Reset mission
b    - Batch comparison (all policies, shared sensor window)
1    - Policy: ML (always full)
2    - Policy: Always-A
3    - Policy: Shortest
4    - Policy: Random
5    - Policy: Oracle
6    - Policy: Adaptive (exit + sleep selection)
```

## Energy Model

All energy figures are **estimates** derived from time measurements on the
board multiplied by the constants below. Concrete absolute values from the
official datasheet (`ABX00031-datasheet.md`, section 1.2) are listed as TBC,
so the constants come from external sources.

| Constant            | Value      | Source                                                  |
|---------------------|------------|---------------------------------------------------------|
| `V_BUS`             | 5.2 V      | USB power-meter on this board                           |
| `I_ACTIVE_MA`       | 32.0 mA    | USB power-meter, board running                          |
| `I_LIGHT_SLEEP_UA`  | 300 µA     | Community-reported, basic `delay()` with timers running |
| `I_DEEP_SLEEP_UA`   | 5 µA       | Community-reported, timers + USB Phy off                |

`E_µJ = V × (I_active × t_active + I_light × t_light + I_deep × t_deep)`.

See `ENERGY_EXTENSION.md` for derivation, sleep API choice, and update path
when direct sleep measurements become available. All printed energy values
are tagged `(est)` to mark them as estimates.

## Results

### Model Sizes

| Model  | Flash  | SRAM Arena | Parameters |
|--------|--------|------------|------------|
| Full   | 3.3 KB | 712 bytes  | ~321       |
| Exit 1 | 2.5 KB | 584 bytes  | ~177       |

### Performance

Single-mission run (4 checkpoints, mixed sensor context). Latency values are
measured; energy values are estimates per the model above. Fill in after a
demo run on the board.

| Policy       | Cost  | Active (ms) | Sleep (ms) | Energy (mJ, est) | Energy saved vs Always-Full+no-sleep |
|--------------|-------|-------------|------------|------------------|--------------------------------------|
| ML (full)    | TBD   | TBD         | TBD        | TBD              | 0% (baseline)                        |
| Adaptive     | TBD   | TBD         | TBD        | TBD              | TBD                                  |
| Always-A     | TBD   | TBD         | TBD        | TBD              | TBD                                  |
| Shortest     | TBD   | TBD         | TBD        | TBD              | TBD                                  |

### Latency

| Inference path | Latency        |
|----------------|----------------|
| Full model     | 0.54 – 0.56 ms |
| Exit 1         | 0.41 ms        |
| Linear formula | 0 ms           |

## Architecture

**Full Model:**
```
Input(10) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

**Exit 1:**
```
Input(10) → Dense(16, relu) → Dense(1, sigmoid)
```

Training uses weighted loss: `{'exit1': 0.4, 'final': 1.0}`

### Input Features

1. Energy budget
2. Path length
3. Turns (normalized)
4. Difficulty
5. Slope
6-10. IMU features (acc_mean, acc_std, acc_peak, gyro_mean, tilt)

### Adaptive Logic

```cpp
// Inference exit
if (budget >= 0.6)       use full_model;
else if (budget >= 0.3)  use exit1_model;
else                     use linear_formula;

// Post-checkpoint sleep
if (budget >= 0.6)       sleep =    0 ms;
else if (budget >= 0.3)  sleep =  500 ms;
else                     sleep = 2000 ms;
```

## Requirements

**Hardware:** Arduino Nano 33 BLE Sense Lite, USB cable, optional USB power
meter for sleep-current validation.

**Software:** Arduino IDE, Google Colab.

**Libraries:** `Arduino_LSM9DS1`, `Harvard_TinyMLx`.
