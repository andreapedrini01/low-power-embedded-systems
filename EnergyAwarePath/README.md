# EnergyAwarePath - Adaptive Multi-Exit Inference

**Authors:** Pedrini, Bellini  
**Hardware:** Arduino Nano 33 BLE Sense Lite (nRF52840)

## Overview

EnergyAwarePath implements adaptive multi-exit inference for energy-aware path selection. The system adjusts inference complexity based on available energy budget.

### Features

- Multi-exit neural network with 3 inference levels
- Adaptive policy that selects appropriate exit based on budget
- On-device TinyML with int8 quantization
- IMU-based context awareness (accelerometer + gyroscope)

### Adaptive Behavior

At each checkpoint, the system evaluates 3 path branches. Inference complexity adapts to remaining budget:

| Budget | Exit | Latency | Accuracy |
|--------|------|---------|----------|
| ≥ 0.6 | Full model | ~0.55 ms | Highest |
| 0.3 - 0.6 | Exit 1 | ~0.41 ms | Good |
| < 0.3 | Linear formula | 0 ms | Acceptable |

## Project Structure

```
EnergyAwarePath/
├── README.md
├── ADAPTIVE_EXTENSION.md
├── EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb
└── EnergyAwarePath/
    ├── Energy-Aware-Path.ino
    ├── config.h
    ├── inference.cpp/h
    ├── planner.cpp/h
    ├── features.cpp/h
    ├── sensors.cpp/h
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
6    - Enable adaptive policy
n    - Execute checkpoint
s    - Show summary
r    - Reset
```

## Results

### Model Sizes

| Model | Flash | SRAM Arena | Parameters |
|-------|-------|------------|------------|
| Full | 3.3 KB | 712 bytes | ~321 |
| Exit 1 | 2.5 KB | 584 bytes | ~177 |

### Performance

| Metric | Value |
|--------|-------|
| Checkpoints completed | 3 / 4 |
| Energy consumed | 1.000 |
| Efficiency vs Oracle | 93.4% |
| Agreement with Oracle | 100% |
| Latency (full) | 0.54-0.56 ms |
| Latency (exit 1) | 0.41 ms |
| Latency reduction | 27% |

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
if (budget >= 0.6)       use full_model;
else if (budget >= 0.3)  use exit1_model;
else                     use linear_formula;
```

## Requirements

**Hardware:** Arduino Nano 33 BLE Sense Lite, USB cable

**Software:** Arduino IDE, Google Colab

**Libraries:** `Arduino_LSM9DS1`, `Harvard_TinyMLx`