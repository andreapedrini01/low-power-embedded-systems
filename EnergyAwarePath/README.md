# EnergyAwarePath

**Low-Power Embedded Systems** — Assignment 3  
Pedrini, Bellini

## What this project is

A small TinyML demo that runs entirely on an Arduino Nano 33 BLE Sense. The board pretends to be a tiny robot navigating a series of checkpoints. At every checkpoint there are three candidate paths to choose from, each with its own length, slope, terrain difficulty and number of turns. The board reads the IMU for about a second to figure out what's happening physically (is it still? is it shaking? tilted?), then asks a quantized neural network to estimate how much energy each path would consume. It picks the cheapest one, subtracts the predicted cost from a simulated battery budget, and moves on.

The whole loop runs on-device, no cloud, no WiFi. Inference takes around half a millisecond.

## Why it matters

The real point isn't navigation. It's showing that a 3 KB neural network running on a 5-euro microcontroller can make decisions that depend on live sensor context. Replace the synthetic data with real telemetry from a robot, drone or wearable, and the same pipeline applies. The architecture is the lesson, not the specific scenario.

For more on what the project demonstrates and what it doesn't, see [`DATASET.md`](DATASET.md).

## Architecture

```
IMU (LSM9DS1) → 5 features → [budget + branch_meta + sensor_features]
                                       ↓
                              TFLite Micro (int8)
                                       ↓
                              predicted energy cost
                                       ↓
                              path selection + budget update
```

## Project layout

```
EnergyAwarePath/
├── EnergyAwarePath_Training.ipynb   Colab notebook (dataset, training, TFLite export)
├── DATASET.md                        How the synthetic dataset is built
├── README.md                         You are here
└── EnergyAwarePath/                 Arduino sketch folder
    ├── EnergyAwarePath.ino          Main loop, serial commands
    ├── config.h                     Branch definitions, normalization, quant params
    ├── sensors.h / sensors.cpp      IMU windowed acquisition (~1.2s @ 100Hz)
    ├── features.h / features.cpp    5-feature extraction
    ├── inference.h / inference.cpp  TFLite Micro setup and invoke
    ├── planner.h / planner.cpp      Branch evaluation + 5 decision policies
    ├── model.h                      Quantized model as C array (from Colab)
    └── model.tflite                 Same model in binary form (reference only)
```

## Workflow

### 1. Train the model on Colab

Open `EnergyAwarePath_Training.ipynb` in Google Colab and run every cell. The notebook generates the synthetic dataset, trains a small MLP, compares it against a linear regression baseline, quantizes to int8, and emits `model.h` ready for Arduino.

When training finishes, download two files from Colab:
- `model.h` → place in `EnergyAwarePath/EnergyAwarePath/`
- `model.tflite` → keep alongside it for documentation

The notebook also prints a block with `FEATURE_MIN` and `FEATURE_MAX` arrays. Copy those values into `config.h` so the on-device normalization matches the training distribution.

### 2. Flash the Arduino

In the Arduino IDE (or Arduino Cloud), install:
- `Arduino_LSM9DS1`
- `Harvard_TinyMLx` — provides the full TensorFlow Lite Micro stack

Select **Arduino Nano 33 BLE Sense** as the board, open `EnergyAwarePath.ino`, then compile and upload.

### 3. Run the demo

Open the Serial Monitor at **115200 baud**. Commands:

| Key | Action |
|---|---|
| `n` or `ENTER` | Trigger next checkpoint |
| `r` | Reset mission (budget back to 1.0) |
| `1` – `5` | Switch decision policy (1=ML, 2=Always-A, 3=Shortest, 4=Random, 5=Oracle) |
| `s` | Print mission summary |

Each `n` starts a 1.2-second IMU acquisition window. Move the board however you want during that window — keep it still, tilt it, shake it, swing it in a circle. The output shows the predicted cost for each branch, the chosen path, the remaining budget, and what an oracle policy (using the exact ground-truth formula) would have picked.

A typical run completes 4 checkpoints under budget when the ML policy works well. If you want to see the comparison clearly, run the same physical sequence under different policies and check the totals at the end.

## Model details

| Property | Value |
|---|---|
| Input | 10 features (1 budget + 4 branch + 5 IMU) |
| Architecture | Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid) |
| Parameters | ~321 |
| Quantization | full int8 |
| Model size | ~3.3 KB |
| Tensor arena | 8 KB allocated, ~712 B used |
| Inference time | ~0.5 ms |
| Test MAE (quantized) | ~0.012 |

## Decision policies

| Policy | What it does |
|---|---|
| ML | Neural network prediction (the main subject of the project) |
| Always-A | Trivial baseline: always picks branch A |
| Shortest | Picks the branch with smallest `length` |
| Random | Uniform random pick |
| Oracle | Uses the ground-truth formula directly — upper bound for comparison |

The point of the baselines is to show that the ML policy actually adds value over naive strategies, while staying close to the oracle.
