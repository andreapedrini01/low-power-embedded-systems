# Quick Start Guide

## 📁 Project Files

```
EnergyAwarePath/
├── README.md                                        ← Start here (overview + results)
├── ADAPTIVE_EXTENSION.md                            ← Adaptive multi-exit spec
├── ENERGY_EXTENSION.md                              ← Energy model and sleep spec
├── QUICKSTART.md                                    ← This file
├── EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb  ← Training notebook
└── EnergyAwarePath/                                 ← Arduino sketch (ready to upload)
```

## 🚀 Reproduce in 3 Steps

### 1. Train Models (Google Colab)
- Open `EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb`
- Runtime → Run all (~20 min)
- Download `model.h` and `model_exit1.h`

### 2. Deploy to Arduino
- Replace model files in `EnergyAwarePath/` folder
- Upload sketch to Arduino Nano 33 BLE Sense Lite
- Open Serial Monitor (115200 baud)

### 3. Test Adaptive Policy
```
6    ← Enable adaptive policy
n    ← Run checkpoint (repeat 4 times)
s    ← Show summary (energy, savings)
b    ← Batch comparison across all policies
```

## 📊 What to Expect

| Checkpoint | Budget | Exit Used      | Inference | Post-checkpoint sleep |
|------------|--------|----------------|-----------|-----------------------|
| 1          | 1.0    | Full model     | ~0.55 ms  | 0 ms                  |
| 2          | ~0.8   | Full model     | ~0.55 ms  | 0 ms                  |
| 3          | ~0.4   | Exit 1         | ~0.41 ms  | 500 ms (deep)         |
| 4          | ~0.2   | Linear formula | 0 ms      | 2000 ms (deep)        |

**Key Result:** System reduces both inference complexity *and* idle current as
the energy budget decreases. Mission summary reports total energy (estimated)
and savings vs an Always-Full + no-sleep baseline.

## 📖 Documentation

- **README.md**: Complete overview, architecture, results
- **ADAPTIVE_EXTENSION.md**: Multi-exit model and adaptive inference details
- **ENERGY_EXTENSION.md**: Power model, sleep API, energy accounting
- **This file**: Quick reference

