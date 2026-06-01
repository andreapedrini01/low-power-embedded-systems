# Quick Start Guide

## 📁 Project Files

```
EnergyAwarePath/
├── README.md                                        ← Start here (overview + results)
├── ADAPTIVE_EXTENSION.md                            ← Technical details
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
s    ← Show summary
```

## 📊 What to Expect

| Checkpoint | Budget | Exit Used | Latency |
|------------|--------|-----------|---------|
| 1 | 1.0 | Full model | ~0.55 ms |
| 2 | ~0.8 | Full model | ~0.55 ms |
| 3 | ~0.4 | Exit 1 | ~0.41 ms |
| 4 | ~0.2 | Linear formula | 0 ms |

**Key Result:** System automatically reduces inference complexity as energy budget decreases.

## 📖 Documentation

- **README.md**: Complete overview, architecture, results
- **ADAPTIVE_EXTENSION.md**: Implementation details, code structure
- **This file**: Quick reference

