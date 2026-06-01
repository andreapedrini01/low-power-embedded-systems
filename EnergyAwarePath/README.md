# EnergyAwarePath - Adaptive Multi-Exit Inference

**Low-Power Embedded Systems - Assignment 3**  
**Authors:** Pedrini, Bellini  
**Hardware:** Arduino Nano 33 BLE Sense Lite (nRF52840)

---

## Overview

EnergyAwarePath implements **adaptive multi-exit inference** for energy-aware path selection on a microcontroller. The system dynamically adjusts inference complexity based on available energy budget, inspired by the ePerceptive paper (SenSys '20).

### Key Features

- **Multi-exit neural network**: 3 inference levels with different accuracy/latency tradeoffs
- **Adaptive policy**: Automatically selects the appropriate exit based on energy budget
- **On-device TinyML**: Quantized int8 models running on 256KB SRAM
- **Context-aware decisions**: Uses IMU sensor data (accelerometer + gyroscope) for path evaluation

### How It Works

At each checkpoint, the system evaluates 3 path branches and selects the one with lowest predicted energy cost. The inference complexity adapts to the remaining budget:

| Budget Level | Exit Used | Latency | Accuracy |
|--------------|-----------|---------|----------|
| ≥ 0.6 | Full model (2 hidden layers) | ~0.55 ms | Highest |
| 0.3 - 0.6 | Exit 1 (1 hidden layer) | ~0.41 ms | Good |
| < 0.3 | Linear formula | 0 ms | Acceptable |

---

## Project Structure

```
EnergyAwarePath/
├── README.md                                        # This file
├── ADAPTIVE_EXTENSION.md                            # Technical specification
├── EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb  # Training notebook
└── EnergyAwarePath/                                 # Arduino sketch
    ├── Energy-Aware-Path.ino                        # Main file
    ├── config.h                                     # Configuration
    ├── inference.cpp/h                              # TFLite inference
    ├── planner.cpp/h                                # Decision logic
    ├── features.cpp/h                               # Feature extraction
    ├── sensors.cpp/h                                # IMU interface
    ├── model.h                                      # Full model (3.3 KB)
    └── model_exit1.h                                # Exit 1 model (2.5 KB)
```

---

## Reproducing the Results

### 1. Training (Google Colab)

1. Open `EnergyAwarePath_Training_MultiExit_COMPLETE.ipynb` in Google Colab
2. Run all cells (Runtime → Run all)
3. Download the generated files:
   - `model.h` (full model)
   - `model_exit1.h` (exit 1 model)

**Training details:**
- Dataset: 5400 synthetic samples (energy cost predictions)
- Architecture: Multi-exit MLP with shared layers
- Quantization: Full int8 for embedded deployment
- Training time: ~20 minutes on Colab

### 2. Arduino Deployment

1. Replace `model.h` and `model_exit1.h` in the Arduino sketch folder
2. Open `Energy-Aware-Path.ino` in Arduino IDE
3. Install required libraries:
   - `Arduino_LSM9DS1`
   - `Harvard_TinyMLx`
4. Compile and upload to Arduino Nano 33 BLE Sense Lite
5. Open Serial Monitor (115200 baud)

### 3. Testing

```
Commands:
  6    - Enable adaptive policy
  n    - Execute next checkpoint (repeat 4 times)
  s    - Show summary
  r    - Reset mission
```

**Expected behavior:**
- Checkpoint 1 (budget 1.0): Uses FULL MODEL
- Checkpoint 2 (budget ~0.8): Uses FULL MODEL
- Checkpoint 3 (budget ~0.4): Uses EXIT 1 (faster)
- Checkpoint 4 (budget ~0.2): Uses LINEAR FORMULA (if reached)

---

## Results

### Model Sizes

| Model | Flash Size | SRAM Arena | Parameters |
|-------|------------|------------|------------|
| Full model | 3.3 KB | 712 bytes | ~321 |
| Exit 1 model | 2.5 KB | 584 bytes | ~177 |

### Performance Metrics

From test run:

| Metric | Value |
|--------|-------|
| Checkpoints completed | 3 / 4 |
| Total energy consumed | 1.000 |
| Efficiency vs Oracle | 93.4% |
| Agreement with Oracle | 100% (3/3) |
| Latency (full model) | 0.54-0.56 ms |
| Latency (exit 1) | 0.41 ms |
| Latency reduction | 27% |

### Adaptive Behavior

The system successfully demonstrates graceful degradation:
- High budget → Maximum accuracy (full model)
- Medium budget → Reduced latency (exit 1)
- Low budget → Zero inference cost (formula)

---

## Technical Details

### Neural Network Architecture

**Full Model:**
```
Input(10) → Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

**Exit 1 Model:**
```
Input(10) → Dense(16, relu) → Dense(1, sigmoid)
```

Both models are trained jointly with weighted loss:
```python
loss_weights = {'exit1': 0.4, 'final': 1.0}
```

### Input Features (10 dimensions)

1. Energy budget (normalized)
2. Path length (normalized)
3. Number of turns (normalized)
4. Path difficulty (normalized)
5. Slope (normalized)
6. Accelerometer mean (from IMU)
7. Accelerometer std (from IMU)
8. Accelerometer peak (from IMU)
9. Gyroscope mean (from IMU)
10. Tilt angle (from IMU)

### Adaptive Policy Logic

```cpp
if (budget >= 0.6)
    use full_model;        // Maximum accuracy
else if (budget >= 0.3)
    use exit1_model;       // Balanced
else
    use linear_formula;    // Minimum cost
```

---

## Requirements

### Hardware
- Arduino Nano 33 BLE Sense Lite
- USB cable for programming

### Software
- Arduino IDE 1.8.x or 2.x
- Google Colab (for training)
- Python 3.x with TensorFlow 2.x (included in Colab)

### Libraries
- `Arduino_LSM9DS1` (IMU driver)
- `Harvard_TinyMLx` (TensorFlow Lite Micro)

---

## References

- **ePerceptive**: Montanari et al., "ePerceptive: Energy Reactive Embedded Intelligence for Batteryless Sensors", SenSys 2020
- **TinyML**: Pete Warden, Daniel Situnayake, "TinyML: Machine Learning with TensorFlow Lite on Arduino and Ultra-Low-Power Microcontrollers"

---

## License

Academic project for Low-Power Embedded Systems course.

