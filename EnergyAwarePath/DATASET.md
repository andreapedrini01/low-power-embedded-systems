# Dataset Construction

This document describes how the synthetic training dataset for `EnergyAwarePath` is built. The dataset is fully synthetic and generated inside the Colab notebook (`EnergyAwarePath_Training.ipynb`).

## Overview

The dataset is constructed at three levels:
1. Synthetic IMU windows simulating different motion conditions
2. Feature extraction from each window (5 features)
3. Cartesian combination with branch metadata and energy budget values
4. Target (energy cost) computed via a known formula plus noise

Final size: **10.800 training samples**, each with **10 input features** and **1 target value**.

## Step 1 — Synthetic IMU Window Generation

For each of 5 motion conditions, we generate 20 windows of 119 samples (~1.2s at 100 Hz). Total: **100 IMU windows**.

| Condition | What it simulates | Characteristics |
|---|---|---|
| `still` | Board resting on a table | acc ≈ (0, 0, 1g) with minimal noise, gyro ≈ 0 with small drift |
| `tilt` | Board tilted 15°–60° | acc rotated according to angle, gyro low |
| `light_shake` | Light shaking | acc oscillating at 4–5 Hz, gyro up to 25 deg/s |
| `heavy_shake` | Hard shaking | acc up to ±2g, gyro up to 100 deg/s |
| `smooth_move` | Smooth circular motion | acc/gyro sinusoidal at 1–1.5 Hz |

Numerical parameters (means, standard deviations, amplitudes) are chosen to approximate what the real LSM9DS1 IMU on the Arduino Nano 33 BLE Sense produces. They were tuned after observing real Serial logs from the board.

## Step 2 — Feature Extraction

From each IMU window we compute 5 features. These are the **same formulas** that run on the Arduino at inference time, ensuring distribution alignment between training and deployment.

| Feature | Formula | What it captures |
|---|---|---|
| `acc_mean` | mean of `sqrt(ax² + ay² + az²)` | average acceleration magnitude |
| `acc_std` | std of acceleration magnitude | vibration level |
| `acc_peak` | max of acceleration magnitude | hard impacts / spikes |
| `gyro_mean` | mean of `sqrt(gx² + gy² + gz²)` | rotation rate |
| `tilt` | `atan2(|mean_ax|, |mean_az|)` in degrees | board orientation |

After this step we have **100 vectors of 5 features**, each tagged with its source condition.

## Step 3 — Combination with Branch Metadata and Budget

Each IMU feature vector is combined with:
- **12 branch configurations** (4 checkpoints × 3 branches per checkpoint, see `config.h`)
- **9 budget values** equispaced from 0.2 to 1.0

Total combinations: `100 × 12 × 9 = 10.800 samples`.

Each sample is a 10-dimensional vector:

```
[budget, length, turns/5, difficulty, slope,    ← 5 scenario values
 acc_mean, acc_std, acc_peak, gyro_mean, tilt]  ← 5 IMU context values
```

### Input semantics

| Input | Concept | Source at runtime |
|---|---|---|
| `budget` | Battery state | Internal state |
| `length` | Path length (0–1) | Static map (`config.h`) |
| `turns/5` | Number of turns, normalized | Static map |
| `difficulty` | Terrain difficulty (0–1) | Static map |
| `slope` | Slope (-1 to +1) | Static map |
| `acc_mean` | Average acceleration magnitude | Live IMU |
| `acc_std` | Acceleration variability (vibration) | Live IMU |
| `acc_peak` | Peak acceleration | Live IMU |
| `gyro_mean` | Rotation rate | Live IMU |
| `tilt` | Tilt from vertical | Live IMU |

## Step 4 — Target (Energy Cost) Calculation

For each sample, the energy cost is computed using a known formula. The model's job is to approximate this function from the inputs alone.

```
motion = (acc_std_norm + gyro_norm) / 2

E = 0.18 × length              ← longer paths cost more
  + 0.12 × turns/5             ← turns cost
  + 0.12 × difficulty          ← rough terrain costs
  + 0.08 × |slope|             ← slope costs (uphill or downhill)
  + 0.05 × acc_std_norm        ← vibration costs
  + 0.05 × gyro_norm           ← rotation costs
  + 0.05 × (1 - budget)        ← low battery = more effort
  + 0.30 × length × motion     ← INTERACTION: long paths under motion cost much more
```

Where `acc_std_norm` and `gyro_norm` are the sensor features rescaled to [0, 1] using the global min/max observed across the 100 IMU windows.

The maximum theoretical sum is ~1.0, so `E ∈ [0.05, 0.95]` after clipping.

### Gaussian noise

We add `N(0, 0.02)` to each target value. This:
- Prevents the model from learning the exact formula
- Simulates realistic measurement uncertainty
- Forces the model to generalize rather than memorize

### Why the interaction term matters

Without the `length × motion` term, the cheapest branch (long, easy, downhill) would always dominate regardless of context, since the static cost of a calm path is always lower than a difficult short one. The interaction term encodes a physical intuition: **long exposed paths become disproportionately expensive when the system is shaking or rotating**, because cumulative wear and energy waste scale with both factors. With this term, the optimal choice flips between branches depending on the live IMU context, which is the whole point of the project.

## Train/Test Split

- 80% train (8640 samples)
- 20% test (2160 samples)
- Random shuffle with fixed seed (42)

## Why This Structure Works

The key insight is that **few sensor contexts are leveraged into a large dataset** by Cartesian combination with scenario parameters. 100 IMU windows + 12 branches + 9 budget levels = 10.800 samples. Each IMU window appears 108 times, paired with all possible scenario parameter combinations.

This forces the model to learn separable contributions:
- What weighs in the path (length, difficulty, slope, turns)
- What weighs in the context (vibration, rotation, residual budget)
- How they interact (length × motion)

## Honest Limitations

These points should be stated clearly in the final report:

- The dataset is synthetic, generated from a known formula. The model is learning to approximate a function we wrote by hand.
- This is a deliberately "circular" exercise: we know the ground truth because we generated it. In a real project, the targets would come from telemetry of past missions where actual energy consumption was logged.
- The motion conditions are simplified models of real sensor behavior. They cover the input range but do not capture all real-world variability (e.g., specific surfaces, mechanical resonances).
- The branch metadata is hardcoded. In a real navigation system it would come from a map or planner.

## What This Project Actually Demonstrates

The value is **not in the dataset** but in the pipeline:

1. Synthetic data → feature extraction → training → int8 quantization → on-device inference → context-aware decision
2. A 3.3 KB model running in 0.5 ms on a Cortex-M4 with 712 bytes of arena
3. A model that adapts decisions to live sensor context, not just to static parameters
4. A baseline comparison (linear regression, oracle, random, fixed) that quantifies what the ML adds

The same pipeline applied to real telemetry data would produce a working energy-aware planner for an actual robot. The architectural pattern transfers directly.

## Reproducibility

Random seeds are fixed:
- `np.random.seed(42)`
- `tf.random.set_seed(42)`

Re-running the notebook with the same code produces identical results, modulo TFLite quantization which can have minor variation across runs.
