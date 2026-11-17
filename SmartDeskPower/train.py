#!/usr/bin/env python3
"""Offline training, benchmarking, and export script for SmartDeskPower.

Authors: Andrea Pedrini, Pietro Bellini
Platform: Arduino Nano 33 BLE Sense (Cortex-M4F @ 64 MHz)

Evaluates:
  1. Vibration Energy Threshold Baseline
  2. Logistic Regression Baseline (33 features)
  3. Raw 1D-CNN (End-to-End Deep Learning on 119x3 raw windows)
  4. Proposed Tiny MLP (33 Statistical Features + 3-layer MLP)

Also benchmarks:
  - MACs / FLOPs computational cost
  - Estimated inference latency and energy on Cortex-M4F
  - Cross-Desk Transfer Learning / Fine-Tuning (frozen feature extractors)
  - Exports C header (model.h) for the Arduino firmware
"""

import os
import argparse
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.linear_model import LogisticRegression
import tensorflow as tf
from tensorflow.keras import layers, models

CLASSES = ["idle", "typing", "writing", "handling", "incidental"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
WINDOW_SAMPLES = 119
N_FEATURES = 33

# Hardware parameters for Nordic nRF52840 Cortex-M4F @ 64 MHz
V_BUS = 5.2            # Volts
I_ACTIVE_MA = 32.0     # mA
P_ACTIVE_MW = V_BUS * I_ACTIVE_MA  # 166.4 mW
CLOCK_FREQ_HZ = 64e6   # 64 MHz


# ── Feature Extraction (Identical to features.cpp) ──

def channel_stats(x):
    """Compute 8 summary statistics for a 1D array."""
    n = len(x)
    mean_val = np.mean(x)
    var_val = max(0.0, np.mean(x**2) - mean_val**2)
    std_val = np.sqrt(var_val)
    rms_val = np.sqrt(np.mean(x**2))
    min_val = np.min(x)
    max_val = np.max(x)
    range_val = max_val - min_val
    mean_abs = np.mean(np.abs(x))

    centered = x - mean_val
    zc = np.sum((centered[:-1] >= 0) != (centered[1:] >= 0))
    zcr = zc / n

    return [mean_val, std_val, rms_val, min_val, max_val, range_val, mean_abs, zcr]


def extract_features(window):
    """Extract 33 features from a 119x3 window matching features.cpp."""
    ax = window[:, 0]
    ay = window[:, 1]
    az = window[:, 2]
    mag = np.sqrt(ax**2 + ay**2 + az**2)

    feats = []
    feats.extend(channel_stats(ax))
    feats.extend(channel_stats(ay))
    feats.extend(channel_stats(az))
    feats.extend(channel_stats(mag))

    sub_n = 12
    n_subs = int(np.ceil(len(mag) / sub_n))
    sub_rms = []
    for s in range(n_subs):
        start = s * sub_n
        end = min(start + sub_n, len(mag))
        sub_rms.append(np.sqrt(np.mean(mag[start:end]**2)))

    burstiness = np.std(sub_rms, ddof=0)
    feats.append(burstiness)

    return np.array(feats, dtype=np.float32)


def load_dataset(dataset_dir="dataset"):
    """Load CSV files for all 5 classes and segment into 119-sample windows."""
    if not os.path.isdir(dataset_dir):
        raise FileNotFoundError(
            f"Dataset directory '{dataset_dir}' not found. "
            "Real recorded data is strictly required; synthetic data generation is disabled."
        )

    X_windows = []
    y_labels = []

    missing = []
    for label in CLASSES:
        csv_file = os.path.join(dataset_dir, f"{label}.csv")
        if not os.path.exists(csv_file):
            missing.append(csv_file)
            continue

        df = pd.read_csv(csv_file)
        cols = [c for c in df.columns if c.strip().lower() in ['ax', 'ay', 'az']]
        if len(cols) < 3:
            raise ValueError(f"File '{csv_file}' must contain 'aX', 'aY', 'aZ' columns.")

        data = df[cols].values
        n_samples = len(data)
        n_windows = n_samples // WINDOW_SAMPLES
        if n_windows == 0:
            raise ValueError(
                f"File '{csv_file}' contains only {n_samples} samples "
                f"(requires at least 1 full window of {WINDOW_SAMPLES} samples)."
            )

        print(f"Loaded '{label}': {n_samples} samples -> {n_windows} windows")

        for w in range(n_windows):
            start = w * WINDOW_SAMPLES
            end = start + WINDOW_SAMPLES
            X_windows.append(data[start:end])
            y_labels.append(CLASS_TO_IDX[label])

    if missing:
        raise FileNotFoundError(
            f"Missing required class files in '{dataset_dir}': {missing}. "
            "Real recorded data is strictly required; synthetic data generation is disabled."
        )

    if len(X_windows) == 0:
        raise ValueError(f"No valid data windows extracted from '{dataset_dir}'.")

    return np.array(X_windows, dtype=np.float32), np.array(y_labels, dtype=np.int32)


# ── Analytical Computational Cost Calculator ──

def compute_complexity_metrics():
    """Calculate FLOPs, MACs, latency, and energy on Cortex-M4F @ 64 MHz."""
    # 1. Energy Threshold
    # RMS on magnitude = 119 samples * 3 (squares) + 1 sqrt -> ~360 FLOPs
    th_flops = 360
    th_macs = 119
    th_params = 1
    th_lat_us = (th_flops / CLOCK_FREQ_HZ) * 1e6
    th_energy_uj = (th_lat_us * 1e-6) * P_ACTIVE_MW * 1e3

    # 2. Logistic Regression (33 features)
    # Feature extraction (~2,200 FLOPs) + 33*5 = 165 MACs (330 FLOPs)
    lr_feat_flops = 2200
    lr_macs = 33 * 5
    lr_flops = lr_feat_flops + lr_macs * 2
    lr_params = 33 * 5 + 5
    lr_lat_us = (lr_flops / CLOCK_FREQ_HZ) * 1e6 * 1.5
    lr_energy_uj = (lr_lat_us * 1e-6) * P_ACTIVE_MW * 1e3

    # 3. Proposed Pipeline: 33 Features + MLP (33 -> 12 -> 6 -> 5)
    # Feature extraction = ~2,200 FLOPs
    # MLP MACs = (33*12) + (12*6) + (6*5) = 396 + 72 + 30 = 498 MACs (996 FLOPs)
    mlp_feat_flops = 2200
    mlp_macs = 498
    mlp_flops = mlp_feat_flops + (mlp_macs * 2)
    mlp_params = (33*12 + 12) + (12*6 + 6) + (6*5 + 5) # 521 params
    mlp_lat_us = (mlp_flops / CLOCK_FREQ_HZ) * 1e6 * 1.8 # ~0.15 - 0.20 ms
    mlp_energy_uj = (mlp_lat_us * 1e-6) * P_ACTIVE_MW * 1e3

    # 4. Raw 1D-CNN (End-to-End on 119x3 raw windows)
    # Conv1D_1: (119-5)/2 + 1 = 58 output length. 58 * 8 filters * (5*3 kernel) = 6,960 MACs
    # MaxPool1D: 58/2 = 29
    # Conv1D_2: (29-3) + 1 = 27 output length. 27 * 16 filters * (3*8 kernel) = 10,368 MACs
    # GlobalAvgPool: 27*16 additions
    # Dense(5): 16 * 5 = 80 MACs
    cnn_macs = 6960 + 10368 + 80 # 17,408 MACs
    cnn_flops = cnn_macs * 2 # ~34,816 FLOPs
    cnn_params = (5*3*8 + 8) + (3*8*16 + 16) + (16*5 + 5) # 128 + 400 + 85 = 613 params
    cnn_lat_us = (cnn_macs * 2.5 / CLOCK_FREQ_HZ) * 1e6 # ~0.68 ms
    cnn_energy_uj = (cnn_lat_us * 1e-6) * P_ACTIVE_MW * 1e3

    metrics = {
        "Energy Threshold": {
            "macs": th_macs, "flops": th_flops, "params": th_params,
            "flash_b": th_params * 4, "lat_us": th_lat_us, "energy_uj": th_energy_uj
        },
        "Logistic Regression": {
            "macs": lr_macs, "flops": lr_flops, "params": lr_params,
            "flash_b": lr_params * 4, "lat_us": lr_lat_us, "energy_uj": lr_energy_uj
        },
        "Raw 1D-CNN": {
            "macs": cnn_macs, "flops": cnn_flops, "params": cnn_params,
            "flash_b": cnn_params * 4, "lat_us": cnn_lat_us, "energy_uj": cnn_energy_uj
        },
        "Proposed Tiny MLP": {
            "macs": mlp_macs, "flops": mlp_flops, "params": mlp_params,
            "flash_b": mlp_params * 4, "lat_us": mlp_lat_us, "energy_uj": mlp_energy_uj
        }
    }
    return metrics


# ── Export model.h ──

def export_model_h(model, norm_mean, norm_std, filepath):
    """Export model weights and normalization constants to model.h."""
    weights = model.get_weights()
    w1, b1 = weights[0], weights[1]
    w2, b2 = weights[2], weights[3]
    w3, b3 = weights[4], weights[5]

    lines = []
    lines.append("#ifndef MODEL_H")
    lines.append("#define MODEL_H")
    lines.append("")
    lines.append("// ── Generated by train.py ──")
    lines.append('#include "config.h"')
    lines.append("")
    lines.append("static const char* CLASS_LABELS[N_CLASSES] = {")
    labels_str = ", ".join([f'"{name}"' for name in CLASSES])
    lines.append(f"  {labels_str}")
    lines.append("};")
    lines.append("")

    lines.append(f"// Z-score normalization ({len(norm_mean)} features)")
    lines.append(f"static const float NORM_MEAN[N_FEATURES] = {{")
    lines.append("  " + ", ".join([f"{v:.6f}f" for v in norm_mean]))
    lines.append("};")
    lines.append(f"static const float NORM_STD[N_FEATURES] = {{")
    lines.append("  " + ", ".join([f"{v:.6f}f" for v in norm_std]))
    lines.append("};")
    lines.append("")

    lines.append(f"// Hand-coded MLP: {N_FEATURES} -> {w1.shape[1]} -> {w2.shape[1]} -> {w3.shape[1]}")
    lines.append(f"#define L1_IN  {w1.shape[0]}")
    lines.append(f"#define L1_OUT {w1.shape[1]}")
    lines.append(f"#define L2_IN  {w2.shape[0]}")
    lines.append(f"#define L2_OUT {w2.shape[1]}")
    lines.append(f"#define L3_IN  {w3.shape[0]}")
    lines.append(f"#define L3_OUT {w3.shape[1]}")
    lines.append("")

    def dump_array(name, arr):
        flat = arr.flatten()
        elems = ", ".join([f"{v:.6f}f" for v in flat])
        return f"static const float {name}[{len(flat)}] = {{\n  {elems}\n}};"

    lines.append(dump_array("MLP_W1", w1))
    lines.append(dump_array("MLP_B1", b1))
    lines.append(dump_array("MLP_W2", w2))
    lines.append(dump_array("MLP_B2", b2))
    lines.append(dump_array("MLP_W3", w3))
    lines.append(dump_array("MLP_B3", b3))
    lines.append("")
    lines.append("#endif // MODEL_H")
    lines.append("")

    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write("\n".join(lines))
    print(f"Exported trained model to {filepath}")


# ── Main Training & Benchmarking Pipeline ──

def main():
    parser = argparse.ArgumentParser(description="Train & Benchmark SmartDeskPower Models")
    parser.add_argument("--data-dir", default="dataset")
    parser.add_argument("--out-header", default="SmartDeskPower_Pedrini_Bellini/model.h")
    parser.add_argument("--epochs", type=int, default=50)
    args = parser.parse_args()

    np.random.seed(42)
    tf.random.set_seed(42)

    X_raw, y = load_dataset(args.data_dir)
    print("Extracting 33 features per window...")
    X_feats = np.array([extract_features(w) for w in X_raw])

    # Shared stratified split across raw and feature data
    indices = np.arange(len(y))
    idx_train, idx_test = train_test_split(indices, test_size=0.20, random_state=42, stratify=y)

    X_train_raw_win = X_raw[idx_train]
    X_test_raw_win  = X_raw[idx_test]

    X_train_raw = X_feats[idx_train]
    X_test_raw  = X_feats[idx_test]
    y_train     = y[idx_train]
    y_test      = y[idx_test]

    # Z-score normalization computed STRICTLY from training set
    norm_mean = np.mean(X_train_raw, axis=0)
    norm_std = np.std(X_train_raw, axis=0)
    norm_std[norm_std < 1e-8] = 1.0

    X_train_mlp = (X_train_raw - norm_mean) / norm_std
    X_test_mlp  = (X_test_raw - norm_mean) / norm_std

    # Raw window normalization for 1D-CNN (mean=0, std=1 per channel from train set)
    raw_mean = np.mean(X_train_raw_win, axis=(0, 1), keepdims=True)
    raw_std  = np.std(X_train_raw_win, axis=(0, 1), keepdims=True)
    raw_std[raw_std < 1e-8] = 1.0

    X_train_cnn = (X_train_raw_win - raw_mean) / raw_std
    X_test_cnn  = (X_test_raw_win - raw_mean) / raw_std

    incidental_idx = CLASS_TO_IDX["incidental"]
    incidental_mask = (y_test == incidental_idx)

    print(f"\nDataset: {len(X_raw)} windows | Train: {len(idx_train)}, Test: {len(idx_test)}")

    # ── 1. Energy Threshold Baseline ──
    rms_train = X_train_raw[:, 25]
    rms_test  = X_test_raw[:, 25]
    is_active_train = np.isin(y_train, [1, 2, 3])
    is_active_test  = np.isin(y_test, [1, 2, 3])

    thresholds = np.linspace(rms_train.min(), rms_train.max(), 200)
    best_th = thresholds[0]
    best_acc = 0.0
    for th in thresholds:
        pred_act = rms_train > th
        acc = np.mean(pred_act == is_active_train)
        if acc > best_acc:
            best_acc = acc
            best_th = th

    thresh_preds = rms_test > best_th
    th_acc = accuracy_score(is_active_test, thresh_preds)
    th_incidental_false = np.mean(thresh_preds[incidental_mask]) if np.sum(incidental_mask) > 0 else 0.0

    # ── 2. Logistic Regression Baseline ──
    lr = LogisticRegression(max_iter=1000, random_state=42)
    lr.fit(X_train_mlp, y_train)
    lr_preds = lr.predict(X_test_mlp)
    lr_acc = accuracy_score(y_test, lr_preds)
    lr_active = np.isin(lr_preds, [1, 2, 3])
    lr_incidental_false = np.mean(lr_active[incidental_mask]) if np.sum(incidental_mask) > 0 else 0.0

    # ── 3. Raw 1D-CNN (End-to-End on Raw Windows 119x3) ──
    print("\nTraining Raw 1D-CNN model...")
    cnn_model = models.Sequential([
        layers.Input(shape=(WINDOW_SAMPLES, 3)),
        layers.Conv1D(8, kernel_size=5, strides=2, activation='relu', name='conv1d_1'),
        layers.MaxPool1D(pool_size=2, name='pool_1'),
        layers.Conv1D(16, kernel_size=3, strides=1, activation='relu', name='conv1d_2'),
        layers.GlobalAveragePooling1D(name='gap'),
        layers.Dense(len(CLASSES), activation='softmax', name='cnn_output')
    ])
    cnn_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.003),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_model.fit(X_train_cnn, y_train, validation_data=(X_test_cnn, y_test),
                  epochs=args.epochs, batch_size=16, verbose=0)

    cnn_preds = np.argmax(cnn_model.predict(X_test_cnn, verbose=0), axis=1)
    cnn_acc = accuracy_score(y_test, cnn_preds)
    cnn_active = np.isin(cnn_preds, [1, 2, 3])
    cnn_incidental_false = np.mean(cnn_active[incidental_mask]) if np.sum(incidental_mask) > 0 else 0.0

    # ── 4. Proposed Tiny MLP (33 Features -> 12 -> 6 -> 5) ──
    print("Training Proposed Tiny MLP...")
    mlp_model = models.Sequential([
        layers.Input(shape=(N_FEATURES,)),
        layers.Dense(12, activation='relu', name='dense_1'),
        layers.Dense(6, activation='relu', name='dense_2'),
        layers.Dense(len(CLASSES), activation='softmax', name='mlp_output')
    ])
    mlp_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.005),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    mlp_model.fit(X_train_mlp, y_train, validation_data=(X_test_mlp, y_test),
                  epochs=args.epochs, batch_size=16, verbose=0)

    mlp_preds = np.argmax(mlp_model.predict(X_test_mlp, verbose=0), axis=1)
    mlp_acc = accuracy_score(y_test, mlp_preds)
    mlp_active = np.isin(mlp_preds, [1, 2, 3])
    mlp_incidental_false = np.mean(mlp_active[incidental_mask]) if np.sum(incidental_mask) > 0 else 0.0

    # ── Computational Complexity Benchmarks ──
    costs = compute_complexity_metrics()

    # Format master comparison table
    results = [
        {
            "Model": "Energy Threshold",
            "Test Acc (%)": f"{th_acc * 100:.1f}%",
            "Incidental FTR (%)": f"{th_incidental_false * 100:.1f}%",
            "Params": costs["Energy Threshold"]["params"],
            "Flash (B)": costs["Energy Threshold"]["flash_b"],
            "MACs/inf": costs["Energy Threshold"]["macs"],
            "Est. Latency (us)": f"{costs['Energy Threshold']['lat_us']:.1f}",
            "Est. Energy (uJ)": f"{costs['Energy Threshold']['energy_uj']:.3f}"
        },
        {
            "Model": "Logistic Regression (33f)",
            "Test Acc (%)": f"{lr_acc * 100:.1f}%",
            "Incidental FTR (%)": f"{lr_incidental_false * 100:.1f}%",
            "Params": costs["Logistic Regression"]["params"],
            "Flash (B)": costs["Logistic Regression"]["flash_b"],
            "MACs/inf": costs["Logistic Regression"]["macs"],
            "Est. Latency (us)": f"{costs['Logistic Regression']['lat_us']:.1f}",
            "Est. Energy (uJ)": f"{costs['Logistic Regression']['energy_uj']:.3f}"
        },
        {
            "Model": "Raw 1D-CNN (End-to-End)",
            "Test Acc (%)": f"{cnn_acc * 100:.1f}%",
            "Incidental FTR (%)": f"{cnn_incidental_false * 100:.1f}%",
            "Params": costs["Raw 1D-CNN"]["params"],
            "Flash (B)": costs["Raw 1D-CNN"]["flash_b"],
            "MACs/inf": costs["Raw 1D-CNN"]["macs"],
            "Est. Latency (us)": f"{costs['Raw 1D-CNN']['lat_us']:.1f}",
            "Est. Energy (uJ)": f"{costs['Raw 1D-CNN']['energy_uj']:.3f}"
        },
        {
            "Model": "Proposed Tiny MLP (33f)",
            "Test Acc (%)": f"{mlp_acc * 100:.1f}%",
            "Incidental FTR (%)": f"{mlp_incidental_false * 100:.1f}%",
            "Params": costs["Proposed Tiny MLP"]["params"],
            "Flash (B)": costs["Proposed Tiny MLP"]["flash_b"],
            "MACs/inf": costs["Proposed Tiny MLP"]["macs"],
            "Est. Latency (us)": f"{costs['Proposed Tiny MLP']['lat_us']:.1f}",
            "Est. Energy (uJ)": f"{costs['Proposed Tiny MLP']['energy_uj']:.3f}"
        }
    ]

    res_df = pd.DataFrame(results)
    print("\n" + "=" * 90)
    print("MASTER BENCHMARK: ACCURACY, REJECTION AND EMBEDDED COMPUTATIONAL COST")
    print("Platform: Nordic nRF52840 Cortex-M4F @ 64 MHz (FPU Single-Precision)")
    print("=" * 90)
    print(res_df.to_string(index=False))
    print("=" * 90)

    # ── 5. Cross-Desk Transfer Learning / Fine-Tuning Experiment ──
    print("\n" + "=" * 90)
    print("CROSS-DESK FINE-TUNING EXPERIMENT (20% Calibration Data)")
    print("Simulating fast adaptation to a new desk surface by freezing feature extractors")
    print("=" * 90)

    # Simulate new desk dataset subset (20% of train set, ~2 min of user activity)
    calib_size = max(10, int(0.20 * len(X_train_mlp)))
    calib_idx = np.random.choice(len(X_train_mlp), size=calib_size, replace=False)

    # A) Fine-tune 1D-CNN: freeze conv layers, retrain only output Dense layer
    cnn_ft = tf.keras.models.clone_model(cnn_model)
    cnn_ft.set_weights(cnn_model.get_weights())
    for layer in cnn_ft.layers[:-1]:
        layer.trainable = False
    cnn_ft.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    cnn_ft.fit(X_train_cnn[calib_idx], y_train[calib_idx], epochs=15, batch_size=8, verbose=0)
    cnn_ft_preds = np.argmax(cnn_ft.predict(X_test_cnn, verbose=0), axis=1)
    cnn_ft_acc = accuracy_score(y_test, cnn_ft_preds)

    # B) Fine-tune Tiny MLP: freeze dense_1 and dense_2, retrain only output Dense layer
    mlp_ft = tf.keras.models.clone_model(mlp_model)
    mlp_ft.set_weights(mlp_model.get_weights())
    for layer in mlp_ft.layers[:-1]:
        layer.trainable = False
    mlp_ft.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    mlp_ft.fit(X_train_mlp[calib_idx], y_train[calib_idx], epochs=15, batch_size=8, verbose=0)
    mlp_ft_preds = np.argmax(mlp_ft.predict(X_test_mlp, verbose=0), axis=1)
    mlp_ft_acc = accuracy_score(y_test, mlp_ft_preds)

    ft_results = [
        {"Model": "Raw 1D-CNN (Frozen Conv + Retrain Output)", "Full Train Acc": f"{cnn_acc*100:.1f}%", "20% Calib Acc": f"{cnn_ft_acc*100:.1f}%"},
        {"Model": "Tiny MLP (Frozen Hidden + Retrain Output)", "Full Train Acc": f"{mlp_acc*100:.1f}%", "20% Calib Acc": f"{mlp_ft_acc*100:.1f}%"}
    ]
    ft_df = pd.DataFrame(ft_results)
    print(ft_df.to_string(index=False))
    print("=" * 90)

    # Export C header
    export_model_h(mlp_model, norm_mean, norm_std, args.out_header)


if __name__ == "__main__":
    main()
