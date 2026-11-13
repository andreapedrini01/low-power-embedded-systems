#!/usr/bin/env python3
"""Cross-Desk Fine-Tuning Pipeline for SmartDeskPower.

Adapts pre-trained MLP weights to a new desk surface by:
1. Extracting 33 time-domain statistical features from calibration windows.
2. Freezing pre-trained Layer 1 (33 -> 12) and Layer 2 (12 -> 6) weights.
3. Optimizing classification output layer weights (6 -> 5) via softmax cross-entropy gradient descent.
4. Exporting the calibrated header model_finetuned.h.

Authors: Andrea Pedrini, Pietro Bellini
"""

import os
import sys
import csv
import math
import argparse

CLASSES = ["idle", "typing", "writing", "handling", "incidental"]
CLASS_TO_IDX = {name: i for i, name in enumerate(CLASSES)}
WINDOW_SAMPLES = 119
N_CLASSES = 5
N_FEATURES = 33

L1_IN, L1_OUT = 33, 12
L2_IN, L2_OUT = 12, 6
L3_IN, L3_OUT = 6, 5

BASE_W1 = [
  0.224753, -0.340493, -0.427353, 0.580892, -0.284568, 0.364996, 0.771003, 0.375143, 0.337088, -0.087446, -0.363316, -0.498702, -0.492120, 0.299211, 0.294776, 0.127978, 0.610462, -0.419682, -0.531450, -0.380620, -0.074976, 0.207738, -0.392793, -0.202382, -0.215294, 0.081874, -0.573735, -0.024544, -0.039131, -0.152563, 0.280131, -0.081301, -0.581081, 0.073733, 0.036788, 0.805411, 0.436791, -0.506805, -0.212421, 0.242166, -0.458407, 0.171450, 0.474682, 0.532960, -0.349489, -0.008099, 0.056189, -0.365530, -0.097991, 0.418085, 0.515504, 0.415120, 0.266748, -0.741363, -0.531479, -0.688550, 0.706637, 0.168490, -0.820603, -0.093898, -0.157233, 0.119598, 0.202448, 0.359198, 0.342832, -0.221973, -0.825325, -0.469304, 0.416498, -0.242572, -0.663610, -0.411690, -0.312250, 0.093893, -1.162797, 0.289637, 0.089955, -0.359760, 0.257538, 0.739411, -0.712111, 0.706415, 0.095180, 0.374326, -0.182306, 0.221811, 0.007170, 0.420315, -0.188066, -0.239381, 0.061821, 0.214981, 0.587234, 0.225937, -0.294890, -0.444628, 0.281673, 0.076360, 0.424545, -0.556346, -0.179534, -0.712057, -0.871967, -0.965632, 0.039183, 0.222072, 0.273341, 0.168812, 0.115161, 0.177610, -0.115355, -0.228440, 0.392222, -0.125064, -0.289249, -0.217640, 0.137389, -0.361683, -0.234110, -0.132790, -0.431498, 0.501957, -0.521403, 0.309723, 0.213075, 0.419244, 1.190923, 0.751939, -0.783258, -0.133709, -0.100596, -0.630344, 0.379580, 0.221932, 0.531844, -0.565207, -0.093890, 0.217798, -0.084806, 0.037812, 0.277949, -0.001936, 0.283416, -0.188334, 0.244414, 0.611372, 0.073565, -0.171494, 0.335528, -0.043930, -0.029240, -0.094659, -0.195784, -0.238616, -0.084977, 0.093090, 0.098841, 0.293391, -0.098006, 0.171777, -0.353874, 0.051054, -0.337911, 0.003261, 0.200507, 0.464969, 0.148086, 0.070841, -0.282582, -0.261978, -0.218671, 0.549350, 0.247801, 0.489911, 0.850950, 0.688852, -0.264022, -0.318552, -0.036074, -0.300751, -0.950700, -0.593207, 0.322058, 0.692480, -0.886428, -0.344830, 0.200486, 0.595225, 0.008885, 0.813582, -0.383555, 0.092806, 0.064243, -0.024878, -0.294330, -0.103911, 0.407047, -0.462000, -0.286091, 0.391795, 0.038576, 0.189034, 0.083962, -0.603318, -0.786333, 0.059706, -0.372289, 0.310817, 0.118364, -0.124211, 0.502672, 0.625587, -0.188628, -0.236864, -0.340358, 0.261762, -0.371359, -0.313506, -0.516627, -0.167260, -0.152417, 0.014169, 0.162795, 0.610632, -0.018710, -0.391565, 0.180999, -0.543066, 0.662714, 0.035175, -0.007833, -0.341119, 0.181261, 0.332744, 0.149286, 0.207150, 0.156445, -0.000862, 0.536228, 0.408492, -0.970746, 0.136905, -0.319054, 0.130783, 0.323197, -0.022774, -0.291967, -0.319656, 0.294423, -0.186031, -0.249795, 0.290484, -0.354544, 0.005173, -0.450998, 0.524345, 0.201262, 0.122587, -0.097474, 0.015612, 0.078365, 0.076241, -0.315373, -0.186571, 0.152982, -0.429703, 0.193760, -0.008889, 0.124599, -0.541330, 0.203939, 0.661864, -0.001801, -0.028435, 0.615608, -0.355018, 0.326258, 0.256205, -0.363973, -0.482583, -0.218644, -0.340044, 0.258123, 0.007009, 0.376743, -0.130819, -0.552292, 0.399148, -0.530898, -0.335908, -0.123958, -0.102978, -0.181034, -0.510959, 0.312531, 0.683638, 0.152042, 0.009665, 0.104772, -0.508694, -1.085302, -0.084658, -0.741911, 0.075734, 0.215417, -0.194577, 0.395436, 0.065237, -0.412862, 0.036176, -0.008123, -0.015158, -0.267762, 0.148401, -0.003612, 0.347573, 0.261136, 0.131107, 0.116212, 0.269339, 0.000469, 0.026075, 0.097871, -0.305932, 0.837189, 0.307966, -0.110825, -0.319110, -0.029776, 0.058046, 0.058886, 0.437855, -0.218045, 0.331171, 0.404737, 0.278193, -0.792614, 0.087104, -0.125934, 0.339910, 0.160204, 0.122032, 0.034580, -0.371246, 0.393104, 0.139390, -0.113478, 0.088620, -0.748613, -0.234578, -0.458021, 0.360579, 0.417906, 0.108067, -0.384984, -0.013726, -0.206312, -0.017359, -0.434091, -0.406052, -0.405727, -0.363983, -0.013764, 0.315972, 0.249904, -0.121653, -0.025551, 0.653386, -0.371807, 0.096711, 0.096202, -0.727255, 0.220788, 0.072671, -0.128243, -0.219428, -0.230487, -0.370325, 0.148893, 0.067916, -0.022741, -0.356464, 0.057787, 0.405629, -0.184203, 0.056522, -0.183935, 0.305859, 0.185614, -0.264214, -0.517238, 0.144566, 0.045629, 0.549316, -0.330777, -0.064075
]
BASE_B1 = [
  0.361418, 0.328234, 0.379742, 0.177983, 0.207679, 0.061723, -0.132509, 0.001520, 0.445738, 0.174032, 0.078259, 0.044829
]
BASE_W2 = [
  0.941797, 0.365915, -0.224973, 0.719027, 0.391589, 0.440823, -0.541181, -0.020770, -0.073870, -0.011520, 0.101991, 0.901394, 0.793722, -0.874321, -0.344711, -0.335750, 0.036894, 0.749828, -0.647637, -0.166440, 0.685843, -0.019943, -0.401799, 0.839831, 0.352408, -0.437768, -0.778979, -0.304145, 0.835017, 0.624919, 0.832041, 0.104720, 0.403303, -0.653189, 0.205522, -0.339727, -0.003759, 0.315742, 0.851956, -0.701416, -0.028620, -0.959173, -0.526332, -0.556562, 0.703968, -0.084093, 0.620659, -0.632200, 0.521194, 0.480524, 0.351905, 0.548757, -1.038748, 1.160620, -0.438899, 0.229461, 0.423195, 1.154771, 0.108876, 0.569147, -0.062325, -0.008647, -0.653632, 0.299544, 0.515917, -0.161795, -0.518611, 0.193620, -0.046497, 0.660004, 0.734969, -1.134480
]
BASE_B2 = [
  0.157305, -0.026466, 0.241102, 0.181928, 0.179344, 0.201083
]
BASE_W3 = [
  -0.641501, -0.957739, 0.407008, 0.231323, 0.150655, -0.120976, -0.581734, -0.203985, 0.197894, -2.034729, -0.553029, 0.686254, -0.570869, 0.027361, -1.237379, 0.485512, -1.623988, -0.937636, 0.038838, -0.336751, 0.052126, -0.515894, 0.606092, -0.694348, -0.239596, -1.430787, -0.704683, -1.818335, 0.429726, 1.038785
]
BASE_B3 = [
  -0.432637, 0.483900, 0.137341, -0.270529, -0.278340
]


def channel_stats(x):
    """Computes 8 time-domain statistical metrics for a 1D sample array."""
    n = len(x)
    mean_val = sum(x) / n
    sq_sum = sum(v * v for v in x)
    var_val = max(0.0, sq_sum / n - mean_val * mean_val)
    std_val = math.sqrt(var_val)
    rms_val = math.sqrt(sq_sum / n)
    min_val = min(x)
    max_val = max(x)
    range_val = max_val - min_val
    mean_abs = sum(abs(v) for v in x) / n

    centered = [v - mean_val for v in x]
    zc = sum(1 for i in range(1, n) if (centered[i - 1] >= 0) != (centered[i] >= 0))
    zcr = zc / n

    return [mean_val, std_val, rms_val, min_val, max_val, range_val, mean_abs, zcr]


def extract_features(window):
    """Extracts the 33-element feature vector matching firmware features_extract()."""
    ax = [r[0] for r in window]
    ay = [r[1] for r in window]
    az = [r[2] for r in window]
    mag = [math.sqrt(ax[i]**2 + ay[i]**2 + az[i]**2) for i in range(len(window))]

    feats = []
    feats.extend(channel_stats(ax))
    feats.extend(channel_stats(ay))
    feats.extend(channel_stats(az))
    feats.extend(channel_stats(mag))

    sub_n = 12
    n_subs = int(math.ceil(len(mag) / sub_n))
    sub_rms = []
    for s in range(n_subs):
        start = s * sub_n
        end = min(start + sub_n, len(mag))
        chunk = mag[start:end]
        sub_rms.append(math.sqrt(sum(v * v for v in chunk) / len(chunk)))

    sub_mean = sum(sub_rms) / len(sub_rms)
    sub_var = sum((v - sub_mean)**2 for v in sub_rms) / len(sub_rms)
    feats.append(math.sqrt(sub_var))

    return feats


def load_dataset(dataset_dir):
    """Loads CSV files for all 5 classes and segments data into 119-sample windows."""
    windows = []
    labels = []
    for label in CLASSES:
        path = os.path.join(dataset_dir, f"{label}.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing '{path}' in calibration directory '{dataset_dir}'")
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.reader(f)
            header = next(reader)
            rows = []
            for r in reader:
                if len(r) >= 3:
                    try:
                        rows.append([float(r[0]), float(r[1]), float(r[2])])
                    except ValueError:
                        pass
        n_win = len(rows) // WINDOW_SAMPLES
        print(f"Loaded '{label}': {n_win} windows ({len(rows)} samples)")
        for w in range(n_win):
            windows.append(rows[w * WINDOW_SAMPLES:(w + 1) * WINDOW_SAMPLES])
            labels.append(CLASS_TO_IDX[label])
    return windows, labels


def forward_frozen_layers(feat_norm):
    """Computes representations h2 through frozen pre-trained Layers 1 and 2."""
    h1 = [0.0] * L1_OUT
    for j in range(L1_OUT):
        s = BASE_B1[j]
        for i in range(L1_IN):
            s += feat_norm[i] * BASE_W1[i * L1_OUT + j]
        h1[j] = s if s > 0.0 else 0.0

    h2 = [0.0] * L2_OUT
    for j in range(L2_OUT):
        s = BASE_B2[j]
        for i in range(L2_IN):
            s += h1[i] * BASE_W2[i * L2_OUT + j]
        h2[j] = s if s > 0.0 else 0.0

    return h2


def softmax(logits):
    """Numerically stable softmax normalization."""
    mx = max(logits)
    exps = [math.exp(v - mx) for v in logits]
    s = sum(exps)
    return [e / s for e in exps]


def train_output_layer(H_train, y_train, epochs=150, lr=0.10, l2_reg=0.001):
    """Trains output layer weights (6 -> 5) via softmax cross-entropy gradient descent."""
    N = len(H_train)
    w = list(BASE_W3)
    b = list(BASE_B3)

    for epoch in range(epochs):
        grad_w = [0.0] * (L3_IN * L3_OUT)
        grad_b = [0.0] * L3_OUT
        total_loss = 0.0
        correct = 0

        for k in range(N):
            h = H_train[k]
            target = y_train[k]

            logits = [b[j] for j in range(L3_OUT)]
            for j in range(L3_OUT):
                for i in range(L3_IN):
                    logits[j] += h[i] * w[i * L3_OUT + j]

            probs = softmax(logits)
            pred = max(range(L3_OUT), key=lambda idx: probs[idx])
            if pred == target:
                correct += 1

            total_loss -= math.log(max(probs[target], 1e-12))

            for j in range(L3_OUT):
                err = probs[j] - (1.0 if j == target else 0.0)
                grad_b[j] += err / N
                for i in range(L3_IN):
                    grad_w[i * L3_OUT + j] += (err * h[i]) / N

        for idx in range(len(w)):
            w[idx] -= lr * (grad_w[idx] + l2_reg * w[idx])
        for idx in range(len(b)):
            b[idx] -= lr * grad_b[idx]

        if (epoch + 1) % 50 == 0 or epoch == epochs - 1:
            acc = (correct / N) * 100.0
            print(f"  Epoch {epoch+1:3d}/{epochs}: Loss = {total_loss/N:.4f}, Calib Accuracy = {acc:.1f}%")

    return w, b


def export_finetuned_header(norm_mean, norm_std, w3, b3, filepath):
    """Exports calibrated weights and normalization constants to a C++ header."""
    lines = []
    lines.append("#ifndef MODEL_FINETUNED_H")
    lines.append("#define MODEL_FINETUNED_H")
    lines.append("")
    lines.append("// Calibrated for target desk surface via finetune.py")
    lines.append('#include "config.h"')
    lines.append("")
    lines.append("static const char* CLASS_LABELS[N_CLASSES] = {")
    labels_str = ", ".join([f'"{name}"' for name in CLASSES])
    lines.append(f"  {labels_str}")
    lines.append("};")
    lines.append("")

    lines.append(f"// Feature normalization parameters ({len(norm_mean)} features)")
    lines.append("static const float NORM_MEAN[N_FEATURES] = {")
    lines.append("  " + ", ".join([f"{v:.6f}f" for v in norm_mean]))
    lines.append("};")
    lines.append("static const float NORM_STD[N_FEATURES] = {")
    lines.append("  " + ", ".join([f"{v:.6f}f" for v in norm_std]))
    lines.append("};")
    lines.append("")

    lines.append(f"// MLP architecture: {N_FEATURES} -> {L1_OUT} -> {L2_OUT} -> {L3_OUT}")
    lines.append(f"#define L1_IN  {L1_IN}")
    lines.append(f"#define L1_OUT {L1_OUT}")
    lines.append(f"#define L2_IN  {L2_IN}")
    lines.append(f"#define L2_OUT {L2_OUT}")
    lines.append(f"#define L3_IN  {L3_IN}")
    lines.append(f"#define L3_OUT {L3_OUT}")
    lines.append("")

    def dump_array(name, arr):
        elems = ", ".join([f"{v:.6f}f" for v in arr])
        return f"static const float {name}[{len(arr)}] = {{\n  {elems}\n}};"

    lines.append(dump_array("MLP_W1", BASE_W1))
    lines.append(dump_array("MLP_B1", BASE_B1))
    lines.append(dump_array("MLP_W2", BASE_W2))
    lines.append(dump_array("MLP_B2", BASE_B2))
    lines.append(dump_array("MLP_W3", w3))
    lines.append(dump_array("MLP_B3", b3))
    lines.append("")
    lines.append("#endif // MODEL_FINETUNED_H")
    lines.append("")

    content = "\n".join(lines)
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(content)
    print(f"\n[OK] Calibrated model header saved to: {filepath}")


def main():
    parser = argparse.ArgumentParser(description="Cross-Desk Fine-Tuning Pipeline")
    parser.add_argument("--calib-dir", default="dataset_cla",
                        help="Calibration dataset folder (default: dataset_cla)")
    parser.add_argument("--out-header", default="SmartDeskPower_Pedrini_Bellini/model_finetuned.h",
                        help="Output header path")
    args = parser.parse_args()

    print(f"Loading calibration dataset from: '{args.calib_dir}'")
    windows, labels = load_dataset(args.calib_dir)

    print(f"Extracting 33 statistical features from {len(windows)} windows...")
    feats = [extract_features(w) for w in windows]

    norm_mean = [0.0] * N_FEATURES
    norm_std  = [0.0] * N_FEATURES
    N = len(feats)

    for i in range(N_FEATURES):
        col = [f[i] for f in feats]
        m = sum(col) / N
        v = sum((x - m)**2 for x in col) / N
        s = math.sqrt(v)
        norm_mean[i] = m
        norm_std[i]  = s if s > 1e-8 else 1.0

    feats_norm = []
    for f in feats:
        norm_f = [(f[i] - norm_mean[i]) / norm_std[i] for i in range(N_FEATURES)]
        feats_norm.append(norm_f)

    print("Computing hidden layer representations through frozen Layers 1 and 2...")
    H2 = [forward_frozen_layers(fn) for fn in feats_norm]

    print("Fine-tuning classification output layer (6 -> 5)...")
    w3, b3 = train_output_layer(H2, labels, epochs=150, lr=0.10, l2_reg=0.001)

    export_finetuned_header(norm_mean, norm_std, w3, b3, args.out_header)


if __name__ == "__main__":
    main()
