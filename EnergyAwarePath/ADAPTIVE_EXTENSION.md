# EnergyAwarePath — Technical Specification

**Inspired by:** ePerceptive (SenSys '20, Montanari et al.)  
**Course:** Low-Power Embedded Systems — Assignment 3  
**Authors:** Pedrini, Bellini  
**Hardware:** Arduino Nano 33 BLE Sense Lite (256 KB SRAM, 1 MB flash)

---

## 1. Objective

This extension implements **adaptive multi-exit inference** inspired by ePerceptive. The system dynamically adjusts inference complexity based on available energy budget:

- **High budget** (≥ 0.6): Full model (2 hidden layers, maximum accuracy)
- **Medium budget** (0.3–0.6): Exit 1 (1 hidden layer, reduced accuracy)
- **Low budget** (< 0.3): Linear formula (zero TFLite cost)

The system **gracefully degrades** inference quality when energy is scarce, instead of using the same computational cost regardless of available budget.

---

## 2. Implementation Overview

### 2.1 File Structure

```
EnergyAwarePath/EnergyAwarePath/
├── Energy-Aware-Path.ino   ← orchestrazione principale, loop seriale
├── config.h                ← costanti, Branch struct, checkpoints[], FEATURE_MIN/MAX
├── sensors.h / sensors.cpp ← IMU init e acquisizione finestra ~1.2s a 100Hz
├── features.h / features.cpp ← estrazione 5 feature da IMUWindow
├── inference.h / inference.cpp ← TFLite Micro setup e invoke
├── planner.h / planner.cpp ← costruzione feature vector, normalizzazione, decisione
└── model.h                 ← modello quantizzato int8 come array C
```

### 2.2 Model Architecture

Multi-layer perceptron (MLP) with Keras:

```python
Dense(16, relu) → Dense(8, relu) → Dense(1, sigmoid)
```

- Input: 10 features (float32 normalized [0,1])
- Output: predicted energy cost (float32, [0,1])
- Quantization: full int8
- Size: ~3.3 KB (flash)
- Arena: ~712 B / 8 KB allocated
- Latency: ~0.5 ms per invocation
- Parameters: ~321

### 2.3 Feature Vector (10 elements)

```
[0] budget          float, [0.2, 1.0] → normalizzato
[1] length          float, [0.2, 0.9] → normalizzato
[2] turns/5         float, [0.0, 1.0] → già normalizzato
[3] difficulty      float, [0.1, 0.9] → normalizzato
[4] slope           float, [-0.3, 0.6] → normalizzato
[5] acc_mean        float, [0.965, 2.005] → normalizzato
[6] acc_std         float, [0.017, 0.749] → normalizzato
[7] acc_peak        float, [1.038, 5.019] → normalizzato
[8] gyro_mean       float, [0.745, 125.75] → normalizzato
[9] tilt            float, [0.016, 44.64] → normalizzato
```

FEATURE_MIN and FEATURE_MAX values are defined in `config.h`.

### 2.4 Oracle Formula (fallback for low budget)

Defined in `planner.cpp → planner_oracle_cost()`:

```
motion = (acc_std_norm + gyro_norm) / 2

E = 0.18 × length
  + 0.12 × turns/5
  + 0.12 × difficulty
  + 0.08 × |slope|
  + 0.05 × acc_std_norm
  + 0.05 × gyro_norm
  + 0.05 × (1 - budget)
  + 0.30 × length × motion
```

Where `acc_std_norm` and `gyro_norm` are features at indices 6 and 8 **after** normalization.

### 2.5 Policy Enum

```cpp
enum Policy {
  POLICY_ML = 0,
  POLICY_ALWAYS_A,
  POLICY_SHORTEST,
  POLICY_RANDOM,
  POLICY_ORACLE
};
```

### 2.6 TFLite Libraries

```cpp
#include <TinyMLShield.h>
#include <TensorFlowLite.h>
#include <tensorflow/lite/micro/all_ops_resolver.h>
#include <tensorflow/lite/micro/micro_error_reporter.h>
#include <tensorflow/lite/micro/micro_interpreter.h>
#include <tensorflow/lite/schema/schema_generated.h>
```

Library: `Harvard_TinyMLx`

---

## 3. Multi-Exit Model Architecture

### 3.1 Training Model Structure

```python
inputs = tf.keras.Input(shape=(10,))
x1 = tf.keras.layers.Dense(16, activation='relu', name='dense_1')(inputs)
# EXIT 1: classificatore leggero attaccato dopo dense_1
exit1_out = tf.keras.layers.Dense(1, activation='sigmoid', name='exit1')(x1)
# Continua verso exit finale
x2 = tf.keras.layers.Dense(8, activation='relu', name='dense_2')(x1)
final_out = tf.keras.layers.Dense(1, activation='sigmoid', name='final')(x2)

model = tf.keras.Model(inputs=inputs, outputs=[exit1_out, final_out])
```

Weighted loss:

```python
model.compile(
    optimizer='adam',
    loss={'exit1': 'mse', 'final': 'mse'},
    loss_weights={'exit1': 0.4, 'final': 1.0},
    metrics={'exit1': 'mae', 'final': 'mae'}
)
```

Weight 0.4 on exit 1 ensures the model optimizes primarily for the final output, while training the intermediate exit to produce reasonable estimates.

### 3.2 Export for Arduino

After training, export **two separate TFLite models**:

**Full model** (final exit):
```python
# Crea un modello che usa solo l'output finale
full_model = tf.keras.Model(inputs=inputs, outputs=final_out)
# Copia i pesi da model
full_model.set_weights(...)  # vedi notebook
# Converti a TFLite int8 → model.h (come già fatto)
```

**Exit 1 model** (dense_1 + exit1 only):
```python
exit1_model = tf.keras.Model(inputs=inputs, outputs=exit1_out)
exit1_model.set_weights(...)  # solo i layer fino a exit1
# Converti a TFLite int8 → model_exit1.h
```

Both `.h` files go in the Arduino sketch folder.

### 3.3 Model Sizes

Expected sizes (`.tflite` binary, actual flash occupation):
- Full model: **3.3 KB**
- Exit 1 model: **~2.5 KB**

The `.h` files are ~6× larger on disk (text representation) but compile to the `.tflite` size.

---

## 4. Arduino Implementation

### 4.1 Configuration (`config.h`)

```cpp
// --- Adaptive inference thresholds ---
#define BUDGET_HIGH_THRESHOLD   0.6f   // sopra: usa modello completo
#define BUDGET_LOW_THRESHOLD    0.3f   // sotto: usa formula oracle (no TFLite)
// tra LOW e HIGH: usa exit 1

// --- Tensor arena per exit 1 (più piccolo) ---
#define TENSOR_ARENA_EXIT1_SIZE 4096   // 4 KB, sufficiente per il modello ridotto

// Aggiungere alla enum Policy:
// POLICY_ADAPTIVE deve essere aggiunto dopo POLICY_ORACLE
```

Add `POLICY_ADAPTIVE` to enum:

```cpp
enum Policy {
  POLICY_ML = 0,
  POLICY_ALWAYS_A,
  POLICY_SHORTEST,
  POLICY_RANDOM,
  POLICY_ORACLE,
  POLICY_ADAPTIVE    // ← nuovo: sceglie exit in base al budget
};
```

### 4.2 Inference Interface (`inference.h`)

```cpp
// Livello di exit usato nell'ultima inferenza adattiva
enum ExitLevel {
  EXIT_FULL   = 2,   // modello completo (budget alto)
  EXIT_MIDDLE = 1,   // exit 1 (budget medio)
  EXIT_LINEAR = 0    // formula oracle (budget basso, no TFLite)
};

// Inizializza anche il modello exit 1
bool inference_init_exit1();

// Inferenza con exit 1 (modello ridotto)
float inference_predict_exit1(const float features[10], unsigned long &latency_us);

// Restituisce il livello di exit usato nell'ultima chiamata adattiva
ExitLevel inference_last_exit_level();

// Inferenza adattiva: sceglie automaticamente l'exit in base al budget
// budget: valore corrente del budget (non normalizzato, [0,1])
// oracle_cost: risultato della formula oracle (pre-calcolato dal planner)
float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us);
```

### 4.3 Inference Implementation (`inference.cpp`)

Add a second set of TFLite variables for exit 1 model:

```cpp
// Secondo interprete per exit 1
namespace exit1 {
  const tflite::Model*      tflModel  = nullptr;
  tflite::MicroInterpreter* interp    = nullptr;
  TfLiteTensor*             input     = nullptr;
  TfLiteTensor*             output    = nullptr;
  float scale_in  = 0.003922f;
  int   zero_in   = -128;
  float scale_out = 0.003906f;
  int   zero_out  = -128;
}

alignas(16) static uint8_t tensor_arena_exit1[TENSOR_ARENA_EXIT1_SIZE];
static ExitLevel last_exit = EXIT_FULL;
```

`inference_init_exit1()` segue la stessa struttura di `inference_init()` ma carica `model_exit1_data` (dall'header `model_exit1.h`) e usa `tensor_arena_exit1`.

`inference_predict_exit1()` segue la stessa struttura di `inference_predict()` ma usa i tensori dell'interprete exit1.

`inference_predict_adaptive()`:

```cpp
float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us) {
  if (budget >= BUDGET_HIGH_THRESHOLD) {
    last_exit = EXIT_FULL;
    return inference_predict(features, latency_us);
  } else if (budget >= BUDGET_LOW_THRESHOLD) {
    last_exit = EXIT_MIDDLE;
    return inference_predict_exit1(features, latency_us);
  } else {
    last_exit = EXIT_LINEAR;
    latency_us = 0;
    return oracle_cost;  // già calcolato dal planner, costo zero
  }
}

ExitLevel inference_last_exit_level() { return last_exit; }
```

**Note:** When `budget < BUDGET_LOW_THRESHOLD`, the planner pre-calculates `oracle_cost` before calling `inference_predict_adaptive`.

### 4.4 Planner Logic (`planner.cpp`)

Add `POLICY_ADAPTIVE` case in `planner_decide()`:

```cpp
else if (policy == POLICY_ADAPTIVE) {
  unsigned long total_lat = 0;
  for (int b = 0; b < NUM_BRANCHES; b++) {
    float feat[NUM_FEATURES];
    planner_build_features(budget, branches[b], sensor, feat);
    planner_normalize(feat);

    // Pre-calcola oracle cost (usato come fallback se budget basso)
    float oracle_cost = planner_oracle_cost(budget, branches[b], sensor);

    unsigned long lat = 0;
    dec.predicted_costs[b] = inference_predict_adaptive(feat, budget, oracle_cost, lat);
    total_lat += lat;
  }
  dec.inference_time_us = total_lat;

  // Selezione del branch migliore (stessa logica di POLICY_ML)
  int best = 0;
  for (int b = 1; b < NUM_BRANCHES; b++) {
    if (dec.predicted_costs[b] < dec.predicted_costs[best]) best = b;
  }

  // Safety margin (stessa logica di POLICY_ML)
  if (budget < SAFETY_MARGIN) {
    if (dec.predicted_costs[best] > 0.5f * budget) {
      int safest = best;
      float min_safe = dec.predicted_costs[best];
      for (int b = 0; b < NUM_BRANCHES; b++) {
        if (b != best && dec.predicted_costs[b] < min_safe) {
          safest = b;
          min_safe = dec.predicted_costs[b];
        }
      }
      if (safest != best) { best = safest; dec.safety_override = true; }
    }
  }

  dec.selected_branch = best;
  dec.selected_cost = dec.predicted_costs[best];
}
```

### 4.5 Main Sketch (`Energy-Aware-Path.ino`)

**Serial loop**: Add command `'6'` for adaptive policy:

```cpp
case '6':
  current_policy = POLICY_ADAPTIVE;
  Serial.println("Policy: Adaptive (exit selection based on budget)");
  break;
```

**In `run_checkpoint()`**: Print exit level when using POLICY_ADAPTIVE:

```cpp
if (current_policy == POLICY_ML || current_policy == POLICY_ADAPTIVE) {
  Serial.print("Inference time: ");
  Serial.print(dec.inference_time_us / 1000.0f, 2);
  Serial.println(" ms");
}

if (current_policy == POLICY_ADAPTIVE) {
  ExitLevel lvl = inference_last_exit_level();
  Serial.print("Exit used: ");
  if (lvl == EXIT_FULL)   Serial.println("FULL MODEL (budget high)");
  if (lvl == EXIT_MIDDLE) Serial.println("EXIT 1 (budget medium)");
  if (lvl == EXIT_LINEAR) Serial.println("LINEAR FORMULA (budget low)");
}
```

**In `setup()`**: Initialize exit1 model:

```cpp
if (!inference_init_exit1()) {
  Serial.println("WARNING: Exit-1 model init failed. Adaptive policy will fall back to full model.");
  // Non è fatale: la policy adattiva può degradare a POLICY_ML
}
```

**Startup message**: Update command help:

```
Press 1-6 to change policy (6=Adaptive)
```

---

## 5. Training Notebook

### 5.1 Multi-Exit Model Structure

```python
import tensorflow as tf

inputs = tf.keras.Input(shape=(10,), name='input')
x1 = tf.keras.layers.Dense(16, activation='relu', name='dense_1')(inputs)
exit1_out = tf.keras.layers.Dense(1, activation='sigmoid', name='exit1')(x1)
x2 = tf.keras.layers.Dense(8, activation='relu', name='dense_2')(x1)
final_out = tf.keras.layers.Dense(1, activation='sigmoid', name='final')(x2)

model_multi = tf.keras.Model(inputs=inputs, outputs=[exit1_out, final_out])

model_multi.compile(
    optimizer='adam',
    loss={'exit1': 'mse', 'final': 'mse'},
    loss_weights={'exit1': 0.4, 'final': 1.0},
    metrics={'exit1': 'mae', 'final': 'mae'}
)
```

Use the same training dataset (X_train, y_train) for both outputs:

```python
history = model_multi.fit(
    X_train,
    {'exit1': y_train, 'final': y_train},
    validation_data=(X_test, {'exit1': y_test, 'final': y_test}),
    epochs=200,
    batch_size=16,
    callbacks=[tf.keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True)]
)
```

### 5.2 Exit 1 Model Extraction and Conversion

```python
# Modello exit 1: input → dense_1 → exit1
exit1_model = tf.keras.Model(
    inputs=model_multi.input,
    outputs=model_multi.get_layer('exit1').output
)

def representative_dataset_gen():
    for i in range(0, len(X_train), len(X_train)//100):
        yield [X_train[i:i+1].astype(np.float32)]

# Converti exit1_model a TFLite int8
converter = tf.lite.TFLiteConverter.from_keras_model(exit1_model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_exit1 = converter.convert()

with open('model_exit1.tflite', 'wb') as f:
    f.write(tflite_exit1)
print(f"Exit1 model size: {len(tflite_exit1)} bytes")
```

### 5.3 C Header Conversion

```python
def tflite_to_c_array(tflite_bytes, var_name):
    hex_array = ', '.join([f'0x{b:02x}' for b in tflite_bytes])
    return (
        f'#ifndef {var_name.upper()}_H\n'
        f'#define {var_name.upper()}_H\n\n'
        f'alignas(8) const unsigned char {var_name}[] = {{\n  {hex_array}\n}};\n'
        f'const unsigned int {var_name}_len = {len(tflite_bytes)};\n\n'
        f'#endif\n'
    )

# model.h (modello completo, come già fatto)
# model_exit1.h (nuovo)
with open('model_exit1.h', 'w') as f:
    f.write(tflite_to_c_array(tflite_exit1, 'model_exit1_data'))
```

### 5.4 Comparative Evaluation

```python
# Confronto MAE: modello completo vs exit 1
_, exit1_preds = model_multi.predict(X_test)  # output finale
exit1_only_preds, _ = model_multi.predict(X_test)  # exit 1

# Oppure usando i modelli separati:
full_preds = full_model.predict(X_test).flatten()
e1_preds   = exit1_model.predict(X_test).flatten()

mae_full  = np.mean(np.abs(full_preds - y_test))
mae_exit1 = np.mean(np.abs(e1_preds - y_test))

print(f"MAE full model: {mae_full:.4f}")
print(f"MAE exit 1:     {mae_exit1:.4f}")
print(f"Accuracy degradation: {(mae_exit1 - mae_full) / mae_full * 100:.1f}%")
```

---

## 6. Expected Serial Output

Example with budget decreasing through thresholds:

```
=== CHECKPOINT 1 ===
Energy budget: 0.850
Sensor context: acc_mean=1.02 acc_std=0.04 gyro=1.23 tilt=3.1°

Branch A: len=0.3 turns=5 diff=0.8 slope=0.4 → predicted cost: 0.412
Branch B: len=0.5 turns=3 diff=0.5 slope=0.1 → predicted cost: 0.331
Branch C: len=0.7 turns=1 diff=0.2 slope=-0.1 → predicted cost: 0.271

SELECTED: Branch C (cost=0.271)
Remaining budget: 0.579
Inference time: 1.48 ms
Exit used: FULL MODEL (budget high)

=== CHECKPOINT 2 ===
Energy budget: 0.579
...
Exit used: EXIT 1 (budget medium)

=== CHECKPOINT 3 ===
Energy budget: 0.241
...
Inference time: 0.00 ms
Exit used: LINEAR FORMULA (budget low)
```

---

## 7. Key Metrics

| Metric | How to Measure |
|--------|----------------|
| MAE full model (offline) | Colab notebook, on X_test |
| MAE exit 1 (offline) | Colab notebook, on X_test |
| Full model size (flash) | `ls -la model.tflite` |
| Exit 1 size (flash) | `ls -la model_exit1.tflite` |
| Arena used (full) | `interpreter->arena_used_bytes()` in setup |
| Arena used (exit 1) | `interp_exit1->arena_used_bytes()` in setup |
| Latency (full model) | `dec.inference_time_us` with POLICY_ML |
| Latency (exit 1) | `dec.inference_time_us` with POLICY_ADAPTIVE at medium budget |
| Total cost comparison | `total_cost_ml` in summary |
| Agreement with oracle | Count `[AGREE]` in logs |

---

## 8. Hardware Constraints

- **SRAM**: 256 KB total. Two arenas (8 KB + 4 KB) + IMU buffer (~2.8 KB) + stack well within limits
- **Flash**: 1 MB total. Two models (~6 KB) + code negligible
- **No dynamic allocation**: All buffers are static
- **No WiFi/BLE**: Serial output only (115200 baud)
- **Library**: `Harvard_TinyMLx` (TensorFlow Lite Micro)
- **Board**: Arduino Nano 33 BLE Sense Lite

---

## 9. Conclusion

This extension demonstrates **adaptive inference** on embedded hardware: the system scales inference cost proportionally to available energy budget, implementing the core concept of ePerceptive's graceful degradation.

**Comparison:**
- **POLICY_ML**: Always full model, ~0.55 ms, lowest MAE
- **POLICY_ADAPTIVE**: Variable exit, lower average latency, slightly higher MAE at medium/low budget, but survives longer by spending fewer computational resources when budget is scarce

The key insight: adaptive policy **degrades gracefully** instead of behaving uniformly regardless of energy context.
