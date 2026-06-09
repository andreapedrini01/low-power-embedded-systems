# EnergyAwarePath - Energy Measurement Extension

**Authors:** Pedrini, Bellini
**Hardware:** Arduino Nano 33 BLE Sense Lite (nRF52840)
**Status:** Specification for the next implementation pass.

This document instructs the implementation LLM to extend the existing
EnergyAwarePath project with **on-board energy estimation** and **real sleep
between operations**, so the demo shows energy savings, not just latency
reduction.

It supplements `ADAPTIVE_EXTENSION.md`. It does not replace it.

---

## 0. Power Numbers Used By This Extension

The official datasheet (`ABX00031-datasheet.md`, section 1.2) lists all
board-level consumption fields as **TBC** and provides no concrete numbers.
The constants below come from two external, declared sources and must be
documented in `config.h` next to their definition.

| Symbol           | Value      | Origin                                                        |
|------------------|------------|---------------------------------------------------------------|
| `V_BUS`          | 5.2 V      | USB power-meter reading on this exact board                   |
| `I_ACTIVE_MA`    | 32.0 mA    | USB power-meter reading, board running                        |
| `I_LIGHT_SLEEP_UA` | 300.0 µA | Community-reported, `delay()` with timers running             |
| `I_DEEP_SLEEP_UA`  | 5.0 µA   | Community-reported, after stopping `micros()` and USB Phy     |

All energy figures derived from these constants are **estimates**. The
implementation must label them as such in serial output and report.

If a future direct measurement of light/deep sleep is taken on the board,
only `config.h` needs to be updated; no other code change is required.

---

## 1. Objective

Add two capabilities to the existing project:

1. **Energy accounting (Layer 1).** Track active time and sleep time around
   every operation, multiply by the constants above, and report energy in µJ
   per checkpoint and mJ per mission.
2. **Real sleep (Layer 2).** Put the board to sleep between checkpoints and,
   where possible, during inter-sample idle in the IMU acquisition loop. The
   adaptive policy must extend sleep duration when the budget is low, so
   "energy-aware" applies both to compute cost and to idle cost.

Outcome to demonstrate in the demo: **Adaptive policy spends measurably less
energy** than `POLICY_ML` (always full inference, no sleep) over the same
checkpoint sequence.

---

## 2. New File: `power.h` / `power.cpp`

A new module is added next to the existing ones. It owns the energy model
and the sleep API. No other module computes energy directly.

### 2.1 `power.h`

```cpp
#ifndef POWER_H
#define POWER_H

#include <Arduino.h>

enum SleepLevel {
  SLEEP_LIGHT = 0,
  SLEEP_DEEP  = 1
};

struct EnergyAccount {
  unsigned long active_us;
  unsigned long light_sleep_us;
  unsigned long deep_sleep_us;
  float         energy_uj;
};

bool power_init();

// Active-time accounting: bracket every active section.
void power_mark_active_begin();
void power_mark_active_end();

// Sleep with accounting. Blocking call.
void power_sleep(SleepLevel level, unsigned long duration_ms);

// Per-checkpoint and total reset/read.
void               power_checkpoint_reset();
EnergyAccount      power_checkpoint_get();
EnergyAccount      power_total_get();

// Energy formula helper, exposed for unit-testable use.
float power_energy_uj(unsigned long active_us,
                      unsigned long light_us,
                      unsigned long deep_us);

#endif
```

### 2.2 `power.cpp` - implementation rules

- All accumulators are `static` module-level variables. No dynamic allocation.
- `power_mark_active_begin/end` use `micros()`; nesting is not supported, but
  the implementation must assert (via `Serial.println("WARN: nested active")`)
  if `begin` is called twice without `end`.
- `power_energy_uj` formula:
  ```
  E_uJ = V_BUS * (I_ACTIVE_MA * active_ms
                + I_LIGHT_SLEEP_UA * 1e-3 * light_ms
                + I_DEEP_SLEEP_UA  * 1e-3 * deep_ms)
  ```
  Units: `V * mA * ms = µJ`.
- `power_sleep(level, duration_ms)`:
  - Calls `power_mark_active_end()` before sleeping if active window is open.
  - `SLEEP_LIGHT`: lowest-cost idle that keeps `micros()` and Serial alive
    (acceptable fallback: `delay(duration_ms)`).
  - `SLEEP_DEEP`: deepest sleep that the platform supports while still being
    wakeable by Serial input (see Section 4).
  - Records the elapsed time in the corresponding accumulator.
  - Re-opens the active window on return.

---

## 3. Configuration Additions (`config.h`)

Add to the existing `config.h`. Do not move existing entries.

```cpp
// Power model. Update only after a direct measurement on the board.
// See ENERGY_EXTENSION.md section 0 for sources.
#define V_BUS              5.2f      // V, USB supply
#define I_ACTIVE_MA        32.0f     // mA, board running
#define I_LIGHT_SLEEP_UA   300.0f    // uA, light sleep
#define I_DEEP_SLEEP_UA    5.0f      // uA, deep sleep

// Sleep policy.
#define SLEEP_BETWEEN_CHECKPOINTS 1   // 0 = busy wait, 1 = deep sleep
#define SLEEP_DURING_SAMPLING     1   // 0 = busy poll, 1 = light sleep

// Adaptive sleep tuning.
#define ADAPTIVE_SLEEP_LOW_MS    2000  // sleep slot when budget < LOW
#define ADAPTIVE_SLEEP_MID_MS     500  // sleep slot at medium budget
#define ADAPTIVE_SLEEP_HIGH_MS      0  // no extra sleep at high budget
```

---

## 4. Sleep API Selection

The Nano 33 BLE Sense runs on Mbed OS. Two compatible options:

1. **`ArduinoLowPower` library.** `LowPower.deepSleep(ms)` and `LowPower.sleep(ms)`.
2. **Mbed sleep primitives.** `mbed::ThisThread::sleep_for(ms)` for light sleep;
   for deep sleep the implementation must stop the system tick and disable USB
   Phy as documented in the Arduino MBED Core.

The implementation picks one of the two and isolates it inside `power.cpp`
behind `power_sleep()`. Calling code never references the underlying API.

Wake source for the demo: **Serial input.** The implementation must
configure the chosen sleep mode so that pressing a key on the Serial Monitor
returns the board to active state. If the chosen API does not support
Serial-wake from deep sleep, fall back to a timed `deepSleep(N)` loop with
`Serial.available()` check after each wake.

---

## 5. Integration With Existing Modules

### 5.1 `sensors.cpp` - `sensors_acquire_window`

The current loop is `while (micros() < next_sample) {}`. Replace with a
sleep-aware wait when `SLEEP_DURING_SAMPLING == 1`:

```cpp
// Inter-sample wait. Sleep accounted for in power module.
unsigned long now_us = micros();
if (next_sample > now_us) {
  unsigned long wait_us = next_sample - now_us;
  if (wait_us > 1500) {
    power_sleep(SLEEP_LIGHT, (wait_us - 500) / 1000);
  }
  while (micros() < next_sample) {}   // tail busy-wait
}
```

Active-time bracketing for the whole acquisition is added by the caller
(see Section 5.3), not inside `sensors.cpp`.

### 5.2 `planner.cpp`

`Decision` already exposes `inference_time_us`. Extend it:

```cpp
struct Decision {
  int   selected_branch;
  float predicted_costs[NUM_BRANCHES];
  float selected_cost;
  unsigned long inference_time_us;
  unsigned long active_time_us;     // total active time of decide()
  float energy_uj;                  // estimated energy of decide()
  bool  safety_override;
};
```

`planner_decide()` must:

1. Call `power_mark_active_begin()` at entry, `power_mark_active_end()` at exit.
2. Snapshot `power_checkpoint_get()` before and after the call, write the
   delta into `dec.active_time_us` and `dec.energy_uj`.

### 5.3 `Energy-Aware-Path.ino` - `run_checkpoint`

Order of operations inside `run_checkpoint`:

1. `power_checkpoint_reset()`.
2. `power_mark_active_begin()`.
3. `sensors_acquire_window(imu_window)` (this contains light sleeps internally).
4. `features_extract(...)`.
5. `power_mark_active_end()`.
6. `Decision dec = planner_decide(...)` (own active bracketing).
7. Print: branches, selected, latency, **energy**.
8. Call `policy_post_checkpoint_sleep(...)` (Section 6.3) which sleeps before
   returning control to the main loop.

### 5.4 Main loop - between checkpoints

When `SLEEP_BETWEEN_CHECKPOINTS == 1` and the mission is not complete, the
loop must sleep instead of busy-polling Serial. Implementation:

```cpp
void loop() {
  if (Serial.available()) {
    handle_serial();
  } else if (!mission_complete) {
    power_sleep(SLEEP_DEEP, 200);   // wake periodically to poll Serial
  }
}
```

Sleep slot of 200 ms is a reasonable compromise between responsiveness and
energy. It is a `#define` (`MAIN_LOOP_SLEEP_MS`) for tuning.

---

## 6. Adaptive Energy Behavior

### 6.1 Compute side (existing)

`POLICY_ADAPTIVE` already routes among full / exit1 / linear based on budget
in `inference_predict_adaptive()`. No change.

### 6.2 Sleep side (new)

Add a function in `planner.cpp`:

```cpp
unsigned long planner_post_checkpoint_sleep_ms(float budget, Policy policy);
```

Returns:
- `0` for non-adaptive policies.
- `ADAPTIVE_SLEEP_HIGH_MS` if `budget >= BUDGET_HIGH_THRESHOLD`.
- `ADAPTIVE_SLEEP_MID_MS`  if `budget >= BUDGET_LOW_THRESHOLD`.
- `ADAPTIVE_SLEEP_LOW_MS`  otherwise.

Called from `run_checkpoint()` after the decision is logged:

```cpp
unsigned long s = planner_post_checkpoint_sleep_ms(energy_budget, current_policy);
if (s > 0) power_sleep(SLEEP_DEEP, s);
```

This is the second axis of "energy awareness": when the budget is low, the
adaptive policy spends less compute *and* idles harder.

### 6.3 Comparison Setup

For a fair comparison the same sequence of policies is run over the same
checkpoint set with the same sensor input. To remove the operator-dependent
delay between `n` commands, add a non-interactive batch mode triggered by
serial command `b`:

- Iterates over all 4 checkpoints.
- For sensor input, uses the **last acquired window** if available, or
  acquires once and reuses it. This makes the energy comparison reproducible.
- Runs each policy in turn (`POLICY_ML`, `POLICY_ADAPTIVE`, `POLICY_ALWAYS_A`,
  `POLICY_SHORTEST`) and prints a summary table at the end.

---

## 7. Output Format

### 7.1 Per Checkpoint

Add to existing log, in the same compact style:

```
=== CHECKPOINT 2 ===
Budget: 0.610
Context: acc=1.10/0.32/2.05 gyro=0.92 tilt=8.4°

Branch A: 0.3/5/0.8/0.4 -> 0.418
Branch B: 0.5/3/0.5/0.1 -> 0.331
Branch C: 0.7/1/0.2/-0.1 -> 0.275

SELECTED: C (0.275)
Remaining: 0.335
Inference: 1.48 ms
Active: 142.3 ms
Sleep:   500 ms (deep)
Energy:  23.7 mJ (est)
```

`(est)` suffix is mandatory wherever an energy figure derived from
`I_*` constants appears.

### 7.2 Mission Summary

Extend `print_summary()`:

```
---------- SUMMARY ----------
Checkpoints: 4 / 4
Final budget: 0.084
Energy consumed (sim): 0.916
Policy cost: 0.916
Oracle cost: 0.901
Efficiency: 98.4%

Active time:  582 ms
Sleep time:   2100 ms (light: 100 ms, deep: 2000 ms)
Energy (est): 102.4 mJ

vs Always-Full+no-sleep: 156.0 mJ
Savings: 34.4%
-----------------------------
```

The "vs Always-Full+no-sleep" baseline is computed analytically from the
recorded active time and an assumed zero sleep time. It does not require a
second run.

---

## 8. README And Documentation Updates

After implementation, update `EnergyAwarePath/README.md`:

- Replace the headline metric "Latency reduction: 27%" with a results section
  that includes both latency *and* energy savings (% vs Always-Full no-sleep).
- Add an "Energy model" subsection that links to this file and lists the four
  power constants with their sources.
- The Performance table must add columns: `Active (ms)`, `Sleep (ms)`,
  `Energy (mJ, est)`, `Energy saved (%)`.

---

## 9. Code Style Constraints

- All identifiers, strings, and comments in **English**.
- Comments are **essential** and **short**. Match the existing style of
  `planner.cpp`, `inference.cpp`, `sensors.cpp` (one-line comments above the
  block, no multi-paragraph explanations).
- No dynamic allocation. All buffers static.
- Power model lives entirely in `power.h` / `power.cpp`. No other module
  reads `I_*` or `V_BUS` constants directly.
- Energy figures must always be printed with the `(est)` suffix or equivalent
  marker so the demo audience knows they are estimated.
- Library additions allowed: `ArduinoLowPower` (or none, if Mbed primitives
  are used). Do not add anything else.

---

## 10. Acceptance Criteria

The implementation is complete when:

1. The sketch builds and uploads to a Nano 33 BLE Sense Lite without warnings.
2. Running `POLICY_ML` for one full mission prints a non-zero energy total.
3. Running `POLICY_ADAPTIVE` over the same mission prints **a strictly lower
   energy total** than `POLICY_ML`. If it does not, threshold tuning in
   `config.h` (`ADAPTIVE_SLEEP_*`) is part of the deliverable.
4. The board observably draws less current during inter-checkpoint waits when
   `SLEEP_BETWEEN_CHECKPOINTS == 1` (verified with the same USB meter used to
   determine `I_ACTIVE_MA`).
5. The summary table at the end of the mission contains all the fields listed
   in Section 7.2.
6. `README.md` is updated as described in Section 8.

---

## 11. Out Of Scope

- Direct hardware measurement of sleep current. Done by the user, separately,
  if a refined `I_*_SLEEP_UA` value is desired.
- Modification of the multi-exit training notebook. The energy extension is
  inference-time only.
- Cutting the power LED solder bridge or any other hardware modification.
