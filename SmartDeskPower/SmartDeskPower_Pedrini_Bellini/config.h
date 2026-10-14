#ifndef CONFIG_H
#define CONFIG_H

// ── Sampling / window ──
#define ACCEL_RATE_HZ        119     // LSM9DS1 default accel ODR
#define WINDOW_SAMPLES       119     // ~1 s window (one classification per second)
#define RECORD_PRE_DELAY_MS  1000    // wait before capture so the activity can start

// ── Classes ──
// 0=idle 1=typing 2=writing 3=handling 4=incidental
// {typing, writing, handling} = activity; {idle, incidental} = no activity.
#define N_CLASSES   5
#define N_FEATURES  33   // 8 stats × 4 channels + 1 burstiness

// ── Decision policy / state machine ──
#define GATE_RMS         0.020f  // vibration RMS gate (g, mean-removed); tune from dataset
#define ACT_CONSECUTIVE  2       // activity windows required before POWER ON
#define IDLE_TIMEOUT_MS  12000   // cut load after this inactivity (short for demo)
#define ACT_THRESHOLD    0.60f   // min softmax confidence to accept a class

// ── Power model (est). Board-level active/sleep is TBC; reused from EnergyAwarePath. ──
#define V_BUS             5.2f    // V, USB supply (measured, as EnergyAwarePath)
#define I_ACTIVE_MA       32.0f   // mA, board running
#define I_LIGHT_SLEEP_UA  300.0f  // uA, light sleep (community)
#define I_DEEP_SLEEP_UA   5.0f    // uA, deep sleep (community)
#define LOAD_WATTS        50.0f   // assumed desk load: monitor + lamp + peripherals (est)

#endif // CONFIG_H
