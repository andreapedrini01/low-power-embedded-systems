#ifndef CONFIG_H
#define CONFIG_H

/*
 * Sampling and Window Configuration
 * LSM9DS1 output data rate is 119 Hz; a 1-second capture window consists of 119 samples.
 */
#define ACCEL_RATE_HZ        119
#define WINDOW_SAMPLES       119
#define RECORD_PRE_DELAY_MS  1000

/*
 * Activity Classes and Feature Dimensions
 * 5 classes: 0=idle, 1=typing, 2=writing, 3=handling, 4=incidental.
 * 33 features: 8 statistical metrics across 4 channels (ax, ay, az, magnitude) + 1 burstiness metric.
 */
#define N_CLASSES   5
#define N_FEATURES  33

/*
 * Model Selection
 * Uncomment to use model_finetuned.h (calibrated for the current surface);
 * keep commented to use model.h (base reference model).
 */
#define USE_FINETUNED_MODEL

/*
 * Decision Policy and State Machine Thresholds
 * GATE_RMS: Mean-removed vibration threshold in g. Skips inference when motion is below this floor.
 * ACT_CONSECUTIVE: Number of consecutive activity windows required before switching LOAD ON.
 * IDLE_TIMEOUT_MS: Inactivity duration before cutting power to the desk load.
 * ACT_THRESHOLD: Minimum softmax probability to accept a predicted activity class.
 */
#define GATE_RMS             0.0015f
#define ACT_CONSECUTIVE      2
#define IDLE_TIMEOUT_MS      12000
#define ACT_THRESHOLD        0.50f

/*
 * Duty-Cycled Sniffing Mode
 * SNIFF_INTERVAL_MS: Inter-sample sleep interval during sniffing state (4 Hz polling).
 * WAKE_DELTA_G: Single-sample acceleration delta that triggers transition to full-rate sensing.
 * SNIFF_FALLBACK_WIN: Consecutive quiet windows before falling back from active sensing to sniffing.
 */
#define SNIFF_INTERVAL_MS    250
#define WAKE_DELTA_G         0.0060f
#define SNIFF_FALLBACK_WIN   3

/*
 * Electrical Power Model Parameters
 * Measured USB voltage, active/sleep currents for the Arduino Nano 33 BLE, and assumed peripheral load.
 */
#define V_BUS             5.2f
#define I_ACTIVE_MA       32.0f
#define I_LIGHT_SLEEP_UA  300.0f
#define I_DEEP_SLEEP_UA   5.0f
#define LOAD_WATTS        50.0f

#endif // CONFIG_H
