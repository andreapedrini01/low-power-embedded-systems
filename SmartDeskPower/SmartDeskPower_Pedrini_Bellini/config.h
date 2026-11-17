#ifndef CONFIG_H
#define CONFIG_H

// Sampling: 119 Hz LSM9DS1 output data rate; 1-second capture window = 119 samples.
#define ACCEL_RATE_HZ        119
#define WINDOW_SAMPLES       119
#define RECORD_PRE_DELAY_MS  1000

// 5 classes: 0=idle, 1=typing, 2=writing, 3=handling, 4=incidental.
// 33 features: 8 stats x 4 channels (ax, ay, az, magnitude) + 1 burstiness metric.
#define N_CLASSES   5
#define N_FEATURES  33

// Model selection: uncomment to use model_finetuned.h; comment out for base model.h.
#define USE_FINETUNED_MODEL

// State machine thresholds
#define GATE_RMS             0.0015f // Noise gate threshold (g); skips inference when below
#define ACT_CONSECUTIVE      2       // Consecutive activity windows required before switching LOAD ON
#define IDLE_TIMEOUT_MS      12000   // Inactivity timeout before turning LOAD OFF
#define ACT_THRESHOLD        0.50f   // Minimum softmax confidence to accept activity prediction

// 4 Hz sniffing parameters
#define SNIFF_INTERVAL_MS    250     // Sleep interval between sniffing samples (ms)
#define WAKE_DELTA_G         0.0060f // Single-sample acceleration delta to trigger 119 Hz acquisition
#define SNIFF_FALLBACK_WIN   3       // Consecutive quiet windows before returning to sniffing

// Power model parameters (measured USB voltage and nRF52840 currents)
#define V_BUS             5.2f
#define I_ACTIVE_MA       32.0f
#define I_LIGHT_SLEEP_UA  300.0f
#define I_DEEP_SLEEP_UA   5.0f
#define LOAD_WATTS        50.0f

#endif // CONFIG_H
