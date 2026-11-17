#ifndef FEATURES_H
#define FEATURES_H

#include "config.h"
#include "sensors.h"

// Extract 33 features: 8 stats x 4 channels (ax, ay, az, magnitude) + 1 burstiness metric.
void features_extract(const AccelWindow &w, float out[N_FEATURES]);

// In-place Z-score normalization: (f - NORM_MEAN) / NORM_STD.
void features_normalize(float f[N_FEATURES]);

// Compute dynamic RMS (standard deviation of magnitude) for the noise gate.
float features_window_rms(const AccelWindow &w);

#endif // FEATURES_H
