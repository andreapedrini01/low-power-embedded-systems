#ifndef FEATURES_H
#define FEATURES_H

#include "config.h"
#include "sensors.h"

// Extract the 33-element feature vector from a raw accelerometer window.
void features_extract(const AccelWindow &w, float out[N_FEATURES]);

// Z-score normalize in place using NORM_MEAN / NORM_STD from model.h.
void features_normalize(float f[N_FEATURES]);

// Quick vibration RMS (std of magnitude) for the cheap gate check.
float features_window_rms(const AccelWindow &w);

#endif // FEATURES_H
