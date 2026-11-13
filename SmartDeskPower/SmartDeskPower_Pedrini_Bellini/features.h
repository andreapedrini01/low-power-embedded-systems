#ifndef FEATURES_H
#define FEATURES_H

#include "config.h"
#include "sensors.h"

/*
 * Computes the 33-element statistical feature vector from an acceleration window:
 * - Features [0..31]: 8 time-domain statistics for ax, ay, az, and Euclidean magnitude.
 * - Feature [32]: Sub-window energy burstiness (standard deviation of chunk RMS).
 */
void features_extract(const AccelWindow &w, float out[N_FEATURES]);

/*
 * Performs in-place Z-score standardization on the 33 features using NORM_MEAN and NORM_STD.
 */
void features_normalize(float f[N_FEATURES]);

/*
 * Computes the mean-removed RMS (standard deviation of magnitude) across the window.
 * Used as an early-exit gate to reject idle noise before executing inference.
 */
float features_window_rms(const AccelWindow &w);

#endif // FEATURES_H
