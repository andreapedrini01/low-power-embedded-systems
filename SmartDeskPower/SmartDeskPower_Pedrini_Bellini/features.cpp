#include "features.h"
#include "active_model.h"
#include <math.h>

/*
 * Computes 8 time-domain statistics for an array of n samples:
 * [0] mean, [1] standard deviation, [2] RMS, [3] minimum,
 * [4] maximum, [5] peak-to-peak range, [6] mean absolute value,
 * [7] zero-crossing rate relative to the mean.
 */
static void channel_stats(const float *data, int n, float out[8]) {
  float sum = 0, sq_sum = 0, abs_sum = 0;
  float mn = data[0], mx = data[0];

  for (int i = 0; i < n; i++) {
    float v = data[i];
    sum     += v;
    sq_sum  += v * v;
    abs_sum += fabsf(v);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }

  float mean = sum / n;
  float var  = sq_sum / n - mean * mean;
  if (var < 0) var = 0;

  int zc = 0;
  for (int i = 1; i < n; i++) {
    if ((data[i - 1] - mean >= 0) != (data[i] - mean >= 0)) {
      zc++;
    }
  }

  out[0] = mean;
  out[1] = sqrtf(var);
  out[2] = sqrtf(sq_sum / n);
  out[3] = mn;
  out[4] = mx;
  out[5] = mx - mn;
  out[6] = abs_sum / n;
  out[7] = (float)zc / n;
}

/*
 * Extracts the 33 features from an acceleration window.
 * Calculates channel_stats for ax, ay, az, and 3D magnitude (features 0..31),
 * then partitions magnitude into ~12-sample chunks to calculate temporal energy
 * variance (burstiness, feature 32) to separate keystrokes from friction.
 */
void features_extract(const AccelWindow &w, float out[N_FEATURES]) {
  int n = w.count;

  float mag[WINDOW_SAMPLES];
  for (int i = 0; i < n; i++) {
    mag[i] = sqrtf(w.ax[i] * w.ax[i] + w.ay[i] * w.ay[i] + w.az[i] * w.az[i]);
  }

  channel_stats(w.ax, n, &out[0]);
  channel_stats(w.ay, n, &out[8]);
  channel_stats(w.az, n, &out[16]);
  channel_stats(mag,  n, &out[24]);

  const int SUB_N = 12;
  int n_subs = (n + SUB_N - 1) / SUB_N;
  float sub_rms[16];
  float sub_sum = 0;

  for (int s = 0; s < n_subs; s++) {
    int start = s * SUB_N;
    int end   = start + SUB_N;
    if (end > n) end = n;
    int cnt = end - start;

    float sq = 0;
    for (int i = start; i < end; i++) {
      sq += mag[i] * mag[i];
    }
    sub_rms[s] = sqrtf(sq / cnt);
    sub_sum += sub_rms[s];
  }

  float sub_mean = sub_sum / n_subs;
  float sub_var  = 0;
  for (int s = 0; s < n_subs; s++) {
    float d = sub_rms[s] - sub_mean;
    sub_var += d * d;
  }
  out[32] = sqrtf(sub_var / n_subs);
}

/*
 * Standardizes each feature in-place using pre-computed normalization parameters:
 * f_norm = (f - NORM_MEAN) / NORM_STD.
 */
void features_normalize(float f[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    float s = NORM_STD[i];
    f[i] = (s > 1e-8f) ? (f[i] - NORM_MEAN[i]) / s : 0.0f;
  }
}

/*
 * Evaluates the root-mean-square amplitude of dynamic acceleration around its gravity mean.
 * Provides a low-cost scalar measure of desk vibration amplitude.
 */
float features_window_rms(const AccelWindow &w) {
  float sum = 0, sq_sum = 0;
  int n = w.count;

  for (int i = 0; i < n; i++) {
    float m = sqrtf(w.ax[i] * w.ax[i] + w.ay[i] * w.ay[i] + w.az[i] * w.az[i]);
    sum    += m;
    sq_sum += m * m;
  }

  float mean = sum / n;
  float var  = sq_sum / n - mean * mean;
  return sqrtf(var > 0.0f ? var : 0.0f);
}
