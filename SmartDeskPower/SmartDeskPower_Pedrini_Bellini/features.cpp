#include "features.h"
#include "model.h"
#include <math.h>

// ── Per-channel statistics (8 features) ──

static void channel_stats(const float *data, int n, float out[8]) {
  float sum = 0, sq_sum = 0, abs_sum = 0;
  float mn = data[0], mx = data[0];

  for (int i = 0; i < n; i++) {
    float v = data[i];
    sum    += v;
    sq_sum += v * v;
    abs_sum += fabsf(v);
    if (v < mn) mn = v;
    if (v > mx) mx = v;
  }

  float mean = sum / n;
  float var  = sq_sum / n - mean * mean;
  if (var < 0) var = 0;  // numerical guard

  // Zero-crossing rate on mean-removed signal (counts oscillations)
  int zc = 0;
  for (int i = 1; i < n; i++) {
    if ((data[i - 1] - mean >= 0) != (data[i] - mean >= 0)) zc++;
  }

  out[0] = mean;                // mean
  out[1] = sqrtf(var);          // std
  out[2] = sqrtf(sq_sum / n);   // RMS
  out[3] = mn;                  // min
  out[4] = mx;                  // max
  out[5] = mx - mn;             // range
  out[6] = abs_sum / n;         // mean absolute
  out[7] = (float)zc / n;       // zero-crossing rate
}

// ── Feature extraction (33 features) ──

void features_extract(const AccelWindow &w, float out[N_FEATURES]) {
  int n = w.count;

  // Magnitude channel
  float mag[WINDOW_SAMPLES];
  for (int i = 0; i < n; i++)
    mag[i] = sqrtf(w.ax[i]*w.ax[i] + w.ay[i]*w.ay[i] + w.az[i]*w.az[i]);

  // 8 stats × 4 channels = features [0..31]
  channel_stats(w.ax, n, &out[0]);
  channel_stats(w.ay, n, &out[8]);
  channel_stats(w.az, n, &out[16]);
  channel_stats(mag,  n, &out[24]);

  // Temporal burstiness on magnitude (feature 32).
  // Std of per-sub-window RMS values (~10 sub-windows of ~12 samples).
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
    for (int i = start; i < end; i++) sq += mag[i] * mag[i];
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

// ── Normalization ──

void features_normalize(float f[N_FEATURES]) {
  for (int i = 0; i < N_FEATURES; i++) {
    float s = NORM_STD[i];
    f[i] = (s > 1e-8f) ? (f[i] - NORM_MEAN[i]) / s : 0.0f;
  }
}

// ── Gate RMS ──
// Std of magnitude (mean-removed RMS) — measures vibration amplitude
// above the gravity baseline. Compared against GATE_RMS in config.h.

float features_window_rms(const AccelWindow &w) {
  float sum = 0, sq_sum = 0;
  int n = w.count;
  for (int i = 0; i < n; i++) {
    float m = sqrtf(w.ax[i]*w.ax[i] + w.ay[i]*w.ay[i] + w.az[i]*w.az[i]);
    sum    += m;
    sq_sum += m * m;
  }
  float mean = sum / n;
  float var  = sq_sum / n - mean * mean;
  return sqrtf(var > 0.0f ? var : 0.0f);
}
