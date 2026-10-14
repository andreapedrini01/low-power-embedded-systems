#include "inference.h"
#include "model.h"
#include <math.h>

// Hand-coded float MLP forward pass (Assignment 1 style).
// Weights laid out row-major (in × out), matching Keras Dense export.

bool inference_init() {
  // No runtime init needed for hand-coded MLP
  return true;
}

// ── Dense layer ──

static void dense(const float *in, const float *w, const float *b,
                  float *out, int n_in, int n_out, bool relu) {
  for (int j = 0; j < n_out; j++) {
    float sum = b[j];
    for (int i = 0; i < n_in; i++)
      sum += in[i] * w[i * n_out + j];
    out[j] = (relu && sum < 0) ? 0 : sum;
  }
}

// ── Softmax (numerically stable) ──

static void softmax(float *v, int n) {
  float mx = v[0];
  for (int i = 1; i < n; i++)
    if (v[i] > mx) mx = v[i];
  float sum = 0;
  for (int i = 0; i < n; i++) {
    v[i] = expf(v[i] - mx);
    sum += v[i];
  }
  for (int i = 0; i < n; i++) v[i] /= sum;
}

// ── Forward pass ──

int inference_run(const float features[N_FEATURES], float scores[N_CLASSES]) {
  float h1[L1_OUT], h2[L2_OUT];

  dense(features, MLP_W1, MLP_B1, h1,     L1_IN, L1_OUT, true);
  dense(h1,       MLP_W2, MLP_B2, h2,     L2_IN, L2_OUT, true);
  dense(h2,       MLP_W3, MLP_B3, scores, L3_IN, L3_OUT, false);
  softmax(scores, N_CLASSES);

  int best = 0;
  for (int i = 1; i < N_CLASSES; i++)
    if (scores[i] > scores[best]) best = i;
  return best;
}
