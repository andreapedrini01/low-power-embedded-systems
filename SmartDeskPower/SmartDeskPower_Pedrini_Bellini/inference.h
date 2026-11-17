#ifndef INFERENCE_H
#define INFERENCE_H

#include "config.h"

// Flash weights require no dynamic allocation.
bool inference_init();

// Forward pass of 3-layer MLP (33 -> 12 -> 6 -> 5). Returns winning class index.
int inference_run(const float features[N_FEATURES], float scores[N_CLASSES]);

#endif // INFERENCE_H
