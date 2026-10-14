#ifndef INFERENCE_H
#define INFERENCE_H

#include "config.h"

bool inference_init();

// Run MLP forward pass. Returns index of winning class; fills scores[].
int  inference_run(const float features[N_FEATURES], float scores[N_CLASSES]);

#endif // INFERENCE_H
