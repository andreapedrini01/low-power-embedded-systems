#ifndef INFERENCE_H
#define INFERENCE_H

#include "config.h"

/*
 * Prepares the inference engine.
 * For the hand-coded floating-point MLP, weights reside in flash and require no heap setup.
 */
bool inference_init();

/*
 * Executes the forward pass of the 3-layer MLP (33 -> 12 -> 6 -> 5):
 * Applies ReLU activation on hidden layers and numerically stable softmax on the output.
 * Populates scores[] with class probabilities and returns the index of the predicted class.
 */
int inference_run(const float features[N_FEATURES], float scores[N_CLASSES]);

#endif // INFERENCE_H
