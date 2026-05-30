#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>

// Initialize TFLite Micro interpreter
bool inference_init();

// Run inference on a feature vector (10 floats, already normalized 0-1)
// Returns predicted energy cost (0-1)
// latency_us is filled with inference time in microseconds
float inference_predict(const float features[10], unsigned long &latency_us);

#endif // INFERENCE_H
