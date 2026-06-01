#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>

// Exit level used in last adaptive inference
enum ExitLevel {
  EXIT_FULL   = 2,   // full model (high budget)
  EXIT_MIDDLE = 1,   // exit 1 (medium budget)
  EXIT_LINEAR = 0    // oracle formula (low budget, no TFLite)
};

// Initialize TFLite Micro interpreter (full model)
bool inference_init();

// Initialize exit 1 model
bool inference_init_exit1();

// Run inference on a feature vector (10 floats, already normalized 0-1)
// Returns predicted energy cost (0-1)
// latency_us is filled with inference time in microseconds
float inference_predict(const float features[10], unsigned long &latency_us);

// Inference with exit 1 (reduced model)
float inference_predict_exit1(const float features[10], unsigned long &latency_us);

// Returns the exit level used in the last adaptive call
ExitLevel inference_last_exit_level();

// Adaptive inference: automatically selects exit based on budget
// budget: current budget value (not normalized, [0,1])
// oracle_cost: result of oracle formula (pre-computed by planner)
float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us);

#endif // INFERENCE_H
