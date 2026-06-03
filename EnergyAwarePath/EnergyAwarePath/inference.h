#ifndef INFERENCE_H
#define INFERENCE_H

#include <Arduino.h>

enum ExitLevel {
  EXIT_FULL   = 2,
  EXIT_MIDDLE = 1,
  EXIT_LINEAR = 0
};

bool inference_init();
bool inference_init_exit1();

float inference_predict(const float features[10], unsigned long &latency_us);
float inference_predict_exit1(const float features[10], unsigned long &latency_us);

ExitLevel inference_last_exit_level();

float inference_predict_adaptive(const float features[10], float budget,
                                 float oracle_cost, unsigned long &latency_us);

#endif
