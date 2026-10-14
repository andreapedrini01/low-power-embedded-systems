#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include "config.h"

// One window of accelerometer samples (contact vibration pickup).
struct AccelWindow {
  float ax[WINDOW_SAMPLES];
  float ay[WINDOW_SAMPLES];
  float az[WINDOW_SAMPLES];
  int   count;
};

bool sensors_init();
bool sensors_acquire_window(AccelWindow &w);

#endif
