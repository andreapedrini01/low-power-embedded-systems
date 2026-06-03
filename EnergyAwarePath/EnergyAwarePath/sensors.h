#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

struct IMUWindow {
  float ax[119];
  float ay[119];
  float az[119];
  float gx[119];
  float gy[119];
  float gz[119];
  int   count;
};

bool sensors_init();
bool sensors_acquire_window(IMUWindow &window);

#endif
