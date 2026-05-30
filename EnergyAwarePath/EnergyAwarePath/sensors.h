#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>

// Raw IMU data buffer
struct IMUWindow {
  float ax[119];
  float ay[119];
  float az[119];
  float gx[119];
  float gy[119];
  float gz[119];
  int   count;
};

// Initialize IMU sensor
bool sensors_init();

// Acquire a full window of IMU data (~1.2s at 100Hz)
// Returns true if successful
bool sensors_acquire_window(IMUWindow &window);

#endif // SENSORS_H
