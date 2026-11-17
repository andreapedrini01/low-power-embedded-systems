#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include "config.h"

// 1-second capture window of 3-axis accelerometer samples (119 Hz).
struct AccelWindow {
  float ax[WINDOW_SAMPLES];
  float ay[WINDOW_SAMPLES];
  float az[WINDOW_SAMPLES];
  int   count;
};

// Initialize LSM9DS1 IMU over I2C. Returns true if communication succeeds.
bool sensors_init();

// Block until WINDOW_SAMPLES (119) are acquired from the IMU.
bool sensors_acquire_window(AccelWindow &w);

// Read a single sample with a 2 ms timeout; returns |mag - prev_mag| for 4 Hz sniffing.
float sensors_read_single_delta();

#endif // SENSORS_H
