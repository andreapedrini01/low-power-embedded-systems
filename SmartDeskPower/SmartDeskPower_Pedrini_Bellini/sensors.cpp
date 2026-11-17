#include "sensors.h"
#include <Arduino_LSM9DS1.h>
#include <math.h>

static float prev_mag = 1.0f;

// Init LSM9DS1 and store baseline gravity reading.
bool sensors_init() {
  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    return false;
  }
  Serial.println("IMU ready (LSM9DS1)");
  prev_mag = 1.0f;
  return true;
}

// Poll IMU at 119 Hz until a complete window (119 samples) is buffered.
bool sensors_acquire_window(AccelWindow &w) {
  w.count = 0;
  while (w.count < WINDOW_SAMPLES) {
    if (IMU.accelerationAvailable()) {
      float ax, ay, az;
      IMU.readAcceleration(ax, ay, az);
      w.ax[w.count] = ax;
      w.ay[w.count] = ay;
      w.az[w.count] = az;
      w.count++;
    }
  }
  return (w.count == WINDOW_SAMPLES);
}

// Single-sample read with 2 ms timeout; returns delta from previous magnitude.
float sensors_read_single_delta() {
  float ax = 0.0f, ay = 0.0f, az = 1.0f;
  unsigned long start_us = micros();

  while (!IMU.accelerationAvailable()) {
    if (micros() - start_us > 2000) {
      return 0.0f;
    }
  }

  IMU.readAcceleration(ax, ay, az);
  float mag = sqrtf(ax * ax + ay * ay + az * az);
  float delta = fabsf(mag - prev_mag);
  prev_mag = mag;
  return delta;
}
