#include "sensors.h"
#include <Arduino_LSM9DS1.h>
#include <math.h>

static float prev_mag = 1.0f;  // resting gravity baseline

bool sensors_init() {
  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    return false;
  }
  Serial.println("IMU ready (LSM9DS1)");
  prev_mag = 1.0f;
  return true;
}

// Fill one window at the native accel rate (~119 Hz). Board is stationary;
// this reads structure-borne vibration from activity on the desk.
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

// Fast single-sample reading (< 0.2 ms active execution).
// Reads one (ax, ay, az) from the LSM9DS1, computes magnitude, and returns
// deviation |mag - prev_mag| to check for mechanical vibration without
// blocking for a full 1-second window.
float sensors_read_single_delta() {
  float ax = 0.0f, ay = 0.0f, az = 1.0f;
  unsigned long start_us = micros();

  // Non-blocking check with tight timeout (~2 ms max)
  while (!IMU.accelerationAvailable()) {
    if (micros() - start_us > 2000) {
      return 0.0f; // No sample ready; stay in sleep
    }
  }

  IMU.readAcceleration(ax, ay, az);
  float mag = sqrtf(ax * ax + ay * ay + az * az);
  float delta = fabsf(mag - prev_mag);
  prev_mag = mag;
  return delta;
}
