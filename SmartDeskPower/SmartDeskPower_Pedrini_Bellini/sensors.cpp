#include "sensors.h"
#include <Arduino_LSM9DS1.h>
#include <math.h>

static float prev_mag = 1.0f;

/*
 * Initializes the LSM9DS1 IMU peripheral and resets the baseline gravity reference.
 */
bool sensors_init() {
  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    return false;
  }
  Serial.println("IMU ready (LSM9DS1)");
  prev_mag = 1.0f;
  return true;
}

/*
 * Fills an AccelWindow by polling IMU.accelerationAvailable() until WINDOW_SAMPLES are stored.
 * Runs at the sensor's hardware output data rate (~119 Hz).
 */
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

/*
 * Fetches one immediate sample from the IMU without waiting for a full window buffer.
 * Calculates |mag - prev_mag| to identify sudden vibration spikes above the noise floor.
 */
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
