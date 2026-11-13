#ifndef SENSORS_H
#define SENSORS_H

#include <Arduino.h>
#include "config.h"

/*
 * Holds one 1-second window of 3-axis accelerometer readings sampled at 119 Hz.
 */
struct AccelWindow {
  float ax[WINDOW_SAMPLES];
  float ay[WINDOW_SAMPLES];
  float az[WINDOW_SAMPLES];
  int   count;
};

/*
 * Initializes the onboard LSM9DS1 IMU via I2C and verifies sensor communication.
 * Returns true if successful, false otherwise.
 */
bool sensors_init();

/*
 * Blocks to acquire WINDOW_SAMPLES consecutive 3-axis accelerometer samples at ~119 Hz.
 * Populates the provided AccelWindow struct and returns true when full.
 */
bool sensors_acquire_window(AccelWindow &w);

/*
 * Reads a single accelerometer sample with a tight 2 ms timeout and computes the
 * absolute difference in acceleration magnitude relative to the previous reading.
 * Used for low-power duty-cycled sniffing (<0.2 ms active execution).
 */
float sensors_read_single_delta();

#endif // SENSORS_H
