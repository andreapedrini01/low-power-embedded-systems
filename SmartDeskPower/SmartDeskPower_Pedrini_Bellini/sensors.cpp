#include "sensors.h"
#include <Arduino_LSM9DS1.h>

bool sensors_init() {
  if (!IMU.begin()) {
    Serial.println("IMU init failed");
    return false;
  }
  Serial.println("IMU ready (LSM9DS1)");
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
