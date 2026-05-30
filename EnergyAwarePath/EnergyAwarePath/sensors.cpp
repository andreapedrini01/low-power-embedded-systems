#include "sensors.h"
#include "config.h"
#include <Arduino_LSM9DS1.h>

bool sensors_init() {
  if (!IMU.begin()) {
    Serial.println("ERROR: IMU initialization failed!");
    return false;
  }
  Serial.println("IMU initialized (LSM9DS1)");
  Serial.print("  Accelerometer sample rate: ");
  Serial.print(IMU.accelerationSampleRate());
  Serial.println(" Hz");
  Serial.print("  Gyroscope sample rate: ");
  Serial.print(IMU.gyroscopeSampleRate());
  Serial.println(" Hz");
  return true;
}

bool sensors_acquire_window(IMUWindow &window) {
  window.count = 0;
  
  unsigned long start_time = micros();
  unsigned long next_sample = start_time;
  
  Serial.println("  Acquiring IMU data...");
  
  while (window.count < NUM_SAMPLES) {
    // Wait for next sample time (non-blocking timing)
    while (micros() < next_sample) {
      // Spin wait — no delay()
    }
    next_sample += SAMPLE_INTERVAL_US;
    
    // Read accelerometer
    if (IMU.accelerationAvailable() && IMU.gyroscopeAvailable()) {
      float ax, ay, az, gx, gy, gz;
      IMU.readAcceleration(ax, ay, az);
      IMU.readGyroscope(gx, gy, gz);
      
      window.ax[window.count] = ax;
      window.ay[window.count] = ay;
      window.az[window.count] = az;
      window.gx[window.count] = gx;
      window.gy[window.count] = gy;
      window.gz[window.count] = gz;
      window.count++;
    }
  }
  
  unsigned long elapsed = micros() - start_time;
  Serial.print("  Window acquired: ");
  Serial.print(window.count);
  Serial.print(" samples in ");
  Serial.print(elapsed / 1000);
  Serial.println(" ms");
  
  return (window.count == NUM_SAMPLES);
}
