#ifndef FEATURES_H
#define FEATURES_H

#include "sensors.h"

// Extracted sensor features
struct SensorFeatures {
  float acc_mean;   // mean acceleration magnitude
  float acc_std;    // std dev of acceleration magnitude
  float acc_peak;   // max acceleration magnitude
  float gyro_mean;  // mean gyroscope magnitude
  float tilt;       // tilt angle in degrees
};

// Extract features from an IMU window
void features_extract(const IMUWindow &window, SensorFeatures &features);

#endif // FEATURES_H
