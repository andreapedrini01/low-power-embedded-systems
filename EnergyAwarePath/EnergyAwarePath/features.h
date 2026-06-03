#ifndef FEATURES_H
#define FEATURES_H

#include "sensors.h"

struct SensorFeatures {
  float acc_mean;
  float acc_std;
  float acc_peak;
  float gyro_mean;
  float tilt;
};

void features_extract(const IMUWindow &window, SensorFeatures &features);

#endif
