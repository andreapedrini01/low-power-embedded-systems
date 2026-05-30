#include "features.h"
#include "config.h"
#include <math.h>

void features_extract(const IMUWindow &window, SensorFeatures &features) {
  float acc_mag[NUM_SAMPLES];
  float gyro_mag[NUM_SAMPLES];
  
  float acc_sum = 0.0f;
  float gyro_sum = 0.0f;
  float acc_max = 0.0f;
  float ax_sum = 0.0f;
  float az_sum = 0.0f;
  
  // First pass: compute magnitudes and sums
  for (int i = 0; i < window.count; i++) {
    float am = sqrtf(window.ax[i] * window.ax[i] +
                     window.ay[i] * window.ay[i] +
                     window.az[i] * window.az[i]);
    float gm = sqrtf(window.gx[i] * window.gx[i] +
                     window.gy[i] * window.gy[i] +
                     window.gz[i] * window.gz[i]);
    
    acc_mag[i] = am;
    gyro_mag[i] = gm;
    
    acc_sum += am;
    gyro_sum += gm;
    
    if (am > acc_max) acc_max = am;
    
    ax_sum += window.ax[i];
    az_sum += window.az[i];
  }
  
  int n = window.count;
  float acc_mean = acc_sum / n;
  float gyro_mean = gyro_sum / n;
  
  // Second pass: compute standard deviation
  float acc_var_sum = 0.0f;
  for (int i = 0; i < n; i++) {
    float diff = acc_mag[i] - acc_mean;
    acc_var_sum += diff * diff;
  }
  float acc_std = sqrtf(acc_var_sum / n);
  
  // Tilt from gravity vector
  float mean_ax = ax_sum / n;
  float mean_az = az_sum / n;
  float tilt_rad = atan2f(fabsf(mean_ax), fabsf(mean_az));
  float tilt_deg = tilt_rad * 180.0f / 3.14159265f;
  
  // Store results
  features.acc_mean  = acc_mean;
  features.acc_std   = acc_std;
  features.acc_peak  = acc_max;
  features.gyro_mean = gyro_mean;
  features.tilt      = tilt_deg;
}
