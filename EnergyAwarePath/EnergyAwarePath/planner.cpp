#include "planner.h"
#include "inference.h"
#include <math.h>

void planner_normalize(float features[NUM_FEATURES]) {
  for (int i = 0; i < NUM_FEATURES; i++) {
    float range = FEATURE_MAX[i] - FEATURE_MIN[i];
    if (range < 0.0001f) range = 1.0f;
    features[i] = (features[i] - FEATURE_MIN[i]) / range;
    // Clamp to [0, 1]
    if (features[i] < 0.0f) features[i] = 0.0f;
    if (features[i] > 1.0f) features[i] = 1.0f;
  }
}

void planner_build_features(float budget, const Branch &branch,
                            const SensorFeatures &sensor, float out[NUM_FEATURES]) {
  out[0] = budget;
  out[1] = branch.length;
  out[2] = (float)branch.turns / 5.0f;  // normalize turns to 0-1
  out[3] = branch.difficulty;
  out[4] = branch.slope;
  out[5] = sensor.acc_mean;
  out[6] = sensor.acc_std;
  out[7] = sensor.acc_peak;
  out[8] = sensor.gyro_mean;
  out[9] = sensor.tilt;
}

float planner_oracle_cost(float budget, const Branch &branch,
                          const SensorFeatures &sensor) {
  // Same formula used to generate training labels.
  // Sensor features are normalized using FEATURE_MIN/MAX to stay in [0,1].
  
  // Index 6 = acc_std, Index 8 = gyro_mean (in FEATURE_MIN/MAX arrays)
  float acc_std_range = FEATURE_MAX[6] - FEATURE_MIN[6];
  float gyro_range    = FEATURE_MAX[8] - FEATURE_MIN[8];
  if (acc_std_range < 0.0001f) acc_std_range = 1.0f;
  if (gyro_range < 0.0001f) gyro_range = 1.0f;
  
  float acc_std_n = (sensor.acc_std - FEATURE_MIN[6]) / acc_std_range;
  float gyro_n    = (sensor.gyro_mean - FEATURE_MIN[8]) / gyro_range;
  if (acc_std_n < 0.0f) acc_std_n = 0.0f;
  if (acc_std_n > 1.0f) acc_std_n = 1.0f;
  if (gyro_n < 0.0f) gyro_n = 0.0f;
  if (gyro_n > 1.0f) gyro_n = 1.0f;
  
  float motion = (acc_std_n + gyro_n) * 0.5f;
  
  float E = 0.18f * branch.length +
            0.12f * ((float)branch.turns / 5.0f) +
            0.12f * branch.difficulty +
            0.08f * fabsf(branch.slope) +
            0.05f * acc_std_n +
            0.05f * gyro_n +
            0.05f * (1.0f - budget) +
            0.30f * branch.length * motion;
  
  if (E < 0.05f) E = 0.05f;
  if (E > 0.95f) E = 0.95f;
  return E;
}

Decision planner_decide(int checkpoint_idx, float budget,
                        const SensorFeatures &sensor, Policy policy) {
  Decision dec;
  dec.safety_override = false;
  dec.inference_time_us = 0;
  
  const Branch* branches = checkpoints[checkpoint_idx];
  
  // --- ML Policy ---
  if (policy == POLICY_ML) {
    for (int b = 0; b < NUM_BRANCHES; b++) {
      float feat[NUM_FEATURES];
      planner_build_features(budget, branches[b], sensor, feat);
      planner_normalize(feat);
      
      unsigned long lat;
      dec.predicted_costs[b] = inference_predict(feat, lat);
      dec.inference_time_us += lat;
    }
    
    // Find minimum cost branch
    int best = 0;
    for (int b = 1; b < NUM_BRANCHES; b++) {
      if (dec.predicted_costs[b] < dec.predicted_costs[best]) {
        best = b;
      }
    }
    
    // Safety margin check
    if (budget < SAFETY_MARGIN) {
      // If best branch costs more than 50% of remaining budget, try alternatives
      if (dec.predicted_costs[best] > 0.5f * budget) {
        // Find the cheapest that doesn't exceed safety threshold
        int safest = best;
        float min_safe_cost = dec.predicted_costs[best];
        for (int b = 0; b < NUM_BRANCHES; b++) {
          if (b != best && dec.predicted_costs[b] < min_safe_cost) {
            safest = b;
            min_safe_cost = dec.predicted_costs[b];
          }
        }
        if (safest != best) {
          best = safest;
          dec.safety_override = true;
        }
      }
    }
    
    dec.selected_branch = best;
    dec.selected_cost = dec.predicted_costs[best];
  }
  // --- Always-A Policy ---
  else if (policy == POLICY_ALWAYS_A) {
    for (int b = 0; b < NUM_BRANCHES; b++) {
      dec.predicted_costs[b] = planner_oracle_cost(budget, branches[b], sensor);
    }
    dec.selected_branch = 0;
    dec.selected_cost = dec.predicted_costs[0];
  }
  // --- Shortest Policy ---
  else if (policy == POLICY_SHORTEST) {
    int shortest = 0;
    for (int b = 0; b < NUM_BRANCHES; b++) {
      dec.predicted_costs[b] = planner_oracle_cost(budget, branches[b], sensor);
      if (branches[b].length < branches[shortest].length) {
        shortest = b;
      }
    }
    dec.selected_branch = shortest;
    dec.selected_cost = dec.predicted_costs[shortest];
  }
  // --- Random Policy ---
  else if (policy == POLICY_RANDOM) {
    for (int b = 0; b < NUM_BRANCHES; b++) {
      dec.predicted_costs[b] = planner_oracle_cost(budget, branches[b], sensor);
    }
    dec.selected_branch = random(0, NUM_BRANCHES);
    dec.selected_cost = dec.predicted_costs[dec.selected_branch];
  }
  // --- Oracle Policy (uses formula directly) ---
  else if (policy == POLICY_ORACLE) {
    for (int b = 0; b < NUM_BRANCHES; b++) {
      dec.predicted_costs[b] = planner_oracle_cost(budget, branches[b], sensor);
    }
    int best = 0;
    for (int b = 1; b < NUM_BRANCHES; b++) {
      if (dec.predicted_costs[b] < dec.predicted_costs[best]) {
        best = b;
      }
    }
    dec.selected_branch = best;
    dec.selected_cost = dec.predicted_costs[best];
  }
  
  return dec;
}
