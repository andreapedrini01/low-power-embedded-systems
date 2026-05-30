#ifndef PLANNER_H
#define PLANNER_H

#include "config.h"
#include "features.h"

// Decision result
struct Decision {
  int   selected_branch;    // 0, 1, or 2
  float predicted_costs[NUM_BRANCHES];
  float selected_cost;
  unsigned long inference_time_us;  // total for all 3 inferences
  bool  safety_override;    // true if safety margin changed the pick
};

// Normalize a raw feature vector to [0, 1]
void planner_normalize(float features[NUM_FEATURES]);

// Build feature vector for a branch
void planner_build_features(float budget, const Branch &branch,
                            const SensorFeatures &sensor, float out[NUM_FEATURES]);

// Evaluate all branches and select the best one
Decision planner_decide(int checkpoint_idx, float budget,
                        const SensorFeatures &sensor, Policy policy);

// Oracle cost formula (same as training target)
float planner_oracle_cost(float budget, const Branch &branch,
                          const SensorFeatures &sensor);

#endif // PLANNER_H
