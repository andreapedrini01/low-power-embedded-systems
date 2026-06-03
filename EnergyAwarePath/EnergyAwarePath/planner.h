#ifndef PLANNER_H
#define PLANNER_H

#include "config.h"
#include "features.h"

struct Decision {
  int   selected_branch;
  float predicted_costs[NUM_BRANCHES];
  float selected_cost;
  unsigned long inference_time_us;
  bool  safety_override;
};

void planner_normalize(float features[NUM_FEATURES]);

void planner_build_features(float budget, const Branch &branch,
                            const SensorFeatures &sensor, float out[NUM_FEATURES]);

Decision planner_decide(int checkpoint_idx, float budget,
                        const SensorFeatures &sensor, Policy policy);

float planner_oracle_cost(float budget, const Branch &branch,
                          const SensorFeatures &sensor);

#endif
