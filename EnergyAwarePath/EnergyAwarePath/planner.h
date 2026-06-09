#ifndef PLANNER_H
#define PLANNER_H

#include "config.h"
#include "features.h"

struct Decision {
  int   selected_branch;
  float predicted_costs[NUM_BRANCHES];
  float selected_cost;
  unsigned long inference_time_us;
  unsigned long active_time_us;     // total active time of decide()
  float         energy_uj;          // estimated energy of decide()
  bool  safety_override;
};

void planner_normalize(float features[NUM_FEATURES]);

void planner_build_features(float budget, const Branch &branch,
                            const SensorFeatures &sensor, float out[NUM_FEATURES]);

Decision planner_decide(int checkpoint_idx, float budget,
                        const SensorFeatures &sensor, Policy policy);

float planner_oracle_cost(float budget, const Branch &branch,
                          const SensorFeatures &sensor);

unsigned long planner_post_checkpoint_sleep_ms(float budget, Policy policy);

#endif
