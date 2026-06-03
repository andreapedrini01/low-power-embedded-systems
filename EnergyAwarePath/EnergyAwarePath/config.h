#ifndef CONFIG_H
#define CONFIG_H

#define INITIAL_BUDGET        1.0f
#define SAFETY_MARGIN         0.2f
#define NUM_CHECKPOINTS       4
#define NUM_BRANCHES          3

#define SAMPLE_RATE_HZ        100
#define SAMPLE_WINDOW_MS      1190
#define NUM_SAMPLES           119
#define SAMPLE_INTERVAL_US    10000

#define NUM_FEATURES          10
#define TENSOR_ARENA_SIZE     8192

struct Branch {
  float length;
  int   turns;
  float difficulty;
  float slope;
  char  label;
};

const Branch checkpoints[NUM_CHECKPOINTS][NUM_BRANCHES] = {
  {
    {0.3f, 5, 0.8f,  0.4f, 'A'},
    {0.5f, 3, 0.5f,  0.1f, 'B'},
    {0.7f, 1, 0.2f, -0.1f, 'C'}
  },
  {
    {0.2f, 4, 0.9f,  0.6f, 'A'},
    {0.6f, 2, 0.4f,  0.0f, 'B'},
    {0.8f, 1, 0.1f, -0.2f, 'C'}
  },
  {
    {0.4f, 3, 0.7f,  0.3f, 'A'},
    {0.5f, 2, 0.5f, -0.1f, 'B'},
    {0.9f, 0, 0.1f, -0.3f, 'C'}
  },
  {
    {0.3f, 4, 0.6f,  0.5f, 'A'},
    {0.4f, 3, 0.4f,  0.2f, 'B'},
    {0.6f, 1, 0.3f,  0.0f, 'C'}
  }
};

const float FEATURE_MIN[NUM_FEATURES] = {
  0.2000f, 0.2000f, 0.0000f, 0.1000f, -0.3000f,
  0.9650f, 0.0173f, 1.0376f, 0.7452f, 0.0159f
};

const float FEATURE_MAX[NUM_FEATURES] = {
  1.0000f, 0.9000f, 1.0000f, 0.9000f, 0.6000f,
  2.0052f, 0.7488f, 5.0186f, 125.7536f, 44.6437f
};

extern float INPUT_SCALE;
extern int   INPUT_ZERO;
extern float OUTPUT_SCALE;
extern int   OUTPUT_ZERO;

#define BUDGET_HIGH_THRESHOLD   0.6f
#define BUDGET_LOW_THRESHOLD    0.3f
#define TENSOR_ARENA_EXIT1_SIZE 4096
enum Policy {
  POLICY_ML = 0,
  POLICY_ALWAYS_A,
  POLICY_SHORTEST,
  POLICY_RANDOM,
  POLICY_ORACLE,
  POLICY_ADAPTIVE    // new: selects exit based on current budget
};

#endif // CONFIG_H
