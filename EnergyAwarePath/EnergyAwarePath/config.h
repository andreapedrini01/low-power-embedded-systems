#ifndef CONFIG_H
#define CONFIG_H

// =============================================================================
// EnergyAwarePath - Configuration
// =============================================================================

// --- System parameters ---
#define INITIAL_BUDGET        1.0f
#define SAFETY_MARGIN         0.2f
#define NUM_CHECKPOINTS       4
#define NUM_BRANCHES          3

// --- IMU sampling ---
#define SAMPLE_RATE_HZ        100
#define SAMPLE_WINDOW_MS      1190    // ~1.2 seconds
#define NUM_SAMPLES           119     // SAMPLE_RATE_HZ * SAMPLE_WINDOW_MS / 1000
#define SAMPLE_INTERVAL_US    10000   // 1000000 / SAMPLE_RATE_HZ

// --- Model parameters ---
#define NUM_FEATURES          10
#define TENSOR_ARENA_SIZE     8192    // 8 KB, increase if needed

// --- Branch struct ---
struct Branch {
  float length;      // 0-1, normalized
  int   turns;       // 0-5
  float difficulty;  // 0-1
  float slope;       // -1 to 1
  char  label;       // 'A', 'B', 'C'
};

// --- Checkpoint configurations (4 checkpoints × 3 branches) ---
const Branch checkpoints[NUM_CHECKPOINTS][NUM_BRANCHES] = {
  // Checkpoint 0
  {
    {0.3f, 5, 0.8f,  0.4f, 'A'},   // short, hard, steep up
    {0.5f, 3, 0.5f,  0.1f, 'B'},   // medium
    {0.7f, 1, 0.2f, -0.1f, 'C'}    // long, easy, slight downhill
  },
  // Checkpoint 1
  {
    {0.2f, 4, 0.9f,  0.6f, 'A'},   // very short, very hard, very steep
    {0.6f, 2, 0.4f,  0.0f, 'B'},   // medium-long, moderate, flat
    {0.8f, 1, 0.1f, -0.2f, 'C'}    // long, very easy, downhill
  },
  // Checkpoint 2
  {
    {0.4f, 3, 0.7f,  0.3f, 'A'},   // medium-short, hard
    {0.5f, 2, 0.5f, -0.1f, 'B'},   // medium, slight downhill
    {0.9f, 0, 0.1f, -0.3f, 'C'}    // very long, trivial, downhill
  },
  // Checkpoint 3
  {
    {0.3f, 4, 0.6f,  0.5f, 'A'},   // short, moderate-hard, steep
    {0.4f, 3, 0.4f,  0.2f, 'B'},   // medium-short, moderate
    {0.6f, 1, 0.3f,  0.0f, 'C'}    // medium, easy, flat
  }
};

// --- Feature normalization constants ---
// From training (Colab notebook output)
// Format: [budget, length, turns_norm, difficulty, slope,
//           acc_mean, acc_std, acc_peak, gyro_mean, tilt]
const float FEATURE_MIN[NUM_FEATURES] = {
  0.2000f, 0.2000f, 0.0000f, 0.1000f, -0.3000f,
  0.9650f, 0.0173f, 1.0376f, 0.7452f, 0.0159f
};

const float FEATURE_MAX[NUM_FEATURES] = {
  1.0000f, 0.9000f, 1.0000f, 0.9000f, 0.6000f,
  2.0052f, 0.7488f, 5.0186f, 125.7536f, 44.6437f
};

// --- Quantization parameters (from TFLite conversion) ---
// Read from model tensors at runtime in inference_init().
// Declared extern here, defined in inference.cpp.
extern float INPUT_SCALE;
extern int   INPUT_ZERO;
extern float OUTPUT_SCALE;
extern int   OUTPUT_ZERO;

// --- Adaptive inference thresholds ---
#define BUDGET_HIGH_THRESHOLD   0.6f   // above: use full model
#define BUDGET_LOW_THRESHOLD    0.3f   // below: use oracle formula (no TFLite)
// between LOW and HIGH: use exit 1 model

// --- Tensor arena for exit 1 model (smaller) ---
#define TENSOR_ARENA_EXIT1_SIZE 4096   // 4 KB, sufficient for the reduced model

// --- Decision policies ---
enum Policy {
  POLICY_ML = 0,
  POLICY_ALWAYS_A,
  POLICY_SHORTEST,
  POLICY_RANDOM,
  POLICY_ORACLE,
  POLICY_ADAPTIVE    // new: selects exit based on current budget
};

#endif // CONFIG_H
