/*
 * EnergyAwarePath - TinyML Energy-Aware Path Selection
 * =====================================================
 * Low-Power Embedded Systems Course
 * Authors: Pedrini, Bellini
 * 
 * Target: Arduino Nano 33 BLE Sense (nRF52840)
 * 
 * The board simulates a navigation scenario where, at each checkpoint,
 * it evaluates 3 candidate path branches and picks the one with the
 * lowest predicted energy cost using a quantized neural network.
 * 
 * Serial commands:
 *   'n' or ENTER  - trigger next checkpoint
 *   'r'           - reset mission
 *   '1'-'5'       - select policy (1=ML, 2=AlwaysA, 3=Shortest, 4=Random, 5=Oracle)
 *   's'           - show summary/comparison of all policies
 */

#include "config.h"
#include "sensors.h"
#include "features.h"
#include "inference.h"
#include "planner.h"

// --- State ---
static float energy_budget = INITIAL_BUDGET;
static int   current_checkpoint = 0;
static Policy current_policy = POLICY_ML;
static bool  mission_complete = false;

// IMU data buffer (static allocation)
static IMUWindow imu_window;
static SensorFeatures sensor_features;

// Logging for comparison
static float total_cost_ml = 0.0f;
static float total_cost_oracle = 0.0f;
static int   checkpoints_completed = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  Serial.println();
  Serial.println("============================================");
  Serial.println("  EnergyAwarePath - TinyML Path Selection");
  Serial.println("  Arduino Nano 33 BLE Sense");
  Serial.println("============================================");
  Serial.println();
  
  // Initialize IMU
  if (!sensors_init()) {
    Serial.println("FATAL: Cannot initialize sensors. Halting.");
    while (1) { delay(1000); }
  }
  
  // Initialize TFLite
  if (!inference_init()) {
    Serial.println("FATAL: Cannot initialize inference engine. Halting.");
    while (1) { delay(1000); }
  }
  
  Serial.println();
  Serial.println("--- System Ready ---");
  Serial.print("Initial energy budget: ");
  Serial.println(INITIAL_BUDGET);
  Serial.print("Checkpoints: ");
  Serial.println(NUM_CHECKPOINTS);
  Serial.print("Policy: ML (press 1-5 to change)");
  Serial.println();
  Serial.println();
  Serial.println("Press 'n' or ENTER to trigger next checkpoint.");
  Serial.println("Press 'r' to reset. Press 's' for summary.");
  Serial.println();
  
  randomSeed(analogRead(0));
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    
    // Consume extra newline/carriage return
    while (Serial.available() && (Serial.peek() == '\n' || Serial.peek() == '\r')) {
      Serial.read();
    }
    
    switch (cmd) {
      case 'n':
      case '\n':
      case '\r':
        if (!mission_complete) {
          run_checkpoint();
        } else {
          Serial.println("Mission complete! Press 'r' to reset.");
        }
        break;
        
      case 'r':
        reset_mission();
        break;
        
      case '1':
        current_policy = POLICY_ML;
        Serial.println("Policy: ML (neural network)");
        break;
      case '2':
        current_policy = POLICY_ALWAYS_A;
        Serial.println("Policy: Always-A");
        break;
      case '3':
        current_policy = POLICY_SHORTEST;
        Serial.println("Policy: Shortest path");
        break;
      case '4':
        current_policy = POLICY_RANDOM;
        Serial.println("Policy: Random");
        break;
      case '5':
        current_policy = POLICY_ORACLE;
        Serial.println("Policy: Oracle (formula)");
        break;
        
      case 's':
        print_summary();
        break;
        
      default:
        break;
    }
  }
}

// =============================================================================
// CHECKPOINT EXECUTION
// =============================================================================

void run_checkpoint() {
  Serial.println();
  Serial.print("=== CHECKPOINT ");
  Serial.print(current_checkpoint + 1);
  Serial.println(" ===");
  Serial.print("Energy budget: ");
  Serial.println(energy_budget, 3);
  
  // 1. Acquire IMU window
  if (!sensors_acquire_window(imu_window)) {
    Serial.println("ERROR: IMU acquisition failed!");
    return;
  }
  
  // 2. Extract features
  features_extract(imu_window, sensor_features);
  
  Serial.print("Sensor context: acc_mean=");
  Serial.print(sensor_features.acc_mean, 2);
  Serial.print(" acc_std=");
  Serial.print(sensor_features.acc_std, 2);
  Serial.print(" acc_peak=");
  Serial.print(sensor_features.acc_peak, 2);
  Serial.print(" gyro=");
  Serial.print(sensor_features.gyro_mean, 2);
  Serial.print(" tilt=");
  Serial.print(sensor_features.tilt, 1);
  Serial.println("°");
  Serial.println();
  
  // 3. Run planner
  Decision dec = planner_decide(current_checkpoint, energy_budget,
                                sensor_features, current_policy);
  
  // 4. Also compute oracle for comparison
  Decision oracle_dec = planner_decide(current_checkpoint, energy_budget,
                                       sensor_features, POLICY_ORACLE);
  
  // 5. Print branch evaluations
  const Branch* branches = checkpoints[current_checkpoint];
  for (int b = 0; b < NUM_BRANCHES; b++) {
    Serial.print("Branch ");
    Serial.print(branches[b].label);
    Serial.print(": len=");
    Serial.print(branches[b].length, 1);
    Serial.print(" turns=");
    Serial.print(branches[b].turns);
    Serial.print(" diff=");
    Serial.print(branches[b].difficulty, 1);
    Serial.print(" slope=");
    Serial.print(branches[b].slope, 1);
    Serial.print(" -> predicted cost: ");
    Serial.println(dec.predicted_costs[b], 3);
  }
  Serial.println();
  
  // 6. Print decision
  Serial.print("SELECTED: Branch ");
  Serial.print(branches[dec.selected_branch].label);
  Serial.print(" (cost=");
  Serial.print(dec.selected_cost, 3);
  Serial.print(")");
  if (dec.safety_override) {
    Serial.print(" [SAFETY OVERRIDE]");
  }
  Serial.println();
  
  // 7. Deduct cost
  energy_budget -= dec.selected_cost;
  if (energy_budget < 0.0f) energy_budget = 0.0f;
  
  Serial.print("Remaining budget: ");
  Serial.println(energy_budget, 3);
  
  if (current_policy == POLICY_ML) {
    Serial.print("Inference time: ");
    Serial.print(dec.inference_time_us / 1000.0f, 2);
    Serial.println(" ms");
  }
  
  // 8. Track comparison metrics
  total_cost_ml += dec.selected_cost;
  total_cost_oracle += oracle_dec.selected_cost;
  checkpoints_completed++;
  
  Serial.print("Oracle would pick: Branch ");
  Serial.print(branches[oracle_dec.selected_branch].label);
  Serial.print(" (cost=");
  Serial.print(oracle_dec.selected_cost, 3);
  Serial.print(")");
  if (dec.selected_branch == oracle_dec.selected_branch) {
    Serial.print(" [AGREE]");
  }
  Serial.println();
  
  // 9. Advance checkpoint
  current_checkpoint++;
  
  if (current_checkpoint >= NUM_CHECKPOINTS || energy_budget <= 0.001f) {
    mission_complete = true;
    Serial.println();
    Serial.println("========== MISSION COMPLETE ==========");
    print_summary();
  } else {
    Serial.println();
    Serial.println("Press 'n' for next checkpoint...");
  }
}

// =============================================================================
// SUMMARY & COMPARISON
// =============================================================================

void print_summary() {
  Serial.println();
  Serial.println("---------- MISSION SUMMARY ----------");
  Serial.print("Checkpoints completed: ");
  Serial.print(checkpoints_completed);
  Serial.print(" / ");
  Serial.println(NUM_CHECKPOINTS);
  Serial.print("Final budget: ");
  Serial.println(energy_budget, 3);
  Serial.print("Total energy consumed: ");
  Serial.println(1.0f - energy_budget, 3);
  Serial.println();
  
  Serial.println("Policy comparison (this run):");
  Serial.print("  Current policy total cost: ");
  Serial.println(total_cost_ml, 3);
  Serial.print("  Oracle total cost:         ");
  Serial.println(total_cost_oracle, 3);
  
  if (total_cost_oracle > 0.001f) {
    float efficiency = total_cost_oracle / total_cost_ml * 100.0f;
    Serial.print("  Efficiency vs oracle:      ");
    Serial.print(efficiency, 1);
    Serial.println("%");
  }
  
  Serial.println("-------------------------------------");
  Serial.println();
}

// =============================================================================
// RESET
// =============================================================================

void reset_mission() {
  energy_budget = INITIAL_BUDGET;
  current_checkpoint = 0;
  mission_complete = false;
  total_cost_ml = 0.0f;
  total_cost_oracle = 0.0f;
  checkpoints_completed = 0;
  
  Serial.println();
  Serial.println("=== MISSION RESET ===");
  Serial.print("Energy budget: ");
  Serial.println(INITIAL_BUDGET);
  Serial.println("Press 'n' to start...");
  Serial.println();
}
