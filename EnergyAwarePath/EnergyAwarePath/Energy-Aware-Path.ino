/*
 * EnergyAwarePath
 * Authors: Pedrini, Bellini
 */

#include "config.h"
#include "sensors.h"
#include "features.h"
#include "inference.h"
#include "planner.h"

static float energy_budget = INITIAL_BUDGET;
static int   current_checkpoint = 0;
static Policy current_policy = POLICY_ML;
static bool  mission_complete = false;

static IMUWindow imu_window;
static SensorFeatures sensor_features;

static float total_cost_ml = 0.0f;
static float total_cost_oracle = 0.0f;
static int   checkpoints_completed = 0;

// =============================================================================
// SETUP
// =============================================================================

void setup() {
  Serial.begin(115200);
  while (!Serial) { delay(10); }
  
  Serial.println("EnergyAwarePath - Pedrini, Bellini");
  Serial.println();
  
  if (!sensors_init()) {
    Serial.println("FATAL: Sensor init failed");
    while (1) { delay(1000); }
  }
  
  if (!inference_init()) {
    Serial.println("FATAL: Inference init failed");
    while (1) { delay(1000); }
  }
  
  if (!inference_init_exit1()) {
    Serial.println("WARNING: Exit-1 model init failed");
  }
  
  Serial.println();
  Serial.println("System ready");
  Serial.print("Budget: ");
  Serial.println(INITIAL_BUDGET);
  Serial.print("Checkpoints: ");
  Serial.println(NUM_CHECKPOINTS);
  Serial.println();
  Serial.println("Commands: n=next, r=reset, s=summary, 1-6=policy");
  Serial.println();
  
  randomSeed(analogRead(0));
}

// =============================================================================
// MAIN LOOP
// =============================================================================

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();
    
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
        Serial.println("Policy: ML");
        break;
      case '2':
        current_policy = POLICY_ALWAYS_A;
        Serial.println("Policy: Always-A");
        break;
      case '3':
        current_policy = POLICY_SHORTEST;
        Serial.println("Policy: Shortest");
        break;
      case '4':
        current_policy = POLICY_RANDOM;
        Serial.println("Policy: Random");
        break;
      case '5':
        current_policy = POLICY_ORACLE;
        Serial.println("Policy: Oracle");
        break;
      case '6':
        current_policy = POLICY_ADAPTIVE;
        Serial.println("Policy: Adaptive");
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
  Serial.print("Budget: ");
  Serial.println(energy_budget, 3);
  
  if (!sensors_acquire_window(imu_window)) {
    Serial.println("ERROR: IMU acquisition failed");
    return;
  }
  
  features_extract(imu_window, sensor_features);
  
  Serial.print("Context: acc=");
  Serial.print(sensor_features.acc_mean, 2);
  Serial.print("/");
  Serial.print(sensor_features.acc_std, 2);
  Serial.print("/");
  Serial.print(sensor_features.acc_peak, 2);
  Serial.print(" gyro=");
  Serial.print(sensor_features.gyro_mean, 2);
  Serial.print(" tilt=");
  Serial.print(sensor_features.tilt, 1);
  Serial.println("°");
  Serial.println();
  
  Decision dec = planner_decide(current_checkpoint, energy_budget,
                                sensor_features, current_policy);
  
  Decision oracle_dec = planner_decide(current_checkpoint, energy_budget,
                                       sensor_features, POLICY_ORACLE);
  
  const Branch* branches = checkpoints[current_checkpoint];
  for (int b = 0; b < NUM_BRANCHES; b++) {
    Serial.print("Branch ");
    Serial.print(branches[b].label);
    Serial.print(": ");
    Serial.print(branches[b].length, 1);
    Serial.print("/");
    Serial.print(branches[b].turns);
    Serial.print("/");
    Serial.print(branches[b].difficulty, 1);
    Serial.print("/");
    Serial.print(branches[b].slope, 1);
    Serial.print(" -> ");
    Serial.println(dec.predicted_costs[b], 3);
  }
  Serial.println();
  
  Serial.print("SELECTED: ");
  Serial.print(branches[dec.selected_branch].label);
  Serial.print(" (");
  Serial.print(dec.selected_cost, 3);
  Serial.print(")");
  if (dec.safety_override) {
    Serial.print(" [SAFETY]");
  }
  Serial.println();
  
  energy_budget -= dec.selected_cost;
  if (energy_budget < 0.0f) energy_budget = 0.0f;
  
  Serial.print("Remaining: ");
  Serial.println(energy_budget, 3);
  
  if (current_policy == POLICY_ML || current_policy == POLICY_ADAPTIVE) {
    Serial.print("Inference: ");
    Serial.print(dec.inference_time_us / 1000.0f, 2);
    Serial.println(" ms");
  }
  
  if (current_policy == POLICY_ADAPTIVE) {
    ExitLevel lvl = inference_last_exit_level();
    Serial.print("Exit: ");
    if (lvl == EXIT_FULL)   Serial.println("FULL");
    if (lvl == EXIT_MIDDLE) Serial.println("EXIT1");
    if (lvl == EXIT_LINEAR) Serial.println("LINEAR");
  }
  
  total_cost_ml += dec.selected_cost;
  total_cost_oracle += oracle_dec.selected_cost;
  checkpoints_completed++;
  
  Serial.print("Oracle: ");
  Serial.print(branches[oracle_dec.selected_branch].label);
  Serial.print(" (");
  Serial.print(oracle_dec.selected_cost, 3);
  Serial.print(")");
  if (dec.selected_branch == oracle_dec.selected_branch) {
    Serial.print(" [AGREE]");
  }
  Serial.println();
  
  current_checkpoint++;
  
  if (current_checkpoint >= NUM_CHECKPOINTS || energy_budget <= 0.001f) {
    mission_complete = true;
    Serial.println();
    Serial.println("========== MISSION COMPLETE ==========");
    print_summary();
  } else {
    Serial.println();
    Serial.println("Press 'n' for next...");
  }
}

// =============================================================================
// SUMMARY & COMPARISON
// =============================================================================

void print_summary() {
  Serial.println();
  Serial.println("---------- SUMMARY ----------");
  Serial.print("Checkpoints: ");
  Serial.print(checkpoints_completed);
  Serial.print(" / ");
  Serial.println(NUM_CHECKPOINTS);
  Serial.print("Final budget: ");
  Serial.println(energy_budget, 3);
  Serial.print("Energy consumed: ");
  Serial.println(1.0f - energy_budget, 3);
  Serial.println();
  
  Serial.print("Policy cost: ");
  Serial.println(total_cost_ml, 3);
  Serial.print("Oracle cost: ");
  Serial.println(total_cost_oracle, 3);
  
  if (total_cost_oracle > 0.001f) {
    float eff = total_cost_oracle / total_cost_ml * 100.0f;
    Serial.print("Efficiency: ");
    Serial.print(eff, 1);
    Serial.println("%");
  }
  
  Serial.println("-----------------------------");
  Serial.println();
}

void reset_mission() {
  energy_budget = INITIAL_BUDGET;
  current_checkpoint = 0;
  mission_complete = false;
  total_cost_ml = 0.0f;
  total_cost_oracle = 0.0f;
  checkpoints_completed = 0;
  
  Serial.println();
  Serial.println("=== RESET ===");
  Serial.print("Budget: ");
  Serial.println(INITIAL_BUDGET);
  Serial.println("Press 'n' to start");
  Serial.println();
}
