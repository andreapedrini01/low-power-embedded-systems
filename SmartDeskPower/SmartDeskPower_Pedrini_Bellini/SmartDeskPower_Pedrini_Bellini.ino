/*
 * SmartDeskPower - Surface Activity Sensing
 * Authors: Pedrini, Bellini
 *
 * Senses structure-borne desk vibration to infer occupancy and manage power.
 * Modes:
 *   'r' - record one window (CSV dump over serial)
 *   'c' - classify single window and print class scores + latency
 *   'm' - toggle continuous monitor mode (state machine + LED + power policy)
 *   's' - print energy and timing summary
 *   'q' - quit monitor mode
 */

#include "config.h"
#include "sensors.h"
#include "features.h"
#include "inference.h"
#include "power.h"
#include "model.h"

static AccelWindow window;
static float       feat[N_FEATURES];
static float       scores[N_CLASSES];

// State machine
static bool          monitoring           = false;
static bool          load_on              = false;
static int           consecutive_activity = 0;
static unsigned long last_activity_ms     = 0;

void record_window();
void classify_single();
void step_monitor();
void set_load_power(bool on);

// SETUP

void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) { delay(10); }

  // RGB LED + onboard LED pins
  pinMode(LEDR, OUTPUT);
  pinMode(LEDG, OUTPUT);
  pinMode(LEDB, OUTPUT);
  pinMode(LED_BUILTIN, OUTPUT);
  digitalWrite(LEDR, HIGH);
  digitalWrite(LEDG, HIGH);
  digitalWrite(LEDB, HIGH);
  digitalWrite(LED_BUILTIN, LOW);

  power_init();
  power_mark_active_begin();

  if (!sensors_init()) {
    Serial.println("FATAL: IMU init failed");
    while (1) { delay(1000); }
  }

  inference_init();

  // Initial state: load ON, activity timer started
  set_load_power(true);
  last_activity_ms = millis();

  Serial.println("SmartDeskPower - Pedrini, Bellini");
  Serial.print("Window: ");
  Serial.print(WINDOW_SAMPLES);
  Serial.println(" samples (~1 s)");
  Serial.println("Commands: r=record, c=classify, m=toggle monitor, s=summary, q=quit");
}

// MAIN LOOP

void loop() {
  if (Serial.available()) {
    char cmd = Serial.read();

    while (Serial.available() && (Serial.peek() == '\n' || Serial.peek() == '\r')) {
      Serial.read();
    }

    switch (cmd) {
      case 'r':
        monitoring = false;
        record_window();
        break;
      case 'c':
        monitoring = false;
        classify_single();
        break;
      case 'm':
        monitoring = !monitoring;
        Serial.print("Monitor mode: ");
        Serial.println(monitoring ? "STARTED" : "STOPPED");
        break;
      case 's':
        power_print_summary();
        break;
      case 'q':
        monitoring = false;
        Serial.println("Monitor mode: STOPPED");
        break;
      default:
        break;
    }
  }

  if (monitoring) {
    step_monitor();
  }
}

// POWER / LED CONTROL

void set_load_power(bool on) {
  load_on = on;
  if (on) {
    power_load_on();
    digitalWrite(LEDG, LOW);   // Green ON (active LOW on Nano 33 BLE)
    digitalWrite(LEDR, HIGH);  // Red OFF
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.println("POWER: ON");
  } else {
    power_load_off();
    digitalWrite(LEDG, HIGH);  // Green OFF
    digitalWrite(LEDR, LOW);   // Red ON (active LOW)
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("POWER: OFF");
  }
}

// RECORD MODE

void record_window() {
  delay(RECORD_PRE_DELAY_MS);

  if (!sensors_acquire_window(window)) {
    Serial.println("ERR: capture failed");
    return;
  }

  Serial.println("START");
  Serial.println("aX,aY,aZ");
  for (int i = 0; i < window.count; i++) {
    Serial.print(window.ax[i], 3);
    Serial.print(',');
    Serial.print(window.ay[i], 3);
    Serial.print(',');
    Serial.println(window.az[i], 3);
  }
  Serial.println("END");
}

// SINGLE CLASSIFY MODE

void classify_single() {
  Serial.println("Capturing window...");
  if (!sensors_acquire_window(window)) {
    Serial.println("ERR: capture failed");
    return;
  }

  float rms = features_window_rms(window);
  Serial.print("Window RMS: ");
  Serial.print(rms, 4);
  Serial.println(" g");

  if (rms < GATE_RMS) {
    Serial.print("Gate: RMS < ");
    Serial.print(GATE_RMS, 3);
    Serial.println(" -> idle (short-circuited, no inference)");
    Serial.println("Result: idle (conf: 1.000)");
    return;
  }

  features_extract(window, feat);
  features_normalize(feat);

  unsigned long t0 = micros();
  int pred = inference_run(feat, scores);
  unsigned long lat_us = micros() - t0;
  power_record_latency(lat_us);

  Serial.print("Inference latency: ");
  Serial.print(lat_us / 1000.0f, 2);
  Serial.print(" ms (");
  Serial.print(lat_us);
  Serial.println(" us)");

  Serial.print("Prediction: ");
  Serial.print(CLASS_LABELS[pred]);
  Serial.print(" (conf: ");
  Serial.print(scores[pred], 3);
  Serial.println(")");

  if (scores[pred] < ACT_THRESHOLD) {
    Serial.print("NOTE: Conf < ");
    Serial.print(ACT_THRESHOLD, 2);
    Serial.println(" -> treated as idle by policy");
  }

  Serial.println("Class scores:");
  for (int i = 0; i < N_CLASSES; i++) {
    Serial.print("  [");
    Serial.print(i);
    Serial.print("] ");
    Serial.print(CLASS_LABELS[i]);
    Serial.print(": ");
    Serial.println(scores[i], 3);
  }
}

// MONITOR MODE (CONTINUOUS)

void step_monitor() {
  if (!sensors_acquire_window(window)) {
    return;
  }

  float rms = features_window_rms(window);
  int   pred = 0;          // 0 = idle
  float conf = 1.0f;
  bool  ran_inference = false;

  // Cheap RMS gate check
  if (rms >= GATE_RMS) {
    features_extract(window, feat);
    features_normalize(feat);

    unsigned long t0 = micros();
    pred = inference_run(feat, scores);
    unsigned long lat_us = micros() - t0;
    power_record_latency(lat_us);

    conf = scores[pred];
    ran_inference = true;

    // Reject low confidence predictions
    if (conf < ACT_THRESHOLD) {
      pred = 0; // treat ambiguous as idle
    }
  } else {
    // Vibration below noise gate -> pure idle
    pred = 0;
    conf = 1.0f;
  }

  // Decision mapping: {typing(1), writing(2), handling(3)} = activity
  //                   {idle(0), incidental(4)} = no activity
  bool is_activity = (pred == 1 || pred == 2 || pred == 3);
  unsigned long now_ms = millis();

  if (is_activity) {
    consecutive_activity++;
    last_activity_ms = now_ms;
    if (!load_on && consecutive_activity >= ACT_CONSECUTIVE) {
      set_load_power(true);
    }
  } else {
    consecutive_activity = 0;
    if (load_on && (now_ms - last_activity_ms > IDLE_TIMEOUT_MS)) {
      set_load_power(false);
    }
  }

  // Monitor status line
  Serial.print("[mon] ");
  Serial.print(CLASS_LABELS[pred]);
  Serial.print(" (");
  Serial.print(conf, 2);
  Serial.print(") rms=");
  Serial.print(rms, 4);
  if (!ran_inference) Serial.print(" [gate]");
  Serial.print(" | LOAD: ");
  Serial.println(load_on ? "ON" : "OFF");

  // Inter-window sleep for low power
  power_sleep(100);
}
