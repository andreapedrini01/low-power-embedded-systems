/*
 * SmartDeskPower - Surface Activity Sensing and Power Management
 * Authors: Andrea Pedrini, Pietro Bellini
 * Platform: Arduino Nano 33 BLE Sense (Nordic nRF52840, LSM9DS1 IMU)
 *
 * Implements structure-borne desk vibration sensing to determine user presence
 * and control desk load power with duty-cycled sniffing and anti-chattering hysteresis.
 */

#include "config.h"
#include "sensors.h"
#include "features.h"
#include "inference.h"
#include "power.h"
#include "active_model.h"

static AccelWindow window;
static float       feat[N_FEATURES];
static float       scores[N_CLASSES];

static bool          monitoring           = false;
static bool          sniffing             = true;
static bool          load_on              = false;
static int           consecutive_activity = 0;
static int           quiet_windows        = 0;
static unsigned long last_activity_ms     = 0;

void record_window();
void classify_single();
void step_monitor();
void set_load_power(bool on);

// Hardware and peripheral setup: serial, LEDs, energy tracking, and LSM9DS1.
void setup() {
  Serial.begin(115200);
  while (!Serial && millis() < 3000) {
    delay(10);
  }

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
    Serial.println("FATAL: IMU initialization failed");
    while (1) {
      delay(1000);
    }
  }

  inference_init();

  set_load_power(true);
  last_activity_ms = millis();

  Serial.println("SmartDeskPower - Pedrini, Bellini");
  Serial.print("Window size: ");
  Serial.print(WINDOW_SAMPLES);
  Serial.println(" samples (~1 s @ 119 Hz)");
  Serial.println("Commands: r=record, c=classify, m=toggle monitor, s=summary, q=quit");
}

// Process serial commands and run monitor cycle if enabled.
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
        sniffing = true;
        quiet_windows = 0;
        consecutive_activity = 0;
        Serial.print("Monitor mode: ");
        Serial.println(monitoring ? "STARTED (Duty-Cycled Sniffing @ 4 Hz)" : "STOPPED");
        break;
      case 's':
        power_print_summary();
        break;
      case 'q':
        monitoring = false;
        sniffing = true;
        quiet_windows = 0;
        consecutive_activity = 0;
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

// Set desk load state and update green (ON) / red (OFF) LEDs.
void set_load_power(bool on) {
  load_on = on;
  if (on) {
    power_load_on();
    digitalWrite(LEDG, LOW);
    digitalWrite(LEDR, HIGH);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, HIGH);
    Serial.println("POWER: ON");
  } else {
    power_load_off();
    digitalWrite(LEDG, HIGH);
    digitalWrite(LEDR, LOW);
    digitalWrite(LEDB, HIGH);
    digitalWrite(LED_BUILTIN, LOW);
    Serial.println("POWER: OFF");
  }
}

// Capture one 119-sample window and print CSV over serial.
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

// Acquire one window, run feature extraction and inference, print scores and latency.
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

// Two-state monitor: 4 Hz sniffing when idle, 119 Hz window inference when motion is sensed.
void step_monitor() {
  if (sniffing) {
    float delta = sensors_read_single_delta();
    if (delta < WAKE_DELTA_G) {
      power_sniff_sleep(SNIFF_INTERVAL_MS);
      return;
    }

    Serial.print("[WAKE] Motion detected (delta=");
    Serial.print(delta, 4);
    Serial.println(" g), capturing 119 Hz window...");
    sniffing = false;
    quiet_windows = 0;
  }

  if (!sensors_acquire_window(window)) {
    return;
  }

  float rms = features_window_rms(window);
  int   pred = 0;
  float conf = 1.0f;
  bool  ran_inference = false;

  if (rms >= GATE_RMS) {
    features_extract(window, feat);
    features_normalize(feat);

    unsigned long t0 = micros();
    pred = inference_run(feat, scores);
    unsigned long lat_us = micros() - t0;
    power_record_latency(lat_us);

    conf = scores[pred];
    ran_inference = true;

    if (conf < ACT_THRESHOLD) {
      pred = 0;
    }
  } else {
    pred = 0;
    conf = 1.0f;
  }

  bool is_activity = (pred == 1 || pred == 2 || pred == 3);
  unsigned long now_ms = millis();

  if (is_activity) {
    consecutive_activity++;
    quiet_windows = 0;
    last_activity_ms = now_ms;
    if (!load_on && consecutive_activity >= ACT_CONSECUTIVE) {
      set_load_power(true);
    }
  } else {
    consecutive_activity = 0;
    quiet_windows++;
    if (load_on && (now_ms - last_activity_ms > IDLE_TIMEOUT_MS)) {
      set_load_power(false);
    }
  }

  Serial.print("[mon] ");
  Serial.print(CLASS_LABELS[pred]);
  Serial.print(" (");
  Serial.print(conf, 2);
  Serial.print(") rms=");
  Serial.print(rms, 4);
  if (!ran_inference) {
    Serial.print(" [gate]");
  }
  Serial.print(" | LOAD: ");
  Serial.println(load_on ? "ON" : "OFF");

  if (!load_on && quiet_windows >= SNIFF_FALLBACK_WIN) {
    sniffing = true;
    Serial.println("[SNIFF] Quiet surface & load OFF -> returning to low-power sniffing (4 Hz)");
  } else {
    power_sleep(100);
  }
}
