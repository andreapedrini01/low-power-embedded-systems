#include "power.h"
#include "config.h"

static bool          active_open = false;
static unsigned long active_start_us;
static unsigned long total_active_us;
static unsigned long total_sleep_us;
static unsigned long total_sniff_us;

static bool          load_flag = false;
static unsigned long load_change_us;
static unsigned long total_load_on_us;
static unsigned long total_load_off_us;
static unsigned long last_latency_us = 0;

/*
 * Resets timing accumulators and initializes timestamps.
 */
void power_init() {
  active_open       = false;
  total_active_us   = 0;
  total_sleep_us    = 0;
  total_sniff_us    = 0;
  load_flag         = false;
  total_load_on_us  = 0;
  total_load_off_us = 0;
  load_change_us    = micros();
}

/*
 * Records the start timestamp of an active CPU processing phase.
 */
void power_mark_active_begin() {
  active_start_us = micros();
  active_open = true;
}

/*
 * Computes elapsed time since the active phase began and accumulates it into total_active_us.
 */
void power_mark_active_end() {
  if (!active_open) return;
  total_active_us += micros() - active_start_us;
  active_open = false;
}

/*
 * Suspends execution for ms milliseconds while accumulating sleep time.
 */
void power_sleep(unsigned long ms) {
  bool was = active_open;
  if (was) power_mark_active_end();

  unsigned long t0 = micros();
  delay(ms);
  total_sleep_us += micros() - t0;

  if (was) power_mark_active_begin();
}

/*
 * Suspends execution during 4 Hz sniffing intervals, accumulating sniffing sleep duration.
 */
void power_sniff_sleep(unsigned long ms) {
  bool was = active_open;
  if (was) power_mark_active_end();

  unsigned long t0 = micros();
  delay(ms);
  total_sniff_us += micros() - t0;

  if (was) power_mark_active_begin();
}

unsigned long power_active_us() { return total_active_us; }
unsigned long power_sleep_us()  { return total_sleep_us; }
unsigned long power_sniff_us()  { return total_sniff_us; }

/*
 * Calculates energy consumed by the board in mJ:
 * E = V_BUS * (I_ACTIVE * t_active + I_SLEEP * t_sleep).
 */
float power_board_energy_mj() {
  float a_ms = total_active_us / 1000.0f;
  float s_ms = (total_sleep_us + total_sniff_us) / 1000.0f;
  float e_uj = V_BUS * (I_ACTIVE_MA * a_ms + I_LIGHT_SLEEP_UA * 0.001f * s_ms);
  return e_uj / 1000.0f;
}

/*
 * Switches state to LOAD ON and accumulates the preceding OFF duration.
 */
void power_load_on() {
  if (load_flag) return;
  unsigned long now = micros();
  total_load_off_us += now - load_change_us;
  load_change_us = now;
  load_flag = true;
}

/*
 * Switches state to LOAD OFF and accumulates the preceding ON duration.
 */
void power_load_off() {
  if (!load_flag) return;
  unsigned long now = micros();
  total_load_on_us += now - load_change_us;
  load_change_us = now;
  load_flag = false;
}

unsigned long power_load_on_us()  { return total_load_on_us; }
unsigned long power_load_off_us() { return total_load_off_us; }

/*
 * Computes energy saved by turning off desk peripherals (in mJ):
 * E_saved = LOAD_WATTS * t_off * 1000.
 */
float power_load_saved_mj() {
  float off_s = total_load_off_us / 1e6f;
  return LOAD_WATTS * off_s * 1000.0f;
}

/*
 * Updates the recorded latency measurement for the last inference cycle.
 */
void power_record_latency(unsigned long lat_us) {
  last_latency_us = lat_us;
}

/*
 * Outputs a diagnostic summary of runtimes, duty cycles, and energy metrics over serial.
 */
void power_print_summary() {
  unsigned long now = micros();
  if (active_open) {
    total_active_us += now - active_start_us;
    active_start_us = now;
  }
  if (load_flag) {
    total_load_on_us += now - load_change_us;
  } else {
    total_load_off_us += now - load_change_us;
  }
  load_change_us = now;

  float total_s = (total_active_us + total_sleep_us + total_sniff_us) / 1e6f;
  float duty    = total_s > 0 ? (total_active_us / 1e6f / total_s * 100.0f) : 0.0f;

  Serial.println("--- System Power Summary ---");
  Serial.print("Session Duration:   "); Serial.print(total_s, 1);           Serial.println(" s");
  Serial.print("Active Processing:  "); Serial.print(total_active_us / 1000); Serial.println(" ms");
  Serial.print("Active-Sense Sleep: "); Serial.print(total_sleep_us / 1000);  Serial.println(" ms");
  Serial.print("Sniffing Sleep:     "); Serial.print(total_sniff_us / 1000);  Serial.println(" ms");
  Serial.print("CPU Duty Cycle:     "); Serial.print(duty, 2);                Serial.println(" %");
  Serial.print("Board Energy:       "); Serial.print(power_board_energy_mj(), 2); Serial.println(" mJ");

  float on_s  = total_load_on_us  / 1e6f;
  float off_s = total_load_off_us / 1e6f;
  Serial.print("Desk Load ON:       "); Serial.print(on_s, 1);  Serial.println(" s");
  Serial.print("Desk Load OFF:      "); Serial.print(off_s, 1); Serial.println(" s");
  Serial.print("Desk Load Saved:    "); Serial.print(power_load_saved_mj(), 1); Serial.println(" mJ");
}
