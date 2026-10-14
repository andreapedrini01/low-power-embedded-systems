#include "power.h"
#include "config.h"

// ── Board active/sleep tracking ──

static bool          active_open = false;
static unsigned long active_start_us;
static unsigned long total_active_us;
static unsigned long total_sleep_us;

// ── Desk load tracking ──

static bool          load_flag = false;
static unsigned long load_change_us;
static unsigned long total_load_on_us;
static unsigned long total_load_off_us;
static unsigned long last_latency_us = 0;

// ── Init ──

void power_init() {
  active_open       = false;
  total_active_us   = 0;
  total_sleep_us    = 0;
  load_flag         = false;
  total_load_on_us  = 0;
  total_load_off_us = 0;
  load_change_us    = micros();
}

// ── Active window bracketing ──

void power_mark_active_begin() {
  active_start_us = micros();
  active_open = true;
}

void power_mark_active_end() {
  if (!active_open) return;
  total_active_us += micros() - active_start_us;
  active_open = false;
}

// Delay-based sleep. Brackets active window automatically (EnergyAwarePath).
void power_sleep(unsigned long ms) {
  bool was = active_open;
  if (was) power_mark_active_end();
  unsigned long t0 = micros();
  delay(ms);
  total_sleep_us += micros() - t0;
  if (was) power_mark_active_begin();
}

unsigned long power_active_us() { return total_active_us; }
unsigned long power_sleep_us()  { return total_sleep_us; }

// Board energy from duty cycle and config.h current constants (est).
float power_board_energy_mj() {
  float a_ms = total_active_us / 1000.0f;
  float s_ms = total_sleep_us  / 1000.0f;
  float e_uj = V_BUS * (I_ACTIVE_MA * a_ms +
                        I_LIGHT_SLEEP_UA * 0.001f * s_ms);
  return e_uj / 1000.0f;
}

// ── Desk load ──

void power_load_on() {
  if (load_flag) return;
  unsigned long now = micros();
  total_load_off_us += now - load_change_us;
  load_change_us = now;
  load_flag = true;
}

void power_load_off() {
  if (!load_flag) return;
  unsigned long now = micros();
  total_load_on_us += now - load_change_us;
  load_change_us = now;
  load_flag = false;
}

unsigned long power_load_on_us()  { return total_load_on_us; }
unsigned long power_load_off_us() { return total_load_off_us; }

// Energy saved = time load was cut × assumed wattage (est).
float power_load_saved_mj() {
  float off_s = total_load_off_us / 1e6f;
  return LOAD_WATTS * off_s * 1000.0f;
}

void power_record_latency(unsigned long lat_us) {
  last_latency_us = lat_us;
}

// ── Summary ──

void power_print_summary() {
  // Flush current periods
  unsigned long now = micros();
  if (active_open) {
    total_active_us += now - active_start_us;
    active_start_us = now;
  }
  if (load_flag)
    total_load_on_us += now - load_change_us;
  else
    total_load_off_us += now - load_change_us;
  load_change_us = now;

  float total_s = (total_active_us + total_sleep_us) / 1e6f;
  float duty    = total_s > 0
                  ? (total_active_us / 1e6f / total_s * 100.0f) : 0;

  Serial.println("── Summary ──");
  Serial.print("Session:      ");
  Serial.print(total_s, 1);           Serial.println(" s");
  Serial.print("Active:       ");
  Serial.print(total_active_us/1000); Serial.println(" ms");
  Serial.print("Sleep:        ");
  Serial.print(total_sleep_us/1000);  Serial.println(" ms");
  Serial.print("Active duty:  ");
  Serial.print(duty, 1);             Serial.println(" % (est)");
  Serial.print("Board energy: ");
  Serial.print(power_board_energy_mj(), 2); Serial.println(" mJ (est)");

  float on_s  = total_load_on_us  / 1e6f;
  float off_s = total_load_off_us / 1e6f;
  Serial.print("Load ON:      ");
  Serial.print(on_s, 1);  Serial.println(" s");
  Serial.print("Load OFF:     ");
  Serial.print(off_s, 1); Serial.println(" s");
  Serial.print("Load saved:   ");
  Serial.print(power_load_saved_mj(), 1); Serial.println(" mJ (est)");

  if (last_latency_us > 0) {
    Serial.print("Inference lat: ");
    Serial.print(last_latency_us / 1000.0f, 2);
    Serial.print(" ms (");
    Serial.print(last_latency_us);
    Serial.println(" us)");
  }
}
