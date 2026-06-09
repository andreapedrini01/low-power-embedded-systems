#include "power.h"
#include "config.h"

// Per-checkpoint accumulators.
static unsigned long ckp_active_us = 0;
static unsigned long ckp_light_us  = 0;
static unsigned long ckp_deep_us   = 0;

// Mission totals.
static unsigned long tot_active_us = 0;
static unsigned long tot_light_us  = 0;
static unsigned long tot_deep_us   = 0;

// Active-window tracking.
static unsigned long active_start_us = 0;
static bool          active_open     = false;

bool power_init() {
  ckp_active_us = ckp_light_us = ckp_deep_us = 0;
  tot_active_us = tot_light_us = tot_deep_us = 0;
  active_start_us = 0;
  active_open = false;
  return true;
}

void power_mark_active_begin() {
  if (active_open) {
    Serial.println("WARN: nested active");
    return;
  }
  active_start_us = micros();
  active_open = true;
}

void power_mark_active_end() {
  if (!active_open) return;
  unsigned long elapsed = micros() - active_start_us;
  ckp_active_us += elapsed;
  tot_active_us += elapsed;
  active_open = false;
}

void power_sleep(SleepLevel level, unsigned long duration_ms) {
  if (duration_ms == 0) return;
  
  bool was_active = active_open;
  if (was_active) power_mark_active_end();
  
  unsigned long t0 = micros();
  // Portable backend: delay() lets the Mbed scheduler idle.
  // Replace with LowPower.deepSleep() for true deep sleep.
  delay(duration_ms);
  unsigned long elapsed = micros() - t0;
  
  if (level == SLEEP_DEEP) {
    ckp_deep_us  += elapsed;
    tot_deep_us  += elapsed;
  } else {
    ckp_light_us += elapsed;
    tot_light_us += elapsed;
  }
  
  if (was_active) power_mark_active_begin();
}

void power_checkpoint_reset() {
  ckp_active_us = ckp_light_us = ckp_deep_us = 0;
}

EnergyAccount power_checkpoint_get() {
  EnergyAccount e;
  e.active_us      = ckp_active_us;
  e.light_sleep_us = ckp_light_us;
  e.deep_sleep_us  = ckp_deep_us;
  e.energy_uj      = power_energy_uj(ckp_active_us, ckp_light_us, ckp_deep_us);
  return e;
}

EnergyAccount power_total_get() {
  EnergyAccount e;
  e.active_us      = tot_active_us;
  e.light_sleep_us = tot_light_us;
  e.deep_sleep_us  = tot_deep_us;
  e.energy_uj      = power_energy_uj(tot_active_us, tot_light_us, tot_deep_us);
  return e;
}

void power_save(PowerState &s) {
  s.ckp_active_us = ckp_active_us;
  s.ckp_light_us  = ckp_light_us;
  s.ckp_deep_us   = ckp_deep_us;
  s.tot_active_us = tot_active_us;
  s.tot_light_us  = tot_light_us;
  s.tot_deep_us   = tot_deep_us;
}

void power_restore(const PowerState &s) {
  ckp_active_us = s.ckp_active_us;
  ckp_light_us  = s.ckp_light_us;
  ckp_deep_us   = s.ckp_deep_us;
  tot_active_us = s.tot_active_us;
  tot_light_us  = s.tot_light_us;
  tot_deep_us   = s.tot_deep_us;
}

float power_energy_uj(unsigned long active_us,
                      unsigned long light_us,
                      unsigned long deep_us) {
  // E_uJ = V * (mA*ms + uA*1e-3*ms + uA*1e-3*ms)
  float active_ms = active_us / 1000.0f;
  float light_ms  = light_us  / 1000.0f;
  float deep_ms   = deep_us   / 1000.0f;
  return V_BUS * (I_ACTIVE_MA      * active_ms
                + I_LIGHT_SLEEP_UA * 0.001f * light_ms
                + I_DEEP_SLEEP_UA  * 0.001f * deep_ms);
}
