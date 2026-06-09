#ifndef POWER_H
#define POWER_H

#include <Arduino.h>

enum SleepLevel {
  SLEEP_LIGHT = 0,
  SLEEP_DEEP  = 1
};

struct EnergyAccount {
  unsigned long active_us;
  unsigned long light_sleep_us;
  unsigned long deep_sleep_us;
  float         energy_uj;
};

struct PowerState {
  unsigned long ckp_active_us;
  unsigned long ckp_light_us;
  unsigned long ckp_deep_us;
  unsigned long tot_active_us;
  unsigned long tot_light_us;
  unsigned long tot_deep_us;
};

bool power_init();

void power_mark_active_begin();
void power_mark_active_end();

void power_sleep(SleepLevel level, unsigned long duration_ms);

void          power_checkpoint_reset();
EnergyAccount power_checkpoint_get();
EnergyAccount power_total_get();

// Save/restore for sub-routines that must not pollute mission totals.
void power_save(PowerState &s);
void power_restore(const PowerState &s);

float power_energy_uj(unsigned long active_us,
                      unsigned long light_us,
                      unsigned long deep_us);

#endif
