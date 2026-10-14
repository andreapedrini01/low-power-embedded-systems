#ifndef POWER_H
#define POWER_H

#include <Arduino.h>

// ── Board active/sleep tracking ──
void power_init();
void power_mark_active_begin();
void power_mark_active_end();
void power_sleep(unsigned long ms);

unsigned long power_active_us();
unsigned long power_sleep_us();
float power_board_energy_mj();   // (est)

// ── Desk load tracking ──
void power_load_on();
void power_load_off();
unsigned long power_load_on_us();
unsigned long power_load_off_us();
float power_load_saved_mj();     // (est)

void power_record_latency(unsigned long lat_us);
void power_print_summary();

#endif // POWER_H
