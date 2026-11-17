#ifndef POWER_H
#define POWER_H

#include <Arduino.h>

// Reset execution and power timing accumulators.
void power_init();

// Track start of active CPU processing.
void power_mark_active_begin();

// Track end of active CPU processing and add elapsed time to total.
void power_mark_active_end();

// Put MCU in sleep for ms milliseconds, tracking sleep time.
void power_sleep(unsigned long ms);

// Sleep for 4 Hz duty-cycled sniffing (250 ms), tracked separately.
void power_sniff_sleep(unsigned long ms);

unsigned long power_active_us();
unsigned long power_sleep_us();
unsigned long power_sniff_us();

// Estimated board energy (mJ) from active/sleep duty cycle.
float power_board_energy_mj();

// Switch load tracker to ON.
void power_load_on();

// Switch load tracker to OFF.
void power_load_off();

unsigned long power_load_on_us();
unsigned long power_load_off_us();

// Estimated load energy saved (mJ) while peripherals were powered OFF.
float power_load_saved_mj();

// Store latency (us) of the last inference call.
void power_record_latency(unsigned long lat_us);

// Print timing, duty cycle, and energy summary to serial.
void power_print_summary();

#endif // POWER_H
