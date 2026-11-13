#ifndef POWER_H
#define POWER_H

#include <Arduino.h>

/*
 * Resets all power and execution time tracking counters.
 */
void power_init();

/*
 * Marks the transition into active processing to accumulate CPU execution microseconds.
 */
void power_mark_active_begin();

/*
 * Marks the transition out of active processing to update total active runtime.
 */
void power_mark_active_end();

/*
 * Puts the microcontroller into low-power sleep for the specified duration in milliseconds,
 * tracking sleep duration and suspending active CPU accounting.
 */
void power_sleep(unsigned long ms);

/*
 * Dedicated sleep function for low-power duty-cycled sniffing intervals (250 ms),
 * tracking sniffing energy separately from active sensing sleep.
 */
void power_sniff_sleep(unsigned long ms);

unsigned long power_active_us();
unsigned long power_sleep_us();
unsigned long power_sniff_us();

/*
 * Estimates total electrical energy consumed by the board in millijoules
 * based on active/sleep duty cycles and current models.
 */
float power_board_energy_mj();

/*
 * Records the desk peripheral load as powered ON, updating cumulative load timers.
 */
void power_load_on();

/*
 * Records the desk peripheral load as powered OFF, updating cumulative load timers.
 */
void power_load_off();

unsigned long power_load_on_us();
unsigned long power_load_off_us();

/*
 * Computes energy saved in millijoules by keeping the assumed peripheral load switched OFF.
 */
float power_load_saved_mj();

/*
 * Stores the latency in microseconds of the most recent inference execution.
 */
void power_record_latency(unsigned long lat_us);

/*
 * Prints a formatted diagnostic summary of session runtime, active duty cycle,
 * estimated board energy, and peripheral energy saved.
 */
void power_print_summary();

#endif // POWER_H
