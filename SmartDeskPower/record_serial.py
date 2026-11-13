#!/usr/bin/env python3
"""Serial dataset recorder for SmartDeskPower.

Streams windows of accelerometer samples (119 Hz) from the Arduino Nano 33 BLE
over serial and appends them to class CSV files.

Authors: Andrea Pedrini, Pietro Bellini
"""

import argparse
import os
import sys
import time
import serial

WINDOW_HEADER = "aX,aY,aZ"


def read_window(ser, timeout=8):
    """Reads one START..END block over serial and returns sample rows."""
    t_start = time.time()

    while True:
        if time.time() - t_start > timeout:
            print(f"  [TIMEOUT] No 'START' marker received after {timeout} s")
            return None

        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        if line == "START":
            break
        if "ERR: capture failed" in line:
            print("  [ERROR] Sensor acquisition failed on board")
            return None

        print(f"  [Arduino]: {line}")

    rows = []
    while True:
        if time.time() - t_start > timeout + 5:
            print("  [TIMEOUT] Transmission interrupted before 'END'")
            return None

        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue
        if line == "END":
            break
        if line == WINDOW_HEADER:
            continue

        parts = line.split(",")
        if len(parts) == 3:
            try:
                float(parts[0])
                float(parts[1])
                float(parts[2])
                rows.append(line)
            except ValueError:
                pass

    return rows


def count_existing_windows(filepath, window_size=119):
    """Counts complete sample windows existing in a target CSV file."""
    if not os.path.exists(filepath):
        return 0
    with open(filepath, "r", encoding="utf-8") as f:
        lines = [ln.strip() for ln in f if ln.strip()]
    data_lines = [ln for ln in lines if ln != WINDOW_HEADER]
    return len(data_lines) // window_size


def append_window(filepath, rows):
    """Appends sample rows to the target CSV file, creating headers if needed."""
    new_file = not os.path.exists(filepath) or os.path.getsize(filepath) == 0
    with open(filepath, "a", encoding="utf-8") as f:
        if new_file:
            f.write(WINDOW_HEADER + "\n")
        for r in rows:
            f.write(r + "\n")


def main():
    parser = argparse.ArgumentParser(description="Serial CSV window recorder")
    parser.add_argument("--port", required=True, help="Serial port (/dev/cu.usbmodem...)")
    parser.add_argument("--baud", type=int, default=115200, help="Baud rate (default: 115200)")
    parser.add_argument("--dir", default="dataset", help="Dataset directory")
    parser.add_argument("--label", required=True,
                        choices=["idle", "typing", "writing", "handling", "incidental"],
                        help="Activity label")
    parser.add_argument("--window-size", type=int, default=119, help="Expected window size")
    args = parser.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    out_path = os.path.join(args.dir, f"{args.label}.csv")

    print(f"Connecting to {args.port} at {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1)
    except serial.SerialException as e:
        print(f"Failed to open port {args.port}: {e}")
        return 1

    time.sleep(2.0)
    ser.reset_input_buffer()

    windows = count_existing_windows(out_path, args.window_size)
    print(f"Recording '{args.label}' -> {out_path} (current windows: {windows})")

    try:
        while True:
            cmd = input(f"[{args.label}] window {windows + 1} - Enter to record, 'q' to quit: ").strip().lower()
            if cmd == "q":
                break

            ser.write(b"r")
            rows = read_window(ser)
            if not rows or len(rows) < args.window_size:
                print(f"  [WARN] Incomplete window: {0 if not rows else len(rows)}/{args.window_size} samples. Discarding.")
                continue

            append_window(out_path, rows)
            windows += 1
            print(f"  -> Saved {len(rows)} samples (total windows: {windows})")

    except KeyboardInterrupt:
        print("\nRecording terminated by user.")
    finally:
        ser.close()

    print(f"Done. '{out_path}' contains {windows} windows.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
