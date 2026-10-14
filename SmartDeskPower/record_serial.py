#!/usr/bin/env python3
"""Serial -> per-class CSV recorder for SmartDeskPower.

Sends 'r' to the board, reads one accelerometer window (START..END block),
and appends its rows to dataset/<label>.csv. Collect each class at different
desk positions and intensities; build a rich 'incidental' set (bumps, passing,
knocks) so the reject class is well represented.

Usage:
    python record_serial.py --port /dev/cu.usbmodemXXXX --label typing

Press Enter to record a window, q + Enter to quit.
"""

import argparse
import os
import sys
import time

import serial  # pyserial

WINDOW_HEADER = "aX,aY,aZ"


def read_window(ser, timeout=8):
    """Read one START..END block; return the list of CSV data rows."""
    t_start = time.time()
    # Skip until START (drops boot text and pre-delay noise).
    while True:
        if time.time() - t_start > timeout:
            print(f"  [TIMEOUT] Nessun marcatore 'START' ricevuto dopo {timeout} s")
            return None

        line = ser.readline().decode(errors="ignore").strip()
        if not line:
            continue

        if line == "START":
            break
        if "ERR: capture failed" in line:
            print("  device reported capture failure")
            return None

        # Print any debug/boot messages sent by the board
        print(f"  [Arduino]: {line}")

    rows = []
    while True:
        if time.time() - t_start > timeout + 5:
            print("  [TIMEOUT] Ricezione dati interrotta prima di 'END'")
            return None

        line = ser.readline().decode(errors="ignore").strip()
        if line == "END":
            break
        if line == "" or line == WINDOW_HEADER:
            continue
        rows.append(line)
    return rows


def main():
    ap = argparse.ArgumentParser(description="Record accelerometer windows to per-class CSV.")
    ap.add_argument("--port", required=True, help="serial port, e.g. /dev/cu.usbmodem101")
    ap.add_argument("--label", required=True,
                    help="class name: idle | typing | writing | handling | incidental")
    ap.add_argument("--dir", default=os.path.join(os.path.dirname(__file__), "dataset"),
                    help="output directory (default: ./dataset)")
    ap.add_argument("--baud", type=int, default=115200)
    args = ap.parse_args()

    os.makedirs(args.dir, exist_ok=True)
    out_path = os.path.join(args.dir, f"{args.label}.csv")
    new_file = not os.path.exists(out_path)

    print(f"Apertura porta {args.port} a {args.baud} baud...")
    try:
        ser = serial.Serial(args.port, args.baud, timeout=1.0)
    except Exception as e:
        print(f"ERRORE apertura porta seriale: {e}")
        print("Verifica che il Serial Monitor di Arduino IDE sia CHIUSO.")
        return 1

    # DTR/RTS e attesa reset USB CDC (fondamentale per Nano 33 BLE)
    ser.dtr = True
    ser.rts = True
    print("Inizializzazione connessione USB con Arduino (2 s)...")
    time.sleep(2.0)

    # Svuota e legge eventuali messaggi di boot iniziali
    while ser.in_waiting > 0:
        init_line = ser.readline().decode(errors="ignore").strip()
        if init_line:
            print(f"  [Arduino boot]: {init_line}")

    windows = 0
    with open(out_path, "a") as f:
        if new_file:
            f.write(WINDOW_HEADER + "\n")

        print(f"\nRecording '{args.label}' -> {out_path}")
        while True:
            cmd = input(f"[{args.label}] window {windows + 1} - Enter per registrare, 'q' per uscire: ")
            if cmd.strip().lower() == "q":
                break

            ser.reset_input_buffer()
            print("  Invio comando 'r'... attendi 1 s di pre-delay e inizia l'attività!")
            ser.write(b"r\n")
            ser.flush()

            rows = read_window(ser, timeout=8)
            if not rows:
                print("  Nessun dato valido acquisito, riprova.")
                continue

            for r in rows:
                f.write(r + "\n")
            f.flush()
            windows += 1
            print(f"  -> Salvati {len(rows)} campioni (finestre totali: {windows})\n")

    ser.close()
    print(f"Fatto. {windows} finestre salvate in {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
