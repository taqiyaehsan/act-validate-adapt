#!/usr/bin/env python3
"""
Calibrate actuator hold times and effect thresholds.
Runs each device for 30 min, logs all sensors each minute.

Usage:
    python calibrate_actuators.py                # All 3 devices
    python calibrate_actuators.py --device Fan   # Single device
"""

import sys
import os
import time
import json
import logging
import argparse
import numpy as np
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policygrid_adapter import read_all_sensors, device_on, device_off, all_off

os.makedirs('logs', exist_ok=True)
os.makedirs('results', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(message)s',
    handlers=[
        logging.FileHandler(f'logs/calibration_{datetime.now().strftime("%Y%m%d_%H%M")}.log'),
        logging.StreamHandler(),
    ]
)
log = logging.getLogger(__name__)


def _print(msg=""):
    """Print to both console and log file."""
    log.info(msg)

DEVICES = ['Heater', 'Humidifier', 'Fan']
TARGETS = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'Satisfaction']
HOLD_MIN = 30
WASHOUT_MIN = 15
PRE_READINGS = 3


def calibrate_device(device, hold_min=HOLD_MIN):
    _print(f"\n{'='*60}")
    _print(f"  CALIBRATING: {device} ({hold_min} min hold)")
    _print(f"{'='*60}")

    all_off()
    time.sleep(5)

    # Pre-measurement
    _print(f"\nPre-measurement ({PRE_READINGS} readings, 1 min apart):")
    pre = []
    for i in range(PRE_READINGS):
        s = read_all_sensors()
        pre.append(s)
        vals = ", ".join(f"{t}={s[t]}" for t in TARGETS)
        _print(f"  [{i+1}] {vals}")
        time.sleep(60)

    # Hold
    _print(f"\nTurning {device} ON...")
    device_on(device)
    _print(f"Holding {hold_min} min...")
    readings = []
    for m in range(hold_min):
        time.sleep(60)
        s = read_all_sensors()
        readings.append(s)
        vals = ", ".join(f"{t}={s[t]}" for t in TARGETS)
        _print(f"  min {m+1:2d}: {vals}")

    # Post-measurement
    _print(f"\nPost-measurement ({PRE_READINGS} readings, 1 min apart):")
    post = []
    for i in range(PRE_READINGS):
        s = read_all_sensors()
        post.append(s)
        vals = ", ".join(f"{t}={s[t]}" for t in TARGETS)
        _print(f"  [{i+1}] {vals}")
        time.sleep(60)

    device_off(device)
    all_off()

    # Analysis
    _print(f"\n{'─'*60}")
    _print(f"  RESULTS: {device}")
    _print(f"{'─'*60}")

    results = {}
    for t in TARGETS:
        pre_vals = [s[t] for s in pre if s[t] is not None]
        post_vals = [s[t] for s in post if s[t] is not None]
        if not pre_vals or not post_vals:
            continue

        pre_mean = np.mean(pre_vals)
        post_mean = np.mean(post_vals)
        delta = post_mean - pre_mean

        # Find when effect first becomes detectable (>50% of final delta)
        onset_min = None
        if abs(delta) > 0.01:
            threshold = pre_mean + delta * 0.5
            for m, s in enumerate(readings):
                if s[t] is not None:
                    if delta > 0 and s[t] >= threshold:
                        onset_min = m + 1
                        break
                    elif delta < 0 and s[t] <= threshold:
                        onset_min = m + 1
                        break

        results[t] = {
            'pre_mean': round(pre_mean, 3),
            'post_mean': round(post_mean, 3),
            'delta': round(delta, 3),
            'abs_delta': round(abs(delta), 3),
            'onset_min': onset_min,
        }

        onset_str = f"{onset_min} min" if onset_min else "not reached"
        _print(f"  {t:20s}: {pre_mean:.2f} → {post_mean:.2f} "
              f"(Δ={delta:+.3f}, onset={onset_str})")

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', choices=DEVICES,
                        help='Single device to calibrate')
    parser.add_argument('--hold', type=int, default=HOLD_MIN,
                        help=f'Hold time in minutes (default: {HOLD_MIN})')
    args = parser.parse_args()

    devices = [args.device] if args.device else DEVICES
    all_results = {}

    _print("=" * 60)
    _print("  ACTUATOR CALIBRATION")
    _print(f"  Devices: {devices}")
    _print(f"  Hold: {args.hold} min per device")
    _print(f"  Washout: {WASHOUT_MIN} min between devices")
    est = len(devices) * (PRE_READINGS + args.hold + PRE_READINGS + WASHOUT_MIN)
    _print(f"  Estimated time: {est} min ({est/60:.1f}h)")
    _print("=" * 60)

    try:
        for i, device in enumerate(devices):
            all_results[device] = calibrate_device(device, args.hold)

            if i < len(devices) - 1:
                _print(f"\nWashout {WASHOUT_MIN} min...")
                time.sleep(WASHOUT_MIN * 60)

        # Summary
        _print(f"\n{'='*60}")
        _print(f"  CALIBRATION SUMMARY")
        _print(f"{'='*60}")
        _print(f"{'Device':15s} {'Target':20s} {'Delta':>8s} {'Onset':>8s}")
        _print(f"{'─'*15} {'─'*20} {'─'*8} {'─'*8}")
        for device, results in all_results.items():
            for target, r in results.items():
                onset = f"{r['onset_min']}m" if r['onset_min'] else "—"
                _print(f"{device:15s} {target:20s} {r['delta']:+8.3f} {onset:>8s}")

        # Recommended thresholds
        _print(f"\n  RECOMMENDED THRESHOLDS (50% of smallest significant delta):")
        for target in TARGETS:
            deltas = [abs(all_results[d][target]['delta'])
                      for d in all_results if target in all_results[d]
                      and abs(all_results[d][target]['delta']) > 0.005]
            if deltas:
                recommended = min(deltas) * 0.5
                _print(f"    {target:20s}: {recommended:.4f}")

        # Recommended hold times
        _print(f"\n  RECOMMENDED HOLD TIMES (onset + 50% buffer):")
        for device in all_results:
            onsets = [r['onset_min'] for r in all_results[device].values()
                      if r['onset_min'] is not None]
            if onsets:
                max_onset = max(onsets)
                recommended = int(max_onset * 1.5)
                _print(f"    {device:15s}: {recommended} min (max onset={max_onset} min)")

        # Save
        os.makedirs('results', exist_ok=True)
        with open('results/calibration.json', 'w') as f:
            json.dump(all_results, f, indent=2)
        _print(f"\n  Saved to results/calibration.json")

    except KeyboardInterrupt:
        _print("\nInterrupted")
    finally:
        all_off()


if __name__ == '__main__':
    main()
