#!/usr/bin/env python3
"""
Hardware smoke test for PolicyGRID integration.
Tests all sensors and actuators before running discovery/policy.

Usage:
    python smoke_test_hardware.py
"""

import asyncio
import sys
import time
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def run(coro):
    loop = asyncio.new_event_loop()
    return loop.run_until_complete(coro)

def main():
    passed = 0
    failed = 0
    total = 0

    def check(name, condition, detail=""):
        nonlocal passed, failed, total
        total += 1
        if condition:
            passed += 1
            print(f"  [PASS] {name} {detail}")
        else:
            failed += 1
            print(f"  [FAIL] {name} {detail}")

    print("=" * 50)
    print("  PolicyGRID Hardware Smoke Test")
    print("=" * 50)

    # ── 1. Sensor reads ──
    print("\n--- Sensors ---")

    # BME680
    try:
        from drivers import read_bme_data
        bme = run(read_bme_data()) or {}
        check("BME680 reachable", bme is not None and len(bme) > 0,
              f"keys={list(bme.keys())}" if bme else "")
        check("BME680 IAQ", bme.get('iaq') is not None,
              f"iaq={bme.get('iaq')}")
        check("BME680 temperature", bme.get('temperature') is not None,
              f"temp={bme.get('temperature')}")
    except Exception as e:
        check("BME680 import/connect", False, str(e))

    # Govee
    try:
        from drivers import read_govee_data
        govee = run(read_govee_data()) or {}
        check("Govee reachable", govee is not None and len(govee) > 0,
              f"keys={list(govee.keys())}" if govee else "")
        temps = [v for k, v in govee.items() if 'Temperature' in k and v is not None]
        check("Govee temp sensors", len(temps) >= 1,
              f"found {len(temps)} temp readings: {temps}")
        humids = [v for k, v in govee.items() if 'Humidity' in k and v is not None]
        check("Govee humidity sensors", len(humids) >= 1,
              f"found {len(humids)} humidity readings: {humids}")
    except Exception as e:
        check("Govee import/connect", False, str(e))

    # Room logger (window/occupancy)
    try:
        from drivers import read_room_state
        room = run(read_room_state())
        check("Room logger reachable", room is not None,
              f"state={room}" if room else "returned None")
        if room:
            check("Window state", 'window_open' in room,
                  f"window_open={room.get('window_open')}")
            check("Occupancy", 'occupancy' in room,
                  f"occupancy={room.get('occupancy')}")
    except Exception as e:
        check("Room logger import/connect", False, str(e))

    # Kasa energy
    try:
        from drivers.kasa_controller import get_all_energy_usage
        energy = run(get_all_energy_usage()) or {}
        check("Kasa energy reachable", energy is not None and len(energy) > 0,
              f"power={energy.get('current_total_power_w', '?')}W")
    except Exception as e:
        check("Kasa energy import/connect", False, str(e))

    # ── 2. PolicyGRID adapter (unified read) ──
    print("\n--- PolicyGRID Adapter ---")
    try:
        from policygrid_adapter import read_all_sensors
        state = read_all_sensors()
        non_null = {k: v for k, v in state.items() if v is not None}
        check("read_all_sensors()", len(non_null) >= 4,
              f"{len(non_null)}/9 non-null: {list(non_null.keys())}")
        check("Temperature", state.get('Temperature') is not None,
              f"{state.get('Temperature')}")
        check("Humidity", state.get('Humidity') is not None,
              f"{state.get('Humidity')}")
        check("AirQuality", state.get('AirQuality') is not None,
              f"{state.get('AirQuality')}")
        check("EnergyConsumption", state.get('EnergyConsumption') is not None,
              f"{state.get('EnergyConsumption')}W")
        check("Satisfaction", state.get('Satisfaction') is not None,
              f"{state.get('Satisfaction')}")
    except Exception as e:
        check("PolicyGRID adapter", False, str(e))

    # ── 3. Actuators ──
    print("\n--- Actuators (Kasa Smart Plugs) ---")
    try:
        from policygrid_adapter import device_on, device_off, all_off, get_device_states

        # Get initial states
        states_before = get_device_states()
        check("get_device_states()", states_before is not None,
              f"{states_before}")

        # Test each device: ON → verify → OFF → verify
        for device in ['Heater', 'Humidifier', 'Fan']:
            print(f"\n  Testing {device}...")

            # Turn ON
            on_result = device_on(device)
            check(f"{device} ON command", on_result is not False,
                  f"result={on_result}")
            time.sleep(3)

            # Verify ON
            states = get_device_states()
            is_on = states.get(device, False)
            check(f"{device} is ON", is_on, f"states={states}")

            # Read energy (should be > 0 if device is actually on)
            state = read_all_sensors()
            power = state.get('EnergyConsumption', 0)
            check(f"{device} draws power", power > 0,
                  f"power={power}W")

            # Turn OFF
            off_result = device_off(device)
            check(f"{device} OFF command", off_result is not False,
                  f"result={off_result}")
            time.sleep(3)

            # Verify OFF
            states = get_device_states()
            is_off = not states.get(device, True)
            check(f"{device} is OFF", is_off, f"states={states}")

        # Final: all off
        all_off()
        time.sleep(2)
        final_states = get_device_states()
        all_are_off = not any(final_states.get(d, False)
                              for d in ['Heater', 'Humidifier', 'Fan'])
        check("all_off()", all_are_off, f"states={final_states}")

    except Exception as e:
        check("Actuator test", False, str(e))
        # Safety: try to turn everything off
        try:
            from policygrid_adapter import all_off
            all_off()
        except:
            pass

    # ── 4. Discovery imports ──
    print("\n--- Discovery Pipeline Imports ---")
    try:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))
        from run_policygrid_discovery import (
            run_hypothesis_generation, run_iterative_discovery,
            GRAPH_VARS, ACTUATORS
        )
        check("Discovery imports", True, f"vars={len(GRAPH_VARS)}, actuators={ACTUATORS}")
    except Exception as e:
        check("Discovery imports", False, str(e))

    # ── Summary ──
    print("\n" + "=" * 50)
    print(f"  RESULTS: {passed}/{total} passed, {failed} failed")
    if failed == 0:
        print("  ✓ All checks passed — ready for PolicyGRID")
    else:
        print("  ✗ Fix failures above before running discovery")
    print("=" * 50)

    return 0 if failed == 0 else 1


if __name__ == '__main__':
    sys.exit(main())
