"""
PolicyGRID adapter for the testbed's IoT drivers.
============================================
Provides synchronous wrappers around the async drivers so PolicyGRID's
discovery pipeline can call them directly. Does NOT modify any driver code.

Key insight: the kasa_controller driver caches device objects at module level
(_discovered_devices). These objects are tied to the event loop they were
created on. We must clear this cache between asyncio.run() calls, or the
stale cached devices cause "Timeout context manager should be used inside
a task" errors.

Usage:
    from policygrid_adapter import read_all_sensors, device_on, device_off, all_off
"""

import asyncio
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def _clear_kasa_cache():
    """Clear python-kasa's cached device objects and sessions.

    kasa_controller.py caches discovered devices at module level.
    These objects are bound to the event loop they were created on.
    Since each asyncio.run() creates a new loop, stale cached devices
    cause errors. Clear them so each call gets fresh connections.
    """
    try:
        from drivers import kasa_controller
        kasa_controller._discovered_devices.clear()
        # Close and reset the shared session too
        if kasa_controller._shared_session and not kasa_controller._shared_session.closed:
            # Can't await here (sync context), just discard it
            pass
        kasa_controller._shared_session = None
    except Exception:
        pass


def _run(coro):
    """Run async coroutine synchronously, clearing Kasa cache first."""
    _clear_kasa_cache()
    return asyncio.run(coro)


# ── Sensor reading ──

def read_all_sensors() -> dict:
    """
    Read all sensors and return a unified dict with PolicyGRID variable names.
    Averages both Govee sensors for Temperature/Humidity.
    """
    async def _do():
        from drivers import read_bme_data, read_govee_data, read_room_state
        from drivers.kasa_controller import get_all_energy_usage

        bme, govee, room, energy = await asyncio.gather(
            read_bme_data(),
            read_govee_data(),
            read_room_state(),
            get_all_energy_usage()
        )
        return bme or {}, govee or {}, room or {}, energy or {}

    bme, govee, room, energy = _run(_do())

    # Average both Govee sensors
    temps = [v for k, v in govee.items() if 'Temperature' in k and v is not None]
    humids = [v for k, v in govee.items() if 'Humidity' in k and v is not None]

    state = {
        'timestamp': datetime.now().isoformat(),
        'Temperature': round(np.mean(temps), 2) if temps else None,
        'Humidity': round(np.mean(humids), 2) if humids else None,
        'AirQuality': bme.get('iaq'),
        'CO2eq': bme.get('co2_equivalent'),
        'EnergyConsumption': energy.get('current_total_power_w', 0),
        'Occupancy': room.get('occupancy', 0) if room else 0,
        'WindowState': room.get('window_open', 0) if room else 0,
    }

    # Satisfaction (ISO 7730 simplified)
    if state['Temperature'] is not None and state['Humidity'] is not None:
        state['Satisfaction'] = _compute_satisfaction(state['Temperature'], state['Humidity'])
    else:
        state['Satisfaction'] = None

    return state


def _compute_satisfaction(temperature, humidity):
    """Simplified ISO 7730 PMV-based satisfaction (0-100)."""
    metabolic_rate = 1.2
    pmv = 0.303 * np.exp(-0.036 * metabolic_rate * 58.15) + 0.028
    t_cl = 35.7 - 0.028 * metabolic_rate * 58.15
    h_c = max(2.38 * abs(t_cl - temperature) ** 0.25, 12.1 * np.sqrt(0.1))
    thermal_load = (
        metabolic_rate * 58.15
        - 3.05e-3 * (5733 - 6.99 * metabolic_rate * 58.15
                      - max(humidity * 0.01 * np.exp(16.6536 - 4030.183 / (temperature + 235)), 0))
        - 0.42 * (metabolic_rate * 58.15 - 58.15)
        - 1.7e-5 * metabolic_rate * 58.15 * (5867
                      - max(humidity * 0.01 * np.exp(16.6536 - 4030.183 / (temperature + 235)), 0))
        - 0.0014 * metabolic_rate * (34 - temperature)
        - 3.96e-8 * ((t_cl + 273) ** 4 - (temperature + 273) ** 4)
        - h_c * (t_cl - temperature)
    )
    pmv_val = max(-3, min(3, pmv * thermal_load))
    return round(max(0, 100 - (abs(pmv_val) / 3.0) * 100), 1)


# ── Device control ──

def device_on(device_name: str) -> bool:
    """Turn a device ON. device_name: 'Heater', 'Humidifier', or 'Fan'."""
    async def _do():
        from drivers.kasa_controller import control_device
        return await control_device(device_name, turn_on=True)
    logger.info(f"  Device ON: {device_name}")
    return _run(_do())


def device_off(device_name: str) -> bool:
    """Turn a device OFF."""
    async def _do():
        from drivers.kasa_controller import control_device
        return await control_device(device_name, turn_on=False)
    logger.info(f"  Device OFF: {device_name}")
    return _run(_do())


def all_off() -> bool:
    """Turn all devices OFF."""
    async def _do():
        from drivers import close_all_devices
        return await close_all_devices()
    logger.info("  All devices OFF")
    return _run(_do())


def get_device_states() -> dict:
    """Get current on/off state of all devices."""
    async def _do():
        from drivers.kasa_controller import get_device_states as _get_states
        return await _get_states()
    states = _run(_do())
    return {
        'Heater': bool(states.get('Heater_On', 0)),
        'Humidifier': bool(states.get('Humidifier_On', 0)),
        'Fan': bool(states.get('Fan_On', 0)),
    }


def cleanup():
    """Clean up Kasa connections."""
    async def _do():
        from drivers.kasa_controller import cleanup_kasa
        return await cleanup_kasa()
    _run(_do())
