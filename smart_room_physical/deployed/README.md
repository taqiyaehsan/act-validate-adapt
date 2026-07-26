# IoT Smart Room Monitoring & Control

Room environment monitoring and control system for causal analysis research. Collects data from multiple sensor types, controls smart home devices via Kasa smart plugs, and logs everything to CSV.

## Hardware

- **ESP32-C6 + BME680** — Temperature, humidity, pressure, gas resistance (IAQ)
- **Govee H5179 x2** — Wireless temperature/humidity sensors (cloud API)
- **TP-Link Kasa Smart Plugs x3** — Heater, humidifier, fan (on/off control + energy monitoring)
- **Room Logger Web App** — Manual window state and occupancy logging

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Start the BME680 sensor server (receives ESP32 data)
python drivers/bme_server/server.py

# 2. Start the room logger web app (window/occupancy tracking)
python drivers/room_logger/server.py

# 3. Run the main data collector
python data_collector.py
```

## Project Structure

```
IOT/
├── data_collector.py              # Main loop: collect data every 8s, randomize devices every 3h
├── drivers/                       # Hardware drivers (all access goes through __init__.py)
│   ├── __init__.py                # Unified interface for all drivers
│   ├── bme_server/server.py       # Flask server: receives ESP32 BME680 data, computes IAQ
│   ├── bme_client.py              # HTTP client for bme_server
│   ├── room_logger/               # Web app for manual window/occupancy logging
│   │   ├── server.py              # Flask/Waitress server
│   │   ├── templates/             # HTML templates
│   │   └── static/                # CSS
│   ├── room_client.py             # HTTP client for room_logger
│   ├── govee_reader.py            # Govee cloud API reader
│   ├── kasa_controller.py         # Kasa smart plug control + energy monitoring
│   └── close.py                   # Turn off all devices
├── data/                          # CSV output (git-ignored)
├── logs/                          # Runtime logs (git-ignored)
└── requirements.txt
```

## How It Works

`data_collector.py` runs a continuous loop that:

1. **Every 8 seconds**: queries all sensors concurrently (BME680, Govee, Kasa, Room Logger), then writes one row to a timestamped CSV file.
2. **Every 3 hours**: randomly toggles device states (heater, humidifier, fan), ensuring at least one device changes.
3. **On exit (Ctrl+C)**: automatically turns off all devices and cleans up connections.

The random device toggling creates varied environmental conditions, producing data suitable for downstream causal discovery analysis.

## CSV Output

Each row contains 19 fields:

| Field | Source |
|-------|--------|
| `timestamp` | System clock |
| `heater_on`, `humidifier_on`, `fan_on` | Kasa smart plugs |
| `bme_temperature`, `bme_humidity`, `bme_pressure`, `bme_gas_resistance` | ESP32 + BME680 |
| `bme_iaq`, `bme_co2_equivalent`, `bme_breath_voc_equivalent` | Computed by bme_server |
| `govee_temp_1`, `govee_humidity_1`, `govee_temp_2`, `govee_humidity_2` | Govee H5179 sensors |
| `kasa_total_energy_kwh`, `kasa_current_power_w` | Kasa energy monitoring |
| `window_open`, `occupancy` | Room logger web app |

## Other Commands

```bash
# Single data collection test (no continuous loop)
python data_collector.py test

# Turn off all devices
python -m drivers.close
```
