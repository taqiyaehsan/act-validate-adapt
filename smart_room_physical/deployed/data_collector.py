import random
import asyncio
from drivers import (
    read_bme_data, read_govee_data, read_room_state,
    control_heater, control_humidifier, control_fan,
    get_device_states, get_all_energy_usage,
    close_all_devices, cleanup_kasa
)
import time
import csv
from datetime import datetime
import os
import logging
STATE_CHANGE_INTERVAL = 3600 * 3  # 3 hrs
DATA_COLLECTION_INTERVAL = 8  # 8 seconds

_start_time = datetime.now().strftime("%Y%m%d_%H%M%S")
CSV_FILENAME = f"data/sensor_data_{_start_time}.csv"
LOG_FILENAME = f"logs/data_collector_{_start_time}.log"

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(LOG_FILENAME),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

# CSV headers
CSV_HEADERS = [
    'timestamp', 'heater_on', 'humidifier_on', 'fan_on',
    'bme_temperature', 'bme_humidity', 'bme_pressure', 'bme_gas_resistance', 
    'bme_iaq', 'bme_co2_equivalent', 'bme_breath_voc_equivalent',
    'govee_temp_1', 'govee_humidity_1', 'govee_temp_2', 'govee_humidity_2',
    'kasa_total_energy_kwh', 'kasa_current_power_w',
    'window_open', 'occupancy'
]

async def random_kasa_state():
    # Get current device states first
    try:
        current_states = await get_current_device_states()
        current_device_states = {
            "Heater": current_states.get("heater_on"),
            "Humidifier": current_states.get("humidifier_on"),
            "Fan": current_states.get("fan_on")
        }
        logger.info(f"Current device states: {current_device_states}")
    except Exception as e:
        logger.warning(f"Could not get current states, using random: {e}")
        current_device_states = {
            "Heater": None,
            "Humidifier": None,
            "Fan": None
        }
    
    # Generate new states ensuring at least one is different
    max_attempts = 10
    for attempt in range(max_attempts):
        random_states = {
            "Heater": random.choice([True, False]),
            "Humidifier": random.choice([True, False]),
            "Fan": random.choice([True, False])
        }
        
        # Check if at least one state is different from current
        if any(current_device_states[device] is None or 
               random_states[device] != current_device_states[device] 
               for device in ["Heater", "Humidifier", "Fan"]):
            break
        
        if attempt == max_attempts - 1:
            # Force change one random device if all attempts failed
            device_to_change = random.choice(["Heater", "Humidifier", "Fan"])
            if current_device_states[device_to_change] is not None:
                random_states[device_to_change] = not current_device_states[device_to_change]
            logger.info(f"Forced change on {device_to_change} after {max_attempts} attempts")
    
    logger.info(f"Applying new states: {random_states}")
    
    try:
        heater_result, humidifier_result, fan_result = await asyncio.gather(
            control_heater(random_states["Heater"]),
            control_humidifier(random_states["Humidifier"]),
            control_fan(random_states["Fan"])
        )
        
        results = {
            "Heater": heater_result,
            "Humidifier": humidifier_result,
            "Fan": fan_result
        }
        
        logger.info(f"Control results: {results}")
        return {
            "current_states": current_device_states,
            "target_states": random_states,
            "control_results": results
        }
        
    except Exception as e:
        logger.error(f"Error controlling devices: {e}")
        return {
            "current_states": current_device_states,
            "target_states": random_states,
            "control_results": {},
            "error": str(e)
        }

async def get_kasa_energy():
    try:
        energy_data = await get_all_energy_usage()
        if energy_data:
            return {
                "total_energy_kwh": energy_data.get("total_energy_kwh", 0),
                "current_total_power_w": energy_data.get("current_total_power_w", 0)
            }
        return None
    except Exception as e:
        logger.error(f"Error getting Kasa energy data: {e}")
        return None

async def query_data():
    bme_data, kasa_energy_data, govee_data, room_data = await asyncio.gather(
        read_bme_data(),
        get_kasa_energy(),
        read_govee_data(),
        read_room_state()
    )

    return {
        "bme_data": bme_data,
        "kasa_energy_data": kasa_energy_data,
        "govee_data": govee_data,
        "room_data": room_data
    }

async def get_current_device_states():
    """Get current on/off states of all devices"""
    try:
        states = await get_device_states()
        return {
            "heater_on": bool(states.get("Heater_On", 0)),
            "humidifier_on": bool(states.get("Humidifier_On", 0)),
            "fan_on": bool(states.get("Fan_On", 0))
        }
    except Exception as e:
        logger.error(f"Error getting device states: {e}")
        return {"heater_on": None, "humidifier_on": None, "fan_on": None}

def write_data_to_csv(data_row):
    """Write a single data row to CSV file"""
    file_exists = os.path.exists(CSV_FILENAME)
    
    with open(CSV_FILENAME, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        
        if not file_exists:
            writer.writeheader()
            logger.info(f"Created new CSV file: {CSV_FILENAME}")
        
        writer.writerow(data_row)

async def collect_and_log_data():
    """Collect all sensor data and device states, then log to CSV"""
    try:
        # Get all data concurrently
        sensor_data, device_states = await asyncio.gather(
            query_data(),
            get_current_device_states()
        )
        
        # Prepare CSV row
        timestamp = datetime.now().isoformat()
        
        # Extract data with safe defaults
        bme = sensor_data.get("bme_data") or {}
        kasa = sensor_data.get("kasa_energy_data") or {}
        govee = sensor_data.get("govee_data") or {}
        room = sensor_data.get("room_data") or {}

        data_row = {
            'timestamp': timestamp,
            'heater_on': device_states.get("heater_on"),
            'humidifier_on': device_states.get("humidifier_on"),
            'fan_on': device_states.get("fan_on"),
            'bme_temperature': bme.get('temperature'),
            'bme_humidity': bme.get('humidity'),
            'bme_pressure': bme.get('pressure'),
            'bme_gas_resistance': bme.get('gas_resistance'),
            'bme_iaq': bme.get('iaq'),
            'bme_co2_equivalent': bme.get('co2_equivalent'),
            'bme_breath_voc_equivalent': bme.get('breath_voc_equivalent'),
            'govee_temp_1': govee.get('Temperature_1'),
            'govee_humidity_1': govee.get('Humidity_1'),
            'govee_temp_2': govee.get('Temperature_2'),
            'govee_humidity_2': govee.get('Humidity_2'),
            'kasa_total_energy_kwh': kasa.get('total_energy_kwh'),
            'kasa_current_power_w': kasa.get('current_total_power_w'),
            'window_open': room.get('window_open'),
            'occupancy': room.get('occupancy')
        }
        
        write_data_to_csv(data_row)
        
        logger.info(f"Data logged - Heater: {device_states.get('heater_on')}, "
                   f"Humidifier: {device_states.get('humidifier_on')}, Fan: {device_states.get('fan_on')}")
        
        return data_row
        
    except Exception as e:
        logger.error(f"Error collecting and logging data: {e}")
        return None

async def main_data_collection():
    """Main function for continuous data collection with periodic state changes"""
    logger.info("Starting continuous data collection...")
    logger.info(f"State change interval: {STATE_CHANGE_INTERVAL} seconds ({STATE_CHANGE_INTERVAL/3600:.1f} hours)")
    logger.info(f"Data collection interval: {DATA_COLLECTION_INTERVAL} seconds")
    logger.info(f"CSV file: {CSV_FILENAME}")
    logger.info(f"Log file: {LOG_FILENAME}")
    
    last_state_change = 0
    
    try:
        while True:
            try:
                current_time = time.time()

                # Check if it's time to change device states
                if current_time - last_state_change >= STATE_CHANGE_INTERVAL:
                    logger.info(f"\n{'='*50}")
                    logger.info("Time for state change!")
                    await random_kasa_state()
                    last_state_change = current_time
                    logger.info(f"{'='*50}\n")

                    # Wait a bit for devices to settle
                    await asyncio.sleep(10)

                # Collect and log data
                await collect_and_log_data()

                # Wait before next data collection
                await asyncio.sleep(DATA_COLLECTION_INTERVAL)

            except KeyboardInterrupt:
                break
            except Exception as e:
                logger.error(f"Error in main loop: {e}")
                await asyncio.sleep(DATA_COLLECTION_INTERVAL)
    finally:
        logger.info("Shutting down: closing all devices...")
        await close_all_devices()
        await cleanup_kasa()
        logger.info("All devices closed. Goodbye.")

if __name__ == "__main__":
    # Test mode - single data collection
    if len(os.sys.argv) > 1 and os.sys.argv[1] == "test":
        t0 = time.time()
        data = asyncio.run(query_data())
        t1 = time.time()
        logger.info(f"Time taken: {t1 - t0} seconds")
        logger.info(f"BME data: {data['bme_data']}")
        logger.info(f"Kasa energy data: {data['kasa_energy_data']}")
        logger.info(f"Govee data: {data['govee_data']}")
        logger.info(f"Room data: {data['room_data']}")
    else:
        # Main continuous collection mode
        asyncio.run(main_data_collection())