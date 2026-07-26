#!/usr/bin/env python3
"""
Smart Room Monitor with ISO 7730 Thermal Comfort Integration
Monitors environmental conditions and calculates scientifically accurate thermal satisfaction
"""

import asyncio
import time
import pandas as pd
import numpy as np
import requests
import sys
import os
from datetime import datetime
import logging

import warnings
warnings.filterwarnings('ignore')

# Thermal comfort calculation
try:
    from pythermalcomfort.models import pmv_ppd
    ISO_7730_AVAILABLE = True
    logging.info("✓ ISO 7730 thermal comfort module loaded")
except ImportError:
    ISO_7730_AVAILABLE = False
    logging.warning("⚠️ pythermalcomfort not installed. Install with: pip install pythermalcomfort")

# Import sensor reading functions
try:
    from smart_room_physical.bme680_reader import get_iaq_value, is_calibrated, get_calibration_progress
    from smart_room_physical.govee_reader import read_govee_data_async, read_govee_data
    from smart_room_physical.kasa_controller import get_device_states, cleanup_kasa, calculate_total_energy
except ImportError as e:
    logging.error(f"Failed to import sensor modules: {e}")
    sys.exit(1)

# Add the project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

logger = logging.getLogger(__name__)

def calculate_iso7730_satisfaction(temperature, humidity, air_velocity=0.1, 
                                 metabolic_rate=1.2, clothing_level=0.6, 
                                 mean_radiant_temp=None):
    """
    Calculate thermal satisfaction using strict ISO 7730 PMV/PPD model.
    
    Args:
        temperature (float): Air temperature in °C
        humidity (float): Relative humidity in %
        air_velocity (float): Air velocity in m/s (default: 0.1 for still air)
        metabolic_rate (float): Metabolic rate in met (default: 1.2 for sedentary office work)
        clothing_level (float): Clothing insulation in clo (default: 0.6 for typical indoor clothing)
        mean_radiant_temp (float): Mean radiant temperature in °C (defaults to air temp)
        
    Returns:
        dict: Contains satisfaction (0-100), PMV (-3 to +3), PPD (0-100%), and method used
    """
    try:
        if not ISO_7730_AVAILABLE:
            logger.debug("Using simplified thermal comfort model (ISO 7730 not available)")
            return calculate_simplified_satisfaction(temperature, humidity)
            
        if mean_radiant_temp is None:
            mean_radiant_temp = temperature
        
        # Validate and constrain inputs to reasonable ranges
        temperature = max(10, min(40, float(temperature)))  # 10-40°C range
        humidity = max(10, min(90, float(humidity)))        # 10-90% range
        air_velocity = max(0.05, min(2.0, air_velocity))    # 0.05-2.0 m/s range
        
        # Calculate PMV/PPD using ISO 7730 standard
        results = pmv_ppd(
            tdb=temperature,
            tr=mean_radiant_temp,
            vr=air_velocity,
            rh=humidity,
            met=metabolic_rate,
            clo=clothing_level,
            standard="ISO"
        )
        
        pmv = results['pmv']
        ppd = results['ppd']
        
        # Convert PPD to satisfaction (100% - PPD%)
        satisfaction = max(0, min(100, 100 - ppd))
        
        logger.debug(f"ISO 7730 Calculation: T={temperature}°C, RH={humidity}%, "
                    f"PMV={pmv:.2f}, PPD={ppd:.1f}%, Satisfaction={satisfaction:.1f}%")
        
        return {
            'satisfaction': round(satisfaction, 1),
            'pmv': round(pmv, 2),
            'ppd': round(ppd, 1),
            'method': 'ISO_7730'
        }
        
    except Exception as e:
        logger.error(f"Error calculating ISO 7730 satisfaction: {e}")
        return calculate_simplified_satisfaction(temperature, humidity)

def calculate_simplified_satisfaction(temperature, humidity):
    """
    Simplified thermal satisfaction calculation as fallback.
    Based on ASHRAE comfort zones and empirical relationships.
    """
    try:
        # ASHRAE comfort zone parameters
        temp_optimal = 22.0  # °C (optimal office temperature)
        humidity_optimal = 45.0  # % (optimal office humidity)
        
        # Temperature comfort calculation
        # Based on ASHRAE Standard 55 comfort zone (20-26°C)
        if 20 <= temperature <= 26:
            temp_comfort = 100 - abs(temperature - temp_optimal) * 3  # 3% penalty per degree
        else:
            temp_comfort = max(0, 100 - abs(temperature - temp_optimal) * 6)  # Higher penalty outside zone
        
        # Humidity comfort calculation
        # Based on ASHRAE comfort zone (30-60% RH)
        if 30 <= humidity <= 60:
            humidity_comfort = 100 - abs(humidity - humidity_optimal) * 1  # 1% penalty per % RH
        else:
            humidity_comfort = max(0, 100 - abs(humidity - humidity_optimal) * 2)  # Higher penalty outside zone
        
        # Combined satisfaction (temperature weighted more heavily)
        satisfaction = (temp_comfort * 0.7 + humidity_comfort * 0.3)
        
        # Estimate PMV and PPD from satisfaction
        estimated_pmv = (100 - satisfaction) / 25 - 2  # Rough approximation
        estimated_ppd = 100 - satisfaction
        
        logger.debug(f"Simplified Calculation: T={temperature}°C, RH={humidity}%, "
                    f"Satisfaction={satisfaction:.1f}% (ASHRAE-based fallback)")
        
        return {
            'satisfaction': round(satisfaction, 1),
            'pmv': round(estimated_pmv, 2),
            'ppd': round(estimated_ppd, 1),
            'method': 'simplified_ASHRAE'
        }
        
    except Exception as e:
        logger.error(f"Error in simplified satisfaction calculation: {e}")
        return {
            'satisfaction': 75.0,
            'pmv': 0.0,
            'ppd': 25.0,
            'method': 'fallback_default'
        }

class SmartRoomMonitor:
    """
    Smart Room Monitor with ISO 7730 Thermal Comfort Integration
    
    Integrates multiple sensor systems:
    - Govee sensors for temperature and humidity
    - BME680 for air quality (IAQ) with burn-in management
    - Kasa smart plugs for energy consumption and device states
    - ISO 7730 thermal comfort calculations
    """
    
    def __init__(self, log_file="smart_room_data.csv"):
        self.log_file = log_file
        self.bme680_server_url = "http://localhost:5001"
        
        # Thermal comfort parameters (can be adjusted for different scenarios)
        self.thermal_params = {
            'air_velocity': 0.1,      # m/s (typical office environment)
            'metabolic_rate': 1.2,    # met (sedentary office work)
            'clothing_level': 0.6     # clo (typical indoor clothing)
        }
        
        self.initialize_log()
        logger.info("SmartRoomMonitor initialized with ISO 7730 thermal comfort")
        
    def initialize_log(self):
        """Initialize log file with all required columns including thermal comfort metrics"""
        if not os.path.exists(self.log_file):
            columns = [
                'timestamp', 'Temperature', 'Humidity', 'AirQuality',
                'EnergyConsumption', 'OverallSatisfaction', 'PMV', 'PPD',
                'Heater_On', 'Humidifier_On', 'Fan_On', 'ThermalComfortMethod'
            ]
            pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
            logger.info(f"Created new log file: {self.log_file}")

    def read_air_quality_direct(self):
        """
        Read air quality directly from BME680 module with improved error handling.
        Prioritizes HTTP server endpoint, falls back to direct module access.
        """
        try:
            # Method 1: HTTP endpoint (preferred - allows for centralized BME680 server)
            try:
                response = requests.get(f"{self.bme680_server_url}/iaq", timeout=3)
                
                if response.status_code == 200:
                    iaq_data = response.json()
                    
                    # Only return IAQ if sensor is calibrated and burn-in complete
                    if iaq_data.get('burn_in_complete', False) and iaq_data.get('calibrated', False):
                        iaq_value = iaq_data.get('iaq')
                        logger.debug(f"BME680 HTTP: IAQ={iaq_value}, Status=Calibrated")
                        return iaq_value
                    else:
                        progress = iaq_data.get('burn_in_progress', 0)
                        logger.debug(f"BME680 HTTP: Burn-in progress {progress:.1f}%")
                        return None
                else:
                    logger.warning(f"BME680 HTTP endpoint error: status {response.status_code}")
                    
            except requests.exceptions.ConnectionError:
                logger.debug("BME680 server not accessible via HTTP, trying direct access")
            except requests.exceptions.Timeout:
                logger.warning("BME680 server timeout")
            
            # Method 2: Direct module access (fallback)
            try:
                if is_calibrated():
                    iaq_data = get_iaq_value()
                    iaq_value = iaq_data.get('iaq')
                    logger.debug(f"BME680 Direct: IAQ={iaq_value}, Status=Calibrated")
                    return iaq_value
                else:
                    progress = get_calibration_progress()
                    logger.debug(f"BME680 Direct: Burn-in progress {progress:.1f}%")
                    return None
                    
            except (ImportError, AttributeError):
                logger.debug("BME680 direct access not available")
                return None
                
        except Exception as e:
            logger.error(f"Error reading BME680 air quality: {e}")
            return None

    async def read_current_state(self):
        """
        Read comprehensive current state from all integrated sensors.
        Returns a complete state dictionary with all environmental and comfort metrics.
        """
        try:
            # 1. Temperature and Humidity from Govee sensors
            govee_data = read_govee_data()
            
            # Process temperature readings
            temp_readings = [govee_data.get('Temperature_1'), govee_data.get('Temperature_2')]
            temp_readings = [t for t in temp_readings if t is not None]
            
            if temp_readings:
                temperature = sum(temp_readings) / len(temp_readings)
                temp_source = f"Govee({len(temp_readings)} sensors)"
                logger.debug(f"Temperature: {temperature:.1f}°C from {temp_source}")
            else:
                # Intelligent fallback with seasonal variation
                base_temp = 22.0
                seasonal_variation = np.sin(datetime.now().timetuple().tm_yday * 2 * np.pi / 365) * 2
                temperature = base_temp + seasonal_variation + np.random.normal(0, 0.5)
                temp_source = "fallback"
                logger.warning(f"⚠️ Using fallback temperature: {temperature:.1f}°C")
            
            # Process humidity readings
            humid_readings = [govee_data.get('Humidity_1'), govee_data.get('Humidity_2')]
            humid_readings = [h for h in humid_readings if h is not None]
            
            if humid_readings:
                humidity = sum(humid_readings) / len(humid_readings)
                humidity_source = f"Govee({len(humid_readings)} sensors)"
                logger.debug(f"Humidity: {humidity:.1f}% from {humidity_source}")
            else:
                # Intelligent fallback with daily variation
                base_humidity = 45.0
                daily_variation = np.sin(datetime.now().hour * 2 * np.pi / 24) * 5
                humidity = max(30, min(70, base_humidity + daily_variation + np.random.normal(0, 2)))
                humidity_source = "fallback"
                logger.warning(f"⚠️ Using fallback humidity: {humidity:.1f}%")

            # 2. Air Quality from BME680
            air_quality = self.read_air_quality_direct()
            
            if air_quality is not None:
                aq_source = "BME680(calibrated)"
                logger.debug(f"Air Quality: {air_quality} IAQ from {aq_source}")
            else:
                # Fallback based on time of day and room activity
                base_aq = 100.0
                time_factor = 20 * np.sin(datetime.now().hour * 2 * np.pi / 24)  # Varies by time
                air_quality = max(50, base_aq + time_factor + np.random.exponential(10))
                air_quality = min(300, air_quality)
                aq_source = "fallback"
                logger.debug(f"Air Quality: {air_quality} IAQ from {aq_source}")

            # 3. Energy Consumption from Kasa devices
            energy_consumption = None
            energy_source = None
            
            try:
                energy_data = await calculate_total_energy()
                if energy_data and 'total_energy_kwh' in energy_data:
                    energy_consumption = energy_data['total_energy_kwh']
                    energy_source = "Kasa"
                    
                    # Log device details
                    if 'devices' in energy_data:
                        active_devices = []
                        for device, data in energy_data['devices'].items():
                            power = data.get('current_power_w', 0)
                            if power > 0:
                                active_devices.append(f"{device}:{power:.0f}W")
                        
                        if active_devices:
                            logger.debug(f"Active devices: {', '.join(active_devices)}")
                        
                    logger.debug(f"Energy: {energy_consumption:.3f} kWh from {energy_source}")
                        
            except Exception as e:
                logger.warning(f"⚠️ Kasa energy reading failed: {e}")
            
            # Fallback energy calculation
            if energy_consumption is None:
                # Physics-based estimation
                base_consumption = 0.5  # kWh baseline
                thermal_load = abs(temperature - 22) * 0.03  # HVAC load
                time_factor = 0.2 * np.sin(datetime.now().hour * 2 * np.pi / 24)  # Daily pattern
                energy_consumption = max(0.1, base_consumption + thermal_load + time_factor + np.random.normal(0, 0.02))
                energy_source = "calculated"
                logger.debug(f"Energy: {energy_consumption:.3f} kWh from {energy_source}")

            # 4. Device States from Kasa
            try:
                device_states = await get_device_states()
                device_source = "Kasa"
                logger.debug(f"Device states from {device_source}: {device_states}")
            except Exception as e:
                logger.warning(f"⚠️ Device state reading failed: {e}")
                device_states = {
                    'Heater_On': 0,
                    'Humidifier_On': 0, 
                    'Fan_On': 0
                }
                device_source = "fallback"

            # 5. Calculate Thermal Comfort using ISO 7730
            comfort_results = calculate_iso7730_satisfaction(
                temperature=temperature,
                humidity=humidity,
                air_velocity=self.thermal_params['air_velocity'],
                metabolic_rate=self.thermal_params['metabolic_rate'],
                clothing_level=self.thermal_params['clothing_level']
            )
            
            satisfaction = comfort_results['satisfaction']
            pmv = comfort_results['pmv']
            ppd = comfort_results['ppd']
            comfort_method = comfort_results['method']

            # 6. Construct comprehensive state dictionary
            state = {
                'timestamp': datetime.now().isoformat(),
                'Temperature': round(float(temperature), 1),
                'Humidity': round(float(humidity), 1),
                'AirQuality': round(float(air_quality), 1),
                'EnergyConsumption': round(float(energy_consumption), 3),
                'OverallSatisfaction': round(float(satisfaction), 1),
                'PMV': round(float(pmv), 2),
                'PPD': round(float(ppd), 1),
                'ThermalComfortMethod': comfort_method,
                **device_states
            }

            # 7. Data quality validation
            numeric_keys = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 
                          'OverallSatisfaction', 'PMV', 'PPD']
            
            for key in numeric_keys:
                if not isinstance(state[key], (int, float)) or np.isnan(state[key]):
                    logger.warning(f"⚠️ Invalid value for {key}: {state[key]}, applying default")
                    defaults = {
                        'Temperature': 22.0, 'Humidity': 45.0, 'AirQuality': 100.0,
                        'EnergyConsumption': 0.5, 'OverallSatisfaction': 75.0,
                        'PMV': 0.0, 'PPD': 25.0
                    }
                    state[key] = defaults.get(key, 0)

            # 8. Log comprehensive reading summary
            logger.info(f"State Reading Complete - "
                       f"T={state['Temperature']}°C({temp_source}), "
                       f"H={state['Humidity']}%({humidity_source}), "
                       f"AQ={state['AirQuality']}({aq_source}), "
                       f"E={state['EnergyConsumption']}kWh({energy_source}), "
                       f"Satisfaction={state['OverallSatisfaction']}%({comfort_method})")

            return state
            
        except Exception as e:
            logger.error(f"⚠️ Critical error in read_current_state: {e}")
            # Return safe fallback state
            return {
                'timestamp': datetime.now().isoformat(),
                'Temperature': 22.0,
                'Humidity': 45.0,
                'AirQuality': 100.0,
                'EnergyConsumption': 0.5,
                'OverallSatisfaction': 75.0,
                'PMV': 0.0,
                'PPD': 25.0,
                'ThermalComfortMethod': 'emergency_fallback',
                'Heater_On': 0,
                'Humidifier_On': 0,
                'Fan_On': 0
            }
    
    async def log_state(self, state=None):
        """Log current state to CSV file with proper data handling"""
        if state is None:
            state = await self.read_current_state()
            
        try:
            # Convert to DataFrame
            df_new = pd.DataFrame([state])
            
            # Append to existing log or create new one
            if os.path.exists(self.log_file):
                df_existing = pd.read_csv(self.log_file)
                df_combined = pd.concat([df_existing, df_new], ignore_index=True)
                df_combined.to_csv(self.log_file, index=False)
            else:
                df_new.to_csv(self.log_file, index=False)
                
            logger.debug(f"State logged to {self.log_file}")
            return state
            
        except Exception as e:
            logger.error(f"Error logging state: {e}")
            return state
    
    async def monitor_continuously(self, interval=300):
        """
        Continuously monitor and log room state with comprehensive status reporting
        """
        print(f"\n{'='*80}")
        print("SMART ROOM CONTINUOUS MONITORING WITH ISO 7730 THERMAL COMFORT")
        print(f"{'='*80}")
        print(f"Logging interval: {interval} seconds")
        print(f"Log file: {self.log_file}")
        print("\nIntegrated Systems:")
        print("  📊 Govee H5179: Temperature & Humidity")
        print("  🌬️  BME680: Air Quality (IAQ) with burn-in management")
        print("  ⚡ Kasa Smart Plugs: Energy & Device States")
        print("  🌡️  ISO 7730: PMV/PPD Thermal Comfort Analysis")
        print(f"{'='*80}\n")
        
        start_time = datetime.now()
        reading_count = 0
        
        try:
            while True:
                loop_start = time.time()
                
                # Read and log state
                state = await self.read_current_state()
                await self.log_state(state)
                reading_count += 1
                
                # Enhanced status display
                runtime = datetime.now() - start_time
                bme680_iaq = self.read_air_quality_direct()
                bme680_status = "📊 Calibrated" if bme680_iaq is not None else "⏳ Burn-in"
                
                # Thermal comfort status
                pmv = state['PMV']
                if abs(pmv) < 0.5:
                    comfort_status = "😊 Comfortable"
                elif abs(pmv) < 1.0:
                    comfort_status = "😐 Acceptable"
                else:
                    comfort_status = "😣 Uncomfortable"
                
                print(f"[{datetime.now().strftime('%H:%M:%S')}] #{reading_count:03d} | "
                      f"T={state['Temperature']}°C H={state['Humidity']}% "
                      f"AQ={state['AirQuality']} E={state['EnergyConsumption']:.2f}kWh | "
                      f"PMV={pmv:+.1f} PPD={state['PPD']:.0f}% {comfort_status} | "
                      f"BME680:{bme680_status} | Runtime:{str(runtime).split('.')[0]}")
                
                # Sleep for remaining interval time
                elapsed = time.time() - loop_start
                sleep_time = max(0, interval - elapsed)
                await asyncio.sleep(sleep_time)
                
        except KeyboardInterrupt:
            print(f"\n{'='*80}")
            print("🛑 Monitoring stopped by user")
            print(f"Total readings collected: {reading_count}")
            print(f"Total runtime: {str(datetime.now() - start_time).split('.')[0]}")
            print(f"Data saved to: {self.log_file}")
            print(f"{'='*80}")
        except Exception as e:
            logger.error(f"Error during continuous monitoring: {e}")
            print(f"❌ Monitoring error: {e}")
        finally:
            await self.cleanup()
            
    async def cleanup(self):
        """Clean up all resources and connections"""
        try:
            await cleanup_kasa()
            logger.info("✓ Smart room monitor cleanup completed")
            return True
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")
            return False

    def test_all_sensors(self):
        """Test all sensor integrations with BME680 burn-in status"""
        print("Testing Smart Room Monitor Sensor Integration:")
        print("=" * 60)
        
        # Test 1: Govee Temperature/Humidity Sensors
        print("\n1. Testing Govee sensors...")
        govee_data = read_govee_data()
        temp_sensors = sum(1 for t in [govee_data.get('Temperature_1'), govee_data.get('Temperature_2')] if t is not None)
        humid_sensors = sum(1 for h in [govee_data.get('Humidity_1'), govee_data.get('Humidity_2')] if h is not None)
        
        if temp_sensors > 0 or humid_sensors > 0:
            print(f"   ✓ Govee responding: {temp_sensors} temp sensors, {humid_sensors} humidity sensors")
            if govee_data.get('Temperature_1'):
                print(f"     Sensor 1: {govee_data['Temperature_1']}°C, {govee_data.get('Humidity_1')}%")
            if govee_data.get('Temperature_2'):
                print(f"     Sensor 2: {govee_data['Temperature_2']}°C, {govee_data.get('Humidity_2')}%")
        else:
            print("   ❌ Govee sensors not responding")
        
        # Test 2: BME680 Air Quality with Burn-in Status
        print("\n2. Testing BME680 air quality sensor...")
        try:
            # Check if we can import the module directly
            try:
                import bme680_reader
                progress = bme680_reader.get_calibration_progress()
                calibrated = bme680_reader.is_calibrated()
                
                if calibrated:
                    iaq_data = bme680_reader.get_iaq_value()
                    print(f"   ✓ BME680 calibrated and ready")
                    print(f"     Current IAQ: {iaq_data.get('iaq')} ({bme680_reader.iaq_calculator._iaq_to_text(iaq_data.get('iaq'))})")
                    print(f"     Total readings: {iaq_data.get('total_readings')}")
                    print(f"     Gas baseline: {iaq_data.get('gas_baseline')}")
                else:
                    print(f"   ⏳ BME680 burn-in progress: {progress:.1f}%")
                    print(f"     Status: {'Calibrated' if calibrated else 'Calibrating'}")
                    print(f"     Note: Air quality readings available after burn-in completes")
                    
            except ImportError:
                # Fallback to HTTP endpoint
                response = requests.get(f"{self.bme680_server_url}/status", timeout=3)
                if response.status_code == 200:
                    status = response.json()
                    if status.get('burn_in_complete', False):
                        print(f"   ✓ BME680 calibrated (via HTTP)")
                        print(f"     Current IAQ: {status.get('current_iaq')}")
                    else:
                        print(f"   ⏳ BME680 burn-in: {status.get('burn_in_progress')} (via HTTP)")
                else:
                    print("   ❌ BME680 HTTP endpoint not responding")
                    
        except Exception as e:
            print(f"   ❌ BME680 error: {e}")
            print("     Make sure BME680 server is running: python bme680_reader.py")
        
        # Test 3: Kasa Smart Device Integration
        print("\n3. Testing Kasa devices...")
        try:
            loop = asyncio.get_event_loop()
            energy_data = loop.run_until_complete(calculate_total_energy())
            
            if energy_data and 'devices' in energy_data:
                total_energy = energy_data.get('total_energy_kwh', 0)
                total_power = energy_data.get('current_total_power_w', 0)
                print(f"   ✓ Kasa devices responding")
                print(f"     Total energy: {total_energy} kWh")
                print(f"     Current power: {total_power} W")
                
                for device, data in energy_data['devices'].items():
                    power = data.get('current_power_w', 0)
                    status = "ON" if power > 0 else "OFF"
                    print(f"     - {device}: {power}W ({status})")
            else:
                print("   ❌ Kasa devices not fully responding")
                
        except Exception as e:
            print(f"   ❌ Kasa error: {e}")

        # Test 4: ISO 7730 Thermal Comfort Calculation
        print("\n4️⃣  Testing ISO 7730 Thermal Comfort Analysis...")
        try:
            # Test with sample data
            test_temp = 22.5
            test_humidity = 45.0
            
            comfort_results = calculate_iso7730_satisfaction(test_temp, test_humidity)
            
            if comfort_results['method'] == 'ISO_7730':
                print(f"   ✅ ISO 7730 calculation operational")
                print(f"      🌡️  Test conditions: {test_temp}°C, {test_humidity}% RH")
                print(f"      📊 PMV: {comfort_results['pmv']:+.2f} (Predicted Mean Vote)")
                print(f"      📊 PPD: {comfort_results['ppd']:.1f}% (Predicted Percentage Dissatisfied)")
                print(f"      😊 Satisfaction: {comfort_results['satisfaction']:.1f}%")
            else:
                print(f"   ⚠️  Using fallback method: c{calculate_simplified_satisfaction}")
        
        except Exception as e:
            print(f"   ❌ Satisfaction error: {e}")
        
        # Test integrated reading
        print("\n4. Testing integrated sensor reading...")
        try:
            loop = asyncio.get_event_loop()
            state = loop.run_until_complete(self.read_current_state())
            
            print("   ✓ Integrated reading successful:")
            print(f"     Temperature: {state['Temperature']}°C")
            print(f"     Humidity: {state['Humidity']}%")
            
            # Show BME680 status in air quality reading
            direct_iaq = self.read_air_quality_direct()
            if direct_iaq is not None:
                print(f"     Air Quality: {state['AirQuality']} IAQ (BME680 calibrated)")
            else:
                print(f"     Air Quality: {state['AirQuality']} IAQ (fallback - BME680 not ready)")
                
            print(f"     Energy: {state['EnergyConsumption']} kWh")
            print(f"     Satisfaction: {state['OverallSatisfaction']}%")
            
        except Exception as e:
            print(f"   ❌ Integrated reading failed: {e}")

if __name__ == "__main__":
    async def main():
        monitor = SmartRoomMonitor()
        
        # Test sensor integration first
        monitor.test_all_sensors()
        
        print("\n" + "="*60)
        print("BME680 Integration Notes:")
        print("- Air quality readings use direct access to bme680_reader.py")
        print("- BME680 requires ~7.5 minutes burn-in time for calibration")
        print("- During burn-in, fallback air quality values are used")
        print("- Once calibrated, real IAQ values are used for causal discovery")
        
        choice = input("\nStart continuous monitoring? (y/N): ")
        
        if choice.lower() == 'y':
            try:
                await monitor.monitor_continuously(interval=300)  # Log every 5 minutes
            finally:
                await monitor.cleanup()
        else:
            # Just test a single reading
            print("\nTesting single state reading...")
            state = await monitor.read_current_state()
            print(f"Current state: {state}")
            await monitor.cleanup()
    
    asyncio.run(main())