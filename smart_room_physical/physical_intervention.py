#!/usr/bin/env python3
"""
Physical Intervention Controller with ISO 7730 Thermal Comfort Integration
Performs physical interventions while monitoring scientific thermal satisfaction
"""

import asyncio
import time
import json
import pandas as pd
import numpy as np
from datetime import datetime
import os
import logging

from smart_room_physical.smart_room_monitor import SmartRoomMonitor, calculate_iso7730_satisfaction
from smart_room_physical.kasa_controller import control_heater, control_humidifier, control_fan

logger = logging.getLogger(__name__)

class PhysicalInterventionController:
    """
    Physical intervention controller with ISO 7730 thermal comfort integration.
    
    Performs interventions on physical devices while monitoring:
    - ISO 7730 PMV/PPD thermal comfort metrics
    - Real-time environmental changes
    - Scientific satisfaction calculations
    """
    
    def __init__(self, log_file="iso7730_interventions_log.csv"):
        self.log_file = log_file
        self.monitor = SmartRoomMonitor()
        
        # Timing configurations (in seconds)
        self.initial_effect_delay = 30  # Time for initial environment changes
        self.stabilization_time = 120   # Total time for environment to stabilize
        
        # ISO 7730 thermal comfort parameters
        self.thermal_params = {
            'air_velocity': 0.1,      # m/s (typical office environment)
            'metabolic_rate': 1.2,    # met (sedentary office work)
            'clothing_level': 0.6     # clo (typical indoor clothing)
        }
        
        self.initialize_log()
        
        # Map variables to intervention functions
        self.intervention_map = {
            'Temperature': {
                'increase': self._heat_room,
                'decrease': self._cool_room,
                'set': self._binary_control_temperature
            },
            'Humidity': {
                'increase': self._humidify_room,
                'decrease': self._dry_room,
                'set': self._binary_control_humidity
            },
            'AirQuality': {
                'increase': self._improve_air,
                'decrease': self._worsen_air,
                'set': self._binary_control_air
            }
        }
    
    def initialize_log(self):
        """Initialize intervention log with ISO 7730 metrics"""
        if not os.path.exists(self.log_file):
            columns = [
                'timestamp', 'variable', 'action', 'target_value',
                'pre_temperature', 'pre_humidity', 'pre_air_quality', 'pre_energy',
                'post_temperature', 'post_humidity', 'post_air_quality', 'post_energy',
                'pre_satisfaction_iso7730', 'post_satisfaction_iso7730',
                'pre_pmv', 'post_pmv', 'pre_ppd', 'post_ppd',
                'thermal_comfort_method', 'air_velocity', 'metabolic_rate', 'clothing_level',
                'satisfaction_change', 'pmv_change', 'ppd_change',
                'success', 'duration_seconds'
            ]
            pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
            logger.info(f"Created ISO 7730 intervention log: {self.log_file}")
    
    def _calculate_iso7730_satisfaction(self, state):
        """Calculate comprehensive satisfaction using ISO 7730 + air quality"""
        temp = state.get('Temperature', 22)
        humidity = state.get('Humidity', 50)
        air_quality = state.get('AirQuality', 100)
        
        # Use ISO 7730 for thermal comfort
        comfort_results = calculate_iso7730_satisfaction(
            temperature=temp,
            humidity=humidity,
            air_velocity=self.thermal_params['air_velocity'],
            metabolic_rate=self.thermal_params['metabolic_rate'],
            clothing_level=self.thermal_params['clothing_level']
        )
        
        # Factor in air quality (ISO 7730 doesn't include air quality)
        thermal_satisfaction = comfort_results['satisfaction']
        air_comfort = max(0, 100 - air_quality * 0.2)  # Lower AQI is better
        
        # Combine thermal and air quality satisfaction
        overall_satisfaction = (thermal_satisfaction * 0.8 + air_comfort * 0.2)
        
        return {
            'satisfaction': max(0, min(100, overall_satisfaction)),
            'thermal_satisfaction': thermal_satisfaction,
            'pmv': comfort_results['pmv'],
            'ppd': comfort_results['ppd'],
            'method': comfort_results['method']
        }
    
    async def _heat_room(self, value=None):
        """Turn heater on, turn fan off"""
        await control_fan(False)
        success = await control_heater(True)
        if success:
            logger.info("🔥 Heating activated: Heater ON, Fan OFF")
        return success
    
    async def _cool_room(self, value=None):
        """Turn heater off, turn fan on"""
        await control_heater(False)
        success = await control_fan(True)
        if success:
            logger.info("❄️ Cooling activated: Heater OFF, Fan ON")
        return success
    
    async def _binary_control_temperature(self, target_value):
        """Convert 'set' command to appropriate binary controls"""
        current_state = await self.monitor.read_current_state()
        current_temp = current_state['Temperature']
        
        if current_temp is None:
            return False
            
        if target_value > current_temp:
            return await self._heat_room()
        else:
            return await self._cool_room()
    
    async def _humidify_room(self, value=None):
        """Turn humidifier on"""
        success = await control_humidifier(True)
        if success:
            logger.info("💧 Humidification activated: Humidifier ON")
        return success
    
    async def _dry_room(self, value=None):
        """Turn humidifier off, turn fan on to circulate air"""
        await control_humidifier(False)
        success = await control_fan(True)
        if success:
            logger.info("🌬️ Dehumidification activated: Humidifier OFF, Fan ON")
        return success
    
    async def _binary_control_humidity(self, target_value):
        """Convert 'set' command to appropriate binary controls"""
        current_state = await self.monitor.read_current_state()
        current_humidity = current_state['Humidity']
        
        if current_humidity is None:
            return False
            
        if target_value > current_humidity:
            return await self._humidify_room()
        else:
            return await self._dry_room()
    
    async def _improve_air(self, value=None):
        """Turn on fan to improve air circulation"""
        success = await control_fan(True)
        if success:
            logger.info("🌪️ Air quality improvement: Fan ON for circulation")
        return success
    
    async def _worsen_air(self, value=None):
        """We don't actively worsen air quality for safety"""
        logger.info("⚠️ Air quality worsening skipped for safety")
        return True
    
    async def _binary_control_air(self, target_value):
        """Convert 'set' command to appropriate binary controls"""
        current_state = await self.monitor.read_current_state()
        current_aq = current_state['AirQuality']
        
        if current_aq is None:
            return False
            
        if target_value < current_aq:  # Lower AQI is better
            return await self._improve_air()
        return True
    
    async def perform_intervention(self, variable, action, value=None):
        """
        Perform physical intervention with comprehensive ISO 7730 satisfaction tracking.
        
        Args:
            variable (str): Variable to intervene on ('Temperature', 'Humidity', 'AirQuality')
            action (str): Action to take ('increase', 'decrease', 'set')
            value (float, optional): Target value for 'set' actions
            
        Returns:
            dict: Comprehensive intervention results with ISO 7730 metrics
        """
        start_time = time.time()
        logger.info(f"🔬 Starting ISO 7730 intervention: {variable} {action}")
        
        # Record pre-intervention state with ISO 7730 calculation
        logger.info("📊 Recording pre-intervention state...")
        pre_state = await self.monitor.read_current_state()
        pre_comfort = self._calculate_iso7730_satisfaction(pre_state)
        
        logger.info(f"Pre-intervention: T={pre_state['Temperature']}°C, "
                   f"H={pre_state['Humidity']}%, AQ={pre_state['AirQuality']}, "
                   f"Satisfaction={pre_comfort['satisfaction']:.1f}% "
                   f"(PMV={pre_comfort['pmv']:+.2f}, PPD={pre_comfort['ppd']:.1f}%)")
        
        # Perform device intervention
        success = False
        if variable in self.intervention_map and action in self.intervention_map[variable]:
            success = await self.intervention_map[variable][action](value)
        else:
            logger.error(f"❌ Invalid intervention: {variable} {action}")
            return None
        
        if not success:
            logger.error(f"❌ Device control failed for {variable} {action}")
            return None
        
        # Wait for initial effects with progress updates
        logger.info(f"⏳ Waiting {self.initial_effect_delay}s for initial environmental changes...")
        for i in range(self.initial_effect_delay):
            if i % 10 == 0:  # Progress update every 10 seconds
                remaining = self.initial_effect_delay - i
                logger.info(f"   {remaining}s remaining for initial effects...")
            await asyncio.sleep(1)
        
        # Measure intermediate state
        intermediate_state = await self.monitor.read_current_state()
        intermediate_comfort = self._calculate_iso7730_satisfaction(intermediate_state)
        
        logger.info(f"📈 Intermediate state ({self.initial_effect_delay}s): "
                   f"T={intermediate_state['Temperature']}°C, "
                   f"H={intermediate_state['Humidity']}%, "
                   f"AQ={intermediate_state['AirQuality']}, "
                   f"Satisfaction={intermediate_comfort['satisfaction']:.1f}% "
                   f"(PMV={intermediate_comfort['pmv']:+.2f})")
        
        # Wait for full stabilization
        remaining_time = self.stabilization_time - self.initial_effect_delay
        logger.info(f"⏳ Waiting additional {remaining_time}s for full environmental stabilization...")
        
        for i in range(remaining_time):
            if i % 30 == 0:  # Progress update every 30 seconds
                remaining = remaining_time - i
                logger.info(f"   {remaining}s remaining for stabilization...")
            await asyncio.sleep(1)
        
        # Record post-intervention state with ISO 7730 calculation
        logger.info("📊 Recording post-intervention state...")
        post_state = await self.monitor.read_current_state()
        post_comfort = self._calculate_iso7730_satisfaction(post_state)
        
        # Calculate changes
        satisfaction_change = post_comfort['satisfaction'] - pre_comfort['satisfaction']
        pmv_change = post_comfort['pmv'] - pre_comfort['pmv']
        ppd_change = post_comfort['ppd'] - pre_comfort['ppd']
        duration = time.time() - start_time
        
        logger.info(f"📊 Post-intervention: T={post_state['Temperature']}°C, "
                   f"H={post_state['Humidity']}%, AQ={post_state['AirQuality']}, "
                   f"Satisfaction={post_comfort['satisfaction']:.1f}% "
                   f"(PMV={post_comfort['pmv']:+.2f}, PPD={post_comfort['ppd']:.1f}%)")
        
        logger.info(f"📈 Changes: Satisfaction={satisfaction_change:+.1f}%, "
                   f"PMV={pmv_change:+.2f}, PPD={ppd_change:+.1f}%")
        
        # Log comprehensive intervention data
        intervention_log = {
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'action': action,
            'target_value': value,
            
            # Pre-intervention environmental data
            'pre_temperature': pre_state['Temperature'],
            'pre_humidity': pre_state['Humidity'],
            'pre_air_quality': pre_state['AirQuality'],
            'pre_energy': pre_state['EnergyConsumption'],
            
            # Post-intervention environmental data
            'post_temperature': post_state['Temperature'],
            'post_humidity': post_state['Humidity'],
            'post_air_quality': post_state['AirQuality'],
            'post_energy': post_state['EnergyConsumption'],
            
            # ISO 7730 thermal comfort metrics
            'pre_satisfaction_iso7730': pre_comfort['satisfaction'],
            'post_satisfaction_iso7730': post_comfort['satisfaction'],
            'pre_pmv': pre_comfort['pmv'],
            'post_pmv': post_comfort['pmv'],
            'pre_ppd': pre_comfort['ppd'],
            'post_ppd': post_comfort['ppd'],
            'thermal_comfort_method': post_comfort['method'],
            
            # Thermal comfort parameters
            'air_velocity': self.thermal_params['air_velocity'],
            'metabolic_rate': self.thermal_params['metabolic_rate'],
            'clothing_level': self.thermal_params['clothing_level'],
            
            # Change metrics
            'satisfaction_change': satisfaction_change,
            'pmv_change': pmv_change,
            'ppd_change': ppd_change,
            
            # Intervention metadata
            'success': success,
            'duration_seconds': duration
        }
        
        # Append to log
        pd.DataFrame([intervention_log]).to_csv(self.log_file, mode='a', header=False, index=False)
        
        # Add calculated satisfaction to states for return data
        pre_state['OverallSatisfaction'] = pre_comfort['satisfaction']
        post_state['OverallSatisfaction'] = post_comfort['satisfaction']
        
        # Summary log message
        effect_magnitude = "significant" if abs(satisfaction_change) > 5 else "moderate" if abs(satisfaction_change) > 2 else "minimal"
        effect_direction = "improved" if satisfaction_change > 0 else "decreased" if satisfaction_change < 0 else "unchanged"
        
        logger.info(f"✅ ISO 7730 intervention completed successfully!")
        logger.info(f"   Duration: {duration:.0f}s")
        logger.info(f"   Environmental change: {variable} "
                   f"{pre_state[variable]:.1f} → {post_state[variable]:.1f}")
        logger.info(f"   Thermal comfort: {effect_magnitude} {effect_direction} "
                   f"({satisfaction_change:+.1f}% satisfaction)")
        logger.info(f"   PMV shift: {pre_comfort['pmv']:+.2f} → {post_comfort['pmv']:+.2f}")
        
        return {
            'preInterventionData': pre_state,
            'postInterventionData': post_state,
            'pre_comfort_metrics': pre_comfort,
            'post_comfort_metrics': post_comfort,
            'intervention_metadata': {
                'variable': variable,
                'action': action,
                'target_value': value,
                'duration_seconds': duration,
                'satisfaction_change': satisfaction_change,
                'pmv_change': pmv_change,
                'ppd_change': ppd_change,
                'thermal_method': post_comfort['method']
            }
        }

async def test_iso7730_intervention():
    """Test ISO 7730 intervention system with real devices"""
    print("🧪 Testing ISO 7730 Physical Intervention System")
    print("=" * 60)
    
    controller = PhysicalInterventionController()
    
    # Safety warning
    print("⚠️  WARNING: This will control real devices!")
    print("   - Heater, humidifier, and fan will be activated")
    print("   - Room temperature and humidity will change")
    print("   - ISO 7730 thermal comfort will be monitored")
    
    confirm = input("\nProceed with physical intervention test? (y/N): ")
    if confirm.lower() != 'y':
        print("❌ Test cancelled by user")
        return
    
    try:
        # Test 1: Temperature increase intervention
        print("\n🔥 Test 1: Temperature increase intervention")
        print("   This will activate the heater and monitor thermal comfort...")
        
        result = await controller.perform_intervention('Temperature', 'increase')
        
        if result:
            pre_comfort = result['pre_comfort_metrics']
            post_comfort = result['post_comfort_metrics']
            
            print(f"   ✅ Temperature intervention completed:")
            print(f"      Temperature: {result['preInterventionData']['Temperature']:.1f}°C → "
                  f"{result['postInterventionData']['Temperature']:.1f}°C")
            print(f"      ISO 7730 Satisfaction: {pre_comfort['satisfaction']:.1f}% → "
                  f"{post_comfort['satisfaction']:.1f}%")
            print(f"      PMV: {pre_comfort['pmv']:+.2f} → {post_comfort['pmv']:+.2f}")
            print(f"      PPD: {pre_comfort['ppd']:.1f}% → {post_comfort['ppd']:.1f}%")
        else:
            print("   ❌ Temperature intervention failed")
        
        # Wait between tests
        print("\n⏳ Waiting 60s before next test...")
        await asyncio.sleep(60)
        
        # Test 2: Humidity increase intervention
        print("\n💧 Test 2: Humidity increase intervention")
        print("   This will activate the humidifier and monitor comfort...")
        
        result = await controller.perform_intervention('Humidity', 'increase')
        
        if result:
            pre_comfort = result['pre_comfort_metrics']
            post_comfort = result['post_comfort_metrics']
            
            print(f"   ✅ Humidity intervention completed:")
            print(f"      Humidity: {result['preInterventionData']['Humidity']:.1f}% → "
                  f"{result['postInterventionData']['Humidity']:.1f}%")
            print(f"      ISO 7730 Satisfaction: {pre_comfort['satisfaction']:.1f}% → "
                  f"{post_comfort['satisfaction']:.1f}%")
            print(f"      PMV: {pre_comfort['pmv']:+.2f} → {post_comfort['pmv']:+.2f}")
        else:
            print("   ❌ Humidity intervention failed")
        
        print(f"\n📊 Intervention log saved to: {controller.log_file}")
        print("✅ ISO 7730 intervention testing completed successfully!")
        
    except Exception as e:
        print(f"❌ Test error: {e}")
        logger.error(f"Test error: {e}")
    finally:
        await controller.monitor.cleanup()

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)
    
    asyncio.run(test_iso7730_intervention())