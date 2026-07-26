import os
import json
import logging
import tempfile
from pathlib import Path
import pandas as pd
from eppy.modeleditor import IDF
import subprocess
import platform
import shutil
import time
import numpy as np

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

class EnergyPlusInterface:
    def __init__(self, idf_template_path, weather_file, eplus_path=None):
        """Initialize EnergyPlus interface with specified paths
        
        Args:
            idf_template_path (str): Path to IDF template file
            weather_file (str): Path to EPW weather file
            eplus_path (str, optional): Path to EnergyPlus installation
        """
        self.data = None
        self.initialized = False
        self.load_lookup_data()

    def load_lookup_data(self):
        """Load pre-calculated EnergyPlus results"""
        try:
            csv_path = os.path.join(os.path.dirname(__file__), 'epluszsz.csv')
            if not os.path.exists(csv_path):
                logger.warning(f"EnergyPlus CSV data not found at {csv_path}")
                return False
                
            self.data = pd.read_csv(csv_path)
            logger.info(f"Loaded EnergyPlus data with {len(self.data)} records")
            self.initialized = True
            return True
        except Exception as e:
            logger.error(f"Error loading EnergyPlus data: {e}")
            return False
            
    def _find_energyplus(self):
        """Find EnergyPlus installation directory"""
        system = platform.system().lower()
        current_dir = Path.cwd().absolute()
        
        possible_paths = []
        if system == "windows":
            possible_paths = [
                Path("C:/EnergyPlus-24.2.0").absolute(),
                Path(os.environ.get("PROGRAMFILES", "C:/Program Files")).absolute() / "EnergyPlus-24.2.0"
            ]
        elif system == "darwin":
            possible_paths = [
                Path("/Applications/EnergyPlus-24-2-0"),
                Path.home() / "Applications/EnergyPlus-24-2-0",
                Path.cwd() / "EnergyPlus-24.2.0"
        ]
        else:  # Linux
            possible_paths = [
                Path("/usr/local/EnergyPlus-24.2.0").absolute(),
                Path.home().absolute() / "EnergyPlus-24.2.0"
            ]
        
        possible_paths.append(current_dir / "EnergyPlus-24.2.0")
        
        logger.debug(f"Searching EnergyPlus in paths: {possible_paths}")
        
        for path in possible_paths:
            exe_name = "energyplus.exe" if system == "windows" else "energyplus"
            exe_path = path / exe_name
            idd_path = path / "Energy+.idd"
            
            if path.exists() and idd_path.exists() and exe_path.exists():
                logger.debug(f"Found valid EnergyPlus at: {path}")
                return path
                    
        logger.error("EnergyPlus installation not found")
        return None
        
    def _init_energyplus(self):
        """Initialize EnergyPlus with IDF template"""
        # Set IDD file location
        idd_path = self.eplus_path / "Energy+.idd"
        if not idd_path.exists():
            raise FileNotFoundError(f"Energy+.idd not found at {idd_path}")
            
        IDF.setiddname(str(idd_path))
        
        # Load IDF template
        if not self.idf_template_path.exists():
            raise FileNotFoundError(f"IDF template not found at {self.idf_template_path}")
            
        self.idf = IDF(str(self.idf_template_path))
        
        # Verify weather file
        if not self.weather_file.exists():
            raise FileNotFoundError(f"Weather file not found at {self.weather_file}")
            
        # Initialize core components
        self._setup_thermal_zones()
        self._create_schedules()
        
    def _ensure_executable_permissions(self):
        """Ensure EnergyPlus executable has proper permissions"""
        if platform.system().lower() == "darwin":
            eplus_exe = self.eplus_path / "energyplus"
            try:
                # Check if file is executable
                if not os.access(str(eplus_exe), os.X_OK):
                    logger.warning(f"Setting executable permissions on {eplus_exe}")
                    subprocess.run(["chmod", "+x", str(eplus_exe)], check=True)
            except Exception as e:
                logger.error(f"Failed to set permissions: {e}")
    
    def test_energyplus_installation(self):
        """Test if EnergyPlus is properly installed and accessible"""
        try:
            # Try using PATH first
            result = subprocess.run(["energyplus", "--version"], 
                                capture_output=True, text=True, timeout=5)
            
            if result.returncode == 0:
                logger.info(f"EnergyPlus found in PATH: {result.stdout.strip()}")
                return True
                
            # Try with full path
            eplus_exe = self.eplus_path / "energyplus"
            result = subprocess.run([str(eplus_exe), "--version"], 
                                capture_output=True, text=True, timeout=5)
                                
            if result.returncode == 0:
                logger.info(f"EnergyPlus found at {eplus_exe}: {result.stdout.strip()}")
                return True
                
            logger.error("EnergyPlus not accessible. Check installation.")
            return False
        except Exception as e:
            logger.error(f"Error testing EnergyPlus: {e}")
            return False
    
    def _setup_thermal_zones(self):
        """Set up thermal zone parameters"""
        # Get or create main zone
        zones = self.idf.idfobjects['ZONE']
        if not zones:
            zone = self.idf.newidfobject('ZONE')
            zone.Name = "MAIN_ZONE"
        else:
            zone = zones[0]
            
        # Set up zone thermostat
        thermostats = self.idf.idfobjects['ZONECONTROL:THERMOSTAT']
        if not thermostats:
            thermostat = self.idf.newidfobject('ZONECONTROL:THERMOSTAT')
            thermostat.Name = "ZONE_THERMOSTAT"
            thermostat.Zone_or_ZoneList_Name = zone.Name
            thermostat.Control_Type_Schedule_Name = "HEATING_SETPOINT_SCHEDULE"
            thermostat.Control_1_Name = "Dual Setpoint"
            thermostat.Control_1_Object_Type = "ThermostatSetpoint:DualSetpoint"
            
        # Set up HVAC template
        hvac = self.idf.idfobjects.get('HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM', [])
        if not hvac:
            hvac = self.idf.newidfobject('HVACTEMPLATE:ZONE:IDEALLOADSAIRSYSTEM')
            hvac.Zone_Name = zone.Name
            hvac.Template_Thermostat_Name = "ZONE_THERMOSTAT"
            
    def _create_schedules(self):
        """Create required schedules"""
        # Create schedule type limits
        if not self.idf.idfobjects.get('SCHEDULETYPELIMITS'):
            self.idf.newidfobject('SCHEDULETYPELIMITS',
                Name='Temperature',
                Lower_Limit_Value=18,
                Upper_Limit_Value=30,
                Numeric_Type='Continuous',
                Unit_Type='Temperature')
                
            self.idf.newidfobject('SCHEDULETYPELIMITS',
                Name='Humidity',
                Lower_Limit_Value=30,
                Upper_Limit_Value=70,
                Numeric_Type='Continuous')
                
        # Create temperature schedules
        self._create_or_update_schedule(
            'HEATING_SETPOINT_SCHEDULE',
            'Temperature',
            22.0
        )
        
        self._create_or_update_schedule(
            'COOLING_SETPOINT_SCHEDULE',
            'Temperature',
            22.0
        )
        
        # Create humidity schedule
        self._create_or_update_schedule(
            'HUMIDITY_SCHEDULE',
            'Humidity',
            50.0
        )
        
    def _create_or_update_schedule(self, name, type_limits, value):
        """Create or update a schedule"""
        schedule = self.idf.getobject('SCHEDULE:COMPACT', name)
        if not schedule:
            schedule = self.idf.newidfobject('SCHEDULE:COMPACT',
                Name=name,
                Schedule_Type_Limits_Name=type_limits,
                Field_1='Through: 12/31',
                Field_2='For: AllDays',
                Field_3='Until: 24:00',
                Field_4=value)
        else:
            schedule.Field_4 = value
            
    def update_zone_conditions(self, temperature, humidity, air_quality):
        """Update zone conditions and return energy consumption
        
        Args:
            temperature (float): Temperature in Celsius
            humidity (float): Relative humidity percentage
            air_quality (float): Air quality index
            
        Returns:
            dict: Calculated results including energy consumption
        """
        if not self.initialized:
            return self._calculate_fallback(temperature, humidity, air_quality)
            
        try:
            # Calculate distances to each data point
            cooling_temp_col = 'NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Temperature [C]'
            cooling_humid_col = 'NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Relative Humidity [%]'
            
            # Sort by the closest temperature/humidity match
            self.data['distance'] = np.sqrt(
                (self.data[cooling_temp_col] - temperature)**2 + 
                (self.data[cooling_humid_col] - humidity)**2
            )
            
            closest = self.data.sort_values('distance').iloc[0]
            
            heating_load_col = 'NORTH_ZONE:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Des Heat Load [W]'
            cooling_load_col = 'NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Des Sens Cool Load [W]'
            
            total_load = closest[heating_load_col] + closest[cooling_load_col]
            max_load = 22753.57  # Maximum load from data
            
            energy_percent = min(100, (total_load / max_load) * 100)
            
            return {
                'EnergyConsumption': float(energy_percent),
                'Temperature': float(temperature),
                'Humidity': float(humidity),
                'AirQuality': float(air_quality)
            }
            
        except Exception as e:
            logger.error(f"Error calculating energy: {e}")
            return self._calculate_fallback(temperature, humidity, air_quality)
        
    def run_simulation(self, temperature, humidity, air_quality):
        """Run EnergyPlus simulation (now just uses lookup)"""
        return self.update_zone_conditions(temperature, humidity, air_quality)
            
    def _parse_results(self, meter_file):
        """Parse EnergyPlus output files"""
        try:
            # Read meter file with more flexible column handling
            df = pd.read_csv(meter_file, skiprows=1, error_bad_lines=False)
            
            # Get energy columns with flexible naming
            energy_cols = [col for col in df.columns if any(x in col.lower() for x in ['electricity', 'heating', 'cooling'])]
            
            if not energy_cols:
                logger.warning("No energy columns found, using fallback")
                return self._calculate_fallback(
                    self.current_conditions['temperature'],
                    self.current_conditions['humidity'],
                    self.current_conditions['air_quality']
                )

            # Sum available energy columns
            results = {}
            for col in energy_cols:
                try:
                    results[col.split(':')[0].lower()] = df[col].sum()
                except Exception as e:
                    logger.warning(f"Error processing column {col}: {e}")
                    results[col.split(':')[0].lower()] = 0

            return results

        except Exception as e:
            logger.warning(f"Error parsing results: {e}")
            return self._calculate_fallback(
                self.current_conditions['temperature'],
                self.current_conditions['humidity'],
                self.current_conditions['air_quality']
            )
            
    def _calculate_energy_consumption(self, results):
        """Calculate total energy consumption percentage"""
        try:
            # Sum available energy values with fallbacks
            total_energy = sum(results.get(k, 0) for k in ['electricity', 'heating', 'cooling'])
            
            # Convert to percentage
            typical_max = 1e9  # 1 GJ reference
            energy_percent = (total_energy / typical_max) * 100
            
            return min(100, max(0, energy_percent))
            
        except Exception as e:
            logger.warning(f"Error calculating energy consumption: {e}")
            return self._calculate_fallback(
                self.current_conditions['temperature'],
                self.current_conditions['humidity'],
                self.current_conditions['air_quality']
            )['EnergyConsumption']
            
    def _calculate_fallback(self, temperature, humidity, air_quality):
        """Fallback calculation when CSV lookup fails"""
        temp_diff = abs(temperature - 22)
        humidity_diff = abs(humidity - 50)
        air_quality_diff = max(0, 100 - air_quality / 5)
        
        base_energy = 30 + (temp_diff * 2) + (humidity_diff * 0.5) + (air_quality_diff * 0.2)
        return {
            'EnergyConsumption': float(min(100, max(0, base_energy))),
            'Temperature': float(temperature),
            'Humidity': float(humidity),
            'AirQuality': float(air_quality)
        }
            
    def cleanup(self):
        """Clean up resources"""
        self.data = None