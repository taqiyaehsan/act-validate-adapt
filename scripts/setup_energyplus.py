import os
import shutil
import urllib.request
import zipfile
import platform
import subprocess
from pathlib import Path

def setup_energyplus():
    # Create necessary directories
    base_dir = Path.cwd()
    templates_dir = base_dir / "templates"
    weather_dir = base_dir / "weather"
    eplus_dir = base_dir / "EnergyPlus-24.2.0"
    
    os.makedirs(templates_dir, exist_ok=True)
    os.makedirs(weather_dir, exist_ok=True)
    
    # Save IDF template
    idf_path = templates_dir / "smart_room.idf"
    with open(idf_path, 'w') as f:
        f.write('''
!- EnergyPlus IDF template for Smart Room
!- Generated for EnergyPlus Version 24.2.0

Version,24.2.0;

!- Building Configuration
Building,
    Smart Room,               !- Name
    0.0,                     !- North Axis {deg}
    City,                    !- Terrain
    0.04,                    !- Loads Convergence Tolerance Value
    0.04,                    !- Temperature Convergence Tolerance Value {deltaC}
    FullExterior,            !- Solar Distribution
    25,                      !- Maximum Number of Warmup Days
    6;                       !- Minimum Number of Warmup Days

!- Simulation Control
SimulationControl,
    No,                      !- Do Zone Sizing Calculation
    No,                      !- Do System Sizing Calculation
    No,                      !- Do Plant Sizing Calculation
    Yes,                     !- Run Simulation for Sizing Periods
    Yes,                     !- Run Simulation for Weather File Run Periods
    No,                      !- Do HVAC Sizing Simulation for Sizing Periods
    1;                       !- Maximum Number of HVAC Sizing Simulation Passes

!- Timestep
Timestep,
    6;                       !- Number of Timesteps per Hour

!- Schedule Type Limits
ScheduleTypeLimits,
    Any Number;              !- Name

!- Temperature Schedule Type
ScheduleTypeLimits,
    Temperature,             !- Name
    -60,                     !- Lower Limit Value
    200,                     !- Upper Limit Value
    Continuous,              !- Numeric Type
    Temperature;             !- Unit Type

!- Humidity Schedule Type
ScheduleTypeLimits,
    Humidity,                !- Name
    0,                       !- Lower Limit Value
    100,                     !- Upper Limit Value
    Continuous;              !- Numeric Type

!- Schedules
Schedule:Compact,
    HEATING_SETPOINT_SCHEDULE,  !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,         !- Field 1
    For: AllDays,           !- Field 2
    Until: 24:00,           !- Field 3
    22.0;                   !- Field 4

Schedule:Compact,
    COOLING_SETPOINT_SCHEDULE,  !- Name
    Temperature,             !- Schedule Type Limits Name
    Through: 12/31,         !- Field 1
    For: AllDays,           !- Field 2
    Until: 24:00,           !- Field 3
    22.0;                   !- Field 4

Schedule:Compact,
    HUMIDITY_SCHEDULE,       !- Name
    Humidity,                !- Schedule Type Limits Name
    Through: 12/31,         !- Field 1
    For: AllDays,           !- Field 2
    Until: 24:00,           !- Field 3
    50.0;                   !- Field 4

!- Zone Definition
Zone,
    MAIN_ZONE,              !- Name
    0,                      !- Direction of Relative North {deg}
    0,                      !- X Origin {m}
    0,                      !- Y Origin {m}
    0,                      !- Z Origin {m}
    1,                      !- Type
    1,                      !- Multiplier
    3.0,                    !- Ceiling Height {m}
    100.0;                  !- Volume {m3}

!- Zone Control:Thermostat
ZoneControl:Thermostat,
    ZONE_THERMOSTAT,        !- Name
    MAIN_ZONE,              !- Zone or ZoneList Name
    HEATING_SETPOINT_SCHEDULE,  !- Control Type Schedule Name
    ThermostatSetpoint:DualSetpoint,  !- Control 1 Object Type
    Dual Setpoint,          !- Control 1 Name
    HEATING_SETPOINT_SCHEDULE,  !- Control 2 Object Type
    COOLING_SETPOINT_SCHEDULE;  !- Control 2 Name

!- Zone HVAC Template
HVACTemplate:Zone:IdealLoadsAirSystem,
    MAIN_ZONE,              !- Zone Name
    HEATING_SETPOINT_SCHEDULE,  !- Template Thermostat Name
    ,                       !- System Availability Schedule Name
    50,                     !- Maximum Heating Supply Air Temperature {C}
    13,                     !- Minimum Cooling Supply Air Temperature {C}
    0.015,                  !- Maximum Heating Supply Air Humidity Ratio {kgWater/kgDryAir}
    0.009,                  !- Minimum Cooling Supply Air Humidity Ratio {kgWater/kgDryAir}
    NoLimit,                !- Heating Limit
    ,                       !- Maximum Heating Air Flow Rate {m3/s}
    NoLimit,                !- Cooling Limit
    ,                       !- Maximum Cooling Air Flow Rate {m3/s}
    ,                       !- Maximum Total Air Flow Rate {m3/s}
    NoOutdoorAir,           !- Outdoor Air Control Type
    ,                       !- Minimum Outdoor Air Schedule Name
    ,                       !- Minimum Outdoor Air Flow Rate {m3/s}
    ,                       !- Maximum Outdoor Air Flow Rate {m3/s}
    ,                       !- Outdoor Air Flow Rate per Person {m3/s}
    ,                       !- Outdoor Air Flow Rate per Zone Floor Area {m3/s-m2}
    ,                       !- Outdoor Air Flow Rate per Zone {m3/s}
    ,                       !- Design Specification Outdoor Air Object Name
    ;                       !- Design Specification Zone Air Distribution Object Name

!- Output Variables
Output:Variable,*,Zone Mean Air Temperature,Hourly;
Output:Variable,*,Zone Air Relative Humidity,Hourly;
Output:Variable,*,Zone Air System Sensible Heating Energy,Hourly;
Output:Variable,*,Zone Air System Sensible Cooling Energy,Hourly;
Output:Variable,*,Zone Thermal Comfort Fanger Model PMV,Hourly;
Output:Variable,*,Zone Thermal Comfort Fanger Model PPD,Hourly;

!- Output Meters
Output:Meter,Electricity:Facility,Hourly;
Output:Meter,Heating:Electricity,Hourly;
Output:Meter,Cooling:Electricity,Hourly;
                ''')  
        
    # Download weather file
    weather_url = "https://energyplus-weather.s3.amazonaws.com/north_and_central_america_wmo_region_4/USA/CA/USA_CA_San.Francisco.Intl.AP.724940_TMY3.zip"
    weather_zip = weather_dir / "sf_weather.zip"
    
    if not os.path.exists(weather_dir / "USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"):
        print("Downloading weather file...")
        urllib.request.urlretrieve(weather_url, weather_zip)
        with zipfile.ZipFile(weather_zip, 'r') as zip_ref:
            zip_ref.extractall(weather_dir)
        os.remove(weather_zip)
    
    # Download and install EnergyPlus if not present
    if not eplus_dir.exists():
        system = platform.system().lower()
        if system == "windows":
            eplus_url = "https://github.com/NREL/EnergyPlus/releases/download/v24.2.0/EnergyPlus-24.2.0-87ed9199d4-Windows-x86_64.exe"
            installer = "EnergyPlus-installer.exe"
        elif system == "darwin":
            eplus_url = "https://github.com/NREL/EnergyPlus/releases/download/v24.2.0/EnergyPlus-24.2.0-87ed9199d4-Darwin-x86_64.dmg"
            installer = "EnergyPlus-installer.dmg"
        elif system == "linux":
            eplus_url = "https://github.com/NREL/EnergyPlus/releases/download/v24.2.0/EnergyPlus-24.2.0-87ed9199d4-Linux-x86_64.sh"
            installer = "EnergyPlus-installer.sh"
        else:
            raise OSError(f"Unsupported operating system: {system}")
            
        print("Downloading EnergyPlus...")
        urllib.request.urlretrieve(eplus_url, installer)
        
        print("Installing EnergyPlus...")
        if system == "windows":
            subprocess.run([installer, "/S", f"/D={eplus_dir}"])
        elif system == "darwin":
            # Mount DMG and copy application
            subprocess.run(["hdiutil", "attach", installer])
            subprocess.run(["cp", "-R", "/Volumes/EnergyPlus-24.2.0/EnergyPlus-24.2.0.app", str(eplus_dir)])
            subprocess.run(["hdiutil", "detach", "/Volumes/EnergyPlus-24.2.0"])
        else:  # Linux
            subprocess.run(["chmod", "+x", installer])
            subprocess.run([f"./{installer}", "--prefix", str(eplus_dir)])
            
        os.remove(installer)
    
    print("\nEnergyPlus setup complete!")
    print(f"IDF template location: {idf_path}")
    print(f"Weather file location: {weather_dir}")
    print(f"EnergyPlus location: {eplus_dir}")
    
if __name__ == "__main__":
    setup_energyplus()