import asyncio
import aiohttp
import datetime
from kasa import Discover
from dateutil.relativedelta import relativedelta
import warnings
warnings.simplefilter('ignore')

# Track all sessions
_all_sessions = []

# Monkey patch ClientSession
original_ClientSession = aiohttp.ClientSession
class TrackedClientSession(original_ClientSession):
   def __init__(self, *args, **kwargs):
       super().__init__(*args, **kwargs)
       _all_sessions.append(self)
aiohttp.ClientSession = TrackedClientSession

# Device configurations
DEVICES = {
  "Heater": {
      "ip": "192.168.0.165",
      "username": "user@example.com",
      "password": "YOUR_KASA_PASSWORD"
  },
#   "AC": {
#       "ip": "192.168.0.232",
#       "username": "user@example.com",
#       "password": "YOUR_KASA_PASSWORD"
#   },
  "Humidifier": {
      "ip": "192.168.0.158",
      "username": "user@example.com",
      "password": "YOUR_KASA_PASSWORD"
  },
  "Fan": {
      "ip": "192.168.0.205",
      "username": "user@example.com",
      "password": "YOUR_KASA_PASSWORD"
  }
}

_discovered_devices = {}
_shared_session = None
_device_energy_log = {}  # Track energy usage over time

async def get_session():
    global _shared_session
    if _shared_session is None or _shared_session.closed:
        _shared_session = aiohttp.ClientSession()
    return _shared_session

async def discover_device(ip, username=None, password=None):
    global _discovered_devices
    key = f"{ip}_{username}"
    
    if key in _discovered_devices:
        return _discovered_devices[key]
        
    try:
        await get_session()  # Ensure shared session exists
        device = await Discover.discover_single(ip, username=username, password=password, timeout=5)
        _discovered_devices[key] = device
        return device
    except Exception as e:
        print(f"Error discovering device at {ip}: {e}")
        return None

async def control_device(device_name, turn_on=True):
    if device_name not in DEVICES:
        print(f"Device {device_name} not found in configuration")
        return False
       
    device_config = DEVICES[device_name]
    
    try:
        device = await discover_device(
            device_config["ip"], 
            username=device_config.get("username"),
            password=device_config.get("password")
        )
        
        if device is None:
            return False
           
        # Get energy state before changing
        if turn_on:
            energy_before = await get_energy_usage(device_name)
            await device.turn_on()
            
            # Initialize energy tracking for this session if turning on
            start_time = datetime.datetime.now()
            if device_name not in _device_energy_log:
                _device_energy_log[device_name] = []
            
            _device_energy_log[device_name].append({
                'start_time': start_time,
                'end_time': None,
                'total_energy_kwh': 0,
                'rated_power_w': device_config.get('rated_power', 0)
            })
        else:
            await device.turn_off()
            
            # Update energy tracking when turning off
            if device_name in _device_energy_log and _device_energy_log[device_name]:
                latest_session = _device_energy_log[device_name][-1]
                if latest_session['end_time'] is None:
                    latest_session['end_time'] = datetime.datetime.now()
                    duration_hours = (latest_session['end_time'] - latest_session['start_time']).total_seconds() / 3600
                    latest_session['total_energy_kwh'] = (latest_session['rated_power_w'] / 1000) * duration_hours
           
        await device.update()
        return device.is_on == turn_on
    except Exception as e:
        print(f"Error controlling {device_name}: {e}")
        return False

async def control_heater(turn_on=True):
    return await control_device("Heater", turn_on)
   
async def control_humidifier(turn_on=True):
    return await control_device("Humidifier", turn_on)
   
async def control_fan(turn_on=True):
    return await control_device("Fan", turn_on)

async def get_device_states():
    states = {}
    
    for device_name in DEVICES:
        try:
            device_config = DEVICES[device_name]
            device = await discover_device(
                device_config["ip"], 
                username=device_config.get("username"),
                password=device_config.get("password")
            )
            
            if device:
                await device.update()
                states[f"{device_name}_On"] = int(device.is_on)
            else:
                states[f"{device_name}_On"] = None
        except Exception as e:
            print(f"Error getting state for {device_name}: {e}")
            states[f"{device_name}_On"] = None
   
    return states

async def cleanup_kasa():
    global _discovered_devices, _all_sessions, _shared_session
    
    # Close all tracked sessions
    for session in _all_sessions:
        if not session.closed:
            try:
                await session.close()
            except Exception as e:
                print(f"Error closing session: {e}")
    
    # Close device connections
    for key, device in _discovered_devices.items():
        try:
            if hasattr(device, 'close') and callable(getattr(device, 'close')):
                await device.close()
        except Exception as e:
            print(f"Error closing device {key}: {e}")
    
    _discovered_devices.clear()
    _all_sessions.clear()
    _shared_session = None
    
    return True

async def get_energy_usage(device_name):
    """Get energy usage data from a Kasa smart plug"""
    if device_name not in DEVICES:
        print(f"Device {device_name} not found in configuration")
        return None
        
    device_config = DEVICES[device_name]
    
    try:
        device = await discover_device(
            device_config["ip"], 
            username=device_config.get("username"),
            password=device_config.get("password")
        )
        
        if device is None:
            return None
            
        await device.update()
        
        # Check if device supports energy monitoring
        if device.has_emeter:
            usage_data = {}
            
            # Get realtime metrics if available
            try:
                realtime = await device.get_emeter_realtime()
                usage_data.update({
                    "current_power_w": realtime.get("power_mw", 0) / 1000 if "power_mw" in realtime else realtime.get("power", 0),
                    "current_voltage_v": realtime.get("voltage_mv", 0) / 1000 if "voltage_mv" in realtime else realtime.get("voltage", 0),
                    "current_current_a": realtime.get("current_ma", 0) / 1000 if "current_ma" in realtime else realtime.get("current", 0)
                })
            except Exception as e:
                print(f"Error getting realtime data: {e}")
                # Fall back to rated power if device is on
                if device.is_on:
                    usage_data["current_power_w"] = device_config.get("rated_power", 0)
                else:
                    usage_data["current_power_w"] = 0
            
            # Get usage module data if available
            if hasattr(device, "modules") and "usage" in device.modules:
                usage = device.modules["usage"]
                usage_data.update({
                    "on_time_today_min": usage.usage_today,
                    "on_time_month_min": usage.usage_this_month
                })
            
            # Get on_since if available
            if hasattr(device, "on_since"):
                usage_data["on_since"] = device.on_since
            
            return usage_data
        else:
            # For devices without energy monitoring
            await device.update()
            usage_data = {
                "current_power_w": device_config.get("rated_power", 0) if device.is_on else 0,
                "current_voltage_v": 120.0,  # Assume standard voltage
                "current_current_a": device_config.get("rated_power", 0) / 120.0 if device.is_on else 0,
                "on_since": device.on_since if device.is_on else None
            }
            return usage_data
    except Exception as e:
        print(f"Error getting energy data from {device_name}: {e}")
        return None

async def calculate_device_energy(device_name, duration_hours=None):
    """Calculate energy consumption for a device based on power readings
    
    Args:
        device_name: Name of the device to calculate energy for
        duration_hours: Optional duration in hours (if not using on_since)
    
    Returns:
        Dict with energy data including kWh consumption
    """
    energy_data = await get_energy_usage(device_name)
    
    if not energy_data:
        return {"energy_kwh": 0, "status": "unavailable"}
    
    # Get power in kW
    power_kw = energy_data.get('current_power_w', 0) / 1000
    
    # Calculate duration the device has been on
    if duration_hours is None and energy_data.get('on_since') is not None:
        # Calculate duration from on_since timestamp
        on_since = energy_data['on_since']
        if on_since:
            now = datetime.datetime.now(on_since.tzinfo)
            duration_hours = (now - on_since).total_seconds() / 3600
        else:
            duration_hours = 0
    elif duration_hours is None:
        duration_hours = 0
    
    # Calculate energy in kWh
    energy_kwh = power_kw * duration_hours
    
    # Add historical energy usage from log
    if device_name in _device_energy_log:
        historical_kwh = sum(session['total_energy_kwh'] for session in _device_energy_log[device_name] 
                            if session['end_time'] is not None)
        energy_kwh += historical_kwh
    
    return {
        **energy_data,
        "energy_kwh": energy_kwh,
        "duration_hours": duration_hours,
        "status": "on" if power_kw > 0 else "off"
    }

async def calculate_total_energy():
    """Calculate total energy consumption across all devices"""
    total_energy = {
        "total_energy_kwh": 0,
        "total_energy_cost": 0,  # Assuming $0.15 per kWh
        "devices": {},
        "timestamp": datetime.datetime.now().isoformat()
    }
    
    for device_name in DEVICES:
        device_energy = await calculate_device_energy(device_name)
        total_energy["devices"][device_name] = device_energy
        total_energy["total_energy_kwh"] += device_energy["energy_kwh"]
    
    # Calculate cost (assuming $0.15 per kWh)
    total_energy["total_energy_cost"] = total_energy["total_energy_kwh"] * 0.15
    
    # Add current power consumption
    total_energy["current_total_power_w"] = sum(
        device.get("current_power_w", 0) 
        for device in total_energy["devices"].values()
    )
    
    return total_energy

async def get_all_energy_usage():
    """Get energy usage from all devices with calculated consumption"""
    energy_data = await calculate_total_energy()
    return energy_data

async def estimate_annual_consumption(days_of_data=1):
    """Estimate annual energy consumption based on current usage pattern"""
    total_energy = await calculate_total_energy()
    
    # Calculate daily consumption
    daily_kwh = total_energy["total_energy_kwh"] / days_of_data
    
    # Estimate annual consumption
    annual_kwh = daily_kwh * 365
    annual_cost = annual_kwh * 0.15  # Assuming $0.15 per kWh
    
    return {
        "daily_kwh": daily_kwh,
        "annual_kwh": annual_kwh,
        "annual_cost": annual_cost,
        "based_on_days": days_of_data
    }

if __name__ == "__main__":
    async def test():
        try:
            print("Testing Kasa Controller...")
            
            print("Getting current device states...")
            states = await get_device_states()
            print(f"Device states: {states}")
            
            print("Testing energy usage monitoring...")
            all_energy = await get_all_energy_usage()
            print(f"All devices energy usage: {all_energy}")
            
            print("Testing Heater...")
            heater_energy = await get_energy_usage("Heater")
            print(f"Heater energy before: {heater_energy}")
            
            heater_on = await control_heater(True)
            print(f"Heater turned ON: {heater_on}")
            
            # Wait a bit to see energy usage change
            await asyncio.sleep(10)
            
            heater_energy = await get_energy_usage("Heater")
            print(f"Heater energy while ON: {heater_energy}")
            
            heater_off = await control_heater(False)
            print(f"Heater turned OFF: {heater_off}")
            
            print("Testing Humidifier...")
            humidifier_energy = await get_energy_usage("Humidifier")
            print(f"Humidifier energy before: {humidifier_energy}")
            
            humidifier_on = await control_humidifier(True)
            print(f"Humidifier turned ON: {humidifier_on}")
            
            # Wait a bit to see energy usage change
            await asyncio.sleep(1)
            
            humidifier_energy = await get_energy_usage("Humidifier")
            print(f"Humidifier energy while ON: {humidifier_energy}")
            
            humidifier_off = await control_humidifier(False)
            print(f"Humidifier turned OFF: {humidifier_off}")
            
            print("Testing Fan...")
            fan_energy = await get_energy_usage("Fan")
            print(f"Fan energy before: {fan_energy}")
            
            fan_on = await control_fan(True)
            print(f"Fan turned ON: {fan_on}")
            
            # Wait a bit to see energy usage change
            await asyncio.sleep(1)
            
            fan_energy = await get_energy_usage("Fan")
            print(f"Fan energy while ON: {fan_energy}")
            
            fan_off = await control_fan(False)
            print(f"Fan turned OFF: {fan_off}")
            
            # Calculate total energy consumption
            print("\nCalculating energy consumption...")
            energy_data = await calculate_total_energy()
            print(f"Total energy consumption: {energy_data['total_energy_kwh']:.4f} kWh")
            print(f"Total energy cost: ${energy_data['total_energy_cost']:.2f}")
            print("Device breakdown:")
            for device, data in energy_data["devices"].items():
                print(f"  {device}: {data['energy_kwh']:.4f} kWh (Power: {data.get('current_power_w', 0):.1f}W)")
            
            # Estimate annual consumption
            annual = await estimate_annual_consumption()
            print(f"\nEstimated annual consumption: {annual['annual_kwh']:.2f} kWh (${annual['annual_cost']:.2f})")
            
            print("Test succeeded")
            
        finally:
            await cleanup_kasa()
    
    asyncio.run(test())
    
# import asyncio
# from kasa import Discover

# async def main():
#     dev = await Discover.discover_single("192.168.0.232",username="user@example.com",password="YOUR_KASA_PASSWORD")
#     await dev.turn_on()
#     await dev.update()

# if __name__ == "__main__":
#     asyncio.run(main())