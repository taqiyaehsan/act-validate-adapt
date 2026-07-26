from .kasa_controller import control_fan, control_humidifier, control_heater, get_device_states
import asyncio


async def close_all_devices():
    print("close all devices...")
    fan_result = await control_fan(False)
    humidifier_result = await control_humidifier(False)
    heater_result = await control_heater(False)
    
    print(f"fan result: {fan_result}")
    print(f"humidifier result: {humidifier_result}")
    print(f"heater result: {heater_result}")
    
    await asyncio.sleep(1)
    
    print("\ncheck device states...")
    states = await get_device_states()
    
    print("device states check result:")
    for device_name, state in states.items():
        if state is None:
            print(f"  {device_name}: cannot get state")
        elif state == 0:
            print(f"  {device_name}: closed")
        else:
            print(f"  {device_name}: still on")
    
    all_closed = all(state == 0 for state in states.values() if state is not None)
    
    if all_closed:
        print("\n✓ all devices closed")
    else:
        print("\n✗ some devices may not be closed, please check")
    
    return all_closed

if __name__ == "__main__":
    asyncio.run(close_all_devices())
