import aiohttp
import json
import time
import os
import asyncio

# API key for Govee
API_KEY = "YOUR_GOVEE_API_KEY"

# Device IDs
DEVICE_1_ID = "FA:85:D1:06:04:46:42:52" # 4252T
DEVICE_2_ID = "E9:EC:D1:06:05:46:59:81" # 5981H
DEVICE_SKU = "H5179"


async def read_govee_data():
    data = {'Temperature_1': None, 'Humidity_1': None, 'Temperature_2': None, 'Humidity_2': None}

    devices = [
        {"device": DEVICE_1_ID, "index": 1},
        {"device": DEVICE_2_ID, "index": 2}
    ]

    async with aiohttp.ClientSession() as session:
        for device_info in devices:
            try:
                state_url = "https://openapi.api.govee.com/router/api/v1/device/state"
                
                payload = {
                    "requestId": str(int(time.time())),
                    "payload": {
                        "sku": DEVICE_SKU,
                        "device": device_info["device"]
                    }
                }
                
                async with session.post(
                    state_url,
                    headers={
                        "Content-Type": "application/json",
                        "Govee-API-Key": API_KEY
                    },
                    json=payload
                ) as response:
                    
                    if response.status == 200:
                        resp_data = await response.json()
                        capabilities = resp_data.get("payload", {}).get("capabilities", [])
                        
                        for prop in capabilities:
                            if prop.get("instance") == "sensorTemperature":
                                # Convert Fahrenheit to Celsius
                                temp_f = prop.get("state", {}).get("value")
                                temp_c = (temp_f - 32) * 5/9
                                data[f'Temperature_{device_info["index"]}'] = round(temp_c, 1)
                            elif prop.get("instance") == "sensorHumidity":
                                data[f'Humidity_{device_info["index"]}'] = prop.get("state", {}).get("value")
            
            except Exception as e:
                print(f"Error reading device {device_info['device']}: {e}")

    return data

if __name__ == "__main__":
    async def main():
        t0 = time.time()
        data = await read_govee_data()
        t1 = time.time()
        print(f"Time taken: {t1 - t0} seconds")
        print("\nCurrent Readings:")
        print(f"Sensor 4252T: {data['Temperature_1']}°C, {data['Humidity_1']}%")
        print(f"Sensor 5981H: {data['Temperature_2']}°C, {data['Humidity_2']}%")
    
    asyncio.run(main())