import requests
import json
import time
import os
from datetime import datetime

# API key for Govee
API_KEY = "YOUR_GOVEE_API_KEY"

# Device IDs
DEVICE_1_ID = "FA:85:D1:06:04:46:42:52" # 4252T
DEVICE_2_ID = "E9:EC:D1:06:05:46:59:81" # 5981H
DEVICE_SKU = "H5179"

def read_govee_data():
    """Read data from Govee sensors using synchronous requests"""
    timestamp = datetime.now().isoformat()
    data = {
        'timestamp': timestamp,
        'Temperature_1': None, 
        'Humidity_1': None, 
        'Temperature_2': None, 
        'Humidity_2': None
    }

    devices = [
        {"device": DEVICE_1_ID, "index": 1},
        {"device": DEVICE_2_ID, "index": 2}
    ]

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
            
            response = requests.post(
                state_url,
                headers={
                    "Content-Type": "application/json",
                    "Govee-API-Key": API_KEY
                },
                json=payload,
                timeout=5
            )
            
            if response.status_code == 200:
                resp_data = response.json()
                capabilities = resp_data.get("payload", {}).get("capabilities", [])
                
                for prop in capabilities:
                    if prop.get("instance") == "sensorTemperature":
                        temp_f = prop.get("state", {}).get("value")
                        if isinstance(temp_f, (int, float)):
                            temp_c = (temp_f - 32) * 5/9
                            data[f'Temperature_{device_info["index"]}'] = round(temp_c, 1)
                    elif prop.get("instance") == "sensorHumidity":
                        humidity = prop.get("state", {}).get("value")
                        if isinstance(humidity, (int, float)):
                            data[f'Humidity_{device_info["index"]}'] = humidity
        
        except Exception as e:
            print(f"Error reading device {device_info['device']}: {e}")
    
    return data

async def read_govee_data_async():
    return read_govee_data()

if __name__ == "__main__":
    data = read_govee_data()
    print(f"\nCurrent Readings ({data['timestamp']}):")
    
    if data['Temperature_1'] is not None:
        print(f"Sensor 4252T: {data['Temperature_1']}°C, {data['Humidity_1']}%")
    else:
        print("Sensor 4252T: No data")
    
    if data['Temperature_2'] is not None:
        print(f"Sensor 5981H: {data['Temperature_2']}°C, {data['Humidity_2']}%")
    else:
        print("Sensor 5981H: No data")