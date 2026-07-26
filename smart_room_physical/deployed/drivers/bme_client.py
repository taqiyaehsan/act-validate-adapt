"""Client for reading BME680 data from bme_server.py via HTTP."""

import aiohttp
import json
import logging

logger = logging.getLogger(__name__)

BME_SERVER_ENDPOINT = "http://192.168.0.208:5001/latest_data"


async def read_bme_data():
    """Fetch latest BME680 sensor data from the Flask server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(BME_SERVER_ENDPOINT) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("status") == "success":
                    return data.get("data")
                else:
                    logger.error(f"Server returned error status: {data}")
                    return None
    except aiohttp.ClientError as e:
        logger.error(f"Error querying BME data: {e}")
        return None
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing JSON response: {e}")
        return None
