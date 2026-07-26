"""Client for reading room state (window/occupancy) from room_logger server."""

import aiohttp
import logging

logger = logging.getLogger(__name__)

ROOM_SERVER_ENDPOINT = "http://192.168.0.208:5000/get_current_state"


async def read_room_state():
    """Fetch current window state and occupancy from the room logger server."""
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(ROOM_SERVER_ENDPOINT) as response:
                response.raise_for_status()
                data = await response.json()
                if data.get("state") == "unknown":
                    return None
                return {
                    "window_open": int(data.get("window_state", 0)),
                    "occupancy": int(data.get("occupancy", 0)),
                }
    except aiohttp.ClientError as e:
        logger.error(f"Error querying room state: {e}")
        return None
