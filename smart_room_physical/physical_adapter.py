#!/usr/bin/env python3
import asyncio
from smart_room_physical.physical_intervention import PhysicalInterventionController

class PhysicalSimulationAdapter:
    """Adapter that mimics smart_room.js interface"""
    def __init__(self):
        self.controller = PhysicalInterventionController()
    
    async def simulateAndGetLatestData(self, duration, interventions, intervention_time):
        """Match signature expected by pipeline"""
        try:
            # Parse interventions format
            variable = interventions[0]['variable']
            action = interventions[0]['action']
            value = interventions[0]['value'] if 'value' in interventions[0] else None
            
            # Handle value conversion
            if value is not None:
                value = float(value)
            
            # Perform intervention
            result = await self.controller.perform_intervention(variable, action, value)
            return result
        except Exception as e:
            print(f"Adapter error: {e}")
            return None
        
# Global adapter instance
_adapter = None

async def _get_adapter():
    global _adapter
    if _adapter is None:
        _adapter = PhysicalSimulationAdapter()
    return _adapter

# For async contexts - direct async call
async def asyncio_simulate(duration, interventions, intervention_time):
    adapter = await _get_adapter()
    return await adapter.simulateAndGetLatestData(duration, interventions, intervention_time)

# For sync contexts - matching the JS interface
def simulateAndGetLatestData(duration, interventions, intervention_time):
    """Entry point function that matches the JS interface"""
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        # No event loop exists, create one
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        return loop.run_until_complete(asyncio_simulate(duration, interventions, intervention_time))
    else:
        # We're already in an event loop
        return asyncio.create_task(asyncio_simulate(duration, interventions, intervention_time))