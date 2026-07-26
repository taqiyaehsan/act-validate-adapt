import os
import signal
import psutil
from src.tester import HypothesisTester
from energyplus_interface import EnergyPlusInterface
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def cleanup_node_processes():
    for proc in psutil.process_iter():
        try:
            if proc.name() == "node":
                proc.kill()
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass

def test_energyplus():
    eplus = EnergyPlusInterface(
        "templates/smart_room.idf",
        "weather/USA_CA_San.Francisco.Intl.AP.724940_TMY3.epw"
    )
    
    try:
        # Test baseline
        logger.info("Testing baseline conditions...")
        results = eplus.run_simulation()
        logger.info(f"Baseline Results: {results}")
        
        # Test with modifications
        logger.info("\nTesting modified conditions...")
        results = eplus.run_simulation(
            temperature=26.0,
            humidity=60.0,
            air_quality=400.0
        )
        logger.info(f"Modified Results: {results}")
        
    except Exception as e:
        logger.error(f"Test failed: {e}")
    finally:
        eplus.cleanup()

if __name__ == "__main__":
    try:
        test_energyplus()
    finally:
        cleanup_node_processes()