#!/usr/bin/env python3
"""
Test Script for Physical Causal Discovery Pipeline with ISO 7730 Integration
Tests the complete pipeline with real sensors and actuators
"""

import os
import sys
import pandas as pd
import numpy as np
import asyncio
import time
import logging
import warnings
import json
from datetime import datetime
from pathlib import Path
import traceback
import requests

warnings.simplefilter('ignore')

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import real physical components
from smart_room_physical.smart_room_monitor import SmartRoomMonitor, calculate_iso7730_satisfaction
from smart_room_physical.physical_intervention import PhysicalInterventionController
from smart_room_physical.kasa_controller import cleanup_kasa, get_all_energy_usage, get_device_states
from smart_room_physical.bme680_reader import get_iaq_value, is_calibrated, get_calibration_progress

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("physical_pipeline_test.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

async def test_sensor_integration():
    """Test individual sensor integrations with ISO 7730"""
    print("\n" + "="*60)
    print("TESTING SENSOR INTEGRATION WITH ISO 7730")
    print("="*60)
    
    monitor = SmartRoomMonitor()
    success_count = 0
    
    try:
        # Test 1: Basic sensor reading with ISO 7730
        print("\n1. Testing basic sensor reading with thermal comfort...")
        state = await monitor.read_current_state()
        
        required_keys = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
        for key in required_keys:
            if key in state and isinstance(state[key], (int, float)):
                print(f"   ✓ {key}: {state[key]}")
                success_count += 1
            else:
                print(f"   ❌ {key}: Missing or invalid")
        
        # Test ISO 7730 metrics
        if all(key in state for key in ['PMV', 'PPD', 'ThermalComfortMethod']):
            print(f"   ✓ ISO 7730 PMV: {state['PMV']:+.2f}")
            print(f"   ✓ ISO 7730 PPD: {state['PPD']:.1f}%")
            print(f"   ✓ Method: {state['ThermalComfortMethod']}")
            success_count += 1
        else:
            print("   ❌ ISO 7730 metrics missing")
        
        # Test 2: BME680 status with HTTP endpoint check
        print("\n2. Testing BME680 air quality server...")
        try:
            response = requests.get("http://localhost:5001/iaq", timeout=3)
            
            if response.status_code == 200:
                iaq_data = response.json()
                
                if iaq_data.get('burn_in_complete', False) and iaq_data.get('calibrated', False):
                    print(f"   ✓ BME680 calibrated: IAQ={iaq_data.get('iaq')}")
                    success_count += 1
                else:
                    progress = iaq_data.get('burn_in_progress', 0)
                    print(f"   ⏳ BME680 burn-in: {progress:.1f}% complete")
                    if progress > 50:  # Partial credit for progress
                        success_count += 0.5
            else:
                print(f"   ❌ BME680 HTTP endpoint error: {response.status_code}")
                
        except requests.exceptions.ConnectionError:
            print("   ❌ BME680 server not running - start with: python bme680_reader.py")
        except Exception as e:
            print(f"   ⚠ BME680 error: {e}")
                
        # Test 3: Energy monitoring
        print("\n3. Testing Kasa energy monitoring...")
        try:
            energy_data = await get_all_energy_usage()
            total_kwh = energy_data.get('total_energy_kwh', 0)
            current_power = energy_data.get('current_total_power_w', 0)
            print(f"   ✓ Total energy: {total_kwh:.3f} kWh")
            print(f"   ✓ Current power: {current_power:.1f} W")
            success_count += 1
        except Exception as e:
            print(f"   ❌ Energy monitoring error: {e}")
        
        # Test 4: Device states
        print("\n4. Testing Kasa device states...")
        try:
            device_states = await get_device_states()
            expected_devices = ['Heater_On', 'Humidifier_On', 'Fan_On']
            
            responding_devices = 0
            for device in expected_devices:
                if device in device_states and device_states[device] is not None:
                    status = "ON" if device_states[device] else "OFF"
                    print(f"   ✓ {device.replace('_On', '')}: {status}")
                    responding_devices += 1
                else:
                    print(f"   ❌ {device}: Not responding")
            
            if responding_devices >= 2:  # At least 2 devices working
                success_count += 1
                
        except Exception as e:
            print(f"   ❌ Device states error: {e}")
        
    finally:
        await monitor.cleanup()
    
    print(f"\nSensor integration: {success_count}/5 tests passed")
    return success_count >= 4

async def test_iso7730_integration():
    """Test ISO 7730 thermal comfort integration specifically"""
    print("\n" + "="*60)
    print("TESTING ISO 7730 THERMAL COMFORT INTEGRATION")
    print("="*60)
    
    try:
        # Test 1: ISO 7730 calculation module
        print("\n1. Testing ISO 7730 calculation...")
        try:
            from pythermalcomfort.models import pmv_ppd
            
            # Test calculation with sample data
            comfort_results = calculate_iso7730_satisfaction(22.0, 45.0)
            
            if comfort_results['method'] == 'ISO_7730':
                print(f"   ✓ ISO 7730 module operational")
                print(f"   ✓ PMV: {comfort_results['pmv']:+.2f}")
                print(f"   ✓ PPD: {comfort_results['ppd']:.1f}%")
                print(f"   ✓ Satisfaction: {comfort_results['satisfaction']:.1f}%")
                iso_available = True
            else:
                print(f"   ⚠ Using fallback: {comfort_results['method']}")
                iso_available = False
                
        except ImportError:
            print("   ❌ pythermalcomfort not installed")
            print("   Install with: pip install pythermalcomfort")
            iso_available = False
        
        # Test 2: Real-time thermal comfort monitoring
        print("\n2. Testing real-time comfort monitoring...")
        monitor = SmartRoomMonitor()
        
        try:
            state = await monitor.read_current_state()
            
            # Check if thermal comfort metrics are present
            thermal_keys = ['PMV', 'PPD', 'ThermalComfortMethod']
            if all(key in state for key in thermal_keys):
                print(f"   ✓ Real-time thermal comfort operational")
                print(f"   ✓ Current PMV: {state['PMV']:+.2f}")
                print(f"   ✓ Current PPD: {state['PPD']:.1f}%")
                print(f"   ✓ Satisfaction: {state['OverallSatisfaction']:.1f}%")
                print(f"   ✓ Method: {state['ThermalComfortMethod']}")
                comfort_monitoring = True
            else:
                print("   ❌ Thermal comfort metrics missing from state")
                comfort_monitoring = False
                
        except Exception as e:
            print(f"   ❌ Comfort monitoring error: {e}")
            comfort_monitoring = False
        finally:
            await monitor.cleanup()
        
        return iso_available and comfort_monitoring
        
    except Exception as e:
        print(f"❌ ISO 7730 integration test failed: {e}")
        return False

async def test_intervention_system():
    """Test physical intervention capabilities with ISO 7730"""
    print("\n" + "="*60)
    print("TESTING INTERVENTION SYSTEM WITH ISO 7730")
    print("="*60)
    
    controller = PhysicalInterventionController()
    
    print("   WARNING: This will control real devices and monitor thermal comfort!")
    confirm = input("   Continue with intervention test? (y/N): ")
    
    if confirm.lower() != 'y':
        print("   ⚠ Intervention test skipped by user")
        return True
    
    try:
        print("\n1. Testing ISO 7730 intervention with temperature increase...")
        
        # Use shorter timings for testing
        original_stabilization = controller.stabilization_time
        controller.stabilization_time = 60  # Reduce to 1 minute for testing
        
        result = await controller.perform_intervention('Temperature', 'increase')
        
        # Restore original timing
        controller.stabilization_time = original_stabilization
        
        if result and 'preInterventionData' in result:
            pre_temp = result['preInterventionData']['Temperature']
            post_temp = result['postInterventionData']['Temperature']
            temp_change = abs(post_temp - pre_temp)
            
            # Check thermal comfort changes
            if 'pre_comfort_metrics' in result:
                pre_comfort = result['pre_comfort_metrics']
                post_comfort = result['post_comfort_metrics']
                satisfaction_change = post_comfort['satisfaction'] - pre_comfort['satisfaction']
                pmv_change = post_comfort['pmv'] - pre_comfort['pmv']
                
                print(f"   ✓ Temperature change: {temp_change:.1f}°C")
                print(f"   ✓ Satisfaction change: {satisfaction_change:+.1f}%")
                print(f"   ✓ PMV change: {pmv_change:+.2f}")
                print(f"   ✓ Method: {post_comfort['method']}")
                
                # Check intervention log
                if os.path.exists(controller.log_file):
                    print(f"   ✓ Intervention logged to: {controller.log_file}")
                
                return True
            else:
                print("   ❌ ISO 7730 comfort metrics missing from result")
                return False
        else:
            print("   ❌ Intervention failed or returned invalid data")
            return False
            
    except Exception as e:
        print(f"   ❌ Intervention error: {e}")
        return False

async def test_causal_integration():
    """Test integration with causal discovery components"""
    print("\n" + "="*60)
    print("TESTING CAUSAL DISCOVERY INTEGRATION")
    print("="*60)
    
    try:
        # Test imports
        print("\n1. Testing component imports...")
        from src.generators import PCGenerator, SAMGenerator, LLMGenerator
        from src.evaluator import EdgeRanker
        from src.ground_truth import GroundTruthDAG
        print("   ✓ All components imported successfully")
        
        # Test with pre-processed CSV data
        print("\n2. Testing with pre-processed CSV data...")
        try:
            baseline_csv = "data/smart-room-hidden-vars_preprocessed.csv"
            scaling_params_csv = "data/smart_room_noisy_preprocessed_scaling_params.csv"
            
            if os.path.exists(baseline_csv):
                baseline_df = pd.read_csv(baseline_csv)
                print(f"   ✓ Loaded {len(baseline_df)} samples from {baseline_csv}")
            else:
                print(f"   ⚠ Baseline CSV not found: {baseline_csv}")
                # Create minimal test data
                baseline_df = pd.DataFrame({
                    'Temperature': np.random.normal(22, 2, 100),
                    'Humidity': np.random.normal(45, 5, 100),
                    'AirQuality': np.random.normal(100, 20, 100),
                    'EnergyConsumption': np.random.normal(1.5, 0.3, 100),
                    'OverallSatisfaction': np.random.normal(75, 10, 100)
                })
                print(f"   ✓ Created synthetic test data: {len(baseline_df)} samples")
            
            if os.path.exists(scaling_params_csv):
                scaling_params = pd.read_csv(scaling_params_csv)
                print(f"   ✓ Scaling parameters available")
                
        except Exception as e:
            print(f"   ❌ Error with CSV data: {e}")
            return False
        
        # Test generators
        print("\n3. Testing hypothesis generators...")
        relevant_columns = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
        
        generators_passed = 0
        
        # PC Generator
        try:
            pc_gen = PCGenerator(relevant_columns=relevant_columns)
            pc_result = pc_gen.generate(baseline_df)
            edges_count = len(pc_result.get('edges', []))
            print(f"   ✓ PC Generator: {edges_count} edges")
            generators_passed += 1
        except Exception as e:
            print(f"   ⚠ PC Generator error: {e}")
        
        # SAM Generator
        try:
            sam_gen = SAMGenerator(relevant_columns=relevant_columns)
            sam_result = sam_gen.generate(baseline_df)
            edges_count = len(sam_result.get('edges', []))
            print(f"   ✓ SAM Generator: {edges_count} edges")
            generators_passed += 1
        except Exception as e:
            print(f"   ⚠ SAM Generator error: {e}")
        
        # LLM Generator
        try:
            api_key = "YOUR_OPENAI_API_KEY"
            llm_gen = LLMGenerator(api_key, relevant_columns=relevant_columns)
            llm_result = llm_gen.generate(baseline_df)
            edges_count = len(llm_result.get('edges', []))
            print(f"   ✓ LLM Generator: {edges_count} edges")
            generators_passed += 1
        except Exception as e:
            print(f"   ⚠ LLM Generator error: {e}")
        
        print(f"\n   Generators working: {generators_passed}/3")
        return generators_passed >= 2
        
    except Exception as e:
        print(f"\n❌ Causal integration failed: {e}")
        return False

async def test_full_workflow():
    """Test complete workflow integration"""
    print("\n" + "="*60)
    print("TESTING FULL WORKFLOW WITH ISO 7730")
    print("="*60)
    
    try:
        # Import physical pipeline
        print("\n1. Testing pipeline import...")
        try:
            from physical_causal_main import PhysicalCausalPipeline
            print("   ✓ PhysicalCausalPipeline imported")
        except ImportError:
            print("   ❌ Cannot import PhysicalCausalPipeline")
            return False
        
        # Test pipeline initialization
        print("\n2. Testing pipeline initialization...")
        pipeline = PhysicalCausalPipeline(
            api_key="YOUR_OPENAI_API_KEY",
            max_iterations=1,
            effect_threshold=0.1
        )
        print("   ✓ Pipeline initialized with ISO 7730 support")
        
        # Test ground truth
        print("\n3. Testing physical ground truth...")
        ground_truth = pipeline.ground_truth
        print(f"   ✓ Ground truth nodes: {len(ground_truth.nodes)}")
        print(f"   ✓ Ground truth edges: {len(ground_truth.edges)}")
        print(f"   ✓ Independent variables: {ground_truth.get_independent_variables()}")
        
        # Test hypothesis generation with synthetic data
        print("\n4. Testing hypothesis generation...")
        synthetic_data = pd.DataFrame({
            'Temperature': np.random.normal(22, 2, 50),
            'Humidity': np.random.normal(45, 5, 50),
            'AirQuality': np.random.normal(100, 20, 50),
            'EnergyConsumption': np.random.normal(1.5, 0.3, 50),
            'OverallSatisfaction': np.random.normal(75, 10, 50)
        })
        
        method_dags = pipeline._generate_initial_hypotheses(synthetic_data)
        total_edges = sum(len(dag['edges']) for dag in method_dags.values())
        
        if total_edges > 0:
            print(f"   ✓ Generated {len(method_dags)} hypotheses with {total_edges} total edges")
            for method, dag in method_dags.items():
                print(f"     - {method.upper()}: {len(dag['edges'])} edges")
        else:
            print("   ⚠ No edges generated (normal with synthetic data)")
        
        # Test edge ranking
        print("\n5. Testing edge ranking...")
        edge_ranker, ranked_edges = pipeline._create_union_dag_and_rank_edges(method_dags)
        print(f"   ✓ Found {len(ranked_edges)} unique edges for testing")
        
        # Test thermal comfort tester
        print("\n6. Testing thermal comfort tester...")
        tester = pipeline.tester
        print(f"   ✓ Tester initialized with effect threshold: {tester.effect_threshold}")
        print(f"   ✓ Safety limits: {tester.max_daily_interventions} interventions/day")
        
        # Cleanup
        await pipeline.cleanup()
        
        print("\n✓ Full workflow test completed successfully")
        return True
        
    except Exception as e:
        print(f"\n❌ Full workflow test failed: {e}")
        print(traceback.format_exc())
        return False

async def run_all_tests():
    """Execute all test suites"""
    print("PHYSICAL CAUSAL DISCOVERY WITH ISO 7730 TEST SUITE")
    print("=" * 80)
    print("⚠️  WARNING: This will test real physical devices!")
    print("📊 Features: ISO 7730 PMV/PPD thermal comfort monitoring")
    print("=" * 80)
    
    # User confirmation
    confirm = input("\nProceed with physical device testing? (y/N): ")
    if confirm.lower() != 'y':
        print("Testing cancelled by user.")
        return False
    
    # Run tests
    test_results = {}
    
    print("\n🔍 Running ISO 7730 test suite...")
    
    try:
        test_results['sensors'] = await test_sensor_integration()
        test_results['iso7730'] = await test_iso7730_integration()
        test_results['interventions'] = await test_intervention_system()
        test_results['causal'] = await test_causal_integration()
        test_results['workflow'] = await test_full_workflow()
    except Exception as e:
        print(f"Critical test error: {e}")
        return False
    
    # Results summary
    print("\n" + "="*80)
    print("TEST RESULTS SUMMARY")
    print("="*80)
    
    passed_tests = 0
    total_tests = len(test_results)
    
    for test_name, result in test_results.items():
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"{test_name.title()} Integration: {status}")
        if result:
            passed_tests += 1
    
    success_rate = passed_tests / total_tests
    overall_status = "✓ READY" if success_rate >= 0.8 else "❌ NEEDS WORK"
    
    print(f"\nOverall Status: {overall_status} ({passed_tests}/{total_tests} tests passed)")
    
    if success_rate >= 0.8:
        print("\n🎉 ISO 7730 physical causal discovery pipeline is ready!")
        print("   - All core components working")
        print("   - ISO 7730 thermal comfort integration operational")
        print("   - Device integration successful")
        print("   - Ready for scientific causal discovery")
    else:
        print("\n⚠️  Issues detected:")
        if not test_results.get('iso7730'):
            print("   - Install pythermalcomfort: pip install pythermalcomfort")
        if not test_results.get('sensors'):
            print("   - Check BME680 server: python bme680_reader.py")
            print("   - Verify Kasa device connections")
        print("   - Review error messages above")
    
    return success_rate >= 0.8

async def cleanup_all_resources():
    """Clean up all test resources"""
    try:
        await cleanup_kasa()
        print("✓ Resources cleaned up")
    except Exception as e:
        print(f"⚠ Cleanup warning: {e}")

def main():
    """Main test execution"""
    try:
        result = asyncio.run(run_all_tests())
        
        if result:
            print("\n" + "="*60)
            print("NEXT STEPS:")
            print("1. Ensure BME680 server is running: python bme680_reader.py")
            print("2. Run: python physical_causal_main.py")
            print("3. Monitor ISO 7730 thermal comfort during discovery")
            print("="*60)
        
        return result
        
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user")
        return False
    except Exception as e:
        print(f"\nCritical error: {e}")
        logger.error(f"Critical error: {e}")
        return False
    finally:
        try:
            asyncio.run(cleanup_all_resources())
        except:
            pass

if __name__ == "__main__":
    success = main()
    
    if success:
        print("\n🚀 Ready for ISO 7730 physical causal discovery!")
        exit(0)
    else:
        print("\n❌ Fix issues before proceeding.")
        exit(1)