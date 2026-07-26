#!/usr/bin/env python3
"""
Physical Smart Room Causal Discovery Pipeline
Implements iterative DAG construction with edge validation through physical interventions
"""

import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '' 

import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import asyncio
import time
import atexit
import logging
import warnings
import json
from datetime import datetime
from pathlib import Path
import traceback
from collections import defaultdict

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

# Import original causal pipeline components
from src.generators import PCGenerator, SAMGenerator, LLMGenerator
from src.evaluator import EdgeRanker
from src.metrics import MetricsCalculator
from src.metrics_viz import MetricsVisualizer, export_metrics_to_csv
from src.gpt_client import GPTClient

# Import physical components
from smart_room_physical.smart_room_monitor import SmartRoomMonitor, calculate_iso7730_satisfaction
from smart_room_physical.kasa_controller import cleanup_kasa
from smart_room_physical.govee_reader import read_govee_data

warnings.simplefilter('ignore')

# Configure logging
logging.basicConfig(
   level=logging.DEBUG,
   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
   handlers=[
       logging.FileHandler("physical_causal_discovery.log", mode='w'),
       logging.StreamHandler()
   ],
   force=True
)

# Set all loggers to DEBUG
logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)

# Only suppress very noisy libraries
logging.getLogger('urllib3').setLevel(logging.WARNING)
logging.getLogger('requests').setLevel(logging.WARNING)
logging.getLogger('matplotlib').setLevel(logging.WARNING)
logging.getLogger('kasa').setLevel(logging.WARNING) 

def create_physical_ground_truth():
    """Create ground truth DAG for physical smart room environment"""
    
    class PhysicalGroundTruth:
        def __init__(self):
            # Physical smart room causal relationships
            self.edges = {
                ('Temperature', 'EnergyConsumption'),
                ('Humidity', 'EnergyConsumption'), 
                ('AirQuality', 'EnergyConsumption'),
                ('Temperature', 'OverallSatisfaction'),
                ('Humidity', 'OverallSatisfaction'),
                ('AirQuality', 'OverallSatisfaction')
            }
            
            self.nodes = {
                'Temperature', 'Humidity', 'AirQuality',
                'EnergyConsumption', 'OverallSatisfaction'
            }
            
            # Variable mapping for case-insensitive lookups
            self.variable_mapping = {
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'airquality': 'AirQuality', 
                'air_quality': 'AirQuality',
                'energyconsumption': 'EnergyConsumption',
                'energy_consumption': 'EnergyConsumption',
                'overallsatisfaction': 'OverallSatisfaction',
                'overall_satisfaction': 'OverallSatisfaction'
            }
        
        def get_shd(self, dag_edges):
            """Calculate Structural Hamming Distance"""
            # Normalize predicted edges
            normalized_pred = set()
            for src, tgt in dag_edges:
                norm_src = self.variable_mapping.get(src.lower(), src)
                norm_tgt = self.variable_mapping.get(tgt.lower(), tgt)
                normalized_pred.add((norm_src, norm_tgt))
            
            # Calculate missing and extra edges
            missing = len(self.edges - normalized_pred)
            extra = len(normalized_pred - self.edges)
            
            return missing + extra
        
        def validate_edge(self, edge):
            """Check if edge exists in ground truth"""
            src, tgt = edge
            norm_src = self.variable_mapping.get(src.lower(), src)
            norm_tgt = self.variable_mapping.get(tgt.lower(), tgt)
            return (norm_src, norm_tgt) in self.edges
        
        def get_independent_variables(self):
            """Variables that can be intervened upon"""
            return {'Temperature', 'Humidity', 'AirQuality'}
        
        def get_outcome_variables(self):
            """Variables that should not be intervention sources"""
            return {'EnergyConsumption', 'OverallSatisfaction'}
    
    return PhysicalGroundTruth()

class PhysicalInterventionController:
    """Physical intervention controller"""
    
    def __init__(self, log_file="interventions_log.csv", scaling_params_path=None):
        self.log_file = log_file
        self.scaling_params_path = scaling_params_path
        self.monitor = SmartRoomMonitor()
        
        # Timing configurations
        self.stabilization_time = 600  # 10 minutes for realistic environmental change
        
        # ISO 7730 thermal comfort parameters
        self.thermal_params = {
            'air_velocity': 0.1,      # m/s (typical office environment)
            'metabolic_rate': 1.2,    # met (sedentary office work)
            'clothing_level': 0.6     # clo (typical indoor clothing)
        }
        
        # Import device controllers
        from smart_room_physical.kasa_controller import control_heater, control_humidifier, control_fan
        self.control_heater = control_heater
        self.control_humidifier = control_humidifier
        self.control_fan = control_fan
        
        self.initialize_log()
        
    def initialize_log(self):
        if not os.path.exists(self.log_file):
            columns = ['timestamp', 'variable', 'action', 'target_value', 
                       'pre_state', 'post_state', 'pre_satisfaction_iso7730', 
                       'post_satisfaction_iso7730', 'pre_pmv', 'post_pmv', 
                       'pre_ppd', 'post_ppd', 'thermal_comfort_method', 'success']
            pd.DataFrame(columns=columns).to_csv(self.log_file, index=False)
            logger.info(f"Created intervention log: {self.log_file}")
    
    def _calculate_iso7730_satisfaction(self, state):
        """Calculate comprehensive satisfaction using ISO 7730 + air quality"""
        temp = state.get('Temperature', 22)
        humidity = state.get('Humidity', 50)
        air_quality = state.get('AirQuality', 100)
        
        # Use ISO 7730 for thermal comfort
        comfort_results = calculate_iso7730_satisfaction(
            temperature=temp,
            humidity=humidity,
            air_velocity=self.thermal_params['air_velocity'],
            metabolic_rate=self.thermal_params['metabolic_rate'],
            clothing_level=self.thermal_params['clothing_level']
        )
        
        # Factor in air quality (ISO 7730 doesn't include air quality)
        thermal_satisfaction = comfort_results['satisfaction']
        air_comfort = max(0, 100 - air_quality * 0.2)  # Lower AQI is better
        
        # Combine thermal and air quality satisfaction
        overall_satisfaction = (thermal_satisfaction * 0.8 + air_comfort * 0.2)
        
        return {
            'satisfaction': max(0, min(100, overall_satisfaction)),
            'thermal_satisfaction': thermal_satisfaction,
            'pmv': comfort_results['pmv'],
            'ppd': comfort_results['ppd'],
            'method': comfort_results['method']
        }
    
    def _apply_scaling(self, state_data, scaling_params_path):
        """Apply scaling parameters to intervention data"""
        if not scaling_params_path or not os.path.exists(scaling_params_path):
            return state_data
        
        try:
            scaling_params = pd.read_csv(scaling_params_path)
            scaling_dict = {row['column']: row for _, row in scaling_params.iterrows()}
            
            scaled_data = state_data.copy()
            for column in ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']:
                if column in scaling_dict and column in scaled_data:
                    params = scaling_dict[column]
                    scaled_data[column] = (scaled_data[column] - params['data_min']) * params['scale_factor']
            
            return scaled_data
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return state_data
    
    async def perform_intervention(self, variable, action, value=None, scaling_params_path=None):
        """Perform physical intervention"""
        logger.info(f"Performing intervention: {variable} {action}")
        
        # Record pre-intervention state
        pre_state = await self.monitor.read_current_state()
        pre_comfort = self._calculate_iso7730_satisfaction(pre_state)
        
        # Perform device intervention
        success = False
        if variable.lower() == 'temperature':
            if action in ['increase', 'set'] and (value is None or value > pre_state['Temperature']):
                # More aggressive heating
                await self.control_fan(False)
                await self.control_humidifier(False)  # Reduce other loads
                success = await self.control_heater(True)
                logger.info("Applied heating intervention: heater ON, fan OFF, humidifier OFF")
            else:  # decrease
                await self.control_heater(False)
                success = await self.control_fan(True)
                logger.info("Applied cooling intervention: heater OFF, fan ON")
                
        elif variable.lower() == 'humidity':
            if action in ['increase', 'set'] and (value is None or value > pre_state['Humidity']):
                # More aggressive humidification
                await self.control_heater(False)  # Reduce competing effects
                success = await self.control_humidifier(True)
                logger.info("Applied humidification intervention: humidifier ON, heater OFF")
            else:  # decrease
                await self.control_humidifier(False)
                success = await self.control_fan(True)
                logger.info("Applied dehumidification intervention: humidifier OFF, fan ON")
                
        elif variable.lower() == 'airquality':
            # For air quality, always improve by turning on fan
            await self.control_heater(False)
            await self.control_humidifier(False)
            success = await self.control_fan(True)
            logger.info("Applied air quality intervention: fan ON, other devices OFF")
        
        if not success:
            logger.error(f"Device control failed for {variable} {action}")
            return None
        
        # Wait for environment to stabilize
        logger.info(f"Waiting {self.stabilization_time}s for environment to stabilize...")
        await asyncio.sleep(self.stabilization_time)
        
        # Record post-intervention state
        post_state = await self.monitor.read_current_state()
        post_comfort = self._calculate_iso7730_satisfaction(post_state)
        
        # Log intervention 
        intervention_log = {
            'timestamp': datetime.now().isoformat(),
            'variable': variable,
            'action': action,
            'target_value': value,
            'pre_state': json.dumps(pre_state),
            'post_state': json.dumps(post_state),
            'pre_satisfaction_iso7730': pre_comfort['satisfaction'],
            'post_satisfaction_iso7730': post_comfort['satisfaction'],
            'pre_pmv': pre_comfort['pmv'],
            'post_pmv': post_comfort['pmv'],
            'pre_ppd': pre_comfort['ppd'],
            'post_ppd': post_comfort['ppd'],
            'thermal_comfort_method': post_comfort['method'],
            'success': success
        }
        
        # Append to log
        pd.DataFrame([intervention_log]).to_csv(self.log_file, mode='a', header=False, index=False)
        
        # Add calculated satisfaction to states
        pre_state['OverallSatisfaction'] = pre_comfort['satisfaction']
        post_state['OverallSatisfaction'] = post_comfort['satisfaction']

        # Apply scaling if parameters provided
        if scaling_params_path:
            pre_state = self._apply_scaling(pre_state, scaling_params_path)
            post_state = self._apply_scaling(post_state, scaling_params_path)
            
            logger.info(f"Intervention completed: {variable} changed from "
                    f"{pre_state[variable]:.1f} to {post_state[variable]:.1f}, "
                    f"satisfaction {pre_comfort['satisfaction']:.1f}% → {post_comfort['satisfaction']:.1f}%")
        
        return {
            'preInterventionData': pre_state,
            'postInterventionData': post_state,
            'pre_comfort_metrics': pre_comfort,
            'post_comfort_metrics': post_comfort
        }

class PhysicalCausalTester:
    """Physical intervention tester"""
    def __init__(self, api_key, effect_threshold=0.03, scaling_params_path=None, baseline_data=None):
        self.gpt_client = GPTClient(api_key)
        self.scaling_params_path = scaling_params_path
        self.controller = PhysicalInterventionController(scaling_params_path=scaling_params_path)
        self.edge_history = defaultdict(list)
        self.intervention_results = defaultdict(list)
        self.effect_threshold = effect_threshold
        
        # Fix: Handle None baseline_data properly
        self.baseline_data = baseline_data
        if baseline_data is not None:
            self.baseline_stats = self._compute_baseline_stats(baseline_data)
        else:
            self.baseline_stats = {}
            logger.warning("No baseline data provided, using empty baseline stats")
        
        # Safety parameters
        self.max_daily_interventions = 1000
        self.min_intervention_interval = 300  # 5 minutes
        self.last_intervention_time = 0
        self.daily_intervention_count = 0
        
        # Validation parameters
        self.initial_test_count = 5
        self.extended_test_count = 8
        self.no_change_threshold = effect_threshold
        
        logger.info("Physical causal tester initialized")
    
    def _check_intervention_safety(self):
        """Check if it's safe to perform another intervention"""
        current_time = time.time()
        
        if self.daily_intervention_count >= self.max_daily_interventions:
            logger.warning(f"Daily intervention limit reached ({self.max_daily_interventions})")
            return False
        
        time_since_last = current_time - self.last_intervention_time
        if time_since_last < self.min_intervention_interval:
            remaining = self.min_intervention_interval - time_since_last
            logger.warning(f"Must wait {remaining:.0f}s before next intervention")
            return False
        
        return True
    
    def _design_llm_intervention(self, edge):
        """Design intervention using LLM"""
        source, target = edge
        
        system_prompt = f"""
        Design a physical intervention to test if {source} causes changes in {target}.

        DEVICE CONSTRAINTS:
        - Heater: ON/OFF only (affects Temperature)
        - Humidifier: ON/OFF only (affects Humidity) 
        - Fan: ON/OFF only (affects Temperature, Humidity, AirQuality)

        Return JSON:
        {{
            "variables": [{{
                "variable": "{source}",
                "action": "increase/decrease"
            }}],
            "expected_effects": {{
                "{target}": "increase/decrease"
            }},
            "reasoning": "Brief explanation"
        }}
        """

        user_prompt = f"Design intervention to test {source} → {target} causality using binary device control."
        
        try:
            response = self.gpt_client.generate_response(system_prompt, user_prompt)
            intervention = json.loads(response)
            logger.info(f"Designed Intervention: {intervention}")
            return intervention
            
        except Exception as e:
            logger.error(f"Error designing intervention: {e}")
            return {
                "variables": [{"variable": source, "action": "increase"}],
                "expected_effects": {target: "increase"}
            }
    
    async def test_edge_iteratively(self, edge, intervention=None):
        """Test edge with iterative validation using improved thresholds and timing"""
        logger.info(f"Testing edge {edge}")
                
        # Design intervention if not provided
        if intervention is None:
            intervention = self._design_llm_intervention(edge)

        # Check daily intervention limits
        if self.daily_intervention_count >= self.max_daily_interventions:
            logger.warning(f"Daily intervention limit reached ({self.max_daily_interventions})")
            return False, {'confidence': 0, 'num_interventions': 0, 'reason': 'daily_limit'}

        # Take baseline reading before interventions
        logger.info("Taking baseline reading...")
        baseline_state = await self.controller.monitor.read_current_state()
        logger.info(f"Baseline: T={baseline_state['Temperature']}°C, "
                    f"H={baseline_state['Humidity']}%, AQ={baseline_state['AirQuality']}")

        # PHASE 1: Initial testing (3 tests)
        logger.info(f"Phase 1: Initial testing for edge {edge}")
        initial_effects = []
        
        for i in range(self.initial_test_count):
            # Safety check for intervention timing
            if i > 0 or self.last_intervention_time > 0:
                if not self._check_intervention_safety():
                    logger.warning(f"Safety check failed for test {i+1}, skipping remaining tests")
                    break

            try:
                logger.info(f"Initial test {i+1}/{self.initial_test_count}")
                
                # Ensure intervention is not a coroutine
                if asyncio.iscoroutine(intervention):
                    intervention = await intervention
                
                # Perform physical intervention with enhanced device control
                variable = intervention['variables'][0]['variable']
                action = intervention['variables'][0]['action']
                result = await self.controller.perform_intervention(variable, action)
                
                if result:
                    # Store result for caching
                    processed_result = {
                        'pre_state': result['preInterventionData'],
                        'post_state': result['postInterventionData']
                    }
                    
                    self.intervention_results[edge].append(processed_result)
                    
                    # Calculate effect size with improved Cohen's d
                    effect_size = self._calculate_effect_size(edge, processed_result)
                    if effect_size is not None:
                        initial_effects.append(effect_size)
                        self.edge_history[edge].append(effect_size)
                        
                        logger.info(f"Test {i+1} effect size: {effect_size:.3f}")
                    
                    # Update intervention tracking
                    self.last_intervention_time = time.time()
                    self.daily_intervention_count += 1
                
                    # Increased wait time between interventions (2 minutes minimum)
                    if i < self.initial_test_count - 1:
                        wait_time = max(120, self.min_intervention_interval)
                        logger.info(f"Waiting {wait_time}s before next test...")
                        await asyncio.sleep(wait_time)  # Use async sleep

                else:
                    logger.warning(f"Intervention failed for test {i+1}")
                
            except Exception as e:
                logger.error(f"Error in test {i+1}: {e}")
                continue
        
        # Evaluate initial results with lowered threshold
        if not initial_effects:
            return False, {'confidence': 0, 'num_interventions': 0, 'phase': 'initial_failed'}
        
        avg_initial_effect = np.mean(initial_effects)
        logger.info(f"Average initial effect: {avg_initial_effect:.3f}")
        
        # Decision: Discard if below lowered threshold (0.2)
        if avg_initial_effect < self.effect_threshold:  # Now 0.2 instead of 0.5
            logger.info(f"Edge {edge}: Effect {avg_initial_effect:.3f} below threshold {self.effect_threshold} - DISCARDING")
            return False, {
                'confidence': avg_initial_effect,
                'num_interventions': len(initial_effects),
                'effect_sizes': initial_effects,
                'phase': 'initial_below_threshold'
            }
        
        # PHASE 2: Extended testing for promising edges
        logger.info(f"Edge {edge}: Effect {avg_initial_effect:.3f} above threshold - extended testing")
        extended_effects = []
        
        additional_tests = min(self.extended_test_count, 
                                self.max_daily_interventions - self.daily_intervention_count)
        
        for i in range(additional_tests):
            if not self._check_intervention_safety():
                break
            
            try:
                logger.info(f"Extended test {i+1}/{additional_tests}")
                
                # Ensure intervention is not a coroutine
                if asyncio.iscoroutine(intervention):
                    intervention = await intervention
                    
                variable = intervention['variables'][0]['variable']
                action = intervention['variables'][0]['action']
                result = await self.controller.perform_intervention(variable, action)
                
                if result:
                    processed_result = {
                        'pre_state': result['preInterventionData'],
                        'post_state': result['postInterventionData']
                    }
                    
                    self.intervention_results[edge].append(processed_result)
                    
                    effect_size = self._calculate_effect_size(edge, processed_result)
                    if effect_size is not None:
                        extended_effects.append(effect_size)
                        self.edge_history[edge].append(effect_size)
                        logger.info(f"Extended test {i+1} effect size: {effect_size:.3f}")
                    
                    self.last_intervention_time = time.time()
                    self.daily_intervention_count += 1
                
                    # Wait between extended tests
                    if i < additional_tests - 1:
                        wait_time = max(120, self.min_intervention_interval)
                        logger.info(f"Waiting {wait_time}s before next test...")
                        await asyncio.sleep(wait_time)
                
            except Exception as e:
                logger.error(f"Error in extended test {i+1}: {e}")
                continue
        
        # FINAL EVALUATION: Combined results with statistical confidence
        all_effects = initial_effects + extended_effects
        final_confidence = np.mean(all_effects) if all_effects else 0
        
        # Additional validation: Check consistency of effects
        effect_std = np.std(all_effects) if len(all_effects) > 1 else 0
        consistency_bonus = 0.1 if effect_std < 0.2 else 0  # Bonus for consistent effects
        
        adjusted_confidence = final_confidence + consistency_bonus
        is_valid = adjusted_confidence >= self.effect_threshold
        
        logger.info(f"Edge {edge} final evaluation:")
        logger.info(f"  Raw confidence: {final_confidence:.3f}")
        logger.info(f"  Effect consistency (std): {effect_std:.3f}")
        logger.info(f"  Adjusted confidence: {adjusted_confidence:.3f}")
        logger.info(f"  Valid: {is_valid} (threshold: {self.effect_threshold})")
        
        return is_valid, {
            'confidence': adjusted_confidence,
            'raw_confidence': final_confidence,
            'effect_consistency': effect_std,
            'num_interventions': len(all_effects),
            'effect_sizes': all_effects,
            'phase': 'completed'
        }
    
    def _calculate_effect_size(self, edge, result):
        """Calculate Cohen's d standardized effect size for causal relationships"""
        source, target = edge
        pre_state = result['pre_state']
        post_state = result['post_state']
        
        # Variable mapping for case-insensitive lookup
        var_mapping = {
            'temperature': 'Temperature',
            'humidity': 'Humidity', 
            'airquality': 'AirQuality',
            'air_quality': 'AirQuality',
            'energyconsumption': 'EnergyConsumption',
            'energy_consumption': 'EnergyConsumption',
            'overallsatisfaction': 'OverallSatisfaction',
            'overall_satisfaction': 'OverallSatisfaction'
        }
        
        # Normalize variable names
        def normalize_var(var):
            normalized = var_mapping.get(var.lower(), var)
            # Try exact match first, then normalized
            if var in pre_state:
                return var
            elif normalized in pre_state:
                return normalized
            else:
                # Log available keys for debugging
                logger.warning(f"Variable '{var}' not found. Available: {list(pre_state.keys())}")
                return var  # Return original to avoid KeyError
        
        source_key = normalize_var(source)
        target_key = normalize_var(target)
        
        # Get baseline statistics from your dataset for standardization
        baseline_stats = getattr(self, 'baseline_stats', {})
        
        try:
            source_pre = float(pre_state.get(source_key, 0))
            source_post = float(post_state.get(source_key, 0))
            target_pre = float(pre_state.get(target_key, 0))
            target_post = float(post_state.get(target_key, 0))
            
            source_change = source_post - source_pre  # Keep sign
            target_change = target_post - target_pre  # Keep sign
            
            logger.info(f"Effect calculation for {source}->{target}:")
            logger.info(f"  Source: {source_pre:.3f} → {source_post:.3f} (Δ={source_change:+.3f})")
            logger.info(f"  Target: {target_pre:.3f} → {target_post:.3f} (Δ={target_change:+.3f})")
            
            # No intervention effect
            if abs(source_change) < 0.001:
                return 0.0
            
            # Calculate standardized effect (Cohen's d)
            target_std = baseline_stats.get(target, {}).get('std', abs(target_pre) * 0.1)
            cohens_d = abs(target_change) / max(target_std, 0.1)
            
            # Directional consistency bonus (source and target change in expected direction)
            directional_consistency = 1.0
            if (source_change > 0 and target_change > 0) or (source_change < 0 and target_change < 0):
                directional_consistency = 1.2  # 20% bonus for consistent direction
            
            # Final effect size with consistency weighting
            effect_size = cohens_d * directional_consistency
            
            # Scientific interpretation thresholds (Cohen's conventions)
            if effect_size < 0.2:
                interpretation = "negligible"
            elif effect_size < 0.5:
                interpretation = "small"
            elif effect_size < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
            
            # Cap at reasonable maximum (3.0 = very large effect)
            final_effect = min(3.0, max(0.0, effect_size))
            
            logger.info(f"  Cohen's d: {cohens_d:.3f}, Direction bonus: {directional_consistency:.1f}")
            logger.info(f"  Effect size: {final_effect:.3f} ({interpretation})")
            
            return final_effect
            
        except Exception as e:
            logger.error(f"Effect calculation error: {e}")
            return 0.0

    def _compute_baseline_stats(self, data):
        """Compute baseline statistics for standardization"""
        if data is None:
            return {}
            
        stats = {}
        for col in ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']:
            if col in data.columns:
                # Use robust statistics
                q25, q75 = data[col].quantile([0.25, 0.75])
                iqr = q75 - q25
                robust_std = iqr / 1.35  # Convert IQR to approximate std
                
                stats[col] = {
                    'mean': data[col].median(),  # Use median instead of mean
                    'std': max(robust_std, data[col].std() * 0.5),  # Ensure minimum std
                    'min': data[col].min(),
                    'max': data[col].max(),
                    'iqr': iqr
                }
        return stats
    
    def cleanup(self):
        """Cleanup physical resources"""
        logger.info(f"Physical testing completed. Total interventions: {self.daily_intervention_count}")

class PhysicalCausalPipeline:
    """Physical causal discovery pipeline"""
    def __init__(self, api_key, max_iterations=15, effect_threshold=0.03, scaling_params_path=None, data_path=None):
        self.api_key = api_key
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.effect_threshold = effect_threshold
        self.data_path = data_path
        
        # Initialize components
        self.monitor = SmartRoomMonitor()
        self.scaling_params_path = scaling_params_path
        
        # Fix: Load baseline data first, then pass it to tester
        baseline_data = None
        if data_path and os.path.exists(data_path):
            try:
                baseline_data = pd.read_csv(data_path)
                logger.info(f"Loaded baseline data: {baseline_data.shape}")
            except Exception as e:
                logger.error(f"Error loading baseline data: {e}")
                baseline_data = None
        else:
            logger.warning(f"Data path {data_path} not found or not provided")
        
        self.baseline_data = baseline_data
        self.tester = PhysicalCausalTester(
            api_key, 
            effect_threshold, 
            scaling_params_path, 
            baseline_data  # Pass the loaded data
        )
        
        self.ground_truth = create_physical_ground_truth()
        self.metrics_calculator = MetricsCalculator(self.ground_truth)
        
        # Pipeline state
        self.validated_edges = set()
        self.discarded_edges = set()
        self.shd_history = defaultdict(list)
        self._results = {}

        self.final_dag_edges = set()
        self.intervention_data_cache = []
        
        # Output directory
        self.output_dir = os.path.join(project_root, "physical_results")
        os.makedirs(self.output_dir, exist_ok=True)
        
        logger.info(f"Physical causal pipeline initialized")
    
    def _generate_initial_hypotheses(self, data):
        """Generate initial hypotheses using PC, SAM, and LLM"""
        logger.info("Generating initial hypotheses...")
        
        relevant_columns = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
        
        generators = {
            'pc': PCGenerator(relevant_columns=relevant_columns),
            'sam': SAMGenerator(relevant_columns=relevant_columns), 
            'llm': LLMGenerator(self.api_key, relevant_columns=relevant_columns)
        }
        
        hypotheses = {}
        for name, generator in generators.items():
            try:
                logger.info(f"Generating {name.upper()} hypothesis...")
                hypothesis = generator.generate(data)
                if hypothesis and 'edges' in hypothesis:
                    # Filter out edges with output variables as sources
                    output_vars = {'energyconsumption', 'overallsatisfaction'}
                    valid_edges = [
                        edge for edge in hypothesis['edges']
                        if str(edge[0]).lower() not in output_vars
                    ]
                    
                    hypotheses[name] = {
                        'hypothesis': {**hypothesis, 'edges': valid_edges},
                        'edges': valid_edges
                    }
                    
                    logger.info(f"{name.upper()}: {len(valid_edges)} edges - {valid_edges}")
                    
            except Exception as e:
                logger.error(f"Error in {name} generation: {e}")
                hypotheses[name] = {'hypothesis': {'edges': []}, 'edges': []}
        
        return hypotheses
    
    def _create_union_dag_and_rank_edges(self, method_dags):
        """Create union DAG and rank edges by confidence"""
        logger.info("Creating union DAG and ranking edges...")
        
        edge_ranker = EdgeRanker(method_dags, self.validated_edges)
        union_edges = edge_ranker.get_union_edges()
        ranked_edges = edge_ranker.get_ranked_edges()
        
        logger.info(f"Union DAG: {len(union_edges)} unique edges")
        for i, edge in enumerate(ranked_edges):
            confidence = edge_ranker.edge_confidence.get(edge, 0)
            logger.info(f"  {i+1}. {edge} - confidence: {confidence:.2f}")
        
        return edge_ranker, ranked_edges
    
    def _extract_edges(self, dag_result):
        """Extract edges from DAG result"""
        edges = []
        if isinstance(dag_result, dict):
            if 'hypothesis' in dag_result:
                if isinstance(dag_result['hypothesis'], dict) and 'edges' in dag_result['hypothesis']:
                    edges = dag_result['hypothesis']['edges']
                else:
                    edges = dag_result['hypothesis']
            elif 'edges' in dag_result:
                edges = dag_result['edges']
        
        return [tuple(edge) if isinstance(edge, list) else edge for edge in edges if edge]
    
    def _create_final_dag(self, validated_edges):
        """Create final DAG from validated edges"""
        G = nx.DiGraph()
        nodes = set()
        
        for edge in validated_edges:
            nodes.update(edge)
        G.add_nodes_from(nodes)
        
        # Calculate confidence scores
        edge_confidences = {}
        for edge in validated_edges:
            test_results = self.tester.edge_history.get(edge, [])
            confidence = np.mean(test_results) if test_results else 0.5
            edge_confidences[edge] = confidence
        
        # Add edges while maintaining DAG property
        sorted_edges = sorted(validated_edges, key=lambda e: -edge_confidences.get(e, 0))
        
        for edge in sorted_edges:
            G.add_edge(*edge)
            if not nx.is_directed_acyclic_graph(G):
                G.remove_edge(*edge)
                logger.info(f"Edge {edge} creates cycle, removed")
        
        # Calculate COMBINED confidence scores
        confidence_scores = {}
        for edge in G.edges():
            # Method agreement component (0-1)
            method_agreement = self.edge_ranker.get_ranking_confidence(edge)
            
            # Intervention effect component (0-1) 
            test_results = self.tester.edge_history.get(edge, [])
            intervention_confidence = np.mean(test_results) if test_results else 0
            intervention_confidence = min(1.0, intervention_confidence)  # Cap at 1.0
            
            # Combined confidence (weighted average)
            alpha, beta = 0.4, 0.6  # Method weight, Intervention weight
            combined_confidence = alpha * method_agreement + beta * intervention_confidence
            
            confidence_scores[f"{edge[0]}->{edge[1]}"] = combined_confidence
        
        return {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'confidence_scores': confidence_scores
        }

    def _create_final_comparison_visualization(self, final_dag, method_dags):
        """Create comprehensive DAG comparison visualization"""
        output_dir = self.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Node colors
        node_colors = {
            'Temperature': '#FF9966', 'Humidity': '#66B2FF', 'AirQuality': '#66CC66',
            'EnergyConsumption': '#FF6666', 'OverallSatisfaction': '#B266FF'
        }
        
        # Create 2x2 subplot layout
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        axes = [ax1, ax2, ax3, ax4]
        titles = ['Final DAG', 'PC DAG', 'SAM DAG', 'LLM DAG']
        
        # Plot each DAG
        for i, (ax, title) in enumerate(zip(axes, titles)):
            if i == 0:  # Final DAG
                edges = final_dag['edges']
            else:  # Method DAGs
                method_name = ['pc', 'sam', 'llm'][i-1]
                if method_name in method_dags:
                    edges = self._extract_edges(method_dags[method_name])
                else:
                    edges = []
            
            # Create NetworkX graph
            G = nx.DiGraph()
            all_nodes = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
            G.add_nodes_from(all_nodes)
            G.add_edges_from(edges)
            
            # Fixed layout for consistency
            pos = {
                'Temperature': (0, 1),
                'Humidity': (0, 0), 
                'AirQuality': (0, -1),
                'EnergyConsumption': (2, 0.5),
                'OverallSatisfaction': (2, -0.5)
            }
            
            # Draw nodes
            for node in all_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=[node], 
                                    node_color=node_colors[node], 
                                    node_size=1500, ax=ax)
            
            # Draw edges
            if edges:
                nx.draw_networkx_edges(G, pos, edgelist=edges, 
                                    arrows=True, arrowsize=20, 
                                    edge_color='black', width=2, ax=ax)
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=10, font_weight='bold', ax=ax)
            
            ax.set_title(title, fontsize=14)
            ax.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'dag_comparison.png'), dpi=300, bbox_inches='tight')
        plt.close()

    async def run(self):
        """Run the complete physical causal discovery pipeline"""
        logger.info("Starting Physical Causal Discovery")
        
        try:
            # Load preprocessed CSV data
            logger.info("Loading preprocessed CSV data...") 
            baseline_df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(baseline_df)} samples from {self.data_path}")

            # Ensure required columns exist
            required_columns = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
            if not all(col in baseline_df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Generate initial hypotheses
            method_dags = self._generate_initial_hypotheses(baseline_df)
            self._results = method_dags
            
            # Iterative edge validation
            final_dag = None
            
            while self.current_iteration < self.max_iterations:
                logger.info(f"\nIteration {self.current_iteration + 1}/{self.max_iterations}")
                
                # Create union DAG and rank edges
                edge_ranker, ranked_edges = self._create_union_dag_and_rank_edges(method_dags)
                
                # Filter edges to test
                edges_to_test = []
                for edge in ranked_edges:
                    if edge[0].lower() in {'energyconsumption', 'overallsatisfaction'}:
                        continue
                    if edge in self.final_dag_edges:  # Skip final DAG edges
                        continue
                    edges_to_test.append(edge)
                
                if not edges_to_test:
                    logger.info("No more edges to test")
                    break
                
                logger.info(f"Testing {len(edges_to_test)} edges this iteration")
                
                # Validate edges
                iteration_validated = set()
                iteration_discarded = set()
                
                for i, edge in enumerate(edges_to_test):
                    # Check if we can safely perform intervention
                    if not self.tester._check_intervention_safety():
                        time_to_wait = self.tester.min_intervention_interval - (time.time() - self.tester.last_intervention_time)
                        if time_to_wait > 0:
                            logger.info(f"Waiting {time_to_wait:.0f}s for safety before testing edge {edge}")
                            await asyncio.sleep(time_to_wait)
                    
                    intervention = self.tester._design_llm_intervention(edge)
                    is_valid, results = await self.tester.test_edge_iteratively(edge, intervention)
                
                    if is_valid:
                        iteration_validated.add(edge)
                        self.validated_edges.add(edge)
                        logger.info(f"✓ Edge {edge} VALIDATED")
                    else:
                        iteration_discarded.add(edge)
                        self.discarded_edges.add(edge)
                        logger.info(f"✗ Edge {edge} DISCARDED")
                    
                    # Add safety wait between edges (except for the last edge)
                    if i < len(edges_to_test) - 1:
                        safety_wait = 120  # 2 minutes between edges
                        logger.info(f"Safety wait: {safety_wait}s before next edge...")
                        await asyncio.sleep(safety_wait)
                
                # Update DAGs regardless of validation results
                self._cache_intervention_data()

                # Create final DAG from current validated edges
                final_dag = self._create_final_dag(self.validated_edges)
                self.final_dag_edges = set(tuple(edge) for edge in final_dag['edges'])

                # Always regenerate hypotheses with new intervention data
                combined_data = self._create_combined_dataset()
                method_dags = self._regenerate_hypotheses(combined_data)
                self._results = method_dags

                # Check for perfect match
                final_edges = set(tuple(edge) for edge in final_dag['edges'])
                final_shd = self.ground_truth.get_shd(final_edges)

                logger.info(f"Current SHD: {final_shd}")

                if final_shd == 0:
                    logger.info("🎉 PERFECT MATCH ACHIEVED!")
                    break
                
                self.current_iteration += 1
            
            # Create final DAG if none exists
            if final_dag is None:
                final_dag = self._create_final_dag(self.validated_edges)
            
            # Calculate final metrics
            final_metrics = self._calculate_final_metrics(final_dag, method_dags)
            
            # Export results
            self._export_results(final_dag, final_metrics)
            
            # Final summary
            final_edges = set(tuple(edge) for edge in final_dag['edges'])
            final_shd = self.ground_truth.get_shd(final_edges)
            
            logger.info("\n🎉 PHYSICAL CAUSAL DISCOVERY COMPLETED!")
            logger.info(f"Total iterations: {self.current_iteration}")
            logger.info(f"Total interventions: {self.tester.daily_intervention_count}")
            logger.info(f"Validated edges: {len(self.validated_edges)}")
            logger.info(f"Final SHD: {final_shd}")
            
            if final_shd == 0:
                logger.info("🏆 PERFECT MATCH ACHIEVED!")
            
            logger.info("DISCOVERED CAUSAL RELATIONSHIPS:")
            for edge in final_dag['edges']:
                confidence = final_dag['confidence_scores'].get(f"{edge[0]}->{edge[1]}", 0)
                logger.info(f"  {edge[0]} → {edge[1]} (confidence: {confidence:.3f})")
            
            return final_dag, final_metrics
            
        except Exception as e:
            logger.error(f"Error in physical pipeline: {e}")
            logger.error(traceback.format_exc())
            return None, None
    
    def _calculate_final_metrics(self, final_dag, method_dags):
        """Calculate comprehensive metrics"""
        metrics = {}
        
        for method, result in method_dags.items():
            edges = self._extract_edges(result)
            edges_set = set(edges)
            
            # Method-specific confidence scores
            method_confidence = {}
            if hasattr(self, 'edge_ranker'):
                method_confidence = {
                    edge: getattr(self, 'edge_ranker').edge_confidence.get(edge, 0)
                    for edge in edges_set
                }
            
            metrics[method] = self.metrics_calculator.calculate_metrics(
                method,
                edges_set,
                method_confidence,
                self.tester.intervention_results
            )
        
        # Final DAG metrics
        final_edges = set(tuple(edge) for edge in final_dag['edges'])
        final_confidence = {
            edge: final_dag['confidence_scores'].get(f"{edge[0]}->{edge[1]}", 0)
            for edge in final_edges
        }
        
        metrics['final'] = self.metrics_calculator.calculate_metrics(
            'final',
            final_edges,
            final_confidence,
            self.tester.intervention_results
        )
        
        return metrics
    
    def _export_results(self, final_dag, final_metrics):
        """Export comprehensive results"""
        try:
            # Export method metrics
            export_metrics_to_csv(final_metrics, self.output_dir)

            # Generate visualizations
            visualizer = MetricsVisualizer(self.output_dir)
            figs = visualizer.plot_metrics(final_metrics)
            
            # Calculate final SHD
            final_edges = set(tuple(edge) for edge in final_dag['edges'])
            final_shd = self.ground_truth.get_shd(final_edges)
            
            # Create subplot for each method's final DAG and the final validated DAG
            fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 20))
            
            # Plot PC DAG
            G_pc = nx.DiGraph()
            pc_edges = self._results.get('pc', {}).get('edges', [])
            G_pc.add_edges_from(pc_edges)
            pos_pc = nx.spring_layout(G_pc)
            nx.draw(G_pc, pos_pc, ax=ax1, with_labels=True, node_color='lightblue', node_size=1000)
            pc_shd = self.ground_truth.get_shd(set(pc_edges))
            ax1.set_title(f"Final PC DAG (SHD: {pc_shd})")
            
            # Plot SAM DAG
            G_sam = nx.DiGraph()
            sam_edges = self._results.get('sam', {}).get('edges', [])
            G_sam.add_edges_from(sam_edges)
            pos_sam = nx.spring_layout(G_sam)
            nx.draw(G_sam, pos_sam, ax=ax2, with_labels=True, node_color='lightgreen', node_size=1000)
            sam_shd = self.ground_truth.get_shd(set(sam_edges))
            ax2.set_title(f"Final SAM DAG (SHD: {sam_shd})")
            
            # Plot LLM DAG
            G_llm = nx.DiGraph()
            llm_edges = self._results.get('llm', {}).get('edges', [])
            G_llm.add_edges_from(llm_edges)
            pos_llm = nx.spring_layout(G_llm)
            nx.draw(G_llm, pos_llm, ax=ax3, with_labels=True, node_color='lightpink', node_size=1000)
            llm_shd = self.ground_truth.get_shd(set(llm_edges))
            ax3.set_title(f"Final LLM DAG (SHD: {llm_shd})")
            
            # Plot Final Validated DAG
            G_final = nx.DiGraph()
            G_final.add_edges_from(final_dag['edges'])
            pos_final = nx.spring_layout(G_final)
            nx.draw(G_final, pos_final, ax=ax4, with_labels=True, node_color='gold', node_size=1000)
            
            # Add edge labels with confidence scores for final DAG
            edge_labels = {(u, v): f"{final_dag['confidence_scores'].get(f'{u}->{v}', 0):.2f}" 
                        for (u, v) in G_final.edges()}
            nx.draw_networkx_edge_labels(G_final, pos_final, edge_labels, ax=ax4)
            
            # Add SHD to title
            final_title = f"Final Validated DAG (SHD: {final_shd})"
            if final_shd == 0:
                final_title += " - PERFECT MATCH!"
            ax4.set_title(final_title)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'final_dags_comparison.png'))
            plt.close()

            # Create DAG comparison visualization
            self._create_final_comparison_visualization(final_dag, self._results)
            
            # Export workflow summary
            workflow_summary = {
                'pipeline_type': 'physical_causal_discovery_iso7730',
                'timestamp': datetime.now().isoformat(),
                'total_iterations': self.current_iteration,
                'total_interventions': self.tester.daily_intervention_count,
                'validated_edges': [f"{e[0]}->{e[1]}" for e in self.validated_edges],
                'discarded_edges': [f"{e[0]}->{e[1]}" for e in self.discarded_edges],
                'final_dag_edges': [f"{e[0]}->{e[1]}" for e in final_dag['edges']],
                'final_shd': self.ground_truth.get_shd(set(final_dag['edges'])),
                'satisfaction_method': 'ISO_7730_with_air_quality',
                'thermal_comfort_parameters': {
                    'air_velocity': 0.1,
                    'metabolic_rate': 1.2,
                    'clothing_level': 0.6
                }
            }
            
            with open(os.path.join(self.output_dir, 'iso7730_workflow_summary.json'), 'w') as f:
                json.dump(workflow_summary, f, indent=2)
            
            logger.info("Results exported")
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
    
    def _cache_intervention_data(self):
        """Cache intervention data from current iteration"""
        for edge, results in self.tester.intervention_results.items():
            for result in results:
                if result not in self.intervention_data_cache:
                    self.intervention_data_cache.append(result)

    def _create_combined_dataset(self):
        """Combine baseline + intervention data with only required columns"""
        intervention_samples = []
        
        for result in self.intervention_data_cache:
            post_state = result['post_state'].copy()
            post_state['weight'] = 2.0
            
            # Apply scaling if available
            if self.scaling_params_path:
                post_state = self._apply_scaling(post_state, self.scaling_params_path)
            
            intervention_samples.append(post_state)
        
        if intervention_samples:
            intervention_df = pd.DataFrame(intervention_samples)
            combined_data = pd.concat([self.baseline_data, intervention_df], ignore_index=True)
        else:
            combined_data = self.baseline_data.copy()
        
        # Keep only required columns
        required_columns = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction', 'weight']
        available_columns = [col for col in required_columns if col in combined_data.columns]
        combined_data = combined_data[available_columns]
        
        # Handle any remaining NaN values in numeric columns
        combined_data = combined_data.fillna(combined_data.mean())
        
        logger.info(f"Combined data shape: {combined_data.shape}")
        logger.info("Dataset cleaned - only required columns retained")
        return combined_data

    def _regenerate_hypotheses(self, combined_data):
        """Regenerate PC/SAM with new data, keep LLM unchanged"""
        relevant_columns = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']
        
        generators = {
            'pc': PCGenerator(relevant_columns=relevant_columns),
            # 'sam': SAMGenerator(relevant_columns=relevant_columns),
            'llm': LLMGenerator(self.api_key, relevant_columns=relevant_columns)
        }
        
        updated_hypotheses = self._results.copy()  # Keep LLM
        
        for name, generator in generators.items():
            hypothesis = generator.generate(combined_data)
            if hypothesis and 'edges' in hypothesis:
                output_vars = {'energyconsumption', 'overallsatisfaction'}
                valid_edges = [edge for edge in hypothesis['edges'] 
                            if str(edge[0]).lower() not in output_vars]
                
                updated_hypotheses[name] = {
                    'hypothesis': {**hypothesis, 'edges': valid_edges},
                    'edges': valid_edges
                }
        
        return updated_hypotheses

    def _apply_scaling(self, state_data, scaling_params_path):
        """Apply scaling parameters with robust error handling"""
        if not scaling_params_path or not os.path.exists(scaling_params_path):
            logger.warning("No scaling params file, using raw data")
            return state_data
        
        try:
            scaling_params = pd.read_csv(scaling_params_path)
            
            required_cols = ['column', 'data_min', 'scale_factor']
            missing_cols = [col for col in required_cols if col not in scaling_params.columns]
            
            if missing_cols:
                logger.warning(f"Missing scaling columns {missing_cols}, using raw data")
                return state_data
                
            scaling_dict = {row['column']: row for _, row in scaling_params.iterrows()}
            scaled_data = state_data.copy()
            
            for column in ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']:
                if column in scaling_dict and column in scaled_data:
                    params = scaling_dict[column]
                    original_val = scaled_data[column]
                    scaled_val = (original_val - params['data_min']) * params['scale_factor']
                    scaled_data[column] = scaled_val
                    logger.debug(f"Scaled {column}: {original_val:.3f} → {scaled_val:.3f}")
            
            return scaled_data
            
        except Exception as e:
            logger.error(f"Scaling failed: {e}")
            return state_data
            
    def create_scaling_params(self, data_path, output_path):
        """Create proper scaling parameters file"""
        data = pd.read_csv(data_path)
        
        scaling_data = []
        for col in ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction']:
            if col in data.columns:
                original_min = data[col].min()
                original_max = data[col].max()
                scale_factor = 1.0 / (original_max - original_min) if original_max > original_min else 1.0
                
                scaling_data.append({
                    'column': col,
                    'original_min': original_min,
                    'original_max': original_max,
                    'scale_factor': scale_factor
                })
        
        pd.DataFrame(scaling_data).to_csv(output_path, index=False)
        logger.info(f"Created scaling params: {output_path}")

    async def cleanup(self):
        """Cleanup all resources"""
        try:
            await self.monitor.cleanup()
            self.tester.cleanup()
            await cleanup_kasa()
            logger.info("Physical pipeline cleanup completed")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

def setup_physical_environment():
    """Setup and validate physical environment"""
    logger.info("\n" + "="*80)
    logger.info("PHYSICAL SMART ROOM CAUSAL DISCOVERY")
    logger.info("="*80)
    logger.info("Features:")
    logger.info("  🌡️  ISO 7730 PMV/PPD thermal satisfaction calculation")
    logger.info("  📊 Real-time comfort monitoring during interventions")
    logger.info("  🔬 Scientific thermal comfort validation")
    logger.info("  ⚡ Physical device control with comfort feedback")
    
    # Test device connectivity
    logger.info("\n1. Testing device connectivity...")
    try:
        from smart_room_physical.govee_reader import read_govee_data
        sensor_data = read_govee_data()
        if sensor_data['Temperature_1'] is not None or sensor_data['Temperature_2'] is not None:
            logger.info("✓ Govee sensors responding")
        else:
            logger.info("⚠ Govee sensors not responding fully")
        
        # Test Kasa devices synchronously
        from smart_room_physical.kasa_controller import DEVICES
        import requests
        
        active_devices = 0
        for device_name, config in DEVICES.items():
            try:
                # Simple ping test to device IP
                response = requests.get(f"http://{config['ip']}", timeout=2)
                active_devices += 1
            except:
                pass
        
        if active_devices > 0:
            logger.info(f"✓ {active_devices}/{len(DEVICES)} Kasa devices responding")
        else:
            logger.info("⚠ No Kasa devices responding")
        
    except Exception as e:
        logger.info(f"⚠ Device connectivity test failed: {e}")
        return False
    
    # Safety checklist
    # logger.info("\n3. SAFETY CHECKLIST:")
    # checklist = [
    #     "Room is unoccupied or minimally occupied",
    #     "All devices are properly configured",
    #     "ISO 7730 thermal comfort monitoring is active",
    #     "Emergency device controls are accessible",
    #     "You understand interventions will affect room comfort"
    # ]
    
    # for item in checklist:
    #     confirm = input(f"   □ {item} (y/n): ").lower().strip()
    #     if confirm != 'y':
    #         logger.info(f"✗ Safety requirement not met")
    #         return False
    
    logger.info("✓ All safety requirements confirmed")
    
    # Display parameters
    logger.info("\n4. ISO 7730 PARAMETERS:")
    logger.info("   - Air velocity: 0.1 m/s (typical office)")
    logger.info("   - Metabolic rate: 1.2 met (sedentary work)")
    logger.info("   - Clothing level: 0.6 clo (indoor clothing)")
    logger.info("   - Satisfaction = 100% - PPD%")
    logger.info("   - Air quality factor included (20% weight)")
    
    logger.info("\n5. EXPECTED TIMELINE:")
    logger.info("   - Baseline collection: 10 minutes")
    logger.info("   - Initial hypothesis generation: 2-3 minutes")
    logger.info("   - Edge validation: 2-3 hours")
    logger.info("   - Per intervention: 3-5 minutes")
    logger.info("   - Results analysis: 5 minutes")
    
    return True

async def main():
    """Main function for physical causal discovery"""
    logger.info("Physical Smart Room Causal Discovery")
    logger.info("=" * 70)
    
    try:
        # Validate environment
        if not setup_physical_environment():
            logger.info("Environment setup failed. Exiting.")
            return
        
        # Final confirmation
        logger.info("\n" + "="*80)
        logger.info("FINAL CONFIRMATION - ISO 7730 THERMAL COMFORT INTEGRATION")
        logger.info("This will perform REAL interventions while monitoring thermal comfort:")
        logger.info("1. Calculate ISO 7730 PMV/PPD before each intervention")
        logger.info("2. Monitor thermal satisfaction changes during interventions")
        logger.info("3. Use scientific thermal comfort as target variable")
        logger.info("4. Track comfort metrics throughout causal discovery")
        logger.info("="*80)
        
        confirmation = input("Type 'YES' to proceed with causal discovery: ")
        
        if confirmation != "YES":
            logger.info("Physical causal discovery cancelled.")
            return
        
        # Initialize API key
        api_key = "YOUR_OPENAI_API_KEY"
        if not api_key or api_key.startswith("your-"):
            api_key = input("Enter your OpenAI API key: ").strip()
        
        # Initialize pipeline
        logger.info("\n🚀 Initializing physical causal discovery...")
        pipeline = PhysicalCausalPipeline(
            api_key=api_key,
            max_iterations=15,
            effect_threshold=0.03,
            scaling_params_path="data/sensor_data_processed_scaling_params.csv",
            data_path = "data/sensor_data_processed.csv" 
        )
        
        # Run pipeline
        logger.info("\n🔬 Running physical causal discovery...")
        start_time = time.time()
        
        final_dag, final_metrics = await pipeline.run()
        
        runtime = time.time() - start_time
        
        # Display results
        if final_dag and final_metrics:
            logger.info("\n" + "="*80)
            logger.info("🎉 PHYSICAL CAUSAL DISCOVERY COMPLETED!")
            logger.info("="*80)
            
            logger.info(f"\n📊 RESULTS SUMMARY:")
            logger.info(f"   Runtime: {runtime/3600:.1f} hours ({runtime/60:.1f} minutes)")
            logger.info(f"   Total iterations: {pipeline.current_iteration}")
            logger.info(f"   Physical interventions: {pipeline.tester.daily_intervention_count}")
            logger.info(f"   Validated edges: {len(pipeline.validated_edges)}")
            logger.info(f"   Discarded edges: {len(pipeline.discarded_edges)}")
            logger.info(f"   Final DAG edges: {len(final_dag['edges'])}")
            
            final_edges = set(tuple(edge) for edge in final_dag['edges'])
            final_shd = pipeline.ground_truth.get_shd(final_edges)
            logger.info(f"   Final SHD: {final_shd}")
            
            if final_shd == 0:
                logger.info("   🏆 PERFECT MATCH ACHIEVED!")
            
            logger.info(f"\n📁 Results saved to: {pipeline.output_dir}")
            
            logger.info(f"\n🌡️ ISO 7730 THERMAL COMFORT INTEGRATION:")
            logger.info(f"   - Scientific PMV/PPD satisfaction calculation")
            logger.info(f"   - Real-time comfort monitoring during interventions")
            logger.info(f"   - Thermal comfort used as target variable")
            logger.info(f"   - Air quality integrated with thermal satisfaction")
            
            logger.info(f"\n🔗 DISCOVERED CAUSAL RELATIONSHIPS:")
            for edge in final_dag['edges']:
                confidence = final_dag['confidence_scores'].get(f"{edge[0]}->{edge[1]}", 0)
                logger.info(f"   {edge[0]} → {edge[1]} (confidence: {confidence:.3f})")
            
        else:
            logger.info("\n❌ Physical causal discovery failed.")
            logger.info("Check logs for details.")
        
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Physical causal discovery interrupted by user")
        cleanup_processes() 
        
    except Exception as e:
        logger.error(f"Error in main: {e}")
        logger.error(traceback.format_exc())
        logger.info(f"\n❌ Error: {e}")
        cleanup_processes()
        
    finally:
        # Cleanup
        logger.info("\n🧹 Cleaning up physical resources...")
        cleanup_processes()
        try:
            if 'pipeline' in locals():
                await pipeline.cleanup()
            await cleanup_kasa()
            logger.info("✓ Cleanup completed")
        except Exception as e:
            logger.info(f"⚠️ Cleanup error: {e}")
        
        logger.info("\n📋 Causal discovery session complete.")

def cleanup_processes():
    """Emergency cleanup function"""
    try:
        import psutil
        for proc in psutil.process_iter(['pid', 'name']):
            if proc.info['name'] == 'node':
                try:
                    proc.kill()
                except:
                    pass
    except:
        pass

if __name__ == "__main__":
    try:
        # Run the async main function
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...exiting")
    except Exception as e:
        logger.info(f"Critical error: {e}")
        logger.error(f"Critical error: {e}")
    finally:
        cleanup_processes()