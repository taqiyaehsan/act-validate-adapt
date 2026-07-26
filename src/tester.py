"""
Intervention-Based Edge Validation (do-Operator Testing)
==========================================================

Tests candidate causal edges by performing do-operator interventions
on the simulator and checking whether the predicted effect materialises.

Components:
  - **EdgeValidator**:      Core validation logic.  Sets the source variable
                            to a fixed value via the simulator, observes the
                            target, and checks effect size against threshold.
  - **HypothesisTester**:   Orchestrates batch validation across all ranked
                            edges.  Manages the simulator subprocess lifecycle
                            (spawn, step, kill) and collects pre/post states.

Intervention protocol per edge (source → target):
  1. Record baseline state (pre-intervention snapshot)
  2. Set source = forced_value via simulator do-operator
  3. Step simulator forward for N timesteps
  4. Record post-intervention state
  5. Compute effect = |target_post − target_pre| / σ_target
  6. Accept edge if effect > effect_threshold

DOMAIN-AGNOSTIC KNOBS:
  - ``effect_threshold``:     Minimum standardised effect to accept (default 0.01).
  - ``smart_room_path``:      Path to the simulator executable.  The tester
                              invokes it as a subprocess with ``--single-step``
                              JSON commands.  Swap for your domain's simulator.
  - ``intervention_values``:  What forced value to apply.  Currently uses
                              0.0 and 1.0 (normalised extremes).
  - ``n_intervention_steps``: How many simulator steps after intervening.
  - ``cleanup``:              Kills orphan simulator processes via psutil.

Paper reference: Section 4.1 (Intervention Validation Protocol).
"""

import json
import random
import numpy as np
from collections import defaultdict
from .gpt_client import GPTClient
import subprocess
import os
import time
import logging
import networkx as nx
import psutil, tempfile
from scipy import stats

logger = logging.getLogger(__name__)


class EdgeValidator:
    """New class to handle edge validation logic"""

    # Physical ranges for range-normalized effect size computation
    _RANGES = {
        'temperature': 17,       # 15-32
        'humidity': 60,          # 20-80
        'co2': 1650,             # 350-2000
        'lightlevel': 1200,      # 0-1200
        'noisedb': 65,           # 25-90
        'airquality': 100,       # 0-100
        'pmv': 6,                # -3 to +3
        'hvacpower': 100,        # 0-100
        'lightingpower': 100,    # 0-100
        'energyconsumption': 100,# 0-100
        'overallsatisfaction': 100,
        'satisfaction': 100,
        'outdoortemp': 16,       # 7-23
        'solarradiation': 800,   # 0-800
        'occupancy': 8,          # 0-8
        'windowposition': 1,     # 0-1
        'occupantcount': 20,
    }

    # Physical minimums (used with _RANGES for physical→[0,1] normalization)
    _MINS = {
        'temperature': 15,
        'humidity': 20,
        'co2': 350,
        'lightlevel': 0,
        'noisedb': 25,
        'airquality': 0,
        'pmv': -3,
        'hvacpower': 0,
        'lightingpower': 0,
        'energyconsumption': 0,
        'overallsatisfaction': 0,
        'satisfaction': 0,
        'outdoortemp': 7,
        'solarradiation': 0,
        'occupancy': 0,
        'windowposition': 0,
        'occupantcount': 0,
        # ASHRAE variables (already [0,1] normalized in data)
        'air_temperature': 0,
        'dew_temperature': 0,
        'sea_level_pressure': 0,
        'meter_reading': 0,
        'square_feet': 0,
        'year_built': 0,
    }

    def __init__(self, effect_threshold=0.01):
        self.effect_threshold = effect_threshold

    def _get_variable_range(self, var_name):
        """Get physical range span for a variable (for normalization)."""
        return self._RANGES.get(var_name.lower(), 1.0)

    @staticmethod
    def _find_key(state, var_name):
        """Find key in state dict case-insensitively with alias support."""
        vl = var_name.lower()
        for key in state.keys():
            kl = key.lower()
            if kl == vl:
                return key
            if vl == 'satisfaction' and kl == 'overallsatisfaction':
                return key
            if vl == 'overallsatisfaction' and kl == 'satisfaction':
                return key
            if vl == 'occupantcount' and kl == 'occupancy':
                return key
            if vl == 'occupancy' and kl == 'occupantcount':
                return key
        return None

    def validate_edge_changes(self, pre_state, post_state, source_var, target_var):
        """
        Compute **signed** effect size for an intervention run.

        Returning a signed value (positive when target co-moves with source,
        negative when it moves against) enables the t-test in
        ``_compute_statistical_confidence`` to distinguish true causal edges
        (consistent direction → mean ≠ 0, low p-value) from confounded edges
        (random direction due to regime resampling → mean ≈ 0, high p-value).

        Reference: Eberhardt & Scheines (2007) — soft interventions shift
        distributions; direction consistency is the causal signal.
        """
        source_key = self._find_key(pre_state, source_var)
        target_key = self._find_key(pre_state, target_var)

        if source_key is None:
            raise KeyError(f"Unknown source variable: {source_var}. Available: {list(pre_state.keys())}")
        if target_key is None:
            raise KeyError(f"Unknown target variable: {target_var}. Available: {list(pre_state.keys())}")

        try:
            source_pre = float(pre_state[source_key])
            source_post = float(post_state[source_key])
            target_pre = float(pre_state[target_key])
            target_post = float(post_state[target_key])

            source_change = source_post - source_pre  # SIGNED
            target_change = target_post - target_pre  # SIGNED

            if abs(source_change) < 0.001:
                return 0.0

            # Normalize by physical ranges
            source_range = self._get_variable_range(source_key)
            target_range = self._get_variable_range(target_key)
            source_norm = source_change / max(source_range, 0.01)
            target_norm = target_change / max(target_range, 0.01)

            # Signed effect: target fraction per source fraction.
            # Positive = target co-moves with source.
            # For true causal edges this has a CONSISTENT sign.
            # For confounded edges the sign FLIPS randomly (regime noise).
            signed_effect = target_norm / max(abs(source_norm), 0.01)

            return signed_effect

        except (KeyError, ValueError, TypeError) as e:
            logger.error(f"Error validating edge changes: {str(e)}")
            return None

    # In EdgeValidator class in tester.py, update validate_edge_changes method:
    # def validate_edge_changes(self, pre_state, post_state, source_var, target_var):
    #     try:
    #         # Handle collision as boolean if needed
    #         if target_var == 'collision':
    #             # Convert to boolean if stored as string or number
    #             target_pre = bool(pre_state[target_var])
    #             target_post = bool(post_state[target_var])
                
    #             # Convert spatial coordinates to floats
    #             source_pre = float(pre_state[source_var])
    #             source_post = float(post_state[source_var])
                
    #             source_change = abs(source_post - source_pre)
    #             # For boolean targets, any change is significant
    #             target_change = 1.0 if target_pre != target_post else 0.0
                
    #             # If target changed with minimal source change, strong effect
    #             if source_change < 0.001 and target_change > 0:
    #                 return 1.0
                    
    #             # Calculate normalized effect size
    #             return target_change if source_change < 0.001 else (target_change / max(source_change, 0.01))
                
    #         # For coordinate variables (handX->ballX, handY->ballY)
    #         elif source_var in ['handX', 'handY'] and target_var in ['ballX', 'ballY']:
    #             source_pre = float(pre_state[source_var])
    #             source_post = float(post_state[source_var])
    #             target_pre = float(pre_state[target_var])
    #             target_post = float(post_state[target_var])
                
    #             source_change = abs(source_post - source_pre)
    #             target_change = abs(target_post - target_pre)
                
    #             # Check if ball is following hand (likely caught)
    #             if source_change > 0 and target_change > 0:
    #                 # Similarity in direction and magnitude indicates causal relationship
    #                 return min(1.0, target_change / max(source_change, 0.01))
    #             return 0.0
                
    #         # Default handling for other relationships
    #         else:
    #             source_pre = float(pre_state[source_var])
    #             source_post = float(post_state[source_var])
    #             target_pre = float(pre_state[target_var])
    #             target_post = float(post_state[target_var])
                
    #             source_change = abs(source_post - source_pre)
    #             target_change = abs(target_post - target_pre)
                
    #             return min(1.0, target_change / max(source_change, 0.01))
    #     except (KeyError, ValueError, TypeError) as e:
    #         logger.error(f"Error validating edge changes: {str(e)}")
    #         return None
        
class HypothesisTester:
    def __init__(self, api_key, smart_room_path, physical_mode=False):
        self.gpt_client = GPTClient(api_key)
        self.smart_room_path = smart_room_path
        # self.robot_arm_path = robot_arm_path
        self.edge_history = defaultdict(list)
        self.intervention_results = defaultdict(list)
        self.edge_validator = EdgeValidator()
        # Principled default: majority rule (>50% of runs must show effect).
        # NOT tuned on any specific dataset — this is the statistical convention
        # for rejecting the null hypothesis of no causal effect.
        self.min_confidence_threshold = 0.5
        self.variance_threshold = 0.15
        self.initial_test_count = 3
        self.extended_test_count = 7
        self.effect_threshold = 0.1
        self.cwm_ref = None
        # self.collision_threshold = 35
        self.physical_mode = physical_mode
        # If in physical mode, print a warning
        if self.physical_mode:
            print("HypothesisTester initialized in PHYSICAL MODE")
            print("Interventions will control actual devices!")
        
        self.variable_ranges = {
            'Temperature': {'min': 18.0, 'max': 30.0},
            'Humidity': {'min': 30.0, 'max': 70.0}, 
            'AirQuality': {'min': 0.0, 'max': 500.0},
            'HVACPower': {'min': 0.0, 'max': 100.0},      # Add this
            'LightingPower': {'min': 0.0, 'max': 100.0},  # Add this
            'EnergyConsumption': {'min': 10.0, 'max': 100.0},  # Add this
            'OverallSatisfaction': {'min': 0.0, 'max': 100.0},
            'Satisfaction': {'min': 0.0, 'max': 100.0},     # open_window alias
            'ThermalComfort': {'min': 0.0, 'max': 100.0},      # Add this
            'VisualComfort': {'min': 0.0, 'max': 100.0},       # Add this
            'AirQualityIndex': {'min': 0.0, 'max': 500.0},     # Add this
            'HVACSetpoint': {'min': 18.0, 'max': 30.0},        # Add this
            'LightingLevel': {'min': 0.0, 'max': 1.0},         # Add this
            'OccupantCount': {'min': 0.0, 'max': 20.0},         # Add this
            # Smart building rich simulator variables
            'OutdoorTemp': {'min': 0.0, 'max': 30.0},
            'SolarRadiation': {'min': 0.0, 'max': 800.0},
            'Occupancy': {'min': 0, 'max': 8},
            'WindowPosition': {'min': 0.0, 'max': 1.0},
            'CO2': {'min': 350.0, 'max': 2000.0},
            'LightLevel': {'min': 0.0, 'max': 1200.0},
            'NoiseDB': {'min': 25.0, 'max': 90.0},
            'PMV': {'min': -3.0, 'max': 3.0},
        }
        
        # self.variable_ranges = {
        #     'Temperature': {'min': 18, 'max': 30, 'step': 0.5},
        #     'Humidity': {'min': 30, 'max': 70, 'step': 1.0},
        #     'AirQuality': {'min': 0, 'max': 500, 'step': 10}
        # }
        # self.variable_ranges = {
        #     'handX': {'min': 0, 'max': 400, 'step': 10},
        #     'handY': {'min': 0, 'max': 300, 'step': 10},
        #     'ballX': {'min': 0, 'max': 400, 'step': 10},
        #     'ballY': {'min': 0, 'max': 300, 'step': 10}
        # }
        
        # Comprehensive variable mapping
        self.variable_mapping = {
            # Temperature variations
            'temperature': 'Temperature',
            'Temperature': 'Temperature',
            'TEMPERATURE': 'Temperature',
            
            # Humidity variations
            'humidity': 'Humidity',
            'Humidity': 'Humidity',
            'HUMIDITY': 'Humidity',
            
            # Air Quality variations
            'airquality': 'AirQuality',
            'air quality': 'AirQuality',
            'air_quality': 'AirQuality',
            'AirQuality': 'AirQuality',
            'Airquality': 'AirQuality',
            'AIRQUALITY': 'AirQuality',
            'AIR_QUALITY': 'AirQuality',
            'airQuality': 'AirQuality',
            
            # HVAC Setpoint variations
            'hvacsetpoint': 'HVACSetpoint',
            'hvac setpoint': 'HVACSetpoint',
            'hvac_setpoint': 'HVACSetpoint',
            'HVACSetpoint': 'HVACSetpoint',
            'Hvacsetpoint': 'HVACSetpoint',
            'HVACSETPOINT': 'HVACSetpoint',
            'HVAC_SETPOINT': 'HVACSetpoint',
            'hvacSetpoint': 'HVACSetpoint',
            
            # Lighting Level variations
            'lightinglevel': 'LightingLevel',
            'lighting level': 'LightingLevel',
            'lighting_level': 'LightingLevel',
            'LightingLevel': 'LightingLevel',
            'Lightinglevel': 'LightingLevel',
            'LIGHTINGLEVEL': 'LightingLevel',
            'LIGHTING_LEVEL': 'LightingLevel',
            'lightingLevel': 'LightingLevel',
            
            # Occupant Count variations
            'occupantcount': 'OccupantCount',
            'occupant count': 'OccupantCount',
            'occupant_count': 'OccupantCount',
            'OccupantCount': 'OccupantCount',
            'Occupantcount': 'OccupantCount',
            'OCCUPANTCOUNT': 'OccupantCount',
            'OCCUPANT_COUNT': 'OccupantCount',
            'occupantCount': 'OccupantCount',
            'occupancy': 'OccupantCount',
            'Occupancy': 'OccupantCount',
            
            # Energy Consumption variations
            'energyconsumption': 'EnergyConsumption',
            'energy consumption': 'EnergyConsumption',
            'energy_consumption': 'EnergyConsumption',
            'EnergyConsumption': 'EnergyConsumption',
            'Energyconsumption': 'EnergyConsumption',
            'ENERGYCONSUMPTION': 'EnergyConsumption',
            'ENERGY_CONSUMPTION': 'EnergyConsumption',
            'energyConsumption': 'EnergyConsumption',
            
            # Overall Satisfaction variations (all map to 'Satisfaction')
            'overallsatisfaction': 'Satisfaction',
            'overall satisfaction': 'Satisfaction',
            'overall_satisfaction': 'Satisfaction',
            'OverallSatisfaction': 'Satisfaction',
            'Overallsatisfaction': 'Satisfaction',
            'OVERALLSATISFACTION': 'Satisfaction',
            'OVERALL_SATISFACTION': 'Satisfaction',
            'overallSatisfaction': 'Satisfaction',
            'satisfaction': 'Satisfaction',
            'Satisfaction': 'Satisfaction',
            'SATISFACTION': 'Satisfaction',
            
            # PMV variations  
            'pmv': 'PMV',
            'PMV': 'PMV',

            # Window Open variations
            'windowopen': 'WindowOpen',
            'window open': 'WindowOpen',
            'WindowOpen': 'WindowOpen',
            'WINDOWOPEN': 'WindowOpen',

            # OutdoorTemperature variations
            'outdoortemperature': 'OutdoorTemperature',
            'outdoor temperature': 'OutdoorTemperature',
            'outdoor_temperature': 'OutdoorTemperature',
            'OutdoorTemperature': 'OutdoorTemperature',
            'Outdoortemperature': 'OutdoorTemperature',
            'OUTDOORTEMPERATURE': 'OutdoorTemperature',
            'OUTDOOR_TEMPERATURE': 'OutdoorTemperature',
            'outdoorTemp': 'OutdoorTemperature',
            'outdoortemp': 'OutdoorTemperature',
    
            # Thermal Comfort variations
            'thermalcomfort': 'ThermalComfort',
            'thermal comfort': 'ThermalComfort',
            'thermal_comfort': 'ThermalComfort',
            'ThermalComfort': 'ThermalComfort',
            'Thermalcomfort': 'ThermalComfort',
            'THERMALCOMFORT': 'ThermalComfort',
            'THERMAL_COMFORT': 'ThermalComfort',
            'thermalComfort': 'ThermalComfort',
            
            # Visual Comfort variations
            'visualcomfort': 'VisualComfort',
            'visual comfort': 'VisualComfort',
            'visual_comfort': 'VisualComfort',
            'VisualComfort': 'VisualComfort',
            'Visualcomfort': 'VisualComfort',
            'VISUALCOMFORT': 'VisualComfort',
            'VISUAL_COMFORT': 'VisualComfort',
            'visualComfort': 'VisualComfort',
            
            # Air Quality Index variations
            'airqualityindex': 'AirQualityIndex',
            'air quality index': 'AirQualityIndex',
            'air_quality_index': 'AirQualityIndex',
            'AirQualityIndex': 'AirQualityIndex',
            'Airqualityindex': 'AirQualityIndex',
            'AIRQUALITYINDEX': 'AirQualityIndex',
            'AIR_QUALITY_INDEX': 'AirQualityIndex',
            'airQualityIndex': 'AirQualityIndex',
            
            # HVAC Power variations
            'hvacpower': 'HVACPower',
            'hvac power': 'HVACPower',
            'hvac_power': 'HVACPower',
            'HVACPower': 'HVACPower',
            'Hvacpower': 'HVACPower',
            'HVACPOWER': 'HVACPower',
            'HVAC_POWER': 'HVACPower',
            'hvacPower': 'HVACPower',
            
            # Lighting Power variations
            'lightingpower': 'LightingPower',
            'lighting power': 'LightingPower',
            'lighting_power': 'LightingPower',
            'LightingPower': 'LightingPower',
            'Lightingpower': 'LightingPower',
            'LIGHTINGPOWER': 'LightingPower',
            'LIGHTING_POWER': 'LightingPower',
            'lightingPower': 'LightingPower',

            # Smart building rich — new variables
            'outdoortemp': 'OutdoorTemp',
            'OutdoorTemp': 'OutdoorTemp',
            'OUTDOORTEMP': 'OutdoorTemp',
            'outdoor_temp': 'OutdoorTemp',

            'solarradiation': 'SolarRadiation',
            'SolarRadiation': 'SolarRadiation',
            'SOLARRADIATION': 'SolarRadiation',
            'solar_radiation': 'SolarRadiation',
            'solarRadiation': 'SolarRadiation',

            'windowposition': 'WindowPosition',
            'WindowPosition': 'WindowPosition',
            'WINDOWPOSITION': 'WindowPosition',
            'window_position': 'WindowPosition',
            'windowPosition': 'WindowPosition',

            'co2': 'CO2',
            'CO2': 'CO2',
            'Co2': 'CO2',

            'lightlevel': 'LightLevel',
            'LightLevel': 'LightLevel',
            'LIGHTLEVEL': 'LightLevel',
            'light_level': 'LightLevel',
            'lightLevel': 'LightLevel',

            'noisedb': 'NoiseDB',
            'NoiseDB': 'NoiseDB',
            'NOISEDB': 'NoiseDB',
            'noise_db': 'NoiseDB',
            'noiseDB': 'NoiseDB',
            'noiseDb': 'NoiseDB',
        }
        # self.variable_mapping = {
        #     # handX variations
        #     'handx': 'handX',
        #     'handX': 'handX',
        #     'HANDX': 'handX',
        #     'hand_x': 'handX',
        #     'hand_X': 'handX',
            
        #     # handY variations
        #     'handy': 'handY',
        #     'handY': 'handY',
        #     'HANDY': 'handY',
        #     'hand_y': 'handY',
        #     'hand_Y': 'handY',
            
        #     # ballX variations
        #     'ballx': 'ballX',
        #     'ballX': 'ballX',
        #     'BALLX': 'ballX',
        #     'ball_x': 'ballX',
        #     'ball_X': 'ballX',
            
        #     # ballY variations
        #     'bally': 'ballY',
        #     'ballY': 'ballY',
        #     'BALLY': 'ballY',
        #     'ball_y': 'ballY',
        #     'ball_Y': 'ballY',
            
        #     # collision variations
        #     'collision': 'collision',
        #     'Collision': 'collision',
        #     'COLLISION': 'collision'
        # }

    def test_hypothesis(self, hypothesis, method_name):
        edge_results = []
        intervention_count = 0
        logger.info(f"Testing hypothesis from {method_name} with {len(hypothesis['edges'])} edges")

        # Non-intervenable variables: use pipeline config if available,
        # plus always-skip metadata columns.
        non_intervenable = set(getattr(self, '_non_intervenable_vars', set()))
        non_intervenable |= {'timestamp', 'elapsedtime', 'interventionapplied'}

        unique_edges = set(tuple(edge) if isinstance(edge, list) else edge 
                         for edge in hypothesis['edges'])
        
        for edge in set(map(tuple, hypothesis['edges'])):
            try:
                source, target = edge
                
                # Skip if source or target is non-intervenable
                if source.lower() in non_intervenable:
                    logger.info(f'Cannot intervene on source variable: {source}')
                    continue
                
                if target.lower() in non_intervenable:
                    logger.info(f'Skipping non-intervenable target variable: {target}')
                    continue
                
                intervention = self._design_llm_intervention(edge)
                logger.info(f"Designed intervention for edge {edge}: {json.dumps(intervention, indent=2)}")
                results = self._run_simulation(edge, intervention)
                
                if results:
                    intervention_count += 1
                    effect_size = self._analyze_simulation_result(edge, results)
                    edge_results.append({
                        'edge': edge,
                        'confidence_score': effect_size,
                        'intervention_count': intervention_count
                    })
            except Exception as e:
                logger.error(f"Error testing edge {edge}: {str(e)}")
        
        return edge_results

    def _design_llm_intervention(self, edge):
        source, target = edge
        
        # Clean and map source variable
        source_clean = source.lower().replace(' ', '').replace('_', '')
        mapped_source = self.variable_mapping.get(source_clean)
        
        if not mapped_source:
            logger.error(f"Unknown source variable: {source} (cleaned: {source_clean})")
            raise ValueError(f"Cannot intervene on variable: {source}")
        
        range_info = self.variable_ranges.get(mapped_source)
        if not range_info:
            logger.error(f"No range information for variable: {mapped_source}")
            raise ValueError(f"Cannot intervene on variable: {mapped_source}")
        
        # Clean and map target variable  
        target_clean = target.lower().replace(' ', '').replace('_', '')
        mapped_target = self.variable_mapping.get(target_clean)
        
        if not mapped_target:
            logger.error(f"Unknown target variable: {target} (cleaned: {target_clean})")
            mapped_target = "EnergyConsumption"  # Fallback target
        
        system_prompt = f"""
        Design an intervention to test if {source} causes changes in {target}.
        Variable constraints:
        - {source}: Range [{range_info['min']}, {range_info['max']}]

        # IMPORTANT: Only use "set", "increase", or "decrease" as action values.
        
        Return a valid JSON object:
        {{
            "variables": [
                {{
                    "variable": "{mapped_source}",
                    "action": "set/increase/decrease", 
                    "value": numerical_value
                }}
            ],
            "expected_effects": {{
                "{mapped_target}": "increase/decrease/unchanged"
            }}
        }}
        """

        user_prompt = f"Design optimal intervention to test if {mapped_source} affects {mapped_target}."

        # Provide more detailed information about the robot arm system
        # system_context = f"""
        #     # Robot Arm System Information:
            
        #     The system is a robot arm simulation with the following key variables:
        #     - handX: X-coordinate of the robot hand (range: {range_info['min']}-{range_info['max']})
        #     - handY: Y-coordinate of the robot hand (range: {range_info['min']}-{range_info['max']})
        #     - ballX: X-coordinate of the ball (range: {range_info['min']}-{range_info['max']})
        #     - ballY: Y-coordinate of the ball (range: {range_info['min']}-{range_info['max']})
        #     - collision: Boolean indicating if hand and ball are colliding (distance < 35 units)
            
        #     # Key Causal Relationships:
        #     - Hand position (handX, handY) affects collision state
        #     - Ball position (ballX, ballY) affects collision state
        #     - When collision occurs, the ball moves with the hand
        #     - Collision happens when distance between hand and ball < 35 units
            
        #     # Physics Rules:
        #     - Distance = sqrt((handX - ballX)^2 + (handY - ballY)^2)
        #     - If Distance < 35: collision = true, else: collision = false
        # """
        
        # Create a more detailed prompt for the LLM
        # system_prompt = f"""
        #     {system_context}
            
        #     Your task: Design an intervention to test if {source} causes changes in {target}.
            
        #     Variable constraints:
        #     - {source}: Range [{range_info['min']}, {range_info['max']}], step {range_info['step']}
            
        #     # IMPORTANT: 
        #     - Only use "set", "increase", or "decrease" as action values
        #     - For collision relationships, include interventions that BOTH cause and prevent collisions
        #     - Design a multi-step intervention that shows clear causality
            
        #     Return a valid JSON object:
        #     {{
        #         "variables": [
        #             {{
        #                 "variable": string,  // Variable to intervene on
        #                 "action": "set/increase/decrease",  // Action to take
        #                 "value": number  // Value to set/increase/decrease by
        #             }},
        #             // Include multiple interventions if needed
        #             ...
        #         ],
        #         "expected_effects": {{
        #             "{mapped_target}": "increase/decrease/unchanged"
        #         }},
        #         "reasoning": "Explanation of why this intervention should test the causal relationship"
        #     }}
        #     """

        # user_prompt = f"""
        #     Design an optimal intervention to test if {mapped_source} causally affects {mapped_target}.
            
        #     For this test, I need to clearly see the effect of changing {mapped_source} on {mapped_target}.
        #     If {mapped_target} is 'collision', remember that:
        #     1. We need to control both hand and ball positions
        #     2. The intervention should create a situation where we can observe the relationship
        #     3. The collision state changes based on distance between hand and ball
            
        #     Please design an intervention that will provide clear evidence of causality.
        #     """

        try:
            response = self.gpt_client.generate_response(system_prompt, user_prompt)
            intervention = json.loads(response)
            
            # Validate intervention structure
            if 'variables' not in intervention or not isinstance(intervention['variables'], list):
                raise ValueError("Invalid intervention: missing or invalid 'variables' field")
            
            if not intervention['variables']:
                raise ValueError("Invalid intervention: empty variables list")
            
            # Process and validate each variable
            for variable_config in intervention['variables']:
                if not isinstance(variable_config, dict):
                    raise ValueError("Invalid variable config: must be dict")
                    
                if 'variable' not in variable_config or 'action' not in variable_config or 'value' not in variable_config:
                    raise ValueError("Invalid variable config: missing required fields")
                
                var_name = variable_config['variable']
                action = variable_config['action']
                value = variable_config['value']
                
                # Validate action
                if action not in ['set', 'increase', 'decrease']:
                    logger.warning(f"Invalid action '{action}', defaulting to 'set'")
                    action = 'set'

                # Validate value is numeric
                if not isinstance(value, (int, float)):
                    raise ValueError(f"Invalid value: {value} must be numeric")

                # Get range info and clamp value
                var_range = self.variable_ranges.get(var_name, range_info)

                # The JS simulator only understands "set" — it ignores the
                # action field.  Convert increase/decrease to absolute set
                # values using extreme values for maximum signal.
                if action == 'increase':
                    variable_config['value'] = var_range['max']
                    variable_config['action'] = 'set'
                elif action == 'decrease':
                    variable_config['value'] = var_range['min']
                    variable_config['action'] = 'set'
                else:
                    variable_config['value'] = max(var_range['min'],
                                                   min(var_range['max'], float(value)))
                
                logger.info(f"Intervention design for {var_name}: {action} to {variable_config['value']}")
            
            # Validate expected_effects if present
            if 'expected_effects' in intervention:
                for effect_var, effect_type in intervention['expected_effects'].items():
                    if effect_type not in ['increase', 'decrease', 'unchanged']:
                        logger.warning(f"Invalid expected effect '{effect_type}' for {effect_var}")
            
            # Remove reasoning if present (not needed for execution)
            if 'reasoning' in intervention:
                logger.info(f"Intervention reasoning: {intervention['reasoning']}")
                del intervention['reasoning']
            
            return intervention
            
        except (json.JSONDecodeError, ValueError, KeyError) as e:
            logger.error(f"Error parsing LLM intervention: {str(e)}")
            return self._create_fallback_intervention(mapped_source, mapped_target, range_info)
        except Exception as e:
            logger.error(f"Unexpected error in LLM intervention design: {str(e)}")
            return self._create_fallback_intervention(mapped_source, mapped_target, range_info)

    def get_current_state(self):
        """Get current simulation state without intervention"""
        try:
            # Run simulation with no intervention to get current state
            temp_intervention = {
                "variables": [{
                    "variable": "Temperature",
                    "action": "set",
                    "value": 22.0  # Neutral value
                }],
                "expected_effects": {}
            }
            result = self._run_simulation(('Temperature', 'Temperature'), temp_intervention)
            if result and len(result) > 0:
                return result[0]['post_state']
            return {}
        except:
            return {}

    def _create_fallback_intervention(self, source, target, range_info):
        """Create a safe fallback intervention"""
        mid_value = (range_info['min'] + range_info['max']) / 2
        return {
            "variables": [{
                "variable": source,
                "action": "set",
                "value": mid_value
            }],
            "expected_effects": {
                target: "unchanged"
            }
        }

    def _run_simulation(self, edge, intervention, max_retries=3, retry_delay=2):
        """
        Run a simulation with the given edge and intervention parameters.
        
        Args:
            edge: Tuple representing the causal edge being tested
            intervention: Dict containing intervention details
            max_retries: Maximum number of retry attempts
            retry_delay: Delay between retries in seconds
        
        Returns:
            List of simulation results containing pre and post intervention states
        """
        # Add a check for ASHRAE data
        if self.smart_room_path is None:
            return self._simulate_ashrae_intervention(edge, intervention)
    
        results = []
        sim_intervention = intervention['variables']
        duration = 2000  # Total simulation duration in ms
        intervention_time = 100  # When to apply intervention in ms

        for attempt in range(max_retries):
            logger.info(f"Attempt {attempt + 1} of {max_retries} retries")
            process = None
            temp_js_path = None

            try:
                with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
                    temp_js_path = temp_js.name
                    temp_js.write(f'''
                    const {{ simulateAndGetLatestData }} = require('{self.smart_room_path}');

                    async function runSimulation() {{
                        try {{
                            const result = await simulateAndGetLatestData(
                                {duration},
                                {json.dumps(sim_intervention)},
                                {intervention_time}
                            );
                            if (!result || !result.preInterventionData || !result.postInterventionData) {{
                                throw new Error('Invalid simulation result');
                            }}
                            console.log("SIMULATION_RESULT:" + JSON.stringify(result));
                            process.exit(0);
                        }} catch (error) {{
                            console.error('SIMULATION_ERROR:', error.message);
                            process.exit(1);
                        }}
                    }}
                    process.on('unhandledRejection', (error) => {{
                        console.error('SIMULATION_ERROR:', error);
                        process.exit(1);
                    }});
                    runSimulation();
                    ''')


                process = subprocess.Popen(
                    ['node', '--max-old-space-size=512', temp_js_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1,
                    universal_newlines=True
                )
                
                simulation_result = None
                output, stderr = process.communicate(timeout=60)  # 30 second timeout

                # Process all output lines
                for line in output.splitlines():
                    print(line)  # Echo Node.js output for debugging
                    if 'SIMULATION_RESULT:' in line:
                        try:
                            # Extract and parse JSON result
                            result_json = line.split('SIMULATION_RESULT:', 1)[1]
                            simulation_result = json.loads(result_json)
                        except (json.JSONDecodeError, IndexError) as e:
                            logger.error(f"JSON parsing error: {e}")
                            continue

                # Log any Node.js errors
                if stderr:
                    logger.error(f"Node.js Errors:\n{stderr}")

                # Process valid simulation results
                if simulation_result and simulation_result.get('preInterventionData') and simulation_result.get('postInterventionData'):
                    results.append({
                        'intervention_value': intervention['variables'][0]['value'],
                        'pre_state': simulation_result['preInterventionData'],
                        'post_state': simulation_result['postInterventionData']
                    })
                    logger.info(f"Successfully completed intervention {attempt + 1} for edge {edge}")
                    # logger.debug(f"\nresult: {results}\n")
                    break  # Exit on successful simulation

            except subprocess.TimeoutExpired:
                logger.error(f"Simulation timed out for edge {edge}, attempt {attempt + 1}")
                if process:
                    try:
                        # Kill process and all child processes
                        parent = psutil.Process(process.pid)
                        for child in parent.children(recursive=True):
                            child.kill()
                        parent.kill()
                    except:
                        process.kill()
            except Exception as e:
                logger.error(f"Simulation error for edge {edge}: {str(e)}")
            finally:
                # Clean up temporary files
                if temp_js_path and os.path.exists(temp_js_path):
                    try:
                        os.remove(temp_js_path)
                    except Exception as e:
                        logger.error(f"Error cleaning up {temp_js_path}: {str(e)}")
                # Ensure process is terminated
                if process:
                    try:
                        process.kill()
                    except:
                        pass

                # Wait before retry if needed
                if attempt < max_retries - 1 and not results:
                    time.sleep(retry_delay)

        return results

    def test_edge_iteratively(self, edge, intervention, additional_interventions=None):
        """Test edge with statistical significance using multiple intervention strategies.

        Args:
            edge: (source, target) tuple
            intervention: primary intervention dict (deterministic, used for initial screening)
            additional_interventions: optional list of additional intervention dicts
                (e.g., paired counterfactual, LLM-designed) used in confirmatory phase.
                If provided, the 7 confirmatory tests are distributed across these strategies.
        """
        logger.info(f"\nTesting edge {edge}")

        # Phase 1: Initial screening (3 tests) — uses primary (deterministic) intervention
        # for maximum signal. If no effect at max, the edge is likely false.
        initial_results = self._run_initial_tests(edge, intervention, self.initial_test_count)
        if not initial_results or not self._has_significant_effect(initial_results):
            result = {'confidence': 0, 'num_interventions': len(initial_results)}
            if edge in self.intervention_results and self.intervention_results[edge]:
                last_sim = self.intervention_results[edge][-1]
                result['preInterventionData'] = last_sim.get('pre_state', {})
                result['postInterventionData'] = last_sim.get('post_state', {})
            return False, result

        time.sleep(0.5)

        # Phase 2: Confirmatory tests (7 tests) — distributed across strategies
        if additional_interventions:
            # Distribute confirmatory tests across all available strategies
            all_strategies = [intervention] + additional_interventions
            extended_results = []
            tests_per_strategy = max(1, self.extended_test_count // len(all_strategies))
            remaining = self.extended_test_count - tests_per_strategy * len(all_strategies)

            for idx, strategy in enumerate(all_strategies):
                n_tests = tests_per_strategy + (1 if idx < remaining else 0)
                strategy_name = strategy.get('strategy_type', f'strategy_{idx}')
                logger.info(f"  Confirmatory: {n_tests} tests with {strategy_name}")
                results = self._run_initial_tests(edge, strategy, n_tests)
                extended_results.extend(results)
        else:
            # Fallback: all confirmatory tests use the same intervention
            extended_results = self._run_extended_tests(edge, intervention, initial_results,
                                                        self.extended_test_count)

        all_results = initial_results + extended_results

        # Compute statistical confidence
        stats_result = self._compute_statistical_confidence(edge, all_results)

        self.edge_history[edge].extend(all_results)

        # Include simulation data from most recent intervention
        if edge in self.intervention_results and self.intervention_results[edge]:
            last_sim = self.intervention_results[edge][-1]
            stats_result['preInterventionData'] = last_sim.get('pre_state', {})
            stats_result['postInterventionData'] = last_sim.get('post_state', {})

        return stats_result['confidence'] >= self.min_confidence_threshold, stats_result
    
    def _compute_statistical_confidence(self, edge, effect_sizes):
        """Compute confidence with effect size + significance"""
        
        if len(effect_sizes) < 3:
            return {'confidence': 0, 'effect_size': 0, 'p_value': 1.0, 
                    'num_interventions': len(effect_sizes)}
        
        # Remove None values
        valid_effects = [e for e in effect_sizes if e is not None]
        
        if not valid_effects:
            return {'confidence': 0, 'effect_size': 0, 'p_value': 1.0, 
                    'num_interventions': 0}
        
        mean_effect = np.mean(valid_effects)
        std_effect = np.std(valid_effects, ddof=1) if len(valid_effects) > 1 else 0
        
        # One-sample t-test: is effect significantly different from 0?
        if std_effect > 0:
            t_stat, p_value = stats.ttest_1samp(valid_effects, 0)
        else:
            # No variance - all values identical
            p_value = 0.0 if abs(mean_effect) > self.effect_threshold else 1.0
        
        # Confidence requires THREE conditions (domain-agnostic):
        #   1. Statistical significance: p < 0.05 (t-test, effect ≠ 0)
        #   2. Practical significance: |effect| > threshold (large enough to matter)
        #   3. Consistency: effects must agree in sign across repeats
        #      (real causal effects are directionally consistent;
        #       noise-driven effects flip sign across runs)
        # This triple gate is standard in causal inference:
        #   - (1) controls Type I error
        #   - (2) ensures practical relevance (Cohen 1988)
        #   - (3) filters noise-inflated effects that pass (1)+(2) by chance
        sign_consistent = all(e > 0 for e in valid_effects) or all(e < 0 for e in valid_effects)

        if p_value < 0.05 and abs(mean_effect) > self.effect_threshold and sign_consistent:
            confidence = 1.0 - p_value
        elif p_value < 0.1 and abs(mean_effect) > self.effect_threshold and sign_consistent:
            confidence = 1.0 - p_value
        else:
            confidence = 0.0
        
        return {
            'confidence': confidence,
            'effect_size': mean_effect,
            'p_value': p_value,
            'num_interventions': len(valid_effects),
            'effect_sizes': valid_effects
        }

    def compute_counterfactual_effect(self, edge, obs_data, intervention_states):
        """
        Leave-one-out counterfactual effect estimation for real-world
        deployment (physical_mode).

        Instead of paired simulations (which require a simulator with seed
        control), we estimate the counterfactual via partial correlation:

        1. Train B̂ = f(Z \\ {A}) on observational data — predicts target
           from all covariates EXCEPT the source variable.
        2. During intervention: compare B_actual with B̂(Z_current).
        3. Effect = B_actual − B̂ — the unexplained component attributable
           to the intervention on A.

        Equivalent to computing the regression coefficient of A in
        B ~ A + Z, which is a partial-correlation test.  Uses only matrix
        operations on the correlation matrix (no sklearn, <1 ms).

        In simulation mode, the paired design gives a perfect counterfactual
        (matched-seed LOW run).  This method provides the same causal
        quantity when a simulator is unavailable.

        Args:
            edge: (source, target) tuple
            obs_data: observational DataFrame for model fitting
            intervention_states: list of dicts with 'source_value' and
                                 'target_value' and covariate values

        Returns:
            List of effect sizes (same format as paired design output)
        """
        source, target = edge
        # Find column names (case-insensitive)
        src_col = next((c for c in obs_data.columns
                        if c.lower() == source.lower()), None)
        tgt_col = next((c for c in obs_data.columns
                        if c.lower() == target.lower()), None)
        if not src_col or not tgt_col:
            return []

        # All covariates except source
        covariate_cols = [c for c in obs_data.columns
                          if c != src_col and c != tgt_col]
        if not covariate_cols:
            # No covariates → marginal effect only
            return [abs(s.get('target_value', 0)) for s in intervention_states]

        # Partial correlation of source and target controlling for Z
        # via precision matrix.  This equals the standardised regression
        # coefficient β₁ in  target ~ source + Z₁ + … + Zₖ.
        cols = [src_col, tgt_col] + covariate_cols
        sub = obs_data[cols].dropna()
        if len(sub) < len(cols) + 3:
            return []

        try:
            C = sub.corr().values
            P = np.linalg.inv(C)
            partial_r = -P[0, 1] / np.sqrt(abs(P[0, 0] * P[1, 1]))
            partial_r = max(-1.0, min(1.0, partial_r))
        except np.linalg.LinAlgError:
            partial_r = 0.0

        # The partial correlation IS the effect size in standardised units.
        # Return one effect per intervention episode, scaled by the
        # source's actual perturbation magnitude for consistency with
        # the paired-design effect sizes.
        target_range = self.edge_validator._get_variable_range(target)
        effects = []
        for state in intervention_states:
            src_val = state.get('source_value', 0)
            src_range = self.edge_validator._get_variable_range(source)
            perturbation_frac = abs(src_val) / max(src_range, 0.01)
            effects.append(abs(partial_r) * perturbation_frac)

        return effects

    def _analyze_simulation_result(self, edge, simulation_results):
        if not simulation_results:
            return 0.0

        source, target = edge
        source_var = self.variable_mapping[source]
        target_var = self.variable_mapping[target]
        
        effect_sizes = []
        
        for result in simulation_results:
            pre_state = result['pre_state']
            post_state = result['post_state']

            # Access using mapped variable names.
            # JS simulators return 'OverallSatisfaction' but the canonical
            # mapped name is now 'Satisfaction'; try the mapped name first,
            # then fall back to the other variant.
            def _get(state, key):
                if key in state:
                    return float(state[key])
                alt = 'OverallSatisfaction' if key == 'Satisfaction' else (
                      'Satisfaction' if key == 'OverallSatisfaction' else None)
                if alt and alt in state:
                    return float(state[alt])
                return float(state.get(key, 0))

            source_change = _get(post_state, source_var) - _get(pre_state, source_var)
            target_change = _get(post_state, target_var) - _get(pre_state, target_var)
            
            if source_change != 0:
                effect_size = abs(target_change / source_change)
                effect_sizes.append(effect_size)

        return min(1.0, np.mean(effect_sizes)) if effect_sizes else 0.0

    def _compute_edge_statistics(self):
        edge_statistics = {}
        
        for edge, confidences in self.edge_history.items():
            if len(confidences) >= 3:
                edge_statistics[edge] = {
                    'mean_confidence': np.mean(confidences),
                    'variance': np.var(confidences),
                    'sample_size': len(confidences),
                    'confidence_interval': np.percentile(confidences, [2.5, 97.5])
                }
        
        return edge_statistics

    def _filter_robust_edges(self, edge_statistics):
        return [
            edge for edge, stats in edge_statistics.items()
            if (stats['mean_confidence'] >= self.min_confidence_threshold and 
                stats['variance'] <= self.variance_threshold and 
                stats['sample_size'] >= 3)
        ]

    def _validate_hypothesis(self, hypothesis, edge_statistics):
        # First ensure networkx can process the edges
        edges = [(str(e[0]), str(e[1])) for e in hypothesis['edges']]
        
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        if not nx.is_directed_acyclic_graph(G):
            while not nx.is_directed_acyclic_graph(G):
                cycle = nx.find_cycle(G)
                min_conf_edge = min(cycle, key=lambda e: edge_statistics[tuple(e)]['mean_confidence'])
                G.remove_edge(*min_conf_edge)
        
        # Convert back to list of edges and return validated hypothesis
        return {
            'nodes': hypothesis['nodes'],
            'edges': list(G.edges()),
            'confidence_scores': hypothesis.get('confidence_scores', {})
        }

    def _create_fallback_hypothesis(self, robust_edges):
        nodes = set()
        for edge in robust_edges:
            nodes.update(edge)
            
        confidence_scores = {}
        for edge in robust_edges:
            stats = self.edge_history[edge]
            confidence_scores[f"{edge[0]}->{edge[1]}"] = np.mean(stats)
        
        return {
            'nodes': list(nodes),
            'edges': list(robust_edges),
            'confidence_scores': confidence_scores
        }
    
    def cleanup(self):
        """Clean up resources used by the tester"""
        self.kill_node_processes()
        
        # Clean up any temporary files
        try:
            import tempfile
            import glob
            
            temp_dir = tempfile.gettempdir()
            temp_js_files = glob.glob(os.path.join(temp_dir, "tmp*.js"))
            
            for file in temp_js_files:
                try:
                    os.remove(file)
                    logger.info(f"Removed temporary file: {file}")
                except (PermissionError, FileNotFoundError):
                    pass
        except Exception as e:
            logger.error(f"Error cleaning up temporary files: {str(e)}")

    def kill_node_processes(self):
        """Kill any lingering Node.js processes"""
        try:
            for proc in psutil.process_iter(['pid', 'name']):
                if proc.info['name'] == 'node':
                    try:
                        parent = psutil.Process(proc.info['pid'])
                        if parent.is_running():
                            logger.info(f"Terminating node process: {parent.pid}")
                            parent.terminate()
                            try:
                                parent.wait(timeout=3)
                            except psutil.TimeoutExpired:
                                parent.kill()
                    except Exception as e:
                        logger.error(f"Error terminating process: {e}")
        except Exception as e:
            logger.error(f"Error in kill_node_processes: {str(e)}")
    
    # def _run_initial_tests(self, edge, intervention, num_tests=3):
    #     """Run initial batch of interventions with proper cleanup"""
    #     effect_sizes = []
    #     for _ in range(num_tests):
    #         try:
    #             result = self._run_single_intervention(edge, intervention)
    #             if result:
    #                 effect_size = self._calculate_effect_size(edge, result[0])
    #                 if effect_size is not None:
    #                     effect_sizes.append(effect_size)
    #             time.sleep(1)  # Add delay between tests
    #         except Exception as e:
    #             logger.error(f"Error in initial test: {e}")
    #             effect_sizes.append(0)
    #     return effect_sizes
    
    def _run_extended_tests(self, edge, intervention, initial_results, min_tests=5, max_tests=10):
        """Run extended testing phase with dynamic stopping"""
        effect_sizes = []
        total_tests = len(initial_results)
        
        while total_tests < max_tests:
            result = self._run_single_intervention(edge, intervention)
            if result:
                if hasattr(self, 'cwm_ref') and self.cwm_ref:
                    self.cwm_ref.intervention_count += 1
                    logger.info(f"Interventions: {self.cwm_ref.intervention_count}/20")
                
                effect_size = self._calculate_effect_size(edge, result[0])
                if effect_size is not None:
                    effect_sizes.append(effect_size)
                    total_tests += 1
                    
                    # Check if we have enough data points
                    if total_tests >= min_tests:
                        current_confidence = np.mean(initial_results + effect_sizes)
                        if current_confidence < 0.5:  # Early stopping condition
                            break
        
        return effect_sizes
    
    def _run_initial_tests(self, edge, intervention, num_tests=3):
        """Run initial batch of interventions"""
        effect_sizes = []
        for test_num in range(num_tests):
            logger.info(f"\nStarting test {test_num + 1} of {num_tests} for edge {edge}")
            try:
                result = self._run_single_intervention(edge, intervention)
                if result:
                    if hasattr(self, 'cwm_ref') and self.cwm_ref:
                        self.cwm_ref.intervention_count += 1
                        logger.info(f"Interventions: {self.cwm_ref.intervention_count}/20")

                    effect_size = self._calculate_effect_size(edge, result[0])
                    if effect_size is not None:
                        effect_sizes.append(effect_size)
                        logger.info(f"Test {test_num + 1} completed with effect size: {effect_size}")
                    else:
                        logger.info(f"Test {test_num + 1} completed but no significant effect detected")
                time.sleep(0.2)
            except Exception as e:
                logger.error(f"Test {test_num + 1} failed: {e}")
                effect_sizes.append(0)
        return effect_sizes
    
    def _run_single_intervention(self, edge, intervention):
        """Run a single intervention with timeout"""
        try:
            simulation_result = self._run_simulation(edge, intervention)
            if simulation_result:
                self.intervention_results[edge].extend(simulation_result)
            return simulation_result
        except Exception as e:
            logger.error(f"Intervention failed: {e}")
            return None
            
    def _has_significant_effect(self, effect_sizes, threshold=None):
        """Check if effect magnitude is above threshold (signed-effect aware)."""
        if not effect_sizes:
            return False
        threshold = threshold or self.edge_validator.effect_threshold
        valid_effects = [e for e in effect_sizes if e is not None]
        # Use absolute mean: signed effects cancel out for confounders
        # but reinforce for true causal edges
        return len(valid_effects) > 0 and abs(np.mean(valid_effects)) > threshold
        
    def _calculate_effect_size(self, edge, result):
        """
        Compute signed effect size for directional consistency testing.

        Combines raw signed effect with back-door adjusted effect (Pearl 2009,
        Ch. 3.3) when observational data is available. The two must agree on
        sign for the effect to be accepted:
          - Same sign → return adjusted effect (cleaner, confounders partialled)
          - Opposite sign → return 0 (contradictory evidence, likely confounded)
          - Back-door unavailable → return raw effect alone

        This prevents the OLS residual error from spuriously confirming edges
        while still benefiting from confounder adjustment on true edges.
        """
        source, target = edge
        source_var = self.variable_mapping.get(source.lower(), source)
        target_var = self.variable_mapping.get(target.lower(), target)

        # Always compute raw signed effect
        raw = self.edge_validator.validate_edge_changes(
            result['pre_state'],
            result['post_state'],
            source_var,
            target_var
        )

        # For intervenable edges (source is an actuator), trust the raw
        # intervention result directly — no back-door adjustment needed.
        # Back-door adjustment is for non-intervenable edges only (Phase 2.4).
        non_intervenable = getattr(self, '_non_intervenable_vars', set())
        if source.lower() not in non_intervenable:
            return raw  # Direct intervention result, no adjustment

        # Try back-door adjusted effect for non-intervenable edges
        obs_data = getattr(self, '_obs_data', None)
        if obs_data is not None and raw is not None and raw != 0:
            adjusted = self._backdoor_adjusted_effect(
                source_var, target_var,
                result['pre_state'], result['post_state'],
                obs_data)
            if adjusted is not None:
                # Sign agreement check: both must point same direction
                if (raw > 0 and adjusted > 0) or (raw < 0 and adjusted < 0):
                    return adjusted  # use cleaner estimate
                else:
                    # Contradictory: raw says one direction, adjusted says other
                    # → likely confounded, return 0 to weaken t-test signal
                    return 0.0

        return raw

    def _backdoor_adjusted_effect(self, source_var, target_var,
                                   pre_state, post_state, obs_data):
        """
        Back-door adjusted **signed** effect estimation.

        Combines two causal inference principles:
          1. **Back-door criterion** (Pearl, 2009, Ch. 3.3): use measured
             confounders to build a counterfactual prediction — "what would
             target be if source stayed at its pre-value but confounders
             took their post-intervention values?"
          2. **Directional consistency** (Eberhardt & Scheines, 2007):
             return a SIGNED effect so that across multiple intervention
             runs, the t-test distinguishes consistent causal effects
             (same sign) from random confounder noise (sign flips).

        The signed adjusted effect = target_post - counterfactual.
        Across runs:
          - Model residual errors are random → cancel in the mean
          - Systematic confounder bias is removed by the counterfactual
          - True causal effect persists with consistent sign

        Returns:
            Signed effect (positive = target co-moves with source
            direction), or None if fitting fails. Falls through to
            raw signed effect in the caller.
        """
        find_key = self.edge_validator._find_key

        # Resolve column names in obs_data
        def find_col(df, var_name):
            vl = var_name.lower()
            for c in df.columns:
                cl = c.lower()
                if cl == vl:
                    return c
                if vl == 'satisfaction' and cl == 'overallsatisfaction':
                    return c
                if vl == 'overallsatisfaction' and cl == 'satisfaction':
                    return c
            return None

        src_col = find_col(obs_data, source_var)
        tgt_col = find_col(obs_data, target_var)
        if not src_col or not tgt_col:
            return None

        src_key = find_key(pre_state, source_var)
        tgt_key = find_key(pre_state, target_var)
        if not src_key or not tgt_key:
            return None

        # Confounder columns: everything except source, target, timestamps
        skip = {src_col.lower(), tgt_col.lower(),
                'timestamp', 'elapsedtime', 'interventionapplied'}
        confounder_cols = [c for c in obs_data.columns
                          if c.lower() not in skip
                          and np.issubdtype(obs_data[c].dtype, np.number)]
        if not confounder_cols:
            return None

        # Fit OLS: target ~ source + confounders
        feature_cols = [src_col] + confounder_cols
        sub = obs_data[[tgt_col] + feature_cols].dropna()
        if len(sub) < len(feature_cols) + 5:
            return None

        try:
            X = sub[feature_cols].values
            y = sub[tgt_col].values
            X_aug = np.column_stack([np.ones(len(X)), X])
            beta = np.linalg.lstsq(X_aug, y, rcond=None)[0]
        except (np.linalg.LinAlgError, ValueError):
            return None

        # Normalize physical state values to [0,1] matching obs_data scale
        def to_norm(state, col):
            key = find_key(state, col)
            if key is None:
                return None
            try:
                raw = float(state[key])
            except (ValueError, TypeError):
                return None
            vl = col.lower()
            phys_range = self.edge_validator._RANGES.get(vl, None)
            phys_min = self.edge_validator._MINS.get(vl, None)
            if phys_range and phys_range > 1e-10 and phys_min is not None:
                return max(0.0, min(1.0, (raw - phys_min) / phys_range))
            cmin, cmax = obs_data[col].min(), obs_data[col].max()
            span = cmax - cmin
            return max(0.0, min(1.0, (raw - cmin) / span)) if span > 1e-10 else 0.5

        # Source change (signed, in [0,1] space)
        src_pre = to_norm(pre_state, src_col)
        src_post = to_norm(post_state, src_col)
        if src_pre is None or src_post is None:
            return None
        source_change = src_post - src_pre  # SIGNED
        # Back-door counterfactual is only reliable when the intervention
        # produced a substantial source change (≥5% of range). For smaller
        # perturbations, the OLS residual error dominates the counterfactual
        # prediction → fall through to raw signed effect in the caller.
        if abs(source_change) < 0.05:
            return None

        # Build counterfactual feature vector:
        # source at PRE value, confounders at POST values
        x_cf = [1.0]  # intercept
        for col in feature_cols:
            val = to_norm(pre_state if col == src_col else post_state, col)
            if val is None:
                return None
            x_cf.append(val)
        target_counterfactual = float(np.array(x_cf) @ beta)

        # Actual post-intervention target (normalized)
        tgt_post = to_norm(post_state, tgt_col)
        if tgt_post is None:
            return None

        # Signed adjusted effect: actual minus counterfactual.
        # This isolates the target change attributable to the source,
        # with confounder contributions partialled out.
        adjusted_delta = tgt_post - target_counterfactual  # SIGNED

        # Signed effect normalized by source change magnitude
        # (same scale as raw signed effect for threshold compatibility)
        signed_effect = adjusted_delta / abs(source_change)

        # Log for diagnostics
        tgt_pre = to_norm(pre_state, tgt_col)
        raw_delta = (tgt_post - tgt_pre) if tgt_pre is not None else 0
        logger.info(f"  Back-door adj {source_var}→{target_var}: "
                    f"raw_signed={raw_delta:.4f}, "
                    f"adj_signed={adjusted_delta:.4f}, "
                    f"effect={signed_effect:.4f}")

        return signed_effect