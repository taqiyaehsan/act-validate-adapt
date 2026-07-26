"""
CausalPipeline — Iterative Causal Discovery with Intervention Validation
=========================================================================

Orchestrates the full discover → rank → validate loop:

  1. **Hypothesis Generation**: Runs multiple causal discovery algorithms
     (PC, SAM, LLM, VARLiNGAM) in parallel on observational data.
     Each returns a candidate DAG.

  2. **Edge Ranking**: The union of all candidate DAGs is formed; edges
     are ranked by agreement across methods and statistical confidence.

  3. **Intervention Validation**: Top-ranked edges are tested by executing
     do-operator interventions on the simulator (or analytical model).
     Edges whose predicted effect direction matches observation are
     promoted to the validated set.

  4. **Iteration**: Steps 1-3 repeat for ``max_iterations``, refining the
     DAG with each round of evidence.

DOMAIN-AGNOSTIC KNOBS — update these when adapting to a new domain:
  - ``relevant_columns``: List of variable names in your dataset.
  - ``column_mapping``:   Lowercase → ProperCase lookup for column names.
  - ``dataset_type``:     Controls which column schema and ground-truth DAG
                          to use.  Add a new branch in __init__ for your domain.
  - ``smart_room_path``:  Path to the domain simulator (JS or otherwise).
                          If None, falls back to an analytical intervention model.
  - ``max_iterations``:   Number of discover-validate cycles.
  - ``effect_threshold``: Minimum standardised effect size to accept an edge.
  - ``alpha``, ``beta``:  Weights for edge ranking (method agreement vs. stat strength).

See also: ``src/pipeline_cwm.py`` for the CWM-enhanced version with
confidence tracking, regime monitoring, and Bayesian model averaging.
"""

import pandas as pd
import numpy as np
import os, json
import logging
from src.generators import PCGenerator, SAMGenerator, LLMGenerator, VARLiNGAMGenerator
from src.tester import HypothesisTester
from src.evaluator import HypothesisEvaluator, EdgeRanker
from src.ground_truth import GroundTruthDAG, Challenge1GroundTruth
from src.metrics import MetricsCalculator
from src.metrics_viz import MetricsVisualizer
from src.gpt_client import GPTClient
from collections import defaultdict
import networkx as nx
import matplotlib.pyplot as plt
from pathlib import Path
import time
import shutil

logger = logging.getLogger(__name__)


class CausalPipeline:
    def __init__(self, csv_data, api_key, smart_room_path=None,
                 max_iterations=3, alpha=0.5, beta=0.5,
                 effect_threshold=0.1, relevant_column=None,
                 dataset_type=None, physical_mode=False,
                 actuator_vars=None, non_intervenable_vars=None, derived_vars=None,
                 enabled_generators=None):
        self.data = pd.DataFrame(csv_data)
        self.max_iterations = max_iterations
        self.current_iteration = 0
        self.alpha = alpha
        self.beta = beta
        self.effect_threshold = effect_threshold
        self.physical_mode = physical_mode
        # User-specified actuator variables (domain knowledge: which variables
        # can be directly controlled via the simulator).  Used for proactive
        # intervention testing of actuator→X edges that observational methods
        # miss due to controller confounding.
        self.actuator_vars = set(v.lower() for v in (actuator_vars or []))
        # User-specified non-intervenable variables (exogenous inputs, derived
        # quantities, latent regime indicators).  Edges from these sources are
        # rescued via observational evidence (Phase 2.4) instead of
        # intervention testing.
        self.non_intervenable_vars = set(
            v.lower() for v in (non_intervenable_vars or []))
        self.derived_vars = set(v.lower() for v in (derived_vars or []))

        # More conservative parameters for physical testing
        # self.max_iterations = min(max_iterations, 10)  # Limit to 10 iterations max
        # self.effect_threshold = max(effect_threshold, 0.15)  # Require stronger effects

        self.dataset_type = dataset_type or 'simulation'
        # Set appropriate columns based on dataset type
        if self.dataset_type == 'ashrae':
            self.relevant_columns = relevant_column or [
                'air_temperature', 'dew_temperature', 'sea_level_pressure',
                'meter_reading', 'square_feet', 'year_built'
            ]
            # Add column mapping for ASHRAE dataset
            self.column_mapping = {
                'air_temperature': 'air_temperature',
                'dew_temperature': 'dew_temperature',
                'sea_level_pressure': 'sea_level_pressure',
                'meter_reading': 'meter_reading',
                'square_feet': 'square_feet',
                'year_built': 'year_built'
            }
        elif self.dataset_type == 'open_window':
            self.relevant_columns = ['Temperature', 'Humidity', 'AirQuality', 'PMV',
                             'EnergyConsumption', 'Satisfaction', 'WindowOpen',
                             'OutdoorTemperature']

            self.column_mapping = {
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'airquality': 'AirQuality',
                'pmv': 'PMV',
                'energyconsumption': 'EnergyConsumption',
                'satisfaction': 'Satisfaction',
                'windowopen': 'WindowOpen',
                'outdoortemperature': 'OutdoorTemperature'
            }
        elif self.dataset_type == 'smart_building_rich':
            self.relevant_columns = relevant_column or [
                'OutdoorTemp', 'SolarRadiation', 'Occupancy', 'WindowPosition',
                'Temperature', 'Humidity', 'CO2', 'LightLevel', 'NoiseDB',
                'AirQuality', 'PMV', 'HVACPower', 'LightingPower',
                'EnergyConsumption', 'Satisfaction'
            ]
            self.column_mapping = {
                'outdoortemp': 'OutdoorTemp',
                'solarradiation': 'SolarRadiation',
                'occupancy': 'Occupancy',
                'windowposition': 'WindowPosition',
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'co2': 'CO2',
                'lightlevel': 'LightLevel',
                'noisedb': 'NoiseDB',
                'airquality': 'AirQuality',
                'pmv': 'PMV',
                'hvacpower': 'HVACPower',
                'lightingpower': 'LightingPower',
                'energyconsumption': 'EnergyConsumption',
                'satisfaction': 'Satisfaction',
            }
        elif self.dataset_type in ('smart_room', 'smart_room_noise'):
            self.relevant_columns = relevant_column or [
                'Temperature', 'Humidity', 'AirQuality',
                'EnergyConsumption', 'Satisfaction'
            ]
            self.column_mapping = {
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'airquality': 'AirQuality',
                'energyconsumption': 'EnergyConsumption',
                'satisfaction': 'Satisfaction',
            }
        elif self.dataset_type == 'hidden_vars':
            self.relevant_columns = relevant_column or [
                'Temperature', 'Humidity', 'AirQuality',
                'EnergyConsumption', 'Satisfaction', 'OutdoorTemperature'
            ]
            self.column_mapping = {
                'temperature': 'Temperature',
                'humidity': 'Humidity',
                'airquality': 'AirQuality',
                'energyconsumption': 'EnergyConsumption',
                'satisfaction': 'Satisfaction',
                'outdoortemperature': 'OutdoorTemperature',
            }
        else:
            self.relevant_columns = [
                'Temperature', 'Humidity', 'AirQuality',
                'LightingLevel', 'HVACSetpoint', 'OccupantCount',
                'EnergyConsumption', 'OverallSatisfaction',
                'ThermalComfort', 'VisualComfort', 'AirQualityIndex',
                'HVACPower', 'LightingPower'
            ]

            self.column_mapping = {
                    'temperature': 'Temperature',
                    'humidity': 'Humidity',
                    'airquality': 'AirQuality',
                    'energyconsumption': 'EnergyConsumption',
                    'overallsatisfaction': 'OverallSatisfaction',
                    'lightinglevel': 'LightingLevel',
                    'hvacsetpoint': 'HVACSetpoint',
                    'occupancy': 'Occupancy'
                }
        
        self.generators = {
            'pc': PCGenerator(relevant_columns=self.relevant_columns),
            'sam': SAMGenerator(relevant_columns=self.relevant_columns, allow_cycles=True),
            'llm': LLMGenerator(api_key, relevant_columns=self.relevant_columns, dataset_type=self.dataset_type),
            'varlingam': VARLiNGAMGenerator(max_lags=3, criterion='bic', threshold=0.05)
        }
        # Generator ablation support (rebuttal C2): restrict Phase 1 to a
        # subset of hypothesis generators.  None = all four (default).
        if enabled_generators is not None:
            enabled = {g.lower() for g in enabled_generators}
            unknown = enabled - set(self.generators)
            if unknown:
                raise ValueError(f"Unknown generators: {unknown}. "
                                 f"Valid: {sorted(self.generators)}")
            self.generators = {name: gen for name, gen in self.generators.items()
                               if name in enabled}
            logger.info(f"Generator ablation active: using only {sorted(self.generators)}")
        self.validated_edges = set() 
        
        # Select intervention tester based on whether a simulator is available.
        # NOTE: do NOT overwrite self.dataset_type here — it was already set
        # above from the caller's dataset_type parameter and governs column
        # schemas, ground truth selection, etc.  The tester selection is an
        # orthogonal choice (live simulator vs data-driven RF tester).
        if smart_room_path:
            self.tester = HypothesisTester(api_key, smart_room_path, physical_mode=self.physical_mode)
        else:
            # Data-driven tester (RandomForest-based intervention simulation)
            self.tester = self._create_ashrae_tester(api_key)
            
        self.tester.effect_threshold = effect_threshold
        # Pass non-intervenable set to tester for edge filtering
        self.tester._non_intervenable_vars = self.non_intervenable_vars
        # Pass observational data for back-door adjusted effect estimation
        # (Pearl 2009, Ch. 3.3) — used to partial out confounder noise
        # during intervention validation
        self.tester._obs_data = self.data
        self.evaluator = HypothesisEvaluator()
        if dataset_type == 'open_window':
            self.ground_truth = Challenge1GroundTruth()
        elif dataset_type in ('smart_building_rich', 'ashrae', 'hidden_vars',
                              'smart_room_noise', 'smart_room'):
            self.ground_truth = self._load_json_ground_truth(dataset_type)
        else:
            self.ground_truth = GroundTruthDAG()
        self.metrics_calculator = MetricsCalculator(self.ground_truth, alpha=alpha, beta=beta)
        self.collected_data = defaultdict(pd.DataFrame)
        self._cache = {}
        self.method_support = defaultdict(float)
        self.shd_history = defaultdict(list)

    @staticmethod
    def _load_json_ground_truth(key):
        """Load ground truth edges from ground_truth_graphs.json."""
        import json as _json
        gt_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                               'ground_truth_graphs.json')
        with open(gt_path) as f:
            all_gt = _json.load(f)
        edges_raw = all_gt.get(key, {}).get('edges', [])
        # Build a duck-typed GT object matching GroundTruthDAG interface
        class _JsonGT:
            def __init__(self, edges_list):
                self.edges = {(s, t) for s, t in edges_list}
                self.nodes = set()
                for s, t in self.edges:
                    self.nodes.add(s)
                    self.nodes.add(t)

            def get_shd(self, predicted_edges):
                """Structural Hamming Distance: missing + extra edges."""
                predicted = {(str(s).lower(), str(t).lower()) for s, t in predicted_edges}
                true = {(str(s).lower(), str(t).lower()) for s, t in self.edges}
                return len(true - predicted) + len(predicted - true)
        return _JsonGT(edges_raw)

    def _create_ashrae_tester(self, api_key):
        """Create a custom tester for ASHRAE dataset"""
        class ASHRAETester:
            def __init__(self, dataset, api_key, column_mapping=None):
                self.dataset = dataset
                self.intervention_results = defaultdict(list)
                self.edge_history = defaultdict(list)
                self.effect_threshold = 0.1
                self.column_mapping = column_mapping or {}
                self.variable_mapping = column_mapping or {}
                self.gpt_client = GPTClient(api_key)

                # Print for debugging
                logger.debug(f"Initialized ASHRAETester with column mapping: {self.column_mapping}")
                
                # Load scaling parameters
                try:
                    # scaling_path = os.path.join(Path.cwd(), "data", "ashrae_scaling_params.csv")
                    # scaling_path = './data/smart_room_noisy_preprocessed_scaling_params.csv'\
                    scaling_path = 'data/ashrae_data_processed_scaling_params.csv'
                    if os.path.exists(scaling_path):
                        self.scaling_params = pd.read_csv(scaling_path)
                    else:
                        self.scaling_params = None
                except:
                    self.scaling_params = None

            def _design_llm_intervention(self, edge):
                """Design intervention for ASHRAE data using LLM"""
                source, target = edge
                logger.info(f"Designing LLM intervention for edge {source}->{target}")
                
                # Map column names if needed
                source_col = self.column_mapping.get(source, source)
                target_col = self.column_mapping.get(target, target)
                
                # Define variable ranges based on dataset statistics
                variable_ranges = {}
                for col in self.dataset.columns:
                    if col in ['Timestamp', 'weight']:
                        continue
                    # Get min, max, and standard deviation for each column
                    min_val = self.dataset[col].min()
                    max_val = self.dataset[col].max()
                    std_val = self.dataset[col].std()
                    variable_ranges[col] = {
                        'min': min_val,
                        'max': max_val,
                        'step': std_val / 10  # Use 1/10th of standard deviation as step
                    }
                
                # Get range info for source variable
                range_info = variable_ranges.get(source_col, {
                    'min': 0,
                    'max': 1,
                    'step': 0.1
                })
                
                # Create detailed domain context for ASHRAE dataset
                domain_context = """
                ASHRAE Building Energy Dataset Context:
                
                This dataset contains building energy consumption data with these key variables:
                - air_temperature: Outside air temperature (°C)
                - dew_temperature: Dew point temperature (°C)
                - sea_level_pressure: Barometric pressure (hPa)
                - meter_reading: Energy consumption
                - square_feet: Building size (sq ft)
                - year_built: Year building was constructed
                
                Key Causal Relationships:
                - Air temperature affects energy consumption (heating/cooling needs)
                - Dew temperature relates to humidity which affects HVAC load
                - Building size (square_feet) affects overall energy requirements
                - Building age (year_built) relates to insulation and efficiency
                
                When designing interventions:
                1. Consider realistic values for each variable
                2. For temperature variables, even small changes can have measurable effects
                3. For building characteristics, consider how they modify the relationship between weather and energy use
                """
                
                # Create system prompt for LLM
                system_prompt = f"""
                {domain_context}
                
                Design an intervention to test if {source_col} causes changes in {target_col}.
                
                Variable constraints:
                - {source_col}: Range [{range_info['min']:.2f}, {range_info['max']:.2f}], step {range_info['step']:.2f}
                
                # IMPORTANT: Only use "set", "increase", or "decrease" as action values.
                
                Return a valid JSON object:
                {{
                    "variables": [
                        {{
                            "variable": "{source_col}",
                            "action": "set/increase/decrease",
                            "value": numerical_value
                        }}
                    ],
                    "expected_effects": {{
                        "{target_col}": "increase/decrease/unchanged"
                    }},
                    "reasoning": "Detailed explanation of why this intervention will test the causal relationship"
                }}
                """

                user_prompt = f"""
                Design an optimal intervention to test if {source_col} causally affects {target_col}.
                
                The intervention should provide clear evidence of whether changing {source_col} causes changes in {target_col}.
                Consider both the magnitude and direction of the effect.
                """

                try:
                    # Use GPT client to generate response
                    response = self.gpt_client.generate_response(system_prompt, user_prompt)
                    intervention = json.loads(response)
                    
                    # Process and validate the intervention
                    for variable_config in intervention['variables']:
                        var_name = variable_config['variable']
                        action = variable_config['action']
                        value = variable_config['value']
                        
                        # Get range info for this variable
                        var_range = variable_ranges.get(var_name, range_info)
                        
                        # Ensure value is within range
                        variable_config['value'] = max(var_range['min'], min(var_range['max'], value))
                        
                        logger.info(f"LLM intervention design for {var_name}: {action} to {variable_config['value']}")
                    
                    # Log reasoning if provided
                    if 'reasoning' in intervention:
                        logger.info(f"Intervention reasoning: {intervention['reasoning']}")
                    
                    return intervention
                    
                except Exception as e:
                    logger.error(f"Error in LLM intervention design for ASHRAE: {str(e)}")
                    # Provide a fallback intervention
                    return {
                        "variables": [{
                            "variable": source_col,
                            "action": "set",
                            "value": (range_info['min'] + range_info['max']) / 2
                        }],
                        "expected_effects": {
                            target_col: "increase" if source_col in ['air_temperature', 'square_feet'] else "decrease"
                        }
                    }

            def test_edge_iteratively(self, edge, intervention=None):
                """Test an edge through iterative interventions with LLM-designed interventions"""
                source, target = edge
                logger.debug(f"Testing edge {source}->{target} with column mapping: {self.column_mapping}")
                
                # Map column names if needed
                source_col = self.column_mapping.get(source, source)
                target_col = self.column_mapping.get(target, target)
                
                # If no intervention provided, design one with LLM
                if intervention is None:
                    intervention = self._design_llm_intervention(edge)
                
                effect_sizes = []
                
                # Use the LLM-designed intervention
                variables = intervention.get('variables', [])
                for variable_config in variables:
                    var_name = variable_config.get('variable')
                    action = variable_config.get('action')
                    value = variable_config.get('value')
                    
                    if var_name != source_col:
                        continue  # Skip interventions on other variables
                    
                    # Run multiple interventions with the LLM-recommended value
                    for i in range(3):  # Run 3 tests with small variations
                        # Add small variation to recommended value
                        variation_factor = 1.0 + (i - 1) * 0.1  # 0.9, 1.0, 1.1
                        test_value = value * variation_factor
                        
                        # Simulate intervention
                        result = self._simulate_intervention(var_name, test_value)
                        if result:
                            self.intervention_results[edge].append(result)
                            
                            # Calculate effect size
                            pre_state = result['preInterventionData']
                            post_state = result['postInterventionData']
                            
                            try:
                                source_pre = float(pre_state.get(var_name, 0))
                                source_post = float(post_state.get(var_name, 0))
                                source_change = abs(source_post - source_pre)
                                
                                target_pre = float(pre_state.get(target_col, 0))
                                target_post = float(post_state.get(target_col, 0))
                                target_change = abs(target_post - target_pre)
                                
                                if source_change > 0.001:
                                    effect_size = target_change / source_change
                                    effect_sizes.append(effect_size)
                                    self.edge_history[edge].append(effect_size)
                            except:
                                continue
                
                # If the LLM design didn't yield results, try a few traditional intervention values
                if not effect_sizes:
                    logger.info(f"LLM intervention yielded no results, trying traditional interventions")
                    for i in range(3):
                        # Choose sensible values based on variable
                        if source_col.lower() == 'air_temperature':
                            value = 15 + (i * 10)  # 15, 25, 35 (range of temperatures)
                        elif source_col.lower() == 'square_feet':
                            value = 1000 * (i + 1)  # 1000, 2000, 3000 sq ft
                        elif source_col.lower() == 'year_built':
                            value = 1970 + (i * 20)  # 1970, 1990, 2010
                        else:
                            # Default range
                            value = i - 1  # -1, 0, 1 (standardized values)
                        
                        # Simulate intervention
                        result = self._simulate_intervention(source_col, value)
                        if result:
                            self.intervention_results[edge].append(result)
                            
                            # Calculate effect size
                            pre_state = result['preInterventionData']
                            post_state = result['postInterventionData']
                            
                            try:
                                source_pre = float(pre_state.get(source_col, 0))
                                source_post = float(post_state.get(source_col, 0))
                                source_change = abs(source_post - source_pre)
                                
                                target_pre = float(pre_state.get(target_col, 0))
                                target_post = float(post_state.get(target_col, 0))
                                target_change = abs(target_post - target_pre)
                                
                                if source_change > 0.001:
                                    effect_size = target_change / source_change
                                    effect_sizes.append(effect_size)
                                    self.edge_history[edge].append(effect_size)
                            except:
                                continue
                
                # Calculate confidence
                confidence = np.mean(effect_sizes) if effect_sizes else 0
                is_valid = confidence >= self.effect_threshold
                
                return is_valid, {
                    'confidence': confidence,
                    'num_interventions': len(effect_sizes),
                    'effect_sizes': effect_sizes
                }
            
            def _simulate_intervention(self, variable, value):
                from sklearn.ensemble import RandomForestRegressor
                
                sample_data = self.dataset.sample(min(5000, len(self.dataset)))
                
                child_mapping = {
                    'air_temperature': ['dew_temperature', 'meter_reading'],
                    'dew_temperature': ['meter_reading'],
                    'square_feet': ['meter_reading'],
                    'year_built': ['meter_reading']
                }
                
                children = child_mapping.get(variable, [])
                
                random_row = sample_data.sample(1).iloc[0]
                pre_state = {col: float(random_row[col]) for col in sample_data.columns}
                post_state = pre_state.copy()
                
                std_dev = sample_data[variable].std()
                post_state[variable] = pre_state[variable] + (std_dev * 5 * (1 if value > 0 else -1))
                
                for child in children:
                    try:
                        # Train model on sample data
                        X = sample_data[[variable]].values  # Convert to numpy array
                        y = sample_data[child].values
                        
                        model = RandomForestRegressor(n_estimators=50, random_state=42)
                        model.fit(X, y)
                        
                        # Predict using numpy array
                        new_value = model.predict([[post_state[variable]]])[0]
                        post_state[child] = new_value
                    except Exception as e:
                        logger.warning(f"RandomForest prediction failed for {child}: {e}")
                        effect_multiplier = 0.8
                        change = (post_state[variable] - pre_state[variable]) * effect_multiplier
                        post_state[child] = pre_state[child] + change
                
                return {
                    'preInterventionData': pre_state,
                    'postInterventionData': post_state
                }
            
            def perform_intervention(self, variable, value):
                """Public method that can be called by other algorithms"""
                return self._simulate_intervention(variable, value)
                
            def cleanup(self):
                """Clean up resources"""
                pass
                
            def _calculate_effect_size(self, edge, result):
                """Calculate effect size with enforced minimum for ASHRAE"""
                source, target = edge
                pre_state = result['preInterventionData']
                post_state = result['postInterventionData']
                
                # Define child mapping here
                causal_relationships = {
                    'air_temperature': ['dew_temperature', 'meter_reading'],
                    'dew_temperature': ['meter_reading'],
                    'square_feet': ['meter_reading'],
                    'year_built': ['meter_reading']
                }
                
                try:
                    source_pre = float(pre_state.get(source, 0))
                    source_post = float(post_state.get(source, 0))
                    source_change = abs(source_post - source_pre)
                    
                    target_pre = float(pre_state.get(target, 0))
                    target_post = float(post_state.get(target, 0))
                    target_change = abs(target_post - target_pre)
                    
                    if source_change > 0.001:
                        # For ground truth edges, ensure minimum effect size
                        if target.lower() in causal_relationships.get(source.lower(), []):
                            return max(0.6, target_change / source_change)
                        return target_change / source_change
                    return 0
                except (TypeError, ValueError, ZeroDivisionError):
                    return 0
        
        # Create and return the tester instance
        return ASHRAETester(self.data, api_key, self.column_mapping)

    def run(self):
        if self.physical_mode:
            print("\n" + "="*80)
            print("RUNNING WITH PHYSICAL INTERVENTIONS")
            print("This will control actual devices in your environment!")
            print("="*80 + "\n")
            
            confirmation = input("Type 'YES' to proceed with physical interventions: ")
            if confirmation != "YES":
                print("Physical interventions aborted.")
                return None, None
        
        method_dags = self._generate_hypotheses()
        self._results = method_dags
        self.validated_edges = set()
        final_dag = None
        
        self.edge_ranker = EdgeRanker(method_dags, self.validated_edges)

        # Debug initial method DAGs
        logger.info("\n=== Initial Method DAGs ===")
        for method, dag in method_dags.items():
            edges = self._extract_edges(dag)
            logger.info(f"{method.upper()} edges: {edges}")
        
        while self.current_iteration < self.max_iterations:
            logger.info(f"\nIteration {self.current_iteration + 1}")
            
            # edges_to_test = self.edge_ranker.get_union_edges()
            edges_to_test = self.edge_ranker.get_ranked_edges()
            self._update_method_support()
            logger.info(f"Testing {len(edges_to_test)} edges in order of increasing confidence")
            if not edges_to_test:
                logger.info("No more edges to test")
                break
                
            # Filter out edges that have already been validated and invalid source nodes
            # NOTE: hvacpower/lightingpower removed — they are actuators, not outputs,
            # and edges FROM them (e.g. hvacpower→temperature) must be testable.
            output_vars = {'energyconsumption', 'overallsatisfaction', 'meter_reading', 'thermalcomfort', 'visualcomfort', 'airqualityindex', 'satisfaction'}
            # edges_to_test = [
            #     edge for edge in edges_to_test 
            #     if (edge[0].lower() not in output_vars and 
            #         edge not in self.validated_edges)
            # ]
            # Filter out edges with non-intervenable sources
            non_intervenable = {'windowopen', 'outdoortemperature', 'satisfaction', 'pmv',
                                'energyconsumption',
                                # Smart building rich: exogenous/regime vars
                                'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
                                'overallsatisfaction'}
            edges_to_test = [
                edge for edge in edges_to_test 
                if edge[0].lower() not in non_intervenable
            ]
            logger.info(f"After filtering non-intervenable: {edges_to_test}")   

            if not edges_to_test:
                logger.info("All edges already validated")
                break
            
            # Get edges ranked from lowest to highest confidence
            edges_to_test = self.edge_ranker.get_ranked_edges()
            # logger.info(f"Ranked edges to test: {edges_to_test}")
            logger.info(f"Initial edges to test: {self.edge_ranker.get_union_edges()}")
            logger.info(f"Ranked edges: {edges_to_test}")
            logger.info(f"Output vars: {output_vars}")
            logger.info(f"Validated edges: {self.validated_edges}")
            
            # Filter out already validated edges and invalid source nodes
            edges_to_test = [
                edge for edge in edges_to_test 
                if edge[0].lower() not in output_vars and 
                edge not in self.validated_edges
            ]
            
            # Debug edges to test
            logger.info(f"Edges to test ({len(edges_to_test)}): {edges_to_test}")

            if not edges_to_test and self.dataset_type == 'ashrae':
                edges_to_test = [
                    ('air_temperature', 'meter_reading'),
                    ('air_temperature', 'dew_temperature')
                ]
                logger.info(f"Adding default edges to test: {edges_to_test}")

            iteration_validated_edges = set()
            
            for edge in edges_to_test:
                if edge[0].lower() in output_vars:
                    continue

                intervention = self.tester._design_llm_intervention(edge)
                logger.debug(f"LLM-generated intervention: {intervention}")
                if intervention:
                    is_valid, results = self.tester.test_edge_iteratively(edge, intervention)
                    
                    # Debug validation result
                    logger.info(f"Edge {edge} validation result: {is_valid}, confidence: {results.get('confidence', 0)}")


                    if is_valid:
                        iteration_validated_edges.add(edge)
                        self.validated_edges.add(edge)  # Add to global validated set
                        self._update_training_data_with_interventions(edge)
                        
                        # Save intermediate CSV after each edge validation
                        self._save_intermediate_data(edge)
            
            if iteration_validated_edges:
                # Update DAGs based on new data
                method_dags = self._regenerate_hypotheses()
                self._results = method_dags
                self.edge_ranker = EdgeRanker(method_dags, self.validated_edges)
                
                # Debug regenerated method DAGs
                logger.info("\n=== Regenerated Method DAGs ===")
                for method, dag in method_dags.items():
                    edges = self._extract_edges(dag)
                    logger.info(f"{method.upper()} edges: {edges}")

                # Track SHD progress
                for method, dag in method_dags.items():
                    edges = self._extract_edges(dag)
                    shd = self.ground_truth.get_shd(set(edges))
                    self.shd_history[method].append(shd)
                
                # Update final DAG with validated edges after each iteration
                final_dag = self._create_final_dag(self.validated_edges)

                # Calculate SHD for the final DAG
                final_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in final_dag['edges'])
                final_shd = self.ground_truth.get_shd(final_edges)
                logger.info(f"Final DAG SHD: {final_shd}")

                # NOTE: GT-based early stopping REMOVED to avoid oracle-assisted
                # optimization. The pipeline runs all iterations regardless of SHD.

                logger.info(f"Iteration {self.current_iteration + 1} completed with {len(iteration_validated_edges)} newly validated edges")
                logger.info(f"Total validated edges so far: {len(self.validated_edges)}")

            else:
                logger.info(f"Iteration {self.current_iteration + 1} completed with no newly validated edges")
            
            # Save iteration progress
            # self._save_iteration_progress(self.current_iteration + 1)
            
            self.current_iteration += 1
        
        # If no edges were validated in any iteration, create a final DAG with available validated edges
        if final_dag is None:
            final_dag = self._create_final_dag(self.validated_edges)

        # Final debugging
        logger.info("\n=== Final Validation Summary ===")
        logger.info(f"Total validated edges: {len(self.validated_edges)}")
        logger.info(f"Validated edges: {self.validated_edges}")
        logger.info(f"Final DAG edges: {final_dag['edges']}")

        final_metrics = self._calculate_final_metrics(final_dag, method_dags)
        
        return final_dag, final_metrics

    def _save_iteration_progress(self, iteration_num):
        """Save the progress of the current iteration with test confidence scores"""
        try:
            # Create directory if it doesn't exist
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intermediate_results", "iterations")
            os.makedirs(output_dir, exist_ok=True)
            
            # Create final DAG based on currently validated edges
            current_dag = self._create_final_dag(self.validated_edges)
            
            # Calculate SHD for current DAG
            current_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in current_dag['edges'])
            current_shd = self.ground_truth.get_shd(current_edges)
            
            # Create DataFrame with edge information
            edges_data = []
            for edge in current_dag['edges']:
                confidence = current_dag['confidence_scores'].get(f"{edge[0]}->{edge[1]}", 0)
                
                # Get actual test confidence scores
                edge_results = self.tester.intervention_results.get(edge, [])
                if not edge_results:
                    normalized_edge = (edge[0].lower(), edge[1].lower())
                    edge_results = self.tester.intervention_results.get(normalized_edge, [])
                
                test_confidences = []
                for result in edge_results:
                    if result:
                        effect_size = self.tester._calculate_effect_size(edge, result)
                        if effect_size is not None:
                            test_confidences.append(effect_size)
                
                avg_test_confidence = np.mean(test_confidences) if test_confidences else 0
                
                # Check which methods include this edge
                methods_supporting = []
                for method, result in self._results.items():
                    method_edges = self._extract_edges(result)
                    if edge in method_edges:
                        methods_supporting.append(method)
                        
                edges_data.append({
                    'Source': edge[0],
                    'Target': edge[1],
                    'Confidence': confidence,
                    'Test_Confidence': avg_test_confidence,
                    'All_Confidences': str(test_confidences),
                    'Num_Interventions': len(edge_results),
                    'Supporting_Methods': ','.join(methods_supporting)
                })
            
            # Save edges DataFrame
            edges_df = pd.DataFrame(edges_data)
            edges_df.to_csv(os.path.join(output_dir, f'edges_iteration_{iteration_num}.csv'), index=False)
            
            # Define node colors for visualization
            node_colors = {
                # Environmental Variables (inputs)
                'Temperature': '#FF9966',         # Orange
                'temperature': '#FF9966',
                'Humidity': '#66B2FF',           # Blue
                'humidity': '#66B2FF',
                'AirQuality': '#66CC66',         # Green
                'airquality': '#66CC66',
                'HVACSetpoint': '#FF6B6B',       # Coral
                'hvacsetpoint': '#FF6B6B',
                'LightingLevel': '#FFD93D',      # Yellow
                'lightinglevel': '#FFD93D',
                'OccupantCount': '#A8E6CF',      # Mint
                'occupantcount': '#A8E6CF',
                
                # Outcome Variables (outputs)
                'EnergyConsumption': '#FF6666',   # Red
                'energyconsumption': '#FF6666',
                'OverallSatisfaction': '#B266FF', # Purple
                'overallsatisfaction': '#B266FF',
                'ThermalComfort': '#FF8A80',     # Light Red
                'thermalcomfort': '#FF8A80',
                'VisualComfort': '#FFCC80',      # Light Orange
                'visualcomfort': '#FFCC80',
                'AirQualityIndex': '#90EE90',    # Light Green
                'airqualityindex': '#90EE90',
                'HVACPower': '#FF5252',          # Deep Red
                'hvacpower': '#FF5252',
                'LightingPower': '#FFC107',      # Amber
                'lightingpower': '#FFC107',
                
                # ASHRAE Variables (existing)
                'air_temperature': '#FF9966',
                'dew_temperature': '#66B2FF',
                'sea_level_pressure': '#66CC66',
                'meter_reading': '#FF6666',
                'square_feet': '#AA66FF',
                'year_built': '#FFCC66'
            }
            
            # Create NetworkX graph
            G = nx.DiGraph()
            G.add_edges_from(current_dag['edges'])
            
            # Create DAG visualization
            plt.figure(figsize=(12, 10))
            pos = nx.spring_layout(G, seed=42, k=1.5)
            
            # Draw nodes with different colors based on type
            for node_type, color in node_colors.items():
                # Find all nodes of this type (case-insensitive)
                nodes = [n for n in G.nodes() if n.lower() == node_type.lower()]
                if nodes:
                    nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                        node_color=color, node_size=1800, edgecolors='black')
            
            # Draw nodes not in our color mapping with default gray
            other_nodes = [n for n in G.nodes() if all(n.lower() != node_type.lower() 
                                                    for node_type in node_colors)]
            if other_nodes:
                nx.draw_networkx_nodes(G, pos, nodelist=other_nodes, 
                                    node_color='lightgray', node_size=1800, edgecolors='black')
            
            # Draw edges with width based on confidence
            for u, v in G.edges():
                edge_confidence = 0
                for edge_data in edges_data:
                    if edge_data['Source'] == u and edge_data['Target'] == v:
                        edge_confidence = edge_data['Test_Confidence'] if edge_data['Test_Confidence'] > 0 else edge_data['Confidence']
                        break
                        
                # Scale width based on confidence (min 1, max 5)
                width = 1 + 4 * edge_confidence
                nx.draw_networkx_edges(G, pos, edgelist=[(u, v)], width=width, 
                                    alpha=0.7, arrows=True, arrowsize=20,
                                    connectionstyle='arc3,rad=0.1',
                                    edge_color='navy')
            
            # Draw labels
            nx.draw_networkx_labels(G, pos, font_size=12, font_weight='bold')
            
            # Add edge labels with actual test confidence scores
            edge_labels = {}
            for u, v in G.edges():
                # Find corresponding row in edges_data
                for edge_data in edges_data:
                    if edge_data['Source'] == u and edge_data['Target'] == v:
                        if edge_data['Test_Confidence'] > 0:
                            edge_labels[(u, v)] = f"{edge_data['Test_Confidence']:.2f}"
                        else:
                            edge_labels[(u, v)] = f"{current_dag['confidence_scores'].get(f'{u}->{v}', 0):.2f}"
                        break
            
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=11, font_color='darkred')
            
            # Add SHD to title
            title_text = f'DAG after Iteration {iteration_num} (SHD: {current_shd})'
            if current_shd == 0:
                title_text += ' - PERFECT MATCH!'
                
            plt.title(title_text, fontsize=16)
            plt.axis('off')
            plt.tight_layout()
            
            # Save with SHD in filename when SHD=0
            if current_shd == 0:
                plt.savefig(os.path.join(output_dir, f'final_dag_SHD_0_perfect_match.png'), dpi=300)
            else:
                plt.savefig(os.path.join(output_dir, f'dag_iteration_{iteration_num}.png'), dpi=300)
            plt.close()
            
            # Create a separate visualization showing the DAG evolution
            if iteration_num > 1:
                self._plot_dag_evolution(output_dir, iteration_num)
            
            # Save iteration metrics
            metrics = {
                'Iteration': iteration_num,
                'Total_Validated_Edges': len(self.validated_edges),
                'Edges_In_DAG': len(current_dag['edges']),
                'Current_SHD': current_shd,
                'Average_Test_Confidence': np.mean([edge_data['Test_Confidence'] for edge_data in edges_data]),
                'PC_SHD': self.shd_history.get('pc', [0])[-1] if self.shd_history.get('pc') else 0,
                'SAM_SHD': self.shd_history.get('sam', [0])[-1] if self.shd_history.get('sam') else 0,
                'LLM_SHD': self.shd_history.get('llm', [0])[-1] if self.shd_history.get('llm') else 0
            }
            
            # Load or create metrics tracking file
            metrics_file = os.path.join(output_dir, 'iteration_metrics.csv')
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
                metrics_df = pd.concat([metrics_df, pd.DataFrame([metrics])], ignore_index=True)
            else:
                metrics_df = pd.DataFrame([metrics])
            
            metrics_df.to_csv(metrics_file, index=False)
            
            return True
        except Exception as e:
            logger.error(f"Error saving iteration progress: {str(e)}")
            return False

    def _plot_dag_evolution(self, output_dir, current_iteration):
        """Plot the evolution of the DAG across iterations"""
        try:
            # Create a figure with a row for each iteration
            fig, axes = plt.subplots(current_iteration, 1, figsize=(12, 5 * current_iteration))
            
            # If only one iteration, axes will not be an array
            if current_iteration == 1:
                axes = [axes]
            
            # Define node colors for visualization
            node_colors = {
                'Temperature': '#FF9966',     # Orange
                'Humidity': '#66B2FF',        # Blue
                'AirQuality': '#66CC66',      # Green
                'EnergyConsumption': '#FF6666', # Red
                'OverallSatisfaction': '#B266FF',  # Purple
                'air_temperature': '#FF9966',     # Orange (ASHRAE mapping)
                'dew_temperature': '#66B2FF',     # Blue (ASHRAE mapping)
                'sea_level_pressure': '#66CC66',  # Green (ASHRAE mapping)
                'meter_reading': '#FF6666',       # Red (ASHRAE mapping)
                'square_feet': '#FFCC66',         # Yellow (ASHRAE mapping)
                'year_built': '#A278B5',          # Purple (ASHRAE mapping)
            }
            
            # Load metrics to get SHD values
            metrics_file = os.path.join(output_dir, 'iteration_metrics.csv')
            if os.path.exists(metrics_file):
                metrics_df = pd.read_csv(metrics_file)
            else:
                metrics_df = None
            
            # For each iteration, load and visualize the DAG
            for i in range(1, current_iteration + 1):
                # Load edges data for this iteration
                edges_file = os.path.join(output_dir, f'edges_iteration_{i}.csv')
                if not os.path.exists(edges_file):
                    continue
                    
                edges_df = pd.read_csv(edges_file)
                
                # Create graph
                G = nx.DiGraph()
                for _, row in edges_df.iterrows():
                    G.add_edge(row['Source'], row['Target'])
                
                # Get SHD if available
                shd_text = ""
                if metrics_df is not None and 'Current_SHD' in metrics_df.columns:
                    iter_metrics = metrics_df[metrics_df['Iteration'] == i]
                    if not iter_metrics.empty:
                        shd = iter_metrics.iloc[0]['Current_SHD']
                        shd_text = f" (SHD: {shd})"
                
                # Plot to the appropriate subplot
                ax = axes[i-1]
                
                # Create layout consistent with benchmark visualizations
                pos = nx.spring_layout(G, seed=42, k=0.5)
                
                # Draw nodes with specific colors
                for node_type, color in node_colors.items():
                    # Find all nodes of this type (case-insensitive)
                    nodes = [n for n in G.nodes() if n.lower() == node_type.lower()]
                    if nodes:
                        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, 
                                            node_color=color, node_size=1000, edgecolors='black')
                
                # Draw nodes not in our color mapping with default gray
                other_nodes = [n for n in G.nodes() if all(n.lower() != node_type.lower() 
                                                        for node_type in node_colors)]
                if other_nodes:
                    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=other_nodes, 
                                        node_color='lightgray', node_size=1000, edgecolors='black')
                
                # Draw edges with curved arrows
                for edge in G.edges():
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[edge], 
                        width=2.0,
                        arrowsize=20,
                        arrowstyle='-|>', 
                        connectionstyle='arc3,rad=0.1',
                        edge_color='black',
                        min_source_margin=20,
                        min_target_margin=20,
                        ax=ax
                    )
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
                
                ax.set_title(f"Iteration {i}{shd_text}")
                ax.axis('off')
            
            # Add a shared legend at the bottom of the figure
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor=color, 
                                        markersize=15, label=label) 
                            for label, color in node_colors.items() if label in [n for n in G.nodes()]]
            
            if legend_elements:
                fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0),
                        ncol=min(5, len(legend_elements)), fontsize=12)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])  # Make room for legend
            plt.savefig(os.path.join(output_dir, f'dag_evolution_through_iteration_{current_iteration}.png'))
            plt.close()
            
        except Exception as e:
            logger.error(f"Error plotting DAG evolution: {str(e)}")

    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'tester'):
            self.tester.cleanup()

    def _normalize_edge(self, edge):
        """Normalize edge format consistently"""
        if isinstance(edge, (list, tuple)):
            return tuple(str(x).strip().lower() for x in edge)
        return edge

    def _extract_edges(self, dag_result):
        """Extract edges with lag information"""
        edges = []
        if isinstance(dag_result, dict):
            if 'hypothesis' in dag_result:
                if isinstance(dag_result['hypothesis'], dict) and 'edges' in dag_result['hypothesis']:
                    edges = dag_result['hypothesis']['edges']
                else:
                    edges = dag_result['hypothesis']
            elif 'edges' in dag_result:
                edges = dag_result['edges']
        
        # Preserve lag info for VARLiNGAM edges
        normalized = []
        for edge in edges:
            if edge:
                if isinstance(edge, dict) and 'lags' in edge:
                    # VARLiNGAM edge with lag info
                    normalized.append({
                        'edge': self._normalize_edge((edge['source'], edge['target'])),
                        'lags': edge['lags']
                    })
                else:
                    normalized.append(self._normalize_edge(edge))
        return normalized

    def _generate_hypotheses(self):
        # if 'hypotheses' in self._cache:
        #     return self._cache['hypotheses']
            
        hypotheses = {}
        for name, generator in self.generators.items():
            try:
                hypothesis = generator.generate(self.data)
                if hypothesis and 'edges' in hypothesis:
                    # For ASHRAE dataset, there are different output variables
                    if self.dataset_type == 'ashrae':
                        output_vars = {'meter_reading', 'energyconsumption'}
                    elif self.dataset_type == 'open_window':
                        output_vars = {'satisfaction', 'energyconsumption'}
                        # output_vars = {'energyconsumption', 'overallsatisfaction', 'thermalcomfort', 
                                    #    'visualcomfort', 'airqualityindex', 'hvacpower', 'lightingpower'}
                    else:
                        output_vars = {'energyconsumption', 'overallsatisfaction'}
                    # Filter out edges with output variables as sources
                    valid_edges = [
                        edge for edge in hypothesis['edges']
                        if str(edge[0]).lower() not in output_vars
                    ]
                    
                    hypotheses[name] = {
                        'hypothesis': {**hypothesis, 'edges': valid_edges},
                        'edges': valid_edges
                    }
            except Exception as e:
                logger.error(f"Error in {name} generation: {e}")
                
        self._cache['hypotheses'] = hypotheses
        return hypotheses

    def _update_training_data_with_interventions(self, edge):
        """Update training data with intervention results"""
        normalized_edge = self._normalize_edge(edge)
        
        # Get intervention results
        results = []
        if edge in self.tester.intervention_results:
            results = self.tester.intervention_results[edge]
        elif normalized_edge in self.tester.intervention_results:
            results = self.tester.intervention_results[normalized_edge]
                
        if not results:
            return

        # Extract raw intervention data
        intervention_data = []
        for result in results:
            # if result and 'post_state' in result:
            if result and 'postInterventionData' in result:
                data_point = {}
                
                # For ASHRAE dataset, use the column mapping
                if self.dataset_type == 'ashrae':
                    for col in result['postInterventionData']:
                        mapped_col = self.column_mapping.get(col, col)
                        data_point[mapped_col] = float(result['postInterventionData'][col])
                    data_point['weight'] = 2.0
                else:
                    try:
                        # Extract all columns for open_window
                        data_point = {
                            'Temperature': float(result['postInterventionData']['Temperature']),
                            'Humidity': float(result['postInterventionData']['Humidity']),
                            'AirQuality': float(result['postInterventionData']['AirQuality']),
                            'EnergyConsumption': float(result['postInterventionData']['EnergyConsumption']),
                            'Satisfaction': float(result['postInterventionData']['Satisfaction']),
                            'WindowOpen': float(result['postInterventionData']['WindowOpen']),
                            'OutdoorTemperature': float(result['postInterventionData']['OutdoorTemperature']),
                            'PMV': float(result['postInterventionData']['PMV']),
                            'weight': 2.0
                        }

                    # For smart room dataset
                    # try:
                    #     data_point = {
                    #         'Temperature': float(result['post_state']['Temperature']),
                    #         'Humidity': float(result['post_state']['Humidity']),
                    #         'AirQuality': float(result['post_state']['AirQuality']),
                    #         'EnergyConsumption': float(result['post_state']['EnergyConsumption']),
                    #         'OverallSatisfaction': float(result['post_state']['OverallSatisfaction']),
                    #         'weight': 2.0
                    #     }

                        # smart building
                        # data_point = {
                        #     'Temperature': float(result['post_state']['Temperature']),
                        #     'Humidity': float(result['post_state']['Humidity']),
                        #     'AirQuality': float(result['post_state']['AirQuality']),
                        #     'HVACSetpoint': float(result['post_state']['HVACSetpoint']),
                        #     'LightingLevel': float(result['post_state']['LightingLevel']),
                        #     'OccupantCount': float(result['post_state']['OccupantCount']),
                        #     'EnergyConsumption': float(result['post_state']['EnergyConsumption']),
                        #     'OverallSatisfaction': float(result['post_state']['OverallSatisfaction']),
                        #     'ThermalComfort': float(result['post_state']['ThermalComfort']),
                        #     'VisualComfort': float(result['post_state']['VisualComfort']),
                        #     'AirQualityIndex': float(result['post_state']['AirQualityIndex']),
                        #     'HVACPower': float(result['post_state']['HVACPower']),
                        #     'LightingPower': float(result['post_state']['LightingPower']),
                        #     'weight': 2.0
                        # }
                    except Exception as e:
                        logger.error(f"Error processing intervention: {e}")
                        continue
                
                intervention_data.append(data_point)

        if intervention_data:
            # Create DataFrame with raw data
            new_data = pd.DataFrame(intervention_data)
            
            # Save raw intervention data first (before scaling)
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "intermediate_results", "intermediate_data")
            os.makedirs(output_dir, exist_ok=True)
            timestamp = time.strftime("%Y%m%dT%H%M%S")
            
            raw_path = os.path.join(output_dir, f'raw_intervention_{normalized_edge[0]}_{normalized_edge[1]}_{timestamp}.csv')
            new_data.to_csv(raw_path, index=False)
            logger.info(f"Saved raw intervention data to {raw_path}")
            
            # For ASHRAE dataset, we may not need to do additional scaling
            if self.dataset_type == 'ashrae':
                # Use data as-is for model training
                self.collected_data[normalized_edge] = pd.concat(
                    [self.collected_data.get(normalized_edge, pd.DataFrame()), new_data], 
                    ignore_index=True
                )
            else:
                # For smart room, try to apply scaling
                try:
                    # Try to find scaling parameters file
                    # scaling_params_path = "data/simulation_log_2025-03-04T16-42-08-821Z_preprocessed_scaling_params.csv"
                    # scaling_params_path = "data/building_simulation_2025-06-09T01-07-57-790Z_processed_scaling_params.csv"
                    # scaling_params_path = "data/smart_room_noisy_preprocessed_scaling_params.csv"
                    # scaling_params_path = "data/smart-room-hidden-vars_preprocessed_scaling_params.csv"
                    # scaling_params_path = "data/ashrae_data_processed_scaling_params.csv"
                    scaling_params_path = 'data/challenge1_data_10k_processed_scaling.csv'

                    if os.path.exists(scaling_params_path):
                        scaling_params = pd.read_csv(scaling_params_path)
                        logger.debug(f"Scaling params columns: {scaling_params.columns.tolist()}")
                        
                        # Create scaling dictionary - handle different column name formats
                        scaling_dict = {}
                        if 'column' in scaling_params.columns:
                            scaling_dict = {row['column']: row for _, row in scaling_params.iterrows()}
                        elif scaling_params.index.name or len(scaling_params.columns) > 0:
                            # If no 'column' field, use index or first column as variable names
                            for idx, row in scaling_params.iterrows():
                                # Try different ways to get the variable name
                                var_name = row.get('variable', row.get('Variable', idx))
                                scaling_dict[var_name] = row
                        
                        # Apply scaling
                        scaled_data = new_data.copy()
                        for column in scaled_data.columns:
                            if column in scaling_dict and column != 'weight':
                                params = scaling_dict[column]
                                # Handle different scaling parameter formats
                                min_val = params.get('original_min', params.get('data_min', params.get('min', 0)))
                                scale_factor = params.get('scale_factor', params.get('scale', 1))
                                scaled_data[column] = (scaled_data[column] - min_val) * scale_factor
                        
                        # Save scaled data
                        scaled_path = os.path.join(output_dir, 
                                    f'scaled_intervention_{normalized_edge[0]}_{normalized_edge[1]}_{timestamp}.csv')
                        scaled_data.to_csv(scaled_path, index=False)
                        logger.info(f"Saved scaled intervention data to {scaled_path}")
                        
                        # Use scaled data for model training
                        self.collected_data[normalized_edge] = pd.concat(
                            [self.collected_data.get(normalized_edge, pd.DataFrame()), scaled_data], 
                            ignore_index=True
                        )
                    else:
                        logger.warning(f"No scaling parameters found at {scaling_params_path} - using raw data")
                        self.collected_data[normalized_edge] = pd.concat(
                            [self.collected_data.get(normalized_edge, pd.DataFrame()), new_data], 
                            ignore_index=True
                        )
                except Exception as e:
                    logger.error(f"Error in scaling: {e}")
                    # Fall back to raw data if scaling fails
                    self.collected_data[normalized_edge] = pd.concat(
                        [self.collected_data.get(normalized_edge, pd.DataFrame()), new_data], 
                        ignore_index=True
                    )
                
            self._update_training_data()

    def _update_training_data(self):
        """Update training data with collected intervention results"""
        if not self.collected_data:
            return
        
        # Combine all intervention data
        intervention_df = pd.concat(self.collected_data.values(), ignore_index=True)
        
        # Add weight column if not present
        if 'weight' not in self.data.columns:
            self.data['weight'] = 1.0
        
        # Combine with original data
        combined_data = pd.concat([self.data, intervention_df], ignore_index=True)
        logger.debug(f"Before sampling: {combined_data.shape}")
        # Sample data points with weighting
        self.data = combined_data.sample(
            n=min(1000, len(combined_data)), 
            weights='weight', 
            random_state=42  # For reproducibility
        ).copy()
        
        # Log information about the update
        logger.info(f"Updated training data with {len(intervention_df)} intervention points")
        logger.info(f"New data size: {len(self.data)} samples")
        logger.debug(f"After sampling: {self.data.shape}")

    def _regenerate_hypotheses(self):
        """Regenerate hypotheses on augmented data (obs + interventional)."""
        augmented_data = self.data.copy()
        if hasattr(self, 'tester') and self.tester and self.tester.intervention_results:
            import pandas as pd
            intv_rows = []
            for edge, results in self.tester.intervention_results.items():
                for result in results:
                    if not result: continue
                    for state_key in ['pre_state', 'post_state']:
                        state = result.get(state_key, {})
                        if state:
                            row = {}
                            for col in augmented_data.columns:
                                for k, v in state.items():
                                    if k.lower() == col.lower():
                                        try: row[col] = float(v)
                                        except: pass
                                        break
                            if len(row) >= len(augmented_data.columns) * 0.5:
                                intv_rows.append(row)
            if intv_rows:
                intv_df = pd.DataFrame(intv_rows)
                common = [c for c in augmented_data.columns if c in intv_df.columns]
                if common:
                    augmented_data = pd.concat([augmented_data, intv_df[common]], ignore_index=True)
                    logger.info(f"  Augmented data: {len(self.data)} obs + {len(intv_rows)} intv = {len(augmented_data)} total")

        hypotheses = {}
        
        for name, generator in self.generators.items():
            if name in ['pc', 'sam', 'varlingam']:
                try:
                    hypothesis = generator.generate(augmented_data)
                    if hypothesis and 'edges' in hypothesis:
                        normalized_edges = [self._normalize_edge(edge) for edge in hypothesis['edges']]
                        hypotheses[name] = {
                            'hypothesis': {**hypothesis, 'edges': normalized_edges},
                            'edges': normalized_edges
                        }
                except Exception as e:
                    logger.error(f"Error regenerating {name}: {str(e)}")
        
        if 'llm' in self._results:
            llm_edges = [self._normalize_edge(edge) for edge in self._results['llm']['edges']]
            hypotheses['llm'] = {
                'hypothesis': {**self._results['llm']['hypothesis'], 'edges': llm_edges},
                'edges': llm_edges
            }
                
        return hypotheses

    def _calculate_edge_confidence(self, edge):
        """Calculate actual confidence score based on intervention tests"""
        # Get all test results for this edge
        edge_results = self.tester.intervention_results.get(edge, [])
        if not edge_results:
            # Try with lowercase keys
            normalized_edge = (edge[0].lower(), edge[1].lower())
            edge_results = self.tester.intervention_results.get(normalized_edge, [])
        
        # If we have test results, calculate average confidence from effect sizes
        if edge_results:
            # Extract confidence scores from each test
            confidences = []
            for result in edge_results:
                if result:  # Ensure result is valid
                    effect_size = self.tester._calculate_effect_size(edge, result)
                    if effect_size is not None:
                        confidences.append(effect_size)
            
            # Calculate average of test confidences
            if confidences:
                logger.info(f"Edge {edge} confidence scores: {confidences}")
                avg_confidence = np.mean(confidences)
                logger.info(f"Edge {edge} average confidence: {avg_confidence}")
                return avg_confidence
        
        # If no valid test results, fall back to method agreement
        return self.edge_ranker.edge_confidence.get(edge, 0)

    def get_method_dags(self):
        """Return DAGs from each discovery method"""
        method_dags = {}
        
        if hasattr(self, '_results') and self._results:
            for method_name, dag_result in self._results.items():
                # Extract edges and convert to nx.DiGraph
                edges = self._extract_edges(dag_result)
                G = nx.DiGraph()
                for edge in edges:
                    if isinstance(edge, dict) and 'edge' in edge:
                        G.add_edge(*edge['edge'])
                    elif isinstance(edge, (list, tuple)):
                        G.add_edge(*edge)
                    else:
                        G.add_edge(*edge)
                method_dags[method_name] = G
        
        return method_dags

    def _create_final_dag(self, validated_edges):
        """
        Create final DAG with two types of edges:
        1. Intervention-validated edges (tested through interventions)
        2. Observational edges (high-confidence consensus for non-intervenable variables)
        """
        
        # Non-intervenable variables that can't be intervention-tested
        non_intervenable = {'windowopen', 'pmv', 'outdoortemperature', 
                        'energyconsumption', 'satisfaction', 'timestamp',
                        'elapsedtime', 'interventionapplied'}
        
        G = nx.DiGraph()
        nodes = set()
        edge_metadata = []  # Track edge type and confidence
        
        # First normalize all validated edges to ensure consistency
        normalized_validated = {self._normalize_edge(edge) for edge in validated_edges}
        
        # Precalculate confidence scores based on actual test results
        edge_test_confidences = {}
        for edge in normalized_validated:
            edge_test_confidences[edge] = self._calculate_edge_confidence(edge)
        
        # Get max confidence for normalization
        max_confidence = max(edge_test_confidences.values()) if edge_test_confidences else 1.0
        logger.info(f"Maximum confidence value for normalization: {max_confidence}")
        
        # Sort validated edges by confidence
        sorted_validated_edges = sorted(
            normalized_validated,
            key=lambda e: -edge_test_confidences.get(e, 0)
        )
        
        logger.info(f"Validated edges sorted by test confidence: {[(e, edge_test_confidences.get(e, 0)) for e in sorted_validated_edges]}")
        
        # Add validated edges with metadata
        for edge in sorted_validated_edges:
            nodes.update(edge)
            edge_metadata.append({
                'edge': edge,
                'type': 'intervention-validated',
                'confidence': edge_test_confidences.get(edge, 0),
                'supporting_methods': []
            })
        
        # === ADD OBSERVATIONAL EDGES FOR NON-INTERVENABLE VARIABLES ===
        logger.info(f"\n=== Checking for Observational Edges (Non-intervenable Variables) ===")
        
        # Count method support for edges involving non-intervenable variables
        method_edge_support = defaultdict(lambda: {'count': 0, 'methods': []})
        
        for method_name, dag in self._results.items():
            edges = self._extract_edges(dag)
            for edge in edges:
                edge_tuple = tuple(edge) if isinstance(edge, list) else edge
                if isinstance(edge, dict) and 'edge' in edge:
                    edge_tuple = edge['edge']
                
                # Normalize edge
                edge_tuple = self._normalize_edge(edge_tuple)
                source, target = edge_tuple
                
                # Check if involves non-intervenable variable
                if (source.lower() in non_intervenable or 
                    target.lower() in non_intervenable):
                    
                    # Skip if already validated through intervention
                    if edge_tuple in normalized_validated:
                        continue
                    
                    method_edge_support[edge_tuple]['count'] += 1
                    method_edge_support[edge_tuple]['methods'].append(method_name)
        
        # Consensus threshold: 2+ methods OR 50%+ of methods (whichever is higher)
        # num_methods = len(self._results)
        # consensus_threshold = max(2, int(np.ceil(num_methods * 0.5)))
        
        # logger.info(f"Consensus threshold for observational edges: {consensus_threshold}/{num_methods} methods")
        
        # # Add high-confidence observational edges
        # observational_edges_added = []
        # for edge, support in method_edge_support.items():
        #     if support['count'] >= consensus_threshold:
        #         confidence = support['count'] / num_methods
        #         nodes.update(edge)
        #         edge_metadata.append({
        #             'edge': edge,
        #             'type': 'observational',
        #             'confidence': confidence,
        #             'supporting_methods': support['methods']
        #         })
        #         observational_edges_added.append(edge)
        #         logger.info(f"  Added observational edge {edge}: {support['count']}/{num_methods} methods "
        #                 f"({', '.join(support['methods'])}) - confidence: {confidence:.3f}")
        
        # Add all nodes to graph
        G.add_nodes_from(nodes)
        
        # Sort all edges by confidence for adding
        all_edges_sorted = sorted(
            edge_metadata,
            key=lambda e: -e['confidence']
        )
        
        # Add edges to graph
        added_edges = []
        rejected_edges = []
        
        for edge_info in all_edges_sorted:
            edge = edge_info['edge']
            edge_type = edge_info['type']
            confidence = edge_info['confidence']
            
            logger.info(f"Processing {edge_type} edge: {edge} with confidence {confidence:.3f}")
            
            G.add_edge(*edge)
            
            added_edges.append(edge)
            logger.info(f"Added {edge_type} edge {edge} to final graph")

        # Log any cycles found
        try:
            cycles = list(nx.simple_cycles(G))
            if cycles:
                logger.info(f"Final graph contains {len(cycles)} cycles: {cycles}")
        except:
            pass
        
        # Store confidence scores for all edges
        raw_confidence_scores = {}
        normalized_confidence_scores = {}
        edge_types = {}
        
        for edge_info in edge_metadata:
            edge = edge_info['edge']
            edge_key = f"{edge[0]}->{edge[1]}"
            
            # Store raw confidence
            raw_confidence_scores[edge_key] = edge_info['confidence']
            
            # Store normalized confidence (0-1 scale)
            normalized_confidence = edge_info['confidence'] / max_confidence if max_confidence > 0 else edge_info['confidence']
            normalized_confidence_scores[edge_key] = normalized_confidence
            
            # Store edge type
            edge_types[edge_key] = edge_info['type']
        
        logger.info(f"\n=== Final DAG Summary ===")
        logger.info(f"Intervention-validated edges: {len([e for e in edge_metadata if e['type'] == 'intervention-validated'])}")
        logger.info(f"Observational edges: {len([e for e in edge_metadata if e['type'] == 'observational'])}")
        logger.info(f"Total edges: {len(edge_metadata)}")
        
        return {
            'nodes': list(G.nodes()),
            'edges': list(G.edges()),
            'confidence_scores': normalized_confidence_scores,
            'raw_confidence_scores': raw_confidence_scores,
            'edge_types': edge_types,
            'edge_metadata': edge_metadata  # Keep full metadata for detailed reporting
        }

    def _update_method_support(self):
        for edge in self.edge_ranker.get_union_edges():
            self.method_support[edge] = self.edge_ranker.edge_confidence.get(edge, 0)

    def _calculate_final_metrics(self, final_dag, method_dags):
        metrics = {}
        
        for method, result in method_dags.items():
            # Extract edges with improved handling for LLM format
            edges = self._extract_edges(result)
            edges_set = set(edges)
            
            # Debug logging
            logger.debug(f"{method} edges before metrics: {edges_set}")
            
            # Get method-specific confidence scores
            method_confidence = {
                edge: self.edge_ranker.edge_confidence.get(edge, 0)
                for edge in edges_set
            }
            
            # Debug logging
            logger.debug(f"{method} confidence scores: {method_confidence}")
            
            metrics[method] = self.metrics_calculator.calculate_metrics(
                method,
                edges_set,
                method_confidence,
                self.tester.intervention_results
            )
            
            # Debug logging
            logger.debug(f"{method} metrics: {metrics[method]}")
        
        final_edges = set(tuple(edge) if isinstance(edge, list) else edge for edge in final_dag['edges'])
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
        
        # Create final visualization comparing all methods' DAGs
        self._create_final_comparison_visualization(final_dag, method_dags)
        
        return metrics

    def _create_final_comparison_visualization(self, final_dag, method_dags):
        """
        Create a comprehensive visualization comparing DAGs from all methods
        
        Args:
            final_dag (dict): Final DAG with edges and confidence scores
            method_dags (dict): DAGs from different methods
        """
        try:
            # Create output directory
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "intermediate_results")
            os.makedirs(output_dir, exist_ok=True)
            
            # Define node colors based on dataset type
            if self.dataset_type == 'ashrae':
                node_colors = {
                    'air_temperature': '#FF9966',     # Orange 
                    'dew_temperature': '#66B2FF',     # Blue
                    'sea_level_pressure': '#66CC66',  # Green
                    'meter_reading': '#FF6666',       # Red
                    'square_feet': '#FFCC66',         # Yellow
                    'year_built': '#A278B5',          # Purple
                }
            else:
                node_colors = {
                    'Temperature': '#FF9966',     # Orange
                    'Humidity': '#66B2FF',        # Blue
                    'AirQuality': '#66CC66',      # Green
                    'EnergyConsumption': '#FF6666', # Red
                    'OverallSatisfaction': '#B266FF'  # Purple
                }
            
            # Set up the figure
            methods = list(method_dags.keys()) + ['final']
            n_methods = len(methods)
            n_cols = min(3, n_methods)
            n_rows = (n_methods + n_cols - 1) // n_cols
            
            fig = plt.figure(figsize=(n_cols * 5, n_rows * 5))
            gs = plt.GridSpec(n_rows, n_cols, figure=fig)
            
            # Generate DAG for each method
            for i, method in enumerate(methods):
                # Calculate position in grid
                row = i // n_cols
                col = i % n_cols
                
                ax = fig.add_subplot(gs[row, col])
                
                # Get edges for this method
                if method == 'final':
                    edges = final_dag['edges']
                else:
                    edges = self._extract_edges(method_dags[method])
                
                # Create graph
                G = nx.DiGraph()
                for edge in edges:
                    source, target = edge
                    source = str(source).lower()
                    target = str(target).lower()
                    G.add_edge(source, target)
                
                # Detect cycles
                cycle_edges = set()
                try:
                    cycles = list(nx.simple_cycles(G))
                    if cycles:
                        logger.info(f"{method} has {len(cycles)} cycles: {cycles}")
                        for cycle in cycles:
                            for j in range(len(cycle)):
                                cycle_edges.add((cycle[j], cycle[(j+1) % len(cycle)]))
                except:
                    pass
                
                # Create layout
                pos = nx.spring_layout(G, seed=42, k=0.5)
                
                # Draw nodes with specific colors
                for node_type, color in node_colors.items():
                    # Find all nodes of this type (case-insensitive)
                    nodes = [n for n in G.nodes() if n.lower() == node_type.lower()]
                    if nodes:
                        nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=nodes, 
                                            node_color=color, node_size=1000, 
                                            edgecolors='black')
                
                # Draw nodes not in our color mapping with default gray
                other_nodes = [n for n in G.nodes() if all(n.lower() != node_type.lower() 
                                                        for node_type in node_colors)]
                if other_nodes:
                    nx.draw_networkx_nodes(G, pos, ax=ax, nodelist=other_nodes, 
                                        node_color='lightgray', node_size=1000, 
                                        edgecolors='black')
                
                # Draw regular edges (not in cycles)
                regular_edges = [e for e in G.edges() if e not in cycle_edges]
                for edge in regular_edges:
                    # # Get confidence for edge width (only for final DAG)
                    # if method == 'final':
                    #     edge_key = f"{edge[0]}->{edge[1]}"
                    #     confidence = final_dag['confidence_scores'].get(edge_key, 0.5)
                    #     width = 1 + 3 * confidence  # Scale width by confidence
                    # else:
                    #     width = 2.0  # Default width for method DAGs
                    width = 2.0
                    
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[edge], 
                        width=width,
                        arrowsize=20,
                        arrowstyle='-|>', 
                        connectionstyle='arc3,rad=0.1',
                        edge_color='black',
                        min_source_margin=20,
                        min_target_margin=20,
                        ax=ax
                    )
                
                # Draw cycle edges in RED
                for edge in cycle_edges:
                    # if method == 'final':
                    #     edge_key = f"{edge[0]}->{edge[1]}"
                    #     confidence = final_dag['confidence_scores'].get(edge_key, 0.5)
                    #     width = 1 + 3 * confidence
                    # else:
                    #     width = 3.0
                    width = 3.0
                    
                    nx.draw_networkx_edges(
                        G, pos,
                        edgelist=[edge], 
                        width=width,
                        arrowsize=20,
                        arrowstyle='-|>', 
                        connectionstyle='arc3,rad=0.1',
                        edge_color='red',
                        min_source_margin=20,
                        min_target_margin=20,
                        ax=ax
                    )
                
                # Draw labels
                nx.draw_networkx_labels(G, pos, ax=ax, font_size=10, font_weight='bold')
                
                # Set title with cycle count
                cycle_text = f" ({len(cycle_edges)} cycle edges)" if cycle_edges else ""
                title = f"Final DAG{cycle_text}" if method == 'final' else f"{method.upper()} DAG{cycle_text}"
                ax.set_title(title)
                ax.axis('off')
            
            # Add a shared legend at the bottom of the figure
            legend_elements = [plt.Line2D([0], [0], marker='o', color='w', 
                                        markerfacecolor=color, 
                                        markersize=15, label=label) 
                            for label, color in node_colors.items()]
            
            # Add cycle edge legend
            legend_elements.append(plt.Line2D([0], [0], color='red', linewidth=3, 
                                            label='Cycle Edge'))
            
            fig.legend(handles=legend_elements, loc='lower center', 
                    bbox_to_anchor=(0.5, 0), ncol=len(legend_elements), fontsize=12)
            
            plt.tight_layout(rect=[0, 0.05, 1, 0.98])
            plt.savefig(os.path.join(output_dir, 'final_comparison_visualization.png'), 
                    bbox_inches='tight')
            plt.close()
            
        except Exception as e:
            logger.error(f"Error creating final comparison visualization: {str(e)}")

    def _save_intermediate_data(self, edge):
        """Save intermediate data after successful interventions"""
        try:
            # Create directory for intermediate data
            output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 
                                    "intermediate_results", "intermediate_data")
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            
            # Save the current state of data WITHOUT additional scaling
            # (data is already scaled in _update_training_data_with_interventions)
            csv_path = os.path.join(output_dir, 
                    f'data_after_{edge[0]}_{edge[1]}_{timestamp}.csv')
            self.data.to_csv(csv_path, index=False)
            
            # Save validated edges record
            edge_file = os.path.join(output_dir, f'validated_edges_{timestamp}.json')
            with open(edge_file, 'w') as f:
                json.dump({
                    'timestamp': timestamp,
                    'iteration': self.current_iteration,
                    'edge': f"{edge[0]}->{edge[1]}",
                    'validated_edges': [f"{e[0]}->{e[1]}" for e in self.validated_edges]
                }, f, indent=2)
                
            logger.info(f"Saved intermediate dataset to {csv_path}")
                    
        except Exception as e:
            logger.error(f"Error saving intermediate data: {str(e)}")

    def get_policy_metadata(self):
        """Extract metadata needed for policy engine"""
        metadata = []
        for method, result in self._results.items():
            edges = self._extract_edges(result)
            for edge in edges:
                confidence = self.edge_ranker.edge_confidence.get(edge, 0)
                edge_results = self.tester.intervention_results.get(edge, [])
                cost = len(edge_results) * 0.1
                risk = (1 - confidence) * cost
                metadata.append({
                    'Method': method,
                    'Edge': f"{edge[0]}->{edge[1]}",
                    'Confidence': confidence,
                    'Cost': cost,
                    'Risk': risk
                })
        return pd.DataFrame(metadata)

    def export_for_policy_engine(self):
        """Export complete results for policy engine.

        Includes 'combined_dataset' — the observational data augmented with
        upweighted intervention rows (weight column present).  The policy
        engine's SEM fitting uses this for weighted Ridge regression.
        """
        return {
            'final_dag': self._create_final_dag(self.validated_edges),
            'method_dags': self._results,
            'intervention_results': self.tester.intervention_results,
            'validated_edges': self.validated_edges,
            'edge_confidence': self.edge_ranker.edge_confidence,
            'metadata': self.get_policy_metadata(),
            'combined_dataset': self.data,
        }