"""
Causal Discovery Algorithm Implementations
============================================

Provides four hypothesis generators, each returning a candidate DAG
(set of directed edges) from the same observational dataset:

  - **PCGenerator**:       Peter-Clark algorithm (constraint-based).
                           Uses Fisher-Z conditional independence tests.
                           Library: ``causal-learn``.
  - **SAMGenerator**:      Structural Agnostic Model (score-based, neural).
                           Fits a generative NN and reads off the adjacency.
                           Library: ``cdt`` (requires PyTorch; optionally R).
  - **LLMGenerator**:      Queries GPT-4 with a domain prompt to propose
                           plausible causal edges.  No statistical test —
                           purely knowledge-driven.
  - **VARLiNGAMGenerator**: Vector-Autoregressive LiNGAM for time-series.
                           Exploits non-Gaussianity to orient edges.
                           Library: ``lingam``.

The pipeline unions all four outputs, ranks edges by method agreement,
and passes the top-K to intervention validation (``tester.py``).

DOMAIN-AGNOSTIC KNOBS:
  - ``relevant_columns``: Passed at init; controls which variables enter
                          each algorithm's adjacency search.
  - ``LLMGenerator``:     The prompt template references smart-building
                          variables.  Update the prompt for a new domain.
  - ``VARLiNGAMGenerator(max_lags, criterion)``: Lag order and model
                          selection criterion (AIC/BIC).  Tune for your
                          data's temporal resolution.
  - ``PCGenerator``:      ``alpha`` (significance level for CI tests),
                          ``indep_test`` (default Fisher-Z; swap for
                          discrete data).
"""

import numpy as np
import pandas as pd
import json
from causallearn.search.ConstraintBased.PC import pc
from cdt.causality.graph import SAM
from causallearn.utils.cit import fisherz
from .gpt_client import GPTClient
import logging
from scipy import stats
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import networkx as nx
from lingam import VARLiNGAM
import warnings

logger = logging.getLogger(__name__)


class BaseGenerator:
    """Base class for all generators to ensure consistent interface"""
    def __init__(self, relevant_columns=None):
        if relevant_columns:
            self.relevant_columns = relevant_columns
        else:
            # self.relevant_columns = [
            #     'Temperature', 'Humidity', 'AirQuality',
            #     'LightingLevel', 'HVACSetpoint', 'OccupantCount',
            #     'EnergyConsumption', 'OverallSatisfaction',
            #     'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
            #     'HVACPower', 'LightingPower'
            # ]
            self.relevant_columns = [
                'Temperature', 'Humidity', 'AirQuality', 'PMV',
                'EnergyConsumption', 'Satisfaction',
                'WindowOpen', 'OutdoorTemperature'
            ]
        self.FALLBACK = {'nodes': [], 'edges': []}
        
    def _validate_data(self, data):
        """Validate input data consistency"""
        missing_cols = set(self.relevant_columns) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        return data[self.relevant_columns].copy()

    def _validate_edges(self, edges):
        """Basic edge validation common to all generators"""
        valid_edges = []
        for edge in edges:
            # Skip invalid source nodes
            if edge[0] in ['EnergyConsumption', 'Satisfaction', 'OverallSatisfaction']:
            # if edge[0] in ['collision']:
                continue
            # Skip self-loops
            if edge[1] == edge[0]:
                continue
            valid_edges.append(edge)
        return valid_edges
    
    def _remove_cycles(self, edges):
        """Remove cycles from edges while preserving strongest relationships"""
        G = nx.DiGraph()
        G.add_edges_from(edges)
        
        while not nx.is_directed_acyclic_graph(G):
            try:
                cycle = nx.find_cycle(G, orientation="original")
                # Remove weakest edge in cycle
                G.remove_edge(*cycle[0])
            except nx.NetworkXNoCycle:
                break
                
        return list(G.edges())
    
class PCGenerator:
    def __init__(self, alpha=0.05, relevant_columns=None):
        self.alpha = alpha
        # Set Fisher-Z test as the independence test
        self.indep_test = fisherz
        self.stable = True
        self.uc_rule = 1  # Changed from 'pc' to 1
        self.uc_priority = 2  # Use priority 2: prioritize existing colliders
        self.mvpc = False       # Add missing attribute
        self.correction_name = None  # Add missing attribute
        self.background_knowledge = None  # Add missing attribute
        self.verbose = False    # Add missing attribute
        self.show_progress = False  # Add missing attribute

        # Store relevant columns if provided, otherwise use defaults
        self.relevant_columns = relevant_columns or [
            'Temperature', 'Humidity', 'AirQuality',
            'EnergyConsumption', 'Satisfaction'
        ]
        # self.relevant_columns = relevant_columns or [
        #     'Temperature', 'Humidity', 'AirQuality',
        #     'LightingLevel', 'HVACSetpoint', 'OccupantCount',
        #     'EnergyConsumption', 'OverallSatisfaction',
        #     'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
        #     'HVACPower', 'LightingPower'
        # ]

    def generate(self, data):
        """
        Generate causal hypothesis using PC algorithm
        
        Args:
            data (pd.DataFrame): Input data
        """
        logger.info("Generating hypothesis using PC method...")

        try:
            if hasattr(self, 'relevant_columns') and self.relevant_columns:
                # Extract only the relevant columns from data
                available_columns = set(data.columns)
                usable_columns = [col for col in self.relevant_columns if col in available_columns]
                
                if not usable_columns:
                    logger.error(f"No relevant columns found in data. Available: {available_columns}")
                    return {'nodes': [], 'edges': []}
                    
                processed_data = data[usable_columns].copy()

            else:
                # Select only the relevant numerical columns
                # relevant_columns = [
                #     'Temperature', 'Humidity', 'AirQuality',
                #     'LightingLevel', 'HVACSetpoint', 'OccupantCount',
                #     'EnergyConsumption', 'OverallSatisfaction',
                #     'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
                #     'HVACPower', 'LightingPower'
                # ]
                relevant_columns = [
                    'Temperature', 'Humidity', 'AirQuality',
                    'EnergyConsumption', 'Satisfaction'
                ]
                # relevant_columns = [
                #     'handX', 'handY', 'ballX', 'ballY',
                #     'collision'
                # ]
                # Extract relevant columns
                processed_data = data[relevant_columns].copy()

            # Check for singular correlation matrix and add noise if needed
            corr_matrix = processed_data.corr()
            if np.linalg.det(corr_matrix) == 0 or np.isclose(np.linalg.det(corr_matrix), 0):
                logger.warning("Singular correlation matrix detected. Adding noise to break multicollinearity.")
                noise_scale = processed_data.std() * 0.001
                for col in processed_data.columns:
                    processed_data[col] += np.random.normal(0, noise_scale[col], len(processed_data))

            # Remove constant columns
            # constant_cols = processed_data.columns[processed_data.std() == 0]
            # if len(constant_cols) > 0:
            #     logger.warning(f"Removing constant columns: {list(constant_cols)}")
            #     processed_data = processed_data.drop(columns=constant_cols)

            # Standardize data
            from sklearn.preprocessing import StandardScaler
            scaler = StandardScaler()
            processed_data = pd.DataFrame(
                scaler.fit_transform(processed_data),
                columns=processed_data.columns
            )

            # Check for missing values
            if processed_data.isnull().values.any():
                logger.warning("Input data contains missing values. Dropping rows with missing values.")
                processed_data = processed_data.dropna()

            data_array = processed_data.values

            # Handle weighted data
            if 'weight' in data.columns:
                weights = data['weight'].values
                weighted_data = processed_data * weights[:, np.newaxis]
                data_array = weighted_data.values
            else:
                data_array = processed_data.values

            logger.info(f"Processed data shape: {data_array.shape}")
            
            # Log parameter values being used
            # logger.info(f"Running PC algorithm with parameters:")
            # logger.info(f"alpha={self.alpha}, indep_test={self.indep_test}")
            # logger.info(f"stable={self.stable}, uc_rule={self.uc_rule}")
            # logger.info(f"uc_priority={self.uc_priority}, mvpc={self.mvpc}")
            
            try:
                cg = pc(data=data_array,
                        alpha=self.alpha,
                        indep_test=self.indep_test,
                        stable=self.stable,
                        uc_rule=self.uc_rule,
                        uc_priority=self.uc_priority,
                        mvpc=self.mvpc,
                        correction_name=self.correction_name,
                        background_knowledge=self.background_knowledge,
                        verbose=False,  
                        show_progress=self.show_progress)
            except Exception as e:
                logger.error(f"Error in PC generation: {str(e)}", exc_info=True)
                raise
            
            # Extract edges from the graph
            edges = self._get_edges(cg.G.graph, processed_data.columns)
            
            logger.info(f"PC algorithm completed. Found {len(edges)} edges: {edges}")
            return {
                'nodes': list(processed_data.columns),
                'edges': edges
            }
            
        except Exception as e:
            logger.error(f"Error in PC generation: {str(e)}")
            return {'nodes': [], 'edges': []}
    
    def _get_edges(self, graph, column_names):
        """Convert PC graph to directed edges only, removing bidirectional edges 
        and retaining only the stronger direction."""
        edges = set()  # Use a set to avoid duplicate edges
        n_nodes = len(column_names)
        
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):  # Iterate only once per pair (i, j)
                if graph[i, j] in {1, -1} and graph[j, i] in {1, -1}:  # Bidirectional case
                    # Retain the stronger direction (higher absolute value)
                    if abs(graph[i, j]) >= abs(graph[j, i]):
                        edges.add((column_names[i], column_names[j]))
                    else:
                        edges.add((column_names[j], column_names[i]))
                elif graph[i, j] in {1, -1}:  # One-way edge (i → j)
                    edges.add((column_names[i], column_names[j]))
                elif graph[j, i] in {1, -1}:  # One-way edge (j → i)
                    edges.add((column_names[j], column_names[i]))
        
        return list(edges)  # Convert back to list for final output

# class SAMGenerator:
#     def __init__(self, relevant_columns=None):
#         super().__init__()
#         self.sparsity_threshold = 0.15
#         self.min_edge_weight = 0.3
#         self.significance_level = 0.05
#         self.variable_ranges = {
#             'Temperature': {'min': 18, 'max': 30},
#             'Humidity': {'min': 30, 'max': 70},
#             'AirQuality': {'min': 0, 'max': 500},
#             # 'HVACSetpoint': {'min': 18, 'max': 30},  # Add this
#             # 'LightingLevel': {'min': 0.0, 'max': 1.0},  # Add this
#             # 'OccupantCount': {'min': 0, 'max': 50}  # Add this
#         }

#         # self.relevant_columns = [
#         #     'handX', 'handY', 'ballX', 'ballY',
#         #     'collision'
#         # ]
#         # # Update variable ranges for robot arm
#         # self.variable_ranges = {
#         #     'handX': {'min': 0, 'max': 400},
#         #     'handY': {'min': 0, 'max': 300},
#         #     'ballX': {'min': 0, 'max': 400},
#         #     'ballY': {'min': 0, 'max': 300}
#         # }
#         # Store relevant columns if provided, otherwise use defaults
#         self.relevant_columns = relevant_columns or [
#             'Temperature', 'Humidity', 'AirQuality',
#             'EnergyConsumption', 'OverallSatisfaction'
#         ]
#         # self.relevant_columns = relevant_columns or [
#         #     'Temperature', 'Humidity', 'AirQuality',
#         #     'LightingLevel', 'HVACSetpoint', 'OccupantCount',
#         #     'EnergyConsumption', 'OverallSatisfaction',
#         #     'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 
#         #     'HVACPower', 'LightingPower'
#         # ]
        
#         # Define variable ranges based on dataset type
#         if 'air_temperature' in (self.relevant_columns or []):
#             self.variable_ranges = {
#                 'air_temperature': {'min': -10, 'max': 40},
#                 'dew_temperature': {'min': -10, 'max': 30},
#                 'sea_level_pressure': {'min': 980, 'max': 1030}
#             }
#         else:
#             self.variable_ranges = {
#                 'Temperature': {'min': 18, 'max': 30},
#                 'Humidity': {'min': 30, 'max': 70},
#                 'AirQuality': {'min': 0, 'max': 500}
#             }

#     def _validate_data(self, data):
#         """Validate input data consistency"""
#         available_columns = set(data.columns)
#         usable_columns = [col for col in self.relevant_columns if col in available_columns]
        
#         if not usable_columns:
#             raise ValueError(f"No relevant columns found in data. Available: {available_columns}")
            
#         return data[usable_columns].copy()

#     def validate_edges(self, edges, data):
#         G = nx.DiGraph()
#         weighted_edges = []
        
#         # Calculate edge weights using partial correlations
#         for edge in edges:
#             source, target = edge
#             partial_corr = self._calculate_partial_correlation(
#                 data[source], data[target], 
#                 data[[col for col in data.columns if col not in [source, target]]]
#             )
#             if abs(partial_corr) >= self.min_edge_weight:
#                 weighted_edges.append((source, target, {'weight': abs(partial_corr)}))
        
#         if not weighted_edges:
#             # Fallback: use strongest correlations to ensure non-empty graph
#             correlations = data[self.relevant_columns].corr()
#             for i, source in enumerate(self.relevant_columns):
#                 for target in self.relevant_columns[i+1:]:
#                     corr = abs(correlations.loc[source, target])
#                     if corr >= self.min_edge_weight:
#                         weighted_edges.append((source, target, {'weight': corr}))
        
#         G.add_edges_from(weighted_edges)
        
#         # Remove cycles while preserving strongest edges
#         while not nx.is_directed_acyclic_graph(G):
#             try:
#                 cycle = nx.find_cycle(G, orientation="original")
#                 # Remove weakest edge in cycle
#                 min_weight_edge = min(cycle, key=lambda x: G[x[0]][x[1]]['weight'])
#                 G.remove_edge(*min_weight_edge[:2])
#             except nx.NetworkXNoCycle:
#                 break

#         # Apply sparsity constraint
#         edges_to_remove = []
#         for edge in G.edges(data=True):
#             if edge[2]['weight'] < self.sparsity_threshold:
#                 edges_to_remove.append((edge[0], edge[1]))
#         G.remove_edges_from(edges_to_remove)
        
#         # Ensure graph remains connected and meaningful
#         if len(G.edges()) < 2:
#             # Add strongest valid edges back
#             sorted_edges = sorted(weighted_edges, key=lambda x: x[2]['weight'], reverse=True)
#             for edge in sorted_edges[:3]:
#                 if not self._creates_cycle(G, edge[0], edge[1]):
#                     G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
        
#         return list(G.edges())

#     def _calculate_partial_correlation(self, x, y, z):
#         try:
#             # Calculate residuals
#             x_resid = self._get_residuals(x, z)
#             y_resid = self._get_residuals(y, z)
            
#             # Calculate partial correlation
#             corr = stats.pearsonr(x_resid, y_resid)[0]
#             return corr if abs(corr) > self.min_edge_weight else 0
            
#         except Exception as e:
#             logger.error(f"Partial correlation calculation failed: {e}")
#             return 0

#     def _get_residuals(self, target, features):
#         try:
#             from sklearn.linear_model import LassoCV
#             model = LassoCV(cv=5, random_state=42)
#             model.fit(features, target)
#             return target - model.predict(features)
#         except Exception:
#             return target

#     def _creates_cycle(self, G, source, target):
#         test_G = G.copy()
#         test_G.add_edge(source, target)
#         try:
#             nx.find_cycle(test_G, orientation="original")
#             return True
#         except nx.NetworkXNoCycle:
#             return False

#     def generate(self, data):
#         if not self.relevant_columns:
#             self.relevant_columns = data.columns.tolist()

#         try:
#             # Filter to only existing columns
#             available_cols = [col for col in self.relevant_columns if col in data.columns]
#             self.relevant_columns = available_cols  # Update for this run
            
#             processed_data = self._validate_data(data)
                
#             # Handle weighted data
#             if 'weight' in data.columns:
#                 weights = data['weight'].values
#                 weighted_df = processed_data * weights[:, np.newaxis]
#                 samples = pd.DataFrame(
#                     StandardScaler().fit_transform(weighted_df),
#                     columns=self.relevant_columns
#                 )
#             else:
#                 samples = pd.DataFrame(
#                     StandardScaler().fit_transform(processed_data),
#                     columns=available_cols
#                 )
            
#             model = SAM(
#                 lr=0.01,
#                 dlr=0.005,
#                 lambda1=0.1,
#                 lambda2=0.01,
#                 train_epochs=200,
#                 test_epochs=40,
#                 batch_size=32,
#                 dagloss=True,
#                 dagstart=0.0,
#                 nruns=3,
#                 njobs=1,
#                 verbose=False
#             )
            
#             G = model.predict(samples)
#             edges = G.edges()
#             valid_edges = self.validate_edges(edges, processed_data)
#             # cycle_free_edges = self._remove_cycles(valid_edges)
#             logger.info(f"SAM algorithm completed. Found {len(valid_edges)} edges: {valid_edges}")

#             return {
#                 'nodes': self.relevant_columns,
#                 'edges': valid_edges
#             }

#         except Exception as e:
#             logger.error(f"Error in SAM generation: {str(e)}")
#             return self._generate_fallback_dag(data)
        
#     def _get_strongest_valid_edge(self, data, source):
#         correlations = data[self.relevant_columns].corr()
#         targets = ['EnergyConsumption', 'OverallSatisfaction']
#         # targets = ['collision']
#         edges = []
        
#         for target in targets:
#             corr = abs(correlations.loc[source, target])
#             if corr >= self.min_edge_weight:
#                 edges.append((source, target, corr))
                
#         return max(edges, key=lambda x: x[2])[:2] if edges else None

#     def _generate_fallback_dag(self, data):
#         """Generate minimal valid DAG based on strongest correlations"""
#         correlations = data[self.relevant_columns].corr()
#         edges = []
        
#         # Add strongest validated edges for core relationships
#         # for source in ['Temperature', 'Humidity', 'AirQuality']:
#         for source in ['handX', 'handY', 'ballX', 'ballY']:
#             edge = self._get_strongest_valid_edge(data, source)
#             if edge:
#                 edges.append(edge)
        
#         return {
#             'nodes': self.relevant_columns,
#             'edges': edges
#         }


class LLMGenerator:
    # Sim-specific physical context for the LLM prompt
    DOMAIN_CONTEXT = {
        'smart_room': """
            A single-zone room where an HVAC system controls temperature,
            humidity, and air quality. The room consumes energy and has
            an occupant satisfaction metric.""",
        'smart_room_noise': """
            A single-zone room where an HVAC system controls temperature,
            humidity, and air quality. Sensors have measurement noise.
            The room consumes energy and has an occupant satisfaction metric.""",
        'hidden_vars': """
            A single-zone room with HVAC control over temperature, humidity,
            and air quality. There may be unobserved confounders 
            not in the variable list.""",
        'open_window': """
            A single-zone room with HVAC control and a window that opens
            and closes. The room has an outdoor environment. PMV measures
            thermal comfort. The window state and outdoor temperature are
            not controllable. The window state cannot be observed either.""",
        'smart_building_rich': """
            A single-zone room with HVAC and lighting systems. The room
            has outdoor weather conditions, solar radiation, occupants
            that come and go, a window that can open, and sensors for
            temperature, humidity, CO2, light level, noise, air quality,
            and thermal comfort (PMV). Occupancy and window position may
            not be directly observable.""",
        'ashrae': """
            Real-world building energy data from the ASHRAE Great Energy
            Predictor dataset. Variables include weather conditions and
            building characteristics.""",
    }

    def __init__(self, api_key, relevant_columns=None, dataset_type=None):
        logger.info("Initializing LLM Generator with API key.")
        self.gpt_client = GPTClient(api_key)
        self.dataset_type = dataset_type or 'smart_room'
        self.relevant_columns = relevant_columns or [
            'Temperature', 'Humidity', 'AirQuality',
            'EnergyConsumption', 'Satisfaction'
        ]
        self.FALLBACK = {
            'nodes': [],
            'edges': []
        }

    def _load_cached_edges(self):
        """Return a cached LLM response for this dataset_type, or None.

        The PROMPT is a fixed template per ``dataset_type`` (it does not read
        the data), but the RESPONSE is stochastic (temperature 1.0): live runs
        draw a fresh response per call, so different seeds generally receive
        different proposals. A cache file replays ONE recorded response for
        every seed --- protocol-faithful, but it removes the per-seed draw
        variance that live runs have. Drop a file at
        ``results/llm_edge_cache/<dataset_type>.json`` with
        ``{"nodes": [...], "edges": [[src, tgt], ...]}`` to bypass the GPT API
        when it is unavailable. Delete the file (or the dir) to resume live
        API calls. Override the dir with the LLM_EDGE_CACHE_DIR env var.
        """
        import os
        cache_dir = os.environ.get('LLM_EDGE_CACHE_DIR', 'results/llm_edge_cache')
        path = os.path.join(cache_dir, f'{self.dataset_type}.json')
        if not os.path.exists(path):
            return None
        try:
            with open(path) as f:
                d = json.load(f)
            edges = [[e[0], e[1]] for e in d.get('edges', [])
                     if isinstance(e, (list, tuple)) and len(e) == 2]
            return {'nodes': d.get('nodes', self.relevant_columns), 'edges': edges}
        except Exception as e:
            logger.warning(f"LLM edge cache at {path} unreadable ({e}); using API")
            return None

    def generate(self, data):
        cached = self._load_cached_edges()
        if cached is not None:
            logger.info(f"LLM: using cached edges for dataset_type="
                        f"'{self.dataset_type}' (GPT API bypass) — "
                        f"{len(cached['edges'])} edges")
            return cached
        try:
            # correlations = data.corr().round(3).to_dict()
            # system_prompt = """
            # You are an expert in causal discovery analyzing a smart room environment with 5 variables.
            # Generate a causal DAG based on physical principles and environmental systems.
            # Return your answer as a JSON object with this format exactly:
            # {"nodes": ['air_temperature', 'dew_temperature', 'sea_level_pressure', 'meter_reading', 'square_feet'], 
            # "edges": [["source", "target"], ...]}
            # """
            # system_prompt = """
            # You are an expert in causal discovery. 
            # Generate a causal DAG based on the environment and energy consumption in buildings.
            # Return your answer as a JSON object with this format exactly:
            # {"nodes": ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction'], "edges": [["source", "target"], ...]}
            # IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks.
            # """
            
            # Updated system prompt for smart building context:
            # system_prompt = """
            # You are an expert in causal discovery analyzing a smart building environment.
            # Generate a causal DAG based on HVAC systems, thermal comfort, and energy efficiency.
            # Variables: 'Temperature', 'Humidity', 'AirQuality', 'LightingLevel', 'HVACSetpoint', 'OccupantCount',
            # 'EnergyConsumption', 'OverallSatisfaction', 'ThermalComfort', 'VisualComfort', 'AirQualityIndex', 'HVACPower', 'LightingPower']
            # Outcomes: 'EnergyConsumption', 'OverallSatisfaction', 'AirQualityIndex',
            # 'ThermalComfort', 'VisualComfort', 'HVACPower', 'LightingPower'
            # Return JSON format: {"nodes": [...], "edges": [["source", "target"], ...]}
            # IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks.
            # """
            # Create prompt with actual column names
            # columns_str = ", ".join(data.columns)
            # user_message = f"""
            # Analyze this dataset with variables: {columns_str}
            
            # Rules:
            # 1. Include directed edges based on likely causal mechanisms
            # 2. Cycles are ok but no self-loops allowed
            # 3. Focus on primary physical relationships, not secondary effects
            # 4. Return only the JSON object, no additional text
            # """

            # Build prompt dynamically from actual column names
            cols = self.relevant_columns
            cols_str = str(cols)

            system_prompt = f"""
            You are an expert in causal discovery for building environments.

            Generate a causal DAG using ONLY these variables:
            {cols_str}

            DO NOT introduce new variables not in the list above.

            Return your answer as a JSON object with this format exactly:
            {{"nodes": {cols_str}, "edges": [["source", "target"], ...]}}

            IMPORTANT: Return ONLY valid JSON, no markdown formatting or code blocks.
            """

            domain = self.DOMAIN_CONTEXT.get(self.dataset_type, self.DOMAIN_CONTEXT['smart_room'])

            user_message = f"""
            Analyze this building dataset with JUST variables:
            {cols_str}

            Domain context:
            {domain}

            Rules:
            1. Consider physical causation, not just correlation
            2. No self-loops, cycles allowed
            3. Controllable variables can affect multiple downstream variables
            4. Outcome/derived variables generally do not cause input variables
            5. Use ONLY the exact variable names listed above

            Return only JSON, no additional text.
            """
            

            # user_message = """
            # Analyze this smart room system with these constraints:

            # Rules:
            # 1. Include directed edges based on likely causal mechanisms
            # 2. No cycles or self-loops allowed
            # 3. Focus on primary physical relationships, not secondary effects
            # """

            # system_prompt = """
            # You are an expert in causal discovery analyzing a robot arm system with 5 variables.
            # Generate a causal DAG based on physical principles and robotic systems.
            # Return your answer as a JSON object with this format exactly:
            # {"nodes": ["handX", "handY", "ballX", "ballY", "collision"], 
            # "edges": [["source", "target"], ...]}
            # """

            # user_message = """
            # Analyze this robot arm system with these constraints:

            # Rules:
            # 1. Include directed edges based on likely causal mechanisms
            # 2. No cycles or self-loops allowed
            # 3. Focus on primary physical relationships, not secondary effects
            # 4. Hand positions (handX, handY) affect the collision detection
            # 5. Ball positions (ballX, ballY) affect the collision detection
            # 6. Collision affects whether the ball is caught
            # """

            response = self.gpt_client.generate_response(system_prompt, user_message, temperature=1.0, top_p=0.8)
            logger.info(f'LLM Hypothesis (unparsed): {response}')

            if not response or not response.strip():
                logger.warning("Received empty response from GPT")
                return self.FALLBACK

            try:
                # Enhanced cleaning of the response string
                cleaned_response = response.strip()
                
                # Remove common markdown formatting
                if cleaned_response.startswith('```json'):
                    cleaned_response = cleaned_response[7:]
                elif cleaned_response.startswith('```'):
                    cleaned_response = cleaned_response[3:]
                elif cleaned_response.startswith('`json'):
                    cleaned_response = cleaned_response[5:]
                elif cleaned_response.startswith('`'):
                    cleaned_response = cleaned_response[1:]
                    
                if cleaned_response.endswith('```'):
                    cleaned_response = cleaned_response[:-3]
                elif cleaned_response.endswith('`'):
                    cleaned_response = cleaned_response[:-1]
                    
                cleaned_response = cleaned_response.strip()
                
                # Log the cleaned response for debugging
                logger.debug(f"Cleaned response: '{cleaned_response}'")
                
                # Check if response is empty after cleaning
                if not cleaned_response:
                    logger.warning("Response is empty after cleaning")
                    return self.FALLBACK

                parsed = json.loads(cleaned_response)
                
                # Validate the parsed response structure
                if not isinstance(parsed, dict):
                    logger.warning("Response is not a dictionary")
                    return self.FALLBACK
                    
                if not all(k in parsed for k in ['nodes', 'edges']):
                    logger.warning(f"Invalid response format. Missing keys. Got: {list(parsed.keys())}")
                    return self.FALLBACK
                    
                # Validate edges format
                if not isinstance(parsed['edges'], list):
                    logger.warning("Edges is not a list")
                    return self.FALLBACK
                    
                # Clean and validate edges
                valid_edges = []
                for edge in parsed['edges']:
                    if isinstance(edge, list) and len(edge) == 2:
                        source, target = edge
                        if isinstance(source, str) and isinstance(target, str):
                            valid_edges.append([source, target])
                        else:
                            logger.warning(f"Invalid edge format: {edge}")
                    else:
                        logger.warning(f"Invalid edge structure: {edge}")
                
                parsed['edges'] = valid_edges
                
                logger.info(f"LLM generation completed. Found {len(parsed.get('edges', []))} edges: {parsed}")
                return parsed 

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse GPT response as JSON: {e}")
                logger.error(f"Problematic response: '{cleaned_response}'")
                
                # Try to extract JSON from response if it contains other text
                try:
                    import re
                    json_match = re.search(r'\{.*\}', cleaned_response, re.DOTALL)
                    if json_match:
                        json_str = json_match.group(0)
                        parsed = json.loads(json_str)
                        logger.info(f"Successfully extracted JSON from response: {parsed}")
                        return parsed
                except:
                    pass
                    
                return self.FALLBACK

        except Exception as e:
            logger.error(f"Error in LLMGenerator: {str(e)}")
            return self.FALLBACK 

class SAMGenerator:
    def __init__(self, relevant_columns=None, allow_cycles=False):
        super().__init__()
        self.allow_cycles = allow_cycles
        self.sparsity_threshold = 0.15
        self.min_edge_weight = 0.3
        self.significance_level = 0.05
        self.relevant_columns = relevant_columns or []

    def _validate_data(self, data):
        """Validate input data consistency"""
        available_columns = set(data.columns)
        usable_columns = [col for col in self.relevant_columns if col in available_columns]
        
        if not usable_columns:
            raise ValueError(f"No relevant columns found in data. Available: {available_columns}")
            
        return data[usable_columns].copy()

    def validate_edges(self, edges, data):
        G = nx.DiGraph()
        weighted_edges = []
        
        # Calculate edge weights using partial correlations
        for edge in edges:
            source, target = edge
            partial_corr = self._calculate_partial_correlation(
                data[source], data[target], 
                data[[col for col in data.columns if col not in [source, target]]]
            )
            if abs(partial_corr) >= self.min_edge_weight:
                weighted_edges.append((source, target, {'weight': abs(partial_corr)}))
        
        if not weighted_edges:
            # Fallback: use strongest correlations to ensure non-empty graph
            correlations = data[self.relevant_columns].corr()
            for i, source in enumerate(self.relevant_columns):
                for target in self.relevant_columns[i+1:]:
                    corr = abs(correlations.loc[source, target])
                    if corr >= self.min_edge_weight:
                        weighted_edges.append((source, target, {'weight': corr}))
        
        G.add_edges_from(weighted_edges)
        
        # Remove cycles only if allow_cycles=False
        if not self.allow_cycles:
            while not nx.is_directed_acyclic_graph(G):
                try:
                    cycle = nx.find_cycle(G, orientation="original")
                    # Remove weakest edge in cycle
                    min_weight_edge = min(cycle, key=lambda x: G[x[0]][x[1]]['weight'])
                    G.remove_edge(*min_weight_edge[:2])
                except nx.NetworkXNoCycle:
                    break

        # Apply sparsity constraint
        edges_to_remove = []
        for edge in G.edges(data=True):
            if edge[2]['weight'] < self.sparsity_threshold:
                edges_to_remove.append((edge[0], edge[1]))
        G.remove_edges_from(edges_to_remove)
        
        # Ensure graph remains connected and meaningful
        if len(G.edges()) < 2:
            # Add strongest valid edges back
            sorted_edges = sorted(weighted_edges, key=lambda x: x[2]['weight'], reverse=True)
            for edge in sorted_edges[:3]:
                # Check cycles only if not allowing them
                if self.allow_cycles or not self._creates_cycle(G, edge[0], edge[1]):
                    G.add_edge(edge[0], edge[1], weight=edge[2]['weight'])
        
        return list(G.edges())

    def _calculate_partial_correlation(self, x, y, z):
        try:
            # Calculate residuals
            x_resid = self._get_residuals(x, z)
            y_resid = self._get_residuals(y, z)
            
            # Calculate partial correlation
            corr = stats.pearsonr(x_resid, y_resid)[0]
            return corr if abs(corr) > self.min_edge_weight else 0
            
        except Exception as e:
            logger.error(f"Partial correlation calculation failed: {e}")
            return 0

    def _get_residuals(self, target, features):
        try:
            from sklearn.linear_model import LassoCV
            model = LassoCV(cv=5, random_state=42)
            model.fit(features, target)
            return target - model.predict(features)
        except Exception:
            return target

    def _creates_cycle(self, G, source, target):
        test_G = G.copy()
        test_G.add_edge(source, target)
        try:
            nx.find_cycle(test_G, orientation="original")
            return True
        except nx.NetworkXNoCycle:
            return False

    def get_cycles(self):
        """Detect cycles in the generated graph"""
        if not hasattr(self, '_graph'):
            return []
        try:
            return list(nx.simple_cycles(self._graph))
        except:
            return []

    def generate(self, data):
        if not self.relevant_columns:
            self.relevant_columns = data.columns.tolist()

        try:
            # Filter to only existing columns
            available_cols = [col for col in self.relevant_columns if col in data.columns]
            self.relevant_columns = available_cols
            
            processed_data = self._validate_data(data)
                
            # Handle weighted data
            if 'weight' in data.columns:
                weights = data['weight'].values
                weighted_df = processed_data * weights[:, np.newaxis]
                samples = pd.DataFrame(
                    StandardScaler().fit_transform(weighted_df),
                    columns=self.relevant_columns
                )
            else:
                samples = pd.DataFrame(
                    StandardScaler().fit_transform(processed_data),
                    columns=available_cols
                )
            
            model = SAM(
                lr=0.01,
                dlr=0.005,
                lambda1=0.1,
                lambda2=0.01,
                train_epochs=200,
                test_epochs=40,
                batch_size=32,
                dagloss=True,
                dagstart=0.0,
                nruns=3,
                njobs=1,
                verbose=False
            )
            
            G = model.predict(samples)
            edges = G.edges()
            valid_edges = self.validate_edges(edges, processed_data)
            
            # Store graph for cycle detection
            self._graph = nx.DiGraph()
            self._graph.add_edges_from(valid_edges)
            
            logger.info(f"SAM algorithm completed. Found {len(valid_edges)} edges: {valid_edges}")

            return {
                'nodes': self.relevant_columns,
                'edges': valid_edges
            }

        except Exception as e:
            logger.error(f"Error in SAM generation: {str(e)}")
            return self._generate_fallback_dag(data)
        
    def _generate_fallback_dag(self, data):
        """Generate minimal valid DAG based on strongest correlations"""
        available_cols = [c for c in self.relevant_columns if c in data.columns]
        if len(available_cols) < 2:
            return {'nodes': available_cols, 'edges': []}
        correlations = data[available_cols].corr().abs()
        # Collect all pairs ranked by |correlation|
        pairs = []
        for i, src in enumerate(available_cols):
            for tgt in available_cols[i+1:]:
                pairs.append((src, tgt, correlations.loc[src, tgt]))
        pairs.sort(key=lambda x: x[2], reverse=True)
        # Take top edges that don't create cycles
        G = nx.DiGraph()
        for src, tgt, _w in pairs[:5]:
            if not self._creates_cycle(G, src, tgt):
                G.add_edge(src, tgt)
        return {
            'nodes': available_cols,
            'edges': list(G.edges())
        }

class VARLiNGAMGenerator(BaseGenerator):
    """VARLiNGAM for time series causal discovery"""
    
    def __init__(self, max_lags=5, criterion='bic', prune=True, threshold=0.01):
        self.max_lags = max_lags
        self.criterion = criterion
        self.prune = prune
        self.threshold = threshold
        self.model = None
        self.var_names = None
        self._graph = None
    
    def fit(self, data: pd.DataFrame) -> 'VARLiNGAMGenerator':
        """Fit VARLiNGAM model"""
        self.var_names = data.columns.tolist()
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.model = VARLiNGAM(
                lags=self.max_lags,
                criterion=self.criterion,
                prune=self.prune
            )
            self.model.fit(data.values)
        
        return self
    
    def generate(self, data: pd.DataFrame):
        """Generate DAG - pipeline compatibility method"""
        # Filter out weight column if present
        data_clean = data.drop(columns=['weight'], errors='ignore')
        
        # Fit model
        self.var_names = data_clean.columns.tolist()
        
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            self.model = VARLiNGAM(
                lags=self.max_lags,
                criterion=self.criterion,
                prune=self.prune
            )
            self.model.fit(data_clean.values)
        
        n_lags = len(self.model.adjacency_matrices_)
        logger.info(f"VARLiNGAM fitted with {n_lags} lags")
        
        # Extract edges exactly like notebook
        edges = []
        for lag_idx in range(n_lags):
            adj = self.model.adjacency_matrices_[lag_idx]
            for i in range(len(self.var_names)):
                for j in range(len(self.var_names)):
                    if i == j:  # Skip self-loops
                        continue
                    
                    w = adj[i, j]
                    if abs(w) > self.threshold:
                        edges.append({
                            'source': self.var_names[i],
                            'target': self.var_names[j],
                            'lag': lag_idx,
                            'weight': w,
                            'abs_weight': abs(w)
                        })
        
        logger.info(f"VARLiNGAM found {len(edges)} total edges (with lags)")
        
        # Convert to simple edge tuples and normalize
        edge_tuples = list(set([
            (e['source'].lower(), e['target'].lower()) for e in edges
        ]))
        logger.info(f"VARLiNGAM unique edges before reversal: {edge_tuples}")
        
        # Reverse edges that have output vars as source
        output_vars = {'energyconsumption', 'overallsatisfaction'}
        corrected_edges = []
        for edge in edge_tuples:
            if edge[0] in output_vars:
                corrected_edges.append((edge[1], edge[0]))  # Reverse
                logger.info(f"Reversed edge: {edge} -> {(edge[1], edge[0])}")
            else:
                corrected_edges.append(edge)
        
        logger.info(f"VARLiNGAM corrected edges: {corrected_edges}")
        
        # Build graph
        G = nx.DiGraph()
        normalized_names = [name.lower() for name in self.var_names]
        G.add_nodes_from(normalized_names)
        G.add_edges_from(corrected_edges)
        self._graph = G
        
        return {
            'nodes': normalized_names,
            'edges': corrected_edges
        }
    
    def get_cycles(self):
        """Detect cycles in generated graph"""
        if self._graph is None:
            return []
        try:
            return list(nx.simple_cycles(self._graph))
        except:
            return []
    
    def get_model_info(self) -> dict:
        """Return model metadata"""
        if self.model is None:
            return {}
        
        return {
            'algorithm': 'VARLiNGAM',
            'n_lags': len(self.model.adjacency_matrices_),
            'causal_order': [self.var_names[i] for i in self.model.causal_order_]
        }