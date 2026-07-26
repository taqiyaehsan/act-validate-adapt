"""
Ground-Truth DAG Definitions
==============================

Defines the known causal structure for each simulator, used to evaluate
discovery quality (SHD, F1, precision, recall).

Classes:
  - **GroundTruthDAG**:         Base smart room (Temperature/Humidity/AQ → Energy/Satisfaction).
  - **Challenge1GroundTruth**:  Open-window variant (adds WindowOpen → Temperature/Humidity).

DOMAIN-AGNOSTIC KNOBS:
  - Add a new class for your domain's ground truth.  Define ``self.edges``
    as a set of (source, target) tuples.  If ground truth is unknown,
    the framework still runs — just skip F1/SHD evaluation.
  - ``ground_truth_graphs.json`` in the repo root stores serialised versions
    for all simulators.
"""

import networkx as nx
from typing import Set, Tuple, Dict
import numpy as np


class GroundTruthDAG:
    def __init__(self):
        """Initialize ground truth DAG based on smart_room.js relationships"""
        self.edges = {
            ('Temperature', 'EnergyConsumption'),
            ('Temperature', 'Satisfaction'),
            ('Humidity', 'EnergyConsumption'),
            ('Humidity', 'Satisfaction'),
            ('AirQuality', 'EnergyConsumption'),
            ('AirQuality', 'Satisfaction')
        }

        # SMart BuilDing
        # self.edges = [
        #     # HVAC load depends on internal and external environmental conditions
        #     ("Temperature", "HVACPower"),         # More deviation from setpoint increases HVAC load
        #     ("Humidity", "HVACPower"),            # Humidity control affects HVAC power
        #     ("HVACSetpoint", "HVACPower"),        # Directly sets the thermal target
        #     ("OccupantCount", "HVACPower"),       # More occupants → more internal heat gain

        #     # Lighting power depends on control and presence
        #     ("LightingLevel", "LightingPower"),   # Higher level → more lighting energy
        #     ("OccupantCount", "LightingPower"),   # Occupancy triggers lighting systems

        #     # Comfort measures depend on sensed conditions
        #     ("Temperature", "ThermalComfort"),    # PMV model — thermal perception
        #     ("Humidity", "ThermalComfort"),       # Humidity affects heat transfer

        #     ("LightingLevel", "VisualComfort"),   # More light → better visual conditions

        #     ("AirQuality", "AirQualityIndex"),    # Direct transformation of AQI to satisfaction

        #     # Energy consumption is driven by all controls and conditions
        #     ("Temperature", "EnergyConsumption"),     # Temperature deviation increases HVAC load
        #     ("Humidity", "EnergyConsumption"),        # Humidity control costs energy
        #     ("HVACSetpoint", "EnergyConsumption"),    # Setpoint defines thermal demand
        #     ("LightingLevel", "EnergyConsumption"),   # Direct lighting energy use
        #     ("OccupantCount", "EnergyConsumption"),   # Both HVAC and lighting respond to people
        #     ("AirQuality", "EnergyConsumption"),      # Ventilation loads affect energy

        #     # Satisfaction depends directly on controllable factors
        #     ("Temperature", "OverallSatisfaction"),   # Impacts thermal comfort
        #     ("Humidity", "OverallSatisfaction"),      # Impacts perceived air quality & comfort
        #     ("HVACSetpoint", "OverallSatisfaction"),  # Affects comfort if poorly configured
        #     ("LightingLevel", "OverallSatisfaction"), # Affects visual experience
        #     ("OccupantCount", "OverallSatisfaction"), # Indirectly affects crowding/comfort
        #     ("AirQuality", "OverallSatisfaction")     # Perceived air quality and health
        # ]
        
        self.nodes = {
            'Temperature', 'Humidity', 'AirQuality',
            'EnergyConsumption', 'Satisfaction'
        }
        # self.nodes = {
        #     # Environmental Variables (interventional)
        #     'Temperature', 'Humidity', 'AirQuality',
        #     'HVACSetpoint', 'LightingLevel', 'OccupantCount',
            
        #     # Outcome Variables (dependent)
        #     'EnergyConsumption', 'OverallSatisfaction', 
        #     'ThermalComfort', 'VisualComfort', 'AirQualityIndex',
        #     'HVACPower', 'LightingPower'
        # }
        
        # Create NetworkX graph for operations
        self.graph = nx.DiGraph()
        self.graph.add_nodes_from(self.nodes)
        self.graph.add_edges_from(self.edges)

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
            
            # Overall Satisfaction variations (all map to 'Satisfaction' — the actual CSV column)
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
            'lightingPower': 'LightingPower'
        }
    
    # def __init__(self):
    #     """Initialize ground truth DAG based on robot_arm.js relationships"""
    #     # Replace the existing edges with robot arm causal relationships
    #     self.edges = {
    #         ('handX', 'ballX'),         # Hand position affects ball position when caught
    #         ('handY', 'ballY'),         # Hand position affects ball position when caught
    #         ('handX', 'collision'),     # Hand X position affects collision detection
    #         ('handY', 'collision'),     # Hand Y position affects collision detection
    #         ('ballX', 'collision'),     # Ball X position affects collision detection
    #         ('ballY', 'collision'),     # Ball Y position affects collision detection
    #     }
        
    #     # Update nodes list
    #     self.nodes = {
    #         'handX', 'handY', 'ballX', 'ballY', 
    #         'collision'
    #     }
        
    #     # Create NetworkX graph for operations
    #     self.graph = nx.DiGraph()
    #     self.graph.add_nodes_from(self.nodes)
    #     self.graph.add_edges_from(self.edges)

    #     # Update variable mapping
    #     self.variable_mapping = {
    #         # handX variations
    #         'handx': 'handX',
    #         'handX': 'handX',
    #         'HANDX': 'handX',
    #         'hand_x': 'handX',
    #         'hand_X': 'handX',
            
    #         # handY variations
    #         'handy': 'handY',
    #         'handY': 'handY',
    #         'HANDY': 'handY',
    #         'hand_y': 'handY',
    #         'hand_Y': 'handY',
            
    #         # ballX variations
    #         'ballx': 'ballX',
    #         'ballX': 'ballX',
    #         'BALLX': 'ballX',
    #         'ball_x': 'ballX',
    #         'ball_X': 'ballX',
            
    #         # ballY variations
    #         'bally': 'ballY',
    #         'ballY': 'ballY',
    #         'BALLY': 'ballY',
    #         'ball_y': 'ballY',
    #         'ball_Y': 'ballY',
            
    #         # collision variations
    #         'collision': 'collision',
    #         'Collision': 'collision',
    #         'COLLISION': 'collision',
    #     }

    def validate_edge(self, edge: Tuple[str, str]) -> bool:
        """Check if an edge exists in ground truth.
        Normalises via variable_mapping so both 'Satisfaction' and
        'OverallSatisfaction' resolve correctly.
        """
        norm_src = self.variable_mapping.get(edge[0], edge[0])
        norm_tgt = self.variable_mapping.get(edge[1], edge[1])
        return (norm_src, norm_tgt) in self.edges
    
    def get_shd(self, dag_edges: Set[Tuple[str, str]]) -> int:
        """Calculate Structural Hamming Distance using NetworkX-style adjacency matrix comparison."""
        # Normalize edges using the mapping
        normalized_edges = set()
        for src, tgt in dag_edges:
            # Normalize source and target, defaulting to original if not found
            norm_src = self.variable_mapping.get(src, src)
            norm_tgt = self.variable_mapping.get(tgt, tgt)
            normalized_edges.add((norm_src, norm_tgt))
        
        # Create NetworkX graphs
        true_graph = nx.DiGraph()
        true_graph.add_nodes_from(self.nodes)
        true_graph.add_edges_from(self.edges)
        
        pred_graph = nx.DiGraph()
        pred_graph.add_nodes_from(self.nodes)
        pred_graph.add_edges_from(normalized_edges)
        
        # Ensure consistent node ordering
        nodes = sorted(list(self.nodes))
        
        # Convert graphs to adjacency matrices
        adj_mat = nx.to_numpy_array(true_graph, nodelist=nodes)
        pred_adj_mat = nx.to_numpy_array(pred_graph, nodelist=nodes)
        
        # Calculate absolute difference
        diff = np.abs(adj_mat - pred_adj_mat)
        
        # Count differences, considering both missing and extra edges
        return int(np.sum(diff))
        
    def get_edge_directions(self) -> Dict[Tuple[str, str], bool]:
        """Get correct edge directions for all edges"""
        return {edge: True for edge in self.edges}
    
class Challenge1GroundTruth(GroundTruthDAG):
    """Ground truth for Challenge 1: Open Window scenario"""
    
    def __init__(self):
        super().__init__()
        self._load_challenge1_dag()
    
    def _load_challenge1_dag(self):
        import networkx as nx
        self.graph = nx.DiGraph()
        
        edges = [
            ('WindowOpen', 'Temperature'),
            ('OutdoorTemperature', 'Temperature'),
            ('WindowOpen', 'Humidity'),
            ('WindowOpen', 'AirQuality'),
            ('Temperature', 'PMV'),
            ('Humidity', 'PMV'),
            ('WindowOpen', 'EnergyConsumption'),
            ('Temperature', 'EnergyConsumption'),
            ('Humidity', 'EnergyConsumption'),
            ('PMV', 'Satisfaction'),
            ('AirQuality', 'Satisfaction'),
            ('EnergyConsumption', 'Satisfaction')
        ]
        
        self.graph.add_edges_from(edges)
        self.true_edges = set(edges)