import numpy as np
from src.ground_truth import GroundTruthDAG

def test_shd_calculation():
    # Create the ground truth DAG
    gt = GroundTruthDAG()
    
    # Define test cases with expected SHD values
    test_cases = [
        # final DAG
        ({('Temperature', 'EnergyConsumption'),
          ('Humidity', 'EnergyConsumption'),
          ('AirQuality', 'EnergyConsumption')}, 3),
        
        # Perfect match (0 SHD)
        (gt.edges, 0),
        
        # SAM-like case: 2 wrong edges, missing all correct ones
        ({('humidity', 'airquality'), ('humidity', 'temperature'), ('energyconsumption', 'overallsatisfaction'), ('overallsatisfaction', 'temperature'), ('temperature', 'airquality'), ('energyconsumption', 'humidity')}, 12),
        
        # PC-like case: Has correct edges but many extra ones
        ({('humidity', 'airquality'), ('temperature', 'overallsatisfaction'), ('energyconsumption', 'overallsatisfaction'), ('temperature', 'airquality'), ('temperature', 'humidity')}, 9),
        
        # ({
        #     ('Humidity', 'Airquality'), 
        #     ('Temperature', 'OverallSatisfaction'), 
        #     ('EnergyConsumption', 'OverallSatisfaction'), 
        #     ('Temperature', 'AirQuality'), 
        #     ('Temperature', 'Humidity')
        # }, 6),
        
        # LLM
        ({
            ('humidity', 'airquality'), ('humidity', 'energyconsumption'), ('temperature', 'overallsatisfaction'), ('airquality', 'overallsatisfaction'), ('temperature', 'airquality'), ('temperature', 'humidity'), ('temperature', 'energyconsumption')
        }, 5)
    ]
    
    # Test each case
    for i, (test_edges, expected_shd) in enumerate(test_cases):
        calculated_shd = gt.get_shd(test_edges)
        print(f"Test {i+1}:")
        print(f"  Edges: {test_edges}")
        print(f"  Expected SHD: {expected_shd}")
        print(f"  Calculated SHD: {calculated_shd}")
        print(f"  {'PASS' if calculated_shd == expected_shd else 'FAIL'}")
        print()

if __name__ == "__main__":
    test_shd_calculation()