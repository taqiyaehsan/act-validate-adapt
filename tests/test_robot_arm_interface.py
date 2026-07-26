import os
import sys
import json
import tempfile
import subprocess
import logging
import time
import numpy as np
import pandas as pd
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class RobotArmCausalTest:
    def __init__(self, robot_arm_path):
        self.robot_arm_path = robot_arm_path
        
        # Define the causal edges we expect to find
        self.causal_edges = [
            ('handX', 'collision'),
            ('handY', 'collision'),
            ('ballX', 'collision'),
            ('ballY', 'collision')
        ]
        
        # Track test results
        self.edge_test_results = {}
        
        # Set collision threshold to match the Matter.js simulation
        self.collision_threshold = 35
    
    def run_all_tests(self):
        """Run all causal relationship tests"""
        logger.info("\n=== Testing Robot Arm Causal Relationships with Matter.js Physics ===")
        
        # Test basic simulation functionality first
        if not self.test_basic_simulation():
            logger.error("❌ Basic simulation test failed. Cannot proceed with causal tests.")
            return False
        
        # Test each causal edge
        all_passed = True
        for source, target in self.causal_edges:
            result = self.test_causal_edge(source, target)
            self.edge_test_results[(source, target)] = result
            all_passed = all_passed and result
        
        # Print summary
        logger.info("\n=== Causal Testing Summary ===")
        for edge, result in self.edge_test_results.items():
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"Edge {edge[0]} → {edge[1]}: {status}")
        
        return all_passed
    
    def test_basic_simulation(self):
        """Test if the Matter.js physics simulation works correctly"""
        logger.info("\n=== Testing Basic Matter.js Physics Simulation ===")
        
        # Simple intervention to verify the simulation works
        intervention = [{
            "variable": "handX",
            "action": "set",
            "value": 230
        }]
        
        result = self._run_simulation(intervention)
        if not result:
            logger.error("❌ Basic Matter.js simulation test failed")
            return False
        
        # Check if intervention was applied correctly
        handX = float(result['postInterventionData']['handX'])
        if abs(handX - 230) < 5:
            logger.info("✅ Basic Matter.js simulation test passed")
            logger.info(f"Post-intervention state: {result['postInterventionData']}")
            return True
        else:
            logger.error(f"❌ Intervention failed to apply. Expected handX≈230, got {handX}")
            return False
    
    def test_causal_edge(self, source, target):
        """Test a specific causal relationship between source and target variables"""
        logger.info(f"\n=== Testing Causal Edge with Matter.js: {source} → {target} ===")
        
        # Run multiple trials since physics can be probabilistic
        num_trials = 3
        no_collision_successes = 0
        collision_successes = 0
        
        # Test 1: Multiple trials with values that should NOT cause a collision
        logger.info(f"\nTest 1: Setting {source} to values that should NOT cause collision")
        for i in range(num_trials):
            no_collision_values = self._get_no_collision_values(source)
            no_collision_result = self._run_simulation(no_collision_values)
            
            if not no_collision_result:
                logger.error(f"❌ No-collision trial {i+1} failed for {source} → {target}")
                continue
            
            # Get collision state and distance
            actual_no_collision = no_collision_result['postInterventionData']['collision']
            distance = self._calculate_distance(
                no_collision_result['postInterventionData']['handX'],
                no_collision_result['postInterventionData']['handY'],
                no_collision_result['postInterventionData']['ballX'],
                no_collision_result['postInterventionData']['ballY']
            )
            
            logger.info(f"Trial {i+1}: Distance = {distance:.2f}, Collision = {actual_no_collision}")
            
            # It should NOT be a collision (distance > threshold)
            if actual_no_collision is False and distance > self.collision_threshold:
                no_collision_successes += 1
        
        # Test 2: Multiple trials with values that SHOULD cause a collision
        logger.info(f"\nTest 2: Setting {source} to values that SHOULD cause collision")
        for i in range(num_trials):
            collision_values = self._get_collision_values(source)
            collision_result = self._run_simulation(collision_values)
            
            if not collision_result:
                logger.error(f"❌ Collision trial {i+1} failed for {source} → {target}")
                continue
            
            # Get collision state and distance
            actual_collision = collision_result['postInterventionData']['collision']
            distance = self._calculate_distance(
                collision_result['postInterventionData']['handX'],
                collision_result['postInterventionData']['handY'],
                collision_result['postInterventionData']['ballX'],
                collision_result['postInterventionData']['ballY']
            )
            
            logger.info(f"Trial {i+1}: Distance = {distance:.2f}, Collision = {actual_collision}")
            
            # For Matter.js physics, verify the relationship based on actual distance
            logger.info(f"Matter.js physics check: Distance({distance:.2f}) < Threshold({self.collision_threshold}): {distance < self.collision_threshold}")
            
            # It should be a collision (distance < threshold)
            if actual_collision is True and distance < self.collision_threshold:
                collision_successes += 1
        
        # A causal relationship exists if we can reliably control the target by manipulating the source
        no_collision_success_rate = no_collision_successes / num_trials
        collision_success_rate = collision_successes / num_trials
        
        logger.info(f"No-collision success rate with Matter.js: {no_collision_success_rate:.2f}")
        logger.info(f"Collision success rate with Matter.js: {collision_success_rate:.2f}")
        
        # Consider the test passed if at least 2/3 of trials behaved as expected
        if no_collision_success_rate >= 0.66 and collision_success_rate >= 0.66:
            logger.info(f"✅ Causal relationship confirmed with Matter.js physics: {source} → {target}")
            return True
        else:
            logger.error(f"❌ Causal relationship NOT consistently confirmed with Matter.js: {source} → {target}")
            return False
    
    def _get_no_collision_values(self, source):
        """Get intervention values that should NOT cause a collision with Matter.js physics"""
        # Make sure positions are well beyond the collision threshold
        safe_distance = self.collision_threshold * 2
        
        # First, set ball to a fixed position
        interventions = [
            {"variable": "ballX", "action": "set", "value": 100},
            {"variable": "ballY", "action": "set", "value": 100}
        ]
        
        # Then set hand position far from ball
        if source in ['handX', 'handY']:
            interventions.append({"variable": "handX", "action": "set", "value": 100 + safe_distance})
            interventions.append({"variable": "handY", "action": "set", "value": 100 + safe_distance})
        else:
            # If testing ball variables, fix hand position and move ball away
            interventions = [
                {"variable": "handX", "action": "set", "value": 200},
                {"variable": "handY", "action": "set", "value": 200},
                {"variable": "ballX", "action": "set", "value": 200 + safe_distance},
                {"variable": "ballY", "action": "set", "value": 200 + safe_distance}
            ]
        
        return interventions
    
    def _get_collision_values(self, source):
        """Get intervention values that SHOULD cause collision with Matter.js physics"""
        if source in ['handX', 'handY']:
            # Testing hand variables - current approach works well
            interventions = [
                {"variable": "ballX", "action": "set", "value": 150},
                {"variable": "ballY", "action": "set", "value": 150},
                {"variable": "handX", "action": "set", "value": 150},
                {"variable": "handY", "action": "set", "value": 150}
            ]
        else:
            # When testing ball variables, set exact same position as hand to ensure collision
            interventions = [
                {"variable": "handX", "action": "set", "value": 175},
                {"variable": "handY", "action": "set", "value": 175},
                {"variable": "ballX", "action": "set", "value": 175},
                {"variable": "ballY", "action": "set", "value": 175}
            ]
        
        return interventions
    
    def _calculate_distance(self, hand_x, hand_y, ball_x, ball_y):
        """Calculate Euclidean distance between hand and ball centers"""
        return np.sqrt((float(hand_x) - float(ball_x))**2 + (float(hand_y) - float(ball_y))**2)
    
    def _run_simulation(self, interventions):
        """Run a Matter.js simulation with the given interventions"""
        temp_js_path = None
        
        try:
            # Create a temporary JavaScript file to run the simulation
            with tempfile.NamedTemporaryFile(mode='w', suffix='.js', delete=False) as temp_js:
                temp_js_path = temp_js.name
                temp_js.write(f'''
                const {{ simulateAndGetLatestData }} = require('{self.robot_arm_path}');
                
                async function runSimulation() {{
                    try {{
                        // Let Matter.js physics engine handle the simulation
                        const result = await simulateAndGetLatestData(
                            3000,  // Duration in ms (shorter since we're using Matter.js)
                            {json.dumps(interventions)},
                            100    // Apply intervention earlier
                        );
                        
                        if (!result || !result.preInterventionData || !result.postInterventionData) {{
                            throw new Error('Invalid Matter.js simulation result');
                        }}
                        
                        // Calculate and add the distance between hand and ball (for debugging)
                        const pre = result.preInterventionData;
                        const post = result.postInterventionData;
                        const preDistance = Math.sqrt(
                            (pre.handX - pre.ballX) ** 2 + 
                            (pre.handY - pre.ballY) ** 2
                        );
                        const postDistance = Math.sqrt(
                            (post.handX - post.ballX) ** 2 + 
                            (post.handY - post.ballY) ** 2
                        );
                        
                        result.preInterventionData.distance = preDistance;
                        result.postInterventionData.distance = postDistance;
                        
                        console.log("SIMULATION_RESULT:" + JSON.stringify(result));
                        process.exit(0);
                    }} catch (error) {{
                        console.error('MATTER_JS_SIMULATION_ERROR:', error.message);
                        process.exit(1);
                    }}
                }}
                
                // Add error handling for unhandled rejections
                process.on('unhandledRejection', (error) => {{
                    console.error('Unhandled promise rejection in Matter.js:', error);
                    process.exit(1);
                }});
                
                runSimulation();
                ''')
            
            # Run the simulation with timeout
            process = subprocess.Popen(
                ['node', '--max-old-space-size=512', temp_js_path],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            
            # Set timeout for Matter.js simulation
            stdout, stderr = process.communicate(timeout=30)
            
            # Check for errors
            if stderr:
                logger.debug(f"Matter.js stderr: {stderr}")
            
            # Parse the result from the output
            result = None
            for line in stdout.splitlines():
                if 'SIMULATION_RESULT:' in line:
                    try:
                        result_json = line.split('SIMULATION_RESULT:', 1)[1]
                        result = json.loads(result_json)
                        break
                    except json.JSONDecodeError:
                        logger.error(f"Failed to parse JSON from Matter.js output: {line}")
            
            if result:
                return result
            else:
                logger.error("No Matter.js simulation result found in output")
                logger.debug(f"Matter.js output: {stdout}")
                return None
        except subprocess.TimeoutExpired:
            logger.error("Matter.js simulation timed out")
            if process:
                process.kill()
            return None
        except Exception as e:
            logger.error(f"Error running Matter.js simulation: {str(e)}")
            return None
        finally:
            # Clean up the temporary file
            if temp_js_path and os.path.exists(temp_js_path):
                try:
                    os.remove(temp_js_path)
                except:
                    pass
    
    def generate_edge_dataset(self):
        """Generate a dataset that demonstrates all causal relationships using Matter.js physics"""
        logger.info("\n=== Generating Causal Edge Dataset with Matter.js Physics ===")
        
        # Number of samples per edge
        samples_per_edge = 20
        
        # Create a list to hold all data points
        all_data = []
        
        # For each causal edge, generate samples that demonstrate the relationship
        for source, target in self.causal_edges:
            logger.info(f"Generating Matter.js data for {source} → {target}")
            
            # Generate samples with varying distances to show relationship
            for i in range(samples_per_edge):
                # Gradually vary the distance from definitely colliding to definitely not
                distance_factor = i / float(samples_per_edge - 1)  # 0.0 to 1.0
                
                # Create interventions with varying distances
                interventions = self._create_variable_distance_intervention(
                    source, 
                    distance_factor * (self.collision_threshold * 3)  # 0 to 3x threshold
                )
                
                # Run the Matter.js simulation
                result = self._run_simulation(interventions)
                
                if result:
                    # Extract the post-intervention state
                    state = result['postInterventionData']
                    
                    # Calculate distance for analysis
                    distance = self._calculate_distance(
                        float(state['handX']), 
                        float(state['handY']),
                        float(state['ballX']), 
                        float(state['ballY'])
                    )
                    
                    # Add distance to the state
                    state['distance'] = distance
                    state['distance_factor'] = distance_factor
                    state['edge_tested'] = f"{source} → {target}"
                    state['expected_collision'] = distance < self.collision_threshold
                    
                    # Add to our dataset
                    all_data.append(state)
                
                # Small delay to let Matter.js engine reset
                time.sleep(0.2)
        
        # Convert to DataFrame
        if all_data:
            df = pd.DataFrame(all_data)
            
            # Save the dataset
            output_file = 'robot_arm_causal_dataset_matter_js.csv'
            df.to_csv(output_file, index=False)
            logger.info(f"✅ Matter.js dataset saved to {output_file}")
            logger.info(f"  - {len(df)} samples total")
            logger.info(f"  - {df['collision'].sum()} collision samples")
            logger.info(f"  - {len(df) - df['collision'].sum()} non-collision samples")
            
            # Show distribution of samples per edge
            edge_counts = df['edge_tested'].value_counts()
            logger.info("Sample distribution by edge:")
            for edge, count in edge_counts.items():
                logger.info(f"  - {edge}: {count} samples")
            
            # Create a distance vs collision plot
            self._analyze_distance_collision_relationship(df)
            
            return df
        else:
            logger.error("❌ Failed to generate Matter.js dataset")
            return None
    
    def _analyze_distance_collision_relationship(self, df):
        """Analyze the relationship between distance and collision in the Matter.js data"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Create a scatter plot of distance vs collision
            plt.figure(figsize=(10, 6))
            sns.scatterplot(data=df, x='distance', y='collision', hue='edge_tested', alpha=0.7)
            plt.axvline(x=self.collision_threshold, color='red', linestyle='--', 
                       label=f'Collision Threshold ({self.collision_threshold})')
            plt.title('Matter.js Physics: Distance vs Collision')
            plt.xlabel('Distance between Hand and Ball')
            plt.ylabel('Collision Detected (1=Yes, 0=No)')
            plt.legend(title='Causal Edge')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            # Save the plot
            plt.savefig('matter_js_distance_collision_relationship.png')
            logger.info("✅ Created distance vs collision analysis plot")
            
        except ImportError:
            logger.warning("Matplotlib/Seaborn not available - skipping plot generation")
    
    def _create_variable_distance_intervention(self, source, distance):
        """Create a test intervention with a specific distance between hand and ball for Matter.js"""
        # Base positions (matching the physics world coordinates used in Matter.js)
        center_x, center_y = 150, 150
        
        if source in ['handX', 'handY']:
            # Fix ball position
            ball_x, ball_y = center_x, center_y
            
            # Set hand position at specified distance (with some randomization)
            angle = np.random.uniform(0, 2 * np.pi)
            hand_x = center_x + distance * np.cos(angle)
            hand_y = center_y + distance * np.sin(angle)
        else:
            # Fix hand position
            hand_x, hand_y = center_x, center_y
            
            # Set ball position at specified distance
            angle = np.random.uniform(0, 2 * np.pi)
            ball_x = center_x + distance * np.cos(angle)
            ball_y = center_y + distance * np.sin(angle)
        
        # Build the interventions for Matter.js
        interventions = [
            {"variable": "handX", "action": "set", "value": hand_x},
            {"variable": "handY", "action": "set", "value": hand_y},
            {"variable": "ballX", "action": "set", "value": ball_x},
            {"variable": "ballY", "action": "set", "value": ball_y}
        ]
        
        return interventions

def main():
    # Get robot_arm.js path from command line or use default
    if len(sys.argv) > 1:
        robot_arm_path = sys.argv[1]
    else:
        # Default path - update to match your actual location
        script_dir = os.path.dirname(os.path.abspath(__file__))
        robot_arm_path = "./robot_arm.js"
    
    if not os.path.exists(robot_arm_path):
        logger.error(f"❌ Robot arm script not found at: {robot_arm_path}")
        sys.exit(1)
    
    logger.info(f"Using Matter.js robot_arm.js at: {robot_arm_path}")
    
    # Create tester
    tester = RobotArmCausalTest(robot_arm_path)
    
    # Run all tests
    all_passed = tester.run_all_tests()
    
    # Generate a dataset if tests passed
    if all_passed:
        tester.generate_edge_dataset()
        logger.info("\n🎉 All Matter.js causal relationship tests passed! Your causal discovery framework should work correctly.")
    else:
        logger.error("\n❌ Some Matter.js causal tests failed. Fix the issues before proceeding with causal discovery.")
    
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()