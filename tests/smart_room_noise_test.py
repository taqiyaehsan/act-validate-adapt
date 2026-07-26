import os
import json
import subprocess
from src.ground_truth import GroundTruthDAG

def run_smart_room_test():
    print("=== Testing Smart Room with Measurement Noise ===\n")

    test_script_path = "noise_test.js"

    with open(test_script_path, "w") as f:
        f.write("""
const { SmartRoom } = require('./smart_room_noise.js');

async function test() {
    const room = new SmartRoom();
    
    console.log('--- TEST 1: No Noise ---');
    room.setNoiseConfiguration(false);
    await room.updateDependentVariables();
    console.log(JSON.stringify({
        type: "baseline",
        temperature: room.temperature,
        humidity: room.humidity,
        airQuality: room.airQuality,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption
    }));
    
    console.log('--- TEST 2: Default Noise ---');
    room.setNoiseConfiguration(true);
    const result1 = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 25 },
        { variable: "Humidity", action: "set", value: 60 },
        { variable: "AirQuality", action: "set", value: 150 }
    ]);
    console.log(JSON.stringify({
        type: "intervention_default_noise",
        pre: result1.preInterventionData,
        post: result1.postInterventionData
    }));
    
    console.log('--- TEST 3: High Noise ---');
    room.setNoiseConfiguration(true, {
        temperature: { mean: 0, stdDev: 1.0, type: 'gaussian' },
        humidity: { mean: 0, stdDev: 5.0, type: 'skewed', skew: 0.5 },
        airQuality: { mean: 0, stdDev: 20.0, type: 'gaussian' }
    });
    const result2 = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 28 },
        { variable: "Humidity", action: "set", value: 45 },
        { variable: "AirQuality", action: "set", value: 250 }
    ]);
    console.log(JSON.stringify({
        type: "intervention_high_noise",
        pre: result2.preInterventionData,
        post: result2.postInterventionData
    }));

    room.cleanup();
    process.exit(0);  
}

test().catch(err => {
    console.error("Test script failed:", err);
    process.exit(1);
});
        """)

    process = subprocess.Popen(
        ["node", test_script_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )

    stdout, stderr = process.communicate()

    # Clean up
    os.remove(test_script_path)

    if stderr:
        print("Script Error:\n", stderr)

    results = []
    for line in stdout.strip().splitlines():
        line = line.strip()
        if line.startswith('{'):
            try:
                data = json.loads(line)
                results.append(data)
                print(f"Received data of type: {data['type']}")
                print(data)
            except json.JSONDecodeError:
                print(f"Could not parse JSON: {line}")

    if not results:
        print("No valid results received")
        return

    print("\n=== Analysis ===")
    for result in results:
        if result['type'] in ('intervention_default_noise', 'intervention_high_noise'):
            pre = result['pre']
            post = result['post']
            print(f"\nAnalysis for {result['type']}:")
            print(f"  Set Temp: {post['Temperature']}, Humidity: {post['Humidity']}, AQ: {post['AirQuality']}")
            print(f"  Energy: {post['EnergyConsumption']}, Satisfaction: {post['OverallSatisfaction']}")
            print(f"  ΔEnergy: {post['EnergyConsumption'] - pre['EnergyConsumption']:.2f}, ΔSatisfaction: {post['OverallSatisfaction'] - pre['OverallSatisfaction']:.2f}")

            print("=== End of Testing ===")

if __name__ == "__main__":
    run_smart_room_test()