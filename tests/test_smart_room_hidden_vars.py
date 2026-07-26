import os
import json
import subprocess

def run_smart_room_test():
    print("=== Testing Smart Room with Hidden Variables ===\n")

    test_script_path = "hidden_vars_test.js"

    with open(test_script_path, "w") as f:
        f.write("""
const { SmartRoom } = require('./smart_room_hidden_vars.js');

async function test() {
    const room = new SmartRoom();
    
    // Wait for initialization
    await room.initPromise;
    
    console.log('--- TEST 1: Initial State ---');
    await room.updateDependentVariables();
    console.log(JSON.stringify({
        type: "baseline",
        temperature: room.temperature,
        humidity: room.humidity,
        airQuality: room.airQuality,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption,
        hidden: {
            hvacEfficiency: room.getHvacEfficiency(),
            windowState: room.getWindowState(),
            occupancyLevel: room.getOccupancyLevel(),
            outdoorTemp: room.getOutdoorTemperature(),
            outdoorHumidity: room.getOutdoorHumidity(),
            timeOfDay: room.getTimeOfDay()
        }
    }));
    
    console.log('--- TEST 2: First Intervention ---');
    const result1 = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 25 },
        { variable: "Humidity", action: "set", value: 60 },
        { variable: "AirQuality", action: "set", value: 150 }
    ]);
    console.log(JSON.stringify({
        type: "intervention_1",
        pre: result1.preInterventionData,
        post: result1.postInterventionData
    }));
    
    console.log('--- TEST 3: Second Intervention ---');
    const result2 = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 28 },
        { variable: "Humidity", action: "set", value: 45 },
        { variable: "AirQuality", action: "set", value: 250 }
    ]);
    console.log(JSON.stringify({
        type: "intervention_2",
        pre: result2.preInterventionData,
        post: result2.postInterventionData
    }));

    // Generate a short simulation dataset
    console.log('--- TEST 4: Short Simulation ---');
    for (let i = 0; i < 3; i++) {
        // Update hidden variables to see their effect
        room._updateHiddenVariables();
        await room.updateDependentVariables();
        room.logData(false);
        
        console.log(JSON.stringify({
            type: "simulation_step",
            step: i,
            temperature: room.temperature,
            humidity: room.humidity,
            airQuality: room.airQuality,
            satisfaction: room.overallSatisfaction,
            energy: room.energyConsumption,
            hidden: {
                hvacEfficiency: room.getHvacEfficiency(),
                windowState: room.getWindowState(),
                occupancyLevel: room.getOccupancyLevel(),
                outdoorTemp: room.getOutdoorTemperature()
            }
        }));
    }

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
                print(json.dumps(data, indent=2))
            except json.JSONDecodeError:
                print(f"Could not parse JSON: {line}")

    if not results:
        print("No valid results received")
        return

    print("\n=== Analysis ===")
    
    # Analyze intervention results
    for result in results:
        if result['type'] in ('intervention_1', 'intervention_2'):
            pre = result['pre']
            post = result['post']
            print(f"\nAnalysis for {result['type']}:")
            print(f"  Set Temp: {post['Temperature']}, Humidity: {post['Humidity']}, AQ: {post['AirQuality']}")
            print(f"  Energy: {post['EnergyConsumption']}, Satisfaction: {post['OverallSatisfaction']}")
            print(f"  ΔEnergy: {post['EnergyConsumption'] - pre['EnergyConsumption']:.2f}, ΔSatisfaction: {float(post['OverallSatisfaction']) - float(pre['OverallSatisfaction']):.2f}")
    
    # Analyze simulation steps if any
    sim_steps = [r for r in results if r['type'] == 'simulation_step']
    if sim_steps:
        print("\nSimulation steps summary:")
        for step in sim_steps:
            print(f"  Step {step['step']}: T={step['temperature']:.1f}°C, H={step['humidity']:.1f}%, AQ={step['airQuality']}")
            print(f"    Window: {step['hidden']['windowState']}, Occupancy: {step['hidden']['occupancyLevel']}")
            print(f"    Satisfaction: {step['satisfaction']:.1f}%, Energy: {step['energy']:.1f}%")

    print("=== End of Testing ===")

if __name__ == "__main__":
    run_smart_room_test()