// Test script to debug the smart building simulation
const { simulateAndGetLatestData } = require('../smart_building.js');

async function debugSimulation() {
    console.log("=== Debugging Smart Building Simulation ===");
    
    // Test 1: Basic function call
    console.log("\n1. Testing basic function availability:");
    console.log("simulateAndGetLatestData exists:", typeof simulateAndGetLatestData === 'function');
    
    // Test 2: Simple intervention
    console.log("\n2. Testing simple temperature intervention:");
    try {
        const interventions = [{"variable": "Temperature", "action": "set", "value": 25}];
        console.log("Calling with interventions:", JSON.stringify(interventions));
        
        const result = await simulateAndGetLatestData(2000, interventions, 100);
        console.log("Result type:", typeof result);
        console.log("Result structure:", Object.keys(result || {}));
        
        if (result) {
            console.log("Has preInterventionData:", !!result.preInterventionData);
            console.log("Has postInterventionData:", !!result.postInterventionData);
            
            if (result.preInterventionData) {
                console.log("Pre-intervention keys:", Object.keys(result.preInterventionData));
                console.log("Pre-intervention Temperature:", result.preInterventionData.Temperature);
            }
            
            if (result.postInterventionData) {
                console.log("Post-intervention keys:", Object.keys(result.postInterventionData));
                console.log("Post-intervention Temperature:", result.postInterventionData.Temperature);
            }
        } else {
            console.log("Result is null or undefined");
        }
        
    } catch (error) {
        console.error("Error during simulation:", error.message);
        console.error("Stack trace:", error.stack);
    }
    
    // Test 3: Check if building initialization works
    console.log("\n3. Testing building initialization:");
    try {
        const { SmartBuilding } = require('../smart_building.js');
        const building = new SmartBuilding();
        
        console.log("Building created");
        await building.initPromise;
        console.log("Building initialized");
        
        const state = building.getAggregatedState();
        console.log("Initial state keys:", Object.keys(state));
        console.log("Initial Temperature:", state.Temperature);
        console.log("Initial Humidity:", state.Humidity);
        
        building.cleanup();
        console.log("Building cleaned up");
        
    } catch (error) {
        console.error("Error during building test:", error.message);
    }
    
    console.log("\n=== Debug Complete ===");
}

// Run the debug
debugSimulation().catch(console.error);