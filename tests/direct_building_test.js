const { SmartBuilding } = require('../smart_building.js');

async function test() {
    const building = new SmartBuilding();
    
    // Wait for initialization to complete
    await building.initPromise;
    
    // Test baseline
    await building.update();
    const baselineState = building.getAggregatedState();
    console.log('Baseline State:', {
        temperature: baselineState.Temperature,
        humidity: baselineState.Humidity,
        airQuality: baselineState.AirQuality,
        satisfaction: baselineState.OverallSatisfaction,
        energy: baselineState.EnergyConsumption,
        hvacSetpoint: baselineState.HVACSetpoint,
        lightingLevel: baselineState.LightingLevel,
        occupants: baselineState.OccupantCount
    });
    
    // Test zone-level access
    console.log('\n--- CHECKING ZONE VARIABLES ---');
    building.zones.forEach((zone, zoneId) => {
        const state = zone.getState();
        console.log(`${zoneId}:`, {
            temp: state.Temperature,
            hvacSetpoint: state.HVACSetpoint,
            occupants: state.OccupantCount,
            satisfaction: state.OverallSatisfaction
        });
    });
    
    // Test intervention
    console.log('\n--- TESTING INTERVENTION ---');
    const interventionResult = await building.applyInterventions([
        { variable: "Temperature", action: "set", value: 25, zoneId: "all" },
        { variable: "Humidity", action: "set", value: 45, zoneId: "all" },
        { variable: "HVACSetpoint", action: "set", value: 24, zoneId: "Office_1" }
    ]);
    
    console.log('\nIntervention Result:', interventionResult);
    // console.log('Pre-intervention:', interventionResult.preState);
    // console.log('Post-intervention:', interventionResult.postState);
    
    // Test multiple dependent variable updates
    console.log('\n--- TESTING MULTIPLE UPDATES ---');
    for (let i = 0; i < 5; i++) {
        // Slightly alter conditions for each test
        building.zones.forEach(zone => {
            zone.temperature += (Math.random() * 2 - 1);
            zone.humidity += (Math.random() * 4 - 2);
            zone.airQuality += (Math.random() * 20 - 10);
        });
        
        await building.update();
        const state = building.getAggregatedState();
        console.log(`\nUpdate ${i+1}:`);
        console.log(`Temperature: ${state.Temperature.toFixed(2)}`);
        console.log(`Humidity: ${state.Humidity.toFixed(2)}`);
        console.log(`Energy Consumption: ${state.EnergyConsumption.toFixed(2)}`);
        console.log(`Overall Satisfaction: ${state.OverallSatisfaction.toFixed(2)}`);
        console.log(`HVAC Power: ${state.HVACPower.toFixed(2)}`);
        console.log(`Lighting Power: ${state.LightingPower.toFixed(2)}`);
    }
    
    // Test data logging
    console.log('\n--- TESTING DATA LOGGING ---');
    building.start();
    
    // Let it run for a few updates
    await new Promise(resolve => setTimeout(resolve, 3000));
    
    building.stop();
    console.log('Data Log Count:', building.dataLog.length);
    if (building.dataLog.length > 0) {
        console.log('Latest Data Entry:', building.dataLog[building.dataLog.length - 1]);
    }
    
    // Test CSV export
    console.log('\n--- TESTING CSV EXPORT ---');
    const filename = building.exportData('test_building_output.csv');
    if (filename) {
        console.log('CSV exported to:', filename);
    }
    
    building.cleanup();
}

test().catch(console.error);