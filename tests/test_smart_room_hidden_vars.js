const { SmartRoom } = require('../smart_room_hidden_vars.js');

async function test() {
    const room = new SmartRoom();
    
    // Wait for initialization to complete
    await room.initPromise;
    
    // Test baseline
    await room.updateDependentVariables();
    console.log('Baseline State:', {
        temperature: room.temperature,
        humidity: room.humidity,
        airQuality: room.airQuality,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption
    });
    
    // Test hidden variables access
    console.log('\n--- CHECKING HIDDEN VARIABLES ---');
    console.log('HVAC Efficiency:', room.getHvacEfficiency());
    console.log('Insulation Quality:', room.getInsulationQuality());
    console.log('Window State:', room.getWindowState());
    console.log('Occupancy Level:', room.getOccupancyLevel());
    console.log('Outdoor Temperature:', room.getOutdoorTemperature());
    
    // Test intervention
    console.log('\n--- TESTING INTERVENTION ---');
    const interventionResult = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 25 },
        { variable: "Humidity", action: "set", value: 45 },
        { variable: "AirQuality", action: "set", value: 150 }
    ]);
    
    console.log('\nIntervention Result:');
    console.log('Pre-intervention:', interventionResult.preInterventionData);
    console.log('Post-intervention:', interventionResult.postInterventionData);
    
    // Test multiple dependent variable updates
    console.log('\n--- TESTING MULTIPLE UPDATES ---');
    for (let i = 0; i < 5; i++) {
        // Slightly alter temperature and humidity for each test
        room.temperature += (Math.random() * 2 - 1);
        room.humidity += (Math.random() * 4 - 2);
        
        await room.updateDependentVariables();
        console.log(`\nUpdate ${i+1}:`);
        console.log(`Temperature: ${room.temperature.toFixed(2)}`);
        console.log(`Humidity: ${room.humidity.toFixed(2)}`);
        console.log(`Energy Consumption: ${room.energyConsumption.toFixed(2)}`);
        console.log(`Overall Satisfaction: ${room.overallSatisfaction.toFixed(2)}`);
    }
    
    // Test data logging
    console.log('\n--- TESTING DATA LOGGING ---');
    room.logData();
    console.log('Data Log Count:', room.dataLog.length);
    console.log('Latest Data Entry:', room.getLatestData());
    
    room.cleanup();
}

test().catch(console.error);