const { SmartRoom } = require('../smart_room_noise.js');

async function test() {
    const room = new SmartRoom();
    
    // Test baseline without noise
    room.setNoiseConfiguration(false);
    await room.updateDependentVariables();
    console.log('Baseline State (No Noise):', {
        temperature: room.temperature,
        humidity: room.humidity,
        airQuality: room.airQuality,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption
    });
    
    // Enable noise with default configuration
    console.log('\n--- ENABLING NOISE (DEFAULT CONFIG) ---');
    room.setNoiseConfiguration(true);
    
    // Test multiple readings with noise to see variation
    for (let i = 0; i < 5; i++) {
        console.log(`\nReading ${i+1} with noise:`);
        console.log(`Temperature: ${room._applyMeasurementNoise(room.temperature, 'temperature')}`);
        console.log(`Humidity: ${room._applyMeasurementNoise(room.humidity, 'humidity')}`);
        console.log(`Air Quality: ${room._applyMeasurementNoise(room.airQuality, 'airQuality')}`);
    }
    
    // Test intervention with noise
    console.log('\n--- TESTING INTERVENTION WITH NOISE ---');
    const interventionResult = await room.applyInterventions([
        { variable: "Temperature", action: "set", value: 35 },
        { variable: "Humidity", action: "set", value: 70 },
        { variable: "AirQuality", action: "set", value: 200 }
    ]);
    
    console.log('\nIntervention Result:');
    console.log('Pre-intervention:', interventionResult.preInterventionData);
    console.log('Post-intervention:', interventionResult.postInterventionData);
    
    // Test with modified config
    console.log('\n--- TESTING WITH MODIFIED NOISE CONFIG ---');
    room.setNoiseConfiguration(true, {
        temperature: { mean: 0, stdDev: 0.5, type: 'gaussian' },  // More temperature noise
        humidity: { mean: 0, stdDev: 4.0, type: 'skewed', skew: 0.5 } // More humidity noise
    });
    
    // Test multiple readings with increased noise
    for (let i = 0; i < 5; i++) {
        console.log(`\nReading ${i+1} with increased noise:`);
        console.log(`Temperature: ${room._applyMeasurementNoise(room.temperature, 'temperature')}`);
        console.log(`Humidity: ${room._applyMeasurementNoise(room.humidity, 'humidity')}`);
        console.log(`Air Quality: ${room._applyMeasurementNoise(room.airQuality, 'airQuality')}`);
    }
    
    room.cleanup();
}

test().catch(console.error);