const { SmartRoom } = require('../smart_room.js');

async function test() {
    const room = new SmartRoom();
    
    // Test baseline
    await room.updateDependentVariables();
    console.log('Baseline State:', {
        temperature: room.temperature,
        humidity: room.humidity,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption
    });
    
    // Test with modifications
    room.temperature = 30;
    room.humidity = 100;
    room.airQuality = 500;
    
    await room.updateDependentVariables();
    console.log('Modified State:', {
        temperature: room.temperature,
        humidity: room.humidity,
        satisfaction: room.overallSatisfaction,
        energy: room.energyConsumption
    });
    
    room.cleanup();
}

test().catch(console.error);