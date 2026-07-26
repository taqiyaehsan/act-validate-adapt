#!/usr/bin/env node
/**
 * Batch simulator — runs N steps in a single process.
 * Usage: node js/batch_sim.js --sim smart_room --steps '[{...}, {...}]'
 *
 * Each step: {temperature, humidity, airQuality}
 * Returns: BATCH_RESULT:[{temperature, humidity, airQuality, energyConsumption, overallSatisfaction}, ...]
 */
const path = require('path');

const SIM_MAP = {
    'smart_room': './smart_room.js',
    'smart_room_noise': './smart_room_noise.js',
    'smart_room_hidden_vars': './smart_room_hidden_vars.js',
    'open_window': './open_window.js',
};

async function main() {
    const args = process.argv.slice(2);
    let simName = 'smart_room';
    let stepsJson = '[]';

    for (let i = 0; i < args.length; i++) {
        if (args[i] === '--sim') simName = args[++i];
        if (args[i] === '--steps') stepsJson = args[++i];
    }

    const simFile = SIM_MAP[simName];
    if (!simFile) {
        console.error(`Unknown sim: ${simName}`);
        process.exit(1);
    }

    const { SmartRoom } = require(simFile);
    const room = new SmartRoom();
    if (room._initialize) await room._initialize();

    const steps = JSON.parse(stepsJson);
    const results = [];

    for (const step of steps) {
        if (step.temperature !== undefined) room.temperature = step.temperature;
        if (step.humidity !== undefined) room.humidity = step.humidity;
        if (step.airQuality !== undefined) room.airQuality = step.airQuality;
        if (room.updateDependentVariables) await room.updateDependentVariables();

        results.push({
            temperature: +room.temperature.toFixed(2),
            humidity: +room.humidity.toFixed(2),
            airQuality: +(room.airQuality || 0).toFixed(2),
            energyConsumption: +(room.energyConsumption || 50).toFixed(2),
            overallSatisfaction: +(room.overallSatisfaction || 50).toFixed(2),
        });
    }

    if (room.cleanup) room.cleanup();
    console.log('BATCH_RESULT:' + JSON.stringify(results));
}

main().then(() => process.exit(0)).catch(e => { console.error(e); process.exit(1); });
