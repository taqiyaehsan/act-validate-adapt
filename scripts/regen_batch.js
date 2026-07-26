#!/usr/bin/env node
/**
 * Batch regenerate training data for small sims.
 * Runs 2000 iterations in-process (no subprocess per row).
 */
const fs = require('fs');
const path = require('path');

// Load the SmartRoom class from each sim
const simFiles = [
    { name: 'smart_room', file: '../js/smart_room.js' },
    { name: 'smart_room_noise', file: '../js/smart_room_noise.js' },
    { name: 'smart_room_hidden_vars', file: '../js/smart_room_hidden_vars.js' },
];

async function generateForSim(simConfig) {
    console.log(`\n=== ${simConfig.name} ===`);

    // Import the sim module
    const simModule = require(simConfig.file);
    const SmartRoom = simModule.SmartRoom || simModule.SmartRoomNoise || simModule.SmartRoomHiddenVars || simModule;

    // Check if SmartRoom is a constructor
    let room;
    if (typeof SmartRoom === 'function') {
        room = new SmartRoom();
    } else {
        console.error(`Cannot find SmartRoom constructor in ${simConfig.file}`);
        console.log('Exports:', Object.keys(simModule));
        return;
    }

    // Initialize
    if (room._initialize) await room._initialize();

    const N = 2000;
    const rows = [];

    for (let i = 0; i < N; i++) {
        // Random input state
        room.temperature = 18 + Math.random() * 12;
        room.humidity = 30 + Math.random() * 40;
        room.airQuality = Math.random() * 500;

        // Update dependent variables
        if (room.updateDependentVariables) {
            await room.updateDependentVariables();
        }

        rows.push({
            Temperature: +room.temperature.toFixed(2),
            Humidity: +room.humidity.toFixed(2),
            AirQuality: +(room.airQuality).toFixed(2),
            EnergyConsumption: +(room.energyConsumption || 50).toFixed(2),
            Satisfaction: +(room.overallSatisfaction || 50).toFixed(2),
        });

        if ((i + 1) % 500 === 0) console.log(`  ${i + 1}/${N}`);
    }

    // Clean up
    if (room.cleanup) room.cleanup();

    // Write CSV
    const cols = Object.keys(rows[0]);
    const csv = [cols.join(','), ...rows.map(r => cols.map(c => r[c]).join(','))].join('\n');
    const outDir = path.join(__dirname, '..', 'data_regen');
    const outPath = path.join(outDir, `${simConfig.name}_processed.csv`);
    fs.writeFileSync(outPath, csv);
    console.log(`  Saved ${rows.length} rows to ${outPath}`);

    // Verify humidity→satisfaction correlation
    const hum = rows.map(r => r.Humidity);
    const sat = rows.map(r => r.Satisfaction);
    const meanH = hum.reduce((a,b) => a+b) / N;
    const meanS = sat.reduce((a,b) => a+b) / N;
    let num = 0, denH = 0, denS = 0;
    for (let i = 0; i < N; i++) {
        num += (hum[i] - meanH) * (sat[i] - meanS);
        denH += (hum[i] - meanH) ** 2;
        denS += (sat[i] - meanS) ** 2;
    }
    const corr = num / Math.sqrt(denH * denS);
    console.log(`  Humidity↔Satisfaction correlation: ${corr.toFixed(4)}`);
}

async function main() {
    for (const sim of simFiles) {
        try {
            await generateForSim(sim);
        } catch (e) {
            console.error(`Error for ${sim.name}: ${e.message}`);
        }
    }
    console.log('\nDone!');
    process.exit(0);
}

main();
