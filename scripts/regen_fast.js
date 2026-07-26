#!/usr/bin/env node
/**
 * Fast data regeneration using in-process simulation.
 * Runs 21 days of simulated time at 15-min intervals = 2016 data points.
 * Uses correlated random walks on inputs (same as original generateSimulationData).
 */
const fs = require('fs');
const path = require('path');

const SIMS = [
    { name: 'smart_room', file: '../js/smart_room.js', className: 'SmartRoom' },
    { name: 'smart_room_noise', file: '../js/smart_room_noise.js', className: 'SmartRoom' },
    { name: 'smart_room_hidden_vars', file: '../js/smart_room_hidden_vars.js', className: 'SmartRoom' },
];

async function generateForSim(simConfig) {
    console.log(`\n=== ${simConfig.name} ===`);

    const simModule = require(simConfig.file);
    const SmartRoom = simModule[simConfig.className] || simModule.SmartRoom || simModule;

    let room;
    try {
        room = new SmartRoom();
        if (room._initialize) await room._initialize();
    } catch (e) {
        console.error(`Failed to init ${simConfig.name}: ${e.message}`);
        return;
    }

    const rows = [];
    const DAYS = 21;
    const INTERVAL_MS = 15 * 60 * 1000; // 15 minutes
    const TOTAL_STEPS = DAYS * 24 * 4;  // 96 steps/day × 21 days = 2016

    for (let i = 0; i < TOTAL_STEPS; i++) {
        // Correlated random walk on inputs (same as original)
        room.temperature += (Math.random() * 2 - 1) * 0.5;
        room.temperature = Math.max(18, Math.min(30, room.temperature));

        room.humidity += (Math.random() * 2 - 1) * 2;
        room.humidity = Math.max(30, Math.min(70, room.humidity));

        room.airQuality += (Math.random() * 2 - 1) * 10;
        room.airQuality = Math.max(0, Math.min(500, room.airQuality));

        // Compute dependent variables
        try {
            await room.updateDependentVariables();
        } catch (e) {
            // Fallback if updateDependentVariables fails
        }

        rows.push({
            Temperature: +room.temperature.toFixed(2),
            Humidity: +room.humidity.toFixed(2),
            AirQuality: +(room.airQuality || 0).toFixed(2),
            EnergyConsumption: +(room.energyConsumption || 50).toFixed(2),
            Satisfaction: +(room.overallSatisfaction || 50).toFixed(2),
        });

        if ((i + 1) % 500 === 0) console.log(`  ${i + 1}/${TOTAL_STEPS}`);
    }

    if (room.cleanup) room.cleanup();

    // Write CSV
    const cols = Object.keys(rows[0]);
    const csv = [cols.join(','), ...rows.map(r => cols.map(c => r[c]).join(','))].join('\n');
    const outPath = path.join(__dirname, '..', 'data_regen', `${simConfig.name}_processed.csv`);
    fs.writeFileSync(outPath, csv);
    console.log(`  Saved ${rows.length} rows to ${outPath}`);

    // Verify correlations
    const hum = rows.map(r => r.Humidity);
    const sat = rows.map(r => r.Satisfaction);
    const meanH = hum.reduce((a, b) => a + b) / rows.length;
    const meanS = sat.reduce((a, b) => a + b) / rows.length;
    let num = 0, denH = 0, denS = 0;
    for (let i = 0; i < rows.length; i++) {
        num += (hum[i] - meanH) * (sat[i] - meanS);
        denH += (hum[i] - meanH) ** 2;
        denS += (sat[i] - meanS) ** 2;
    }
    const corrHS = num / Math.sqrt(denH * denS);

    const temp = rows.map(r => r.Temperature);
    const meanT = temp.reduce((a, b) => a + b) / rows.length;
    let numT = 0, denT = 0;
    for (let i = 0; i < rows.length; i++) {
        numT += (temp[i] - meanT) * (sat[i] - meanS);
        denT += (temp[i] - meanT) ** 2;
    }
    const corrTS = numT / Math.sqrt(denT * denS);

    console.log(`  Humidity→Satisfaction corr: ${corrHS.toFixed(4)}`);
    console.log(`  Temperature→Satisfaction corr: ${corrTS.toFixed(4)}`);

    // Write scaling
    const scaling = cols.map(col => {
        const vals = rows.map(r => r[col]);
        const mn = Math.min(...vals);
        const mx = Math.max(...vals);
        return `${col},${mn},${mx},${mx - mn > 0 ? 1 / (mx - mn) : 1}`;
    });
    const scalingPath = outPath.replace('_processed.csv', '_processed_scaling.csv');
    fs.writeFileSync(scalingPath, 'feature,data_min,data_max,scale\n' + scaling.join('\n'));
    console.log(`  Scaling saved`);
}

async function main() {
    for (const sim of SIMS) {
        await generateForSim(sim);
    }
    console.log('\nDone!');
    process.exit(0);
}

main();
