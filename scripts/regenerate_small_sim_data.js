#!/usr/bin/env node
/**
 * Regenerate training data for small sims with updated physics.
 * Generates 2000 data points per sim with random walks on input vars.
 */

const fs = require('fs');
const path = require('path');

const SIMS = [
    { name: 'smart_room', file: 'smart_room.js', n: 2000 },
    { name: 'smart_room_noise', file: 'smart_room_noise.js', n: 2000 },
    { name: 'smart_room_hidden_vars', file: 'smart_room_hidden_vars.js', n: 2000 },
];

async function generateData(simConfig) {
    const simPath = path.join(__dirname, '..', 'js', simConfig.file);
    const { simulateAndGetLatestData } = require(simPath);

    console.log(`\nGenerating ${simConfig.n} points for ${simConfig.name}...`);

    const data = [];

    for (let i = 0; i < simConfig.n; i++) {
        // Random walk on input variables
        const temp = 18 + Math.random() * 12;      // 18-30
        const humidity = 30 + Math.random() * 40;   // 30-70
        const aq = Math.random() * 500;             // 0-500

        try {
            const result = await simulateAndGetLatestData(
                200,  // short duration
                [
                    { variable: 'Temperature', action: 'set', value: temp },
                    { variable: 'Humidity', action: 'set', value: humidity },
                    { variable: 'AirQuality', action: 'set', value: aq },
                ],
                10  // intervention at 10ms
            );

            if (result && result.postInterventionData) {
                const d = result.postInterventionData;
                data.push({
                    Temperature: d.Temperature || d.temperature || temp,
                    Humidity: d.Humidity || d.humidity || humidity,
                    AirQuality: d.AirQuality || d.airQuality || aq,
                    EnergyConsumption: d.EnergyConsumption || d.energyConsumption || 50,
                    Satisfaction: d.OverallSatisfaction || d.overallSatisfaction || d.Satisfaction || 50,
                });
            }
        } catch (e) {
            // Fallback: just record the inputs
            data.push({
                Temperature: temp,
                Humidity: humidity,
                AirQuality: aq,
                EnergyConsumption: 50,
                Satisfaction: 50,
            });
        }

        if ((i + 1) % 200 === 0) {
            console.log(`  ${i + 1}/${simConfig.n}`);
        }
    }

    // Add OutdoorTemperature for hidden_vars
    if (simConfig.name === 'smart_room_hidden_vars') {
        for (const row of data) {
            row.OutdoorTemperature = 15 + Math.random() * 15; // 15-30
        }
    }

    // Write CSV
    const cols = Object.keys(data[0]);
    const csv = [cols.join(','), ...data.map(r => cols.map(c => r[c]).join(','))].join('\n');
    const outPath = path.join(__dirname, '..', 'data_regen',
        simConfig.name === 'smart_room_hidden_vars'
            ? 'hidden_vars_processed.csv'
            : `${simConfig.name}_processed.csv`);
    fs.writeFileSync(outPath, csv);
    console.log(`  Saved ${data.length} rows to ${outPath}`);

    // Write scaling params (min-max)
    const scaling = cols.map(col => {
        const vals = data.map(r => r[col]);
        const mn = Math.min(...vals);
        const mx = Math.max(...vals);
        return `${col},${mn},${mx},${mx - mn > 0 ? 1 / (mx - mn) : 1}`;
    });
    const scalingPath = outPath.replace('_processed.csv', '_processed_scaling.csv');
    fs.writeFileSync(scalingPath, 'feature,data_min,data_max,scale\n' + scaling.join('\n'));
    console.log(`  Saved scaling to ${scalingPath}`);
}

async function main() {
    for (const sim of SIMS) {
        await generateData(sim);
    }
    console.log('\nDone!');
    process.exit(0);
}

main().catch(e => { console.error(e); process.exit(1); });
