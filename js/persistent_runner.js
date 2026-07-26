#!/usr/bin/env node
/**
 * Persistent simulator runner — keeps sim alive across steps.
 * Reads JSON commands from stdin, writes RESULT: lines to stdout.
 * Same physics as --single-step, just no process respawn.
 *
 * Usage: node js/persistent_runner.js <sim_module_path>
 * stdin:  {"intervention": {"temperature": 24}, "state": {...}}  (one JSON per line)
 * stdout: RESULT:{...}
 * stdin:  QUIT
 */

const readline = require('readline');
const path = require('path');

const simPath = process.argv[2];
if (!simPath) {
    process.stderr.write('Usage: node persistent_runner.js <sim_module_path>\n');
    process.exit(1);
}

// Redirect console.log to stderr during module load (suppresses CSV loading messages)
const origLog = console.log;
console.log = (...args) => process.stderr.write(args.join(' ') + '\n');

const simModule = require(path.resolve(simPath));

// Detect sim type from exports
const isSmartRoom = !!simModule.SmartRoom;
const isRichBuilding = !!simModule.SmartBuildingRich;

let sim = null;

async function initSim() {
    if (isSmartRoom) {
        sim = new simModule.SmartRoom();
        await sim._initialize();
    } else if (isRichBuilding) {
        sim = new simModule.SmartBuildingRich();
        if (sim.initPromise) await sim.initPromise;
    }
}

async function stepSim(cmd) {
    // Set elapsed time for window schedule (open_window sim)
    if (cmd.elapsed_ms !== undefined && sim.simulationStartTime !== undefined) {
        sim.simulationStartTime = Date.now() - cmd.elapsed_ms;
    }

    // Apply state
    if (cmd.state) {
        for (const [k, v] of Object.entries(cmd.state)) {
            if (sim[k] !== undefined) sim[k] = v;
        }
    }

    // Apply intervention
    if (cmd.intervention) {
        for (const [k, v] of Object.entries(cmd.intervention)) {
            if (sim[k] !== undefined) sim[k] = v;
        }
    }

    // Run physics
    if (sim.updateDependentVariables) {
        await sim.updateDependentVariables();
    } else if (sim.update) {
        await sim.update();
    }

    // Build output — mirror exactly what --single-step does
    if (isSmartRoom) {
        const result = {
            temperature: sim.temperature,
            humidity: sim.humidity,
            airQuality: sim.airQuality,
            energyConsumption: sim.energyConsumption,
            overallSatisfaction: sim.overallSatisfaction,
        };
        // Include open_window fields if they exist
        if (sim.windowConfig !== undefined) {
            result.windowOpen = sim.windowConfig.isOpen;
            result.outdoorTemperature = sim.outdoorTemperature;
            result.pmv = sim.pmv || 0;
        }
        // Satisfaction field: some sims use 'satisfaction', others 'overallSatisfaction'
        if (result.overallSatisfaction === undefined && sim.satisfaction !== undefined) {
            result.overallSatisfaction = sim.satisfaction;
        }
        return result;
    } else if (isRichBuilding) {
        return sim.getState ? sim.getState() : {};
    }
    return {};
}

(async () => {
    await initSim();
    // Keep console.log suppressed — only use process.stdout.write for RESULT lines
    // This prevents sim debug messages from polluting the output stream
    process.stdout.write('READY\n');

    const rl = readline.createInterface({ input: process.stdin, terminal: false });

    rl.on('line', async (line) => {
        const trimmed = line.trim();
        if (trimmed === 'QUIT') {
            if (sim && sim.cleanup) sim.cleanup();
            process.exit(0);
        }
        try {
            const cmd = JSON.parse(trimmed);
            if (cmd.reset) await initSim();
            const state = await stepSim(cmd);
            process.stdout.write('RESULT:' + JSON.stringify(state) + '\n');
        } catch (e) {
            process.stdout.write('ERROR:' + e.message + '\n');
        }
    });
})();
