#!/usr/bin/env node
/**
 * Smart Building Rich Simulator — 15-variable environment for sensor ablation study.
 *
 * Pure JS, no external dependencies. Implements a realistic office building with:
 *   - 2 exogenous variables (outdoor temp, solar radiation)
 *   - 2 regime variables (occupancy, window position)
 *   - 6 sensor variables (temperature, humidity, CO2, light, noise, air quality)
 *   - 1 derived variable (PMV)
 *   - 2 actuators (HVAC power, lighting power)
 *   - 2 outcomes (energy consumption, overall satisfaction)
 *
 * Ground truth DAG: 28 directed edges (see ground_truth_graphs.json).
 *
 * Interface: same --single-step protocol as open_window.js
 *   node js/smart_building_rich.js --single-step \
 *     --elapsed-ms 36000000 \
 *     --state '{"temperature":22,"humidity":45}' \
 *     --intervention '{"hvacPower":80}'
 *
 * Output: RESULT:{json with all 15 variables}\n
 */

'use strict';

// ── Utilities ────────────────────────────────────────────────────────────────

/** Poisson random variable (Knuth algorithm). */
function poissonSample(lambda) {
    if (lambda <= 0) return 0;
    const L = Math.exp(-Math.min(lambda, 30)); // cap to avoid underflow
    let k = 0, p = 1;
    do { k++; p *= Math.random(); } while (p > L);
    return k - 1;
}

/** Clamp value to [lo, hi]. */
function clamp(val, lo, hi) {
    return Math.max(lo, Math.min(hi, val));
}

/** Seed-able PRNG (xoshiro128**). Allows reproducible data generation. */
class PRNG {
    constructor(seed) {
        // Splitmix32 to initialize state from a single seed
        let s = seed | 0;
        const sm = () => { s = (s + 0x9e3779b9) | 0; let t = s ^ (s >>> 16); t = Math.imul(t, 0x21f0aaad); t = t ^ (t >>> 15); t = Math.imul(t, 0x735a2d97); return (t ^ (t >>> 15)) >>> 0; };
        this.s = [sm(), sm(), sm(), sm()];
    }
    /** Returns float in [0, 1). */
    next() {
        const s = this.s;
        const result = Math.imul(s[1] * 5, 1 << 7 | 1) >>> 0;
        const t = s[1] << 9;
        s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3];
        s[2] ^= t; s[3] = (s[3] << 11 | s[3] >>> 21);
        return (result >>> 0) / 4294967296;
    }
    /** Gaussian via Box-Muller. */
    gaussian(mean, std) {
        const u1 = this.next() || 1e-10;
        const u2 = this.next();
        return mean + std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }
    /** Poisson. */
    poisson(lambda) {
        if (lambda <= 0) return 0;
        const L = Math.exp(-Math.min(lambda, 30));
        let k = 0, p = 1;
        do { k++; p *= this.next(); } while (p > L);
        return k - 1;
    }
}

// ── Physical Constants ───────────────────────────────────────────────────────

const HVAC_MAX_WATTS  = 5000;  // 5 kW max HVAC
const LIGHT_MAX_WATTS = 500;   // 500 W max lighting
const ROOM_VOLUME_M3  = 200;   // m³
const OUTDOOR_CO2_PPM = 420;   // baseline outdoor CO2
const OUTDOOR_HUMIDITY = 40;   // baseline outdoor humidity %

// ── SmartBuildingRich Class ──────────────────────────────────────────────────

class SmartBuildingRich {
    constructor(seed) {
        this.rng = seed !== undefined ? new PRNG(seed) : null;

        // Outdoor temp baseline offset (°C).  Default 0 = standard 15°C base.
        // Set via --outdoor-offset CLI flag for multi-day weather scenarios.
        // Scientifically justified: represents seasonal/weather variation
        // that a real building would experience over a deployment period.
        this.outdoorTempOffset = 0;

        // Exogenous
        this.OutdoorTemp     = 15.0;   // °C
        this.SolarRadiation  = 0.0;    // W/m²

        // Regime
        this.Occupancy       = 0;      // 0-8 people
        this.WindowPosition  = 0.0;    // 0.0-1.0

        // Actuators (policy-controlled)
        this.HVACPower       = 50.0;   // 0-100 %
        this.LightingPower   = 30.0;   // 0-100 %

        // Endogenous sensors
        this.Temperature     = 22.0;   // °C
        this.Humidity        = 45.0;   // %RH
        this.CO2             = 450.0;  // ppm
        this.LightLevel      = 300.0;  // lux
        this.NoiseDB         = 35.0;   // dB
        this.AirQuality      = 80.0;   // 0-100 (higher = better)

        // Derived
        this.PMV             = 0.0;    // -3 to +3

        // Outcomes
        this.EnergyConsumption    = 0.0;  // 0-100 %
        this.Satisfaction  = 75.0; // 0-100

        // Simulation timing
        this.elapsedMs = 0;
    }

    /** Random float in [0,1). Uses seeded PRNG if available, else Math.random. */
    _rand() {
        return this.rng ? this.rng.next() : Math.random();
    }

    /** Gaussian noise. */
    _noise(std) {
        if (this.rng) return this.rng.gaussian(0, std);
        // Box-Muller fallback
        const u1 = Math.random() || 1e-10;
        const u2 = Math.random();
        return std * Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
    }

    /** Poisson sample. */
    _poisson(lambda) {
        return this.rng ? this.rng.poisson(lambda) : poissonSample(lambda);
    }

    // ── Exogenous update ─────────────────────────────────────────────────────

    updateExogenous(hour) {
        // Daily sinusoidal outdoor temperature: (15 + offset) ± 8°C, peak at 15:00
        this.OutdoorTemp = (15 + this.outdoorTempOffset) + 8 * Math.sin((hour - 6) * Math.PI / 12);

        // Solar radiation: peaks at noon, zero at night
        this.SolarRadiation = Math.max(0, 800 * Math.sin((hour - 6) * Math.PI / 12));
    }

    // ── Regime variable update ───────────────────────────────────────────────
    // Regime vars have temporal inertia: people don't teleport in/out every
    // minute, and windows aren't opened/closed every step.  Instead, the
    // current value drifts toward a schedule-driven target with small random
    // perturbations.  This creates realistic autocorrelation that makes
    // causal relationships discoverable.

    updateRegimeVars(hour) {
        // ── Occupancy: smooth drift toward office schedule ───────────────────
        let targetOcc;
        if      (hour < 7)  targetOcc = 0;
        else if (hour < 9)  targetOcc = ((hour - 7) / 2) * 6;   // morning ramp
        else if (hour < 12) targetOcc = 6;                        // morning work
        else if (hour < 13) targetOcc = 3;                        // lunch dip
        else if (hour < 17) targetOcc = 6;                        // afternoon work
        else if (hour < 19) targetOcc = ((19 - hour) / 2) * 6;   // evening ramp-down
        else                targetOcc = 0;

        // Drift toward target: ~30% of the way per step, with small noise.
        // People arrive/leave over ~3-5 minute windows, not instantaneously.
        const occDrift = 0.3 * (targetOcc - this.Occupancy);
        const occNoise = (this._rand() - 0.5) * 0.8;
        this.Occupancy = clamp(Math.round(this.Occupancy + occDrift + occNoise), 0, 8);

        // ── Window position: smooth drift toward temperature-driven target ──
        // Mildness: 1.0 when outdoor is 21.5°C, drops to 0 outside [18, 25]
        const mildness = Math.max(0, 1 - Math.pow(
            Math.max(0, Math.abs(this.OutdoorTemp - 21.5) - 3.5) / 5, 2));

        // More likely open during daytime
        const daytimeFactor = (hour >= 8 && hour <= 18) ? 0.7 : 0.1;
        const targetWin = mildness * daytimeFactor;

        // Smooth drift: window position changes slowly (someone adjusting it)
        const winDrift = 0.15 * (targetWin - this.WindowPosition);
        const winNoise = (this._rand() - 0.5) * 0.05;
        this.WindowPosition = clamp(this.WindowPosition + winDrift + winNoise, 0, 1);
    }

    // ── Endogenous sensor update ─────────────────────────────────────────────

    updateEndogenous() {
        const occ  = this.Occupancy;
        const win  = this.WindowPosition;
        const hvac = this.HVACPower / 100;   // normalise to [0,1]
        const light = this.LightingPower / 100;

        // ── Temperature ──────────────────────────────────────────────────────
        // Parents: OutdoorTemp, WindowPosition, Occupancy, HVACPower
        // Gains calibrated so each parent produces ~1-3°C variation at typical
        // operating points, ensuring detectable signal for causal discovery.
        const outdoorCoupling = win * 0.15 * (this.OutdoorTemp - this.Temperature);
        const bodyHeat = occ * 0.05;          // ~80W/person ≈ 0.05°C/step
        const hvacEffect = hvac * 0.5 * (22.0 - this.Temperature);
        this.Temperature += outdoorCoupling + bodyHeat + hvacEffect + this._noise(0.1);
        this.Temperature = clamp(this.Temperature, 15, 32);

        // ── Humidity ─────────────────────────────────────────────────────────
        // Parents: WindowPosition, Occupancy, HVACPower
        const outdoorHumMix = win * 0.1 * (OUTDOOR_HUMIDITY - this.Humidity);
        const respiration = occ * 0.15;       // ~50 g/hr/person, scaled up for detectability
        const dehumidification = -hvac * 0.1 * (this.Humidity - 45);
        this.Humidity += outdoorHumMix + respiration + dehumidification + this._noise(0.3);
        this.Humidity = clamp(this.Humidity, 20, 80);

        // ── CO2 ──────────────────────────────────────────────────────────────
        // Parents: Occupancy, WindowPosition
        // Reduced exhale rate and stronger decay to keep CO2 in [400, 1200]
        // range with meaningful variation (not saturated at cap).
        const co2Exhale = occ * 2.0;         // ~2 ppm/step/person (tuned down)
        const ventilation = win * 0.03 * (OUTDOOR_CO2_PPM - this.CO2);
        const co2Decay = -0.01 * (this.CO2 - OUTDOOR_CO2_PPM); // stronger natural decay
        this.CO2 += co2Exhale + ventilation + co2Decay + this._noise(2);
        this.CO2 = clamp(this.CO2, 350, 2000);

        // ── LightLevel ───────────────────────────────────────────────────────
        // Parents: SolarRadiation, WindowPosition, LightingPower
        const naturalLight = this.SolarRadiation * win * 0.5;  // lux from sun
        const artificialLight = light * 500;                     // max 500 lux
        this.LightLevel = naturalLight + artificialLight + this._noise(10);
        this.LightLevel = clamp(this.LightLevel, 0, 1200);

        // ── NoiseDB ──────────────────────────────────────────────────────────
        // Parents: Occupancy, HVACPower, WindowPosition
        // dB addition: 10*log10(sum of 10^(each/10))
        const baseNoise = 30;                                    // ambient
        const speechNoise = occ > 0 ? 40 + 10 * Math.log10(Math.max(1, occ)) : 0;
        const hvacNoise = hvac * 15;                             // fan 0-15 dB
        const outdoorNoise = win * 10;                           // street noise

        const sources = [baseNoise, speechNoise, hvacNoise, outdoorNoise]
            .filter(db => db > 0);
        const totalLinear = sources.reduce(
            (sum, db) => sum + Math.pow(10, db / 10), 0);
        this.NoiseDB = 10 * Math.log10(Math.max(1e-10, totalLinear)) + this._noise(1);
        this.NoiseDB = clamp(this.NoiseDB, 25, 90);

        // ── AirQuality ───────────────────────────────────────────────────────
        // Parents: Occupancy, WindowPosition, HVACPower
        // Scale 0-100 (higher = better).  Mean-reverting toward 70 (decent
        // indoor air) so the variable stays in a useful range with variation.
        const aqTarget = 70;
        const aqMeanRevert = 0.02 * (aqTarget - this.AirQuality);
        const particleLoad     = -occ * 1.0;     // people generate particles
        const ventilationBenefit = win * 3;       // fresh air
        const filtrationBenefit  = hvac * 1.5;    // HEPA filtration
        const outdoorPollution   = -win * 1;      // outdoor pollution
        this.AirQuality += aqMeanRevert + particleLoad + ventilationBenefit
                         + filtrationBenefit + outdoorPollution + this._noise(1);
        this.AirQuality = clamp(this.AirQuality, 0, 100);

        // ── PMV (simplified ASHRAE-55) ───────────────────────────────────────
        // Parents: Temperature, Humidity
        // Linear approximation around comfort zone (22°C, 50%RH)
        const tempDev = this.Temperature - 22.0;
        const humDev  = (this.Humidity - 50) / 100;
        this.PMV = 0.5 * tempDev + 0.1 * humDev * tempDev;
        this.PMV = clamp(this.PMV, -3, 3);

        // ── EnergyConsumption ────────────────────────────────────────────────
        // Parents: HVACPower, LightingPower
        const hvacWatts  = hvac * HVAC_MAX_WATTS;
        const lightWatts = light * LIGHT_MAX_WATTS;
        const maxWatts   = HVAC_MAX_WATTS + LIGHT_MAX_WATTS;
        this.EnergyConsumption = ((hvacWatts + lightWatts) / maxWatts) * 100;
        this.EnergyConsumption = clamp(this.EnergyConsumption, 0, 100);

        // ── OverallSatisfaction ──────────────────────────────────────────────
        // Parents: PMV, Humidity, CO2, LightLevel, NoiseDB, AirQuality, EnergyConsumption
        const pmvSat      = Math.max(0, 100 - Math.abs(this.PMV) * 30);
        const humiditySat = Math.max(0, 100 - Math.abs(this.Humidity - 50) * 2.5);
        const co2Sat      = Math.max(0, 100 - Math.max(0, (this.CO2 - 800)) / 12);
        const lightSat    = this._lightSatisfaction(this.LightLevel);
        const noiseSat    = Math.max(0, 100 - Math.max(0, (this.NoiseDB - 40)) * 2);
        const aqSat       = this.AirQuality;  // already 0-100
        const energyPenalty = this.EnergyConsumption * 0.1;

        this.Satisfaction =
            0.25 * pmvSat +
            0.15 * humiditySat +
            0.15 * co2Sat +
            0.15 * lightSat +
            0.15 * noiseSat +
            0.10 * aqSat +
            0.05 * (100 - energyPenalty);
        this.Satisfaction = clamp(this.Satisfaction, 0, 100);
    }

    /** Bell-curve light satisfaction: optimal ~400-500 lux. */
    _lightSatisfaction(lux) {
        if (lux < 100) return (lux / 100) * 50;
        if (lux < 300) return 50 + ((lux - 100) / 200) * 40;
        if (lux <= 600) return 90 + (1 - Math.abs(lux - 450) / 150) * 10;
        return Math.max(40, 100 - (lux - 600) / 10);
    }

    // ── Full physics step ────────────────────────────────────────────────────

    step() {
        const hour = (this.elapsedMs / 3600000) % 24;
        this.updateExogenous(hour);
        this.updateRegimeVars(hour);
        this.updateEndogenous();
    }

    /**
     * Stabilization step: only update endogenous sensors (temperature, humidity,
     * etc.) while keeping exogenous and regime variables frozen.
     *
     * Used after interventions to let the system reach steady state without
     * confounding from regime changes.  Real-world analogy: you set the HVAC
     * to 80% and wait 5 minutes for the room to stabilize before measuring
     * the temperature — occupancy and window state don't magically change
     * during that 5-minute window.
     */
    stepEndogenousOnly() {
        this.updateEndogenous();
    }

    // ── State I/O ────────────────────────────────────────────────────────────

    getState() {
        return {
            outdoorTemp:          +this.OutdoorTemp.toFixed(2),
            solarRadiation:       +this.SolarRadiation.toFixed(1),
            occupancy:            this.Occupancy,
            windowPosition:       +this.WindowPosition.toFixed(3),
            temperature:          +this.Temperature.toFixed(2),
            humidity:             +this.Humidity.toFixed(2),
            co2:                  +this.CO2.toFixed(1),
            lightLevel:           +this.LightLevel.toFixed(1),
            noiseDB:              +this.NoiseDB.toFixed(1),
            airQuality:           +this.AirQuality.toFixed(1),
            pmv:                  +this.PMV.toFixed(3),
            hvacPower:            +this.HVACPower.toFixed(1),
            lightingPower:        +this.LightingPower.toFixed(1),
            energyConsumption:    +this.EnergyConsumption.toFixed(2),
            satisfaction:  +this.Satisfaction.toFixed(2),
            elapsedTime:          +(this.elapsedMs / 1000).toFixed(1),
        };
    }

    /** Restore state from a camelCase JSON object. */
    restoreState(obj) {
        const map = {
            outdoorTemp:         'OutdoorTemp',
            solarRadiation:      'SolarRadiation',
            occupancy:           'Occupancy',
            windowPosition:      'WindowPosition',
            temperature:         'Temperature',
            humidity:            'Humidity',
            co2:                 'CO2',
            lightLevel:          'LightLevel',
            noiseDB:             'NoiseDB',
            airQuality:          'AirQuality',
            pmv:                 'PMV',
            hvacPower:           'HVACPower',
            lightingPower:       'LightingPower',
            energyConsumption:   'EnergyConsumption',
            satisfaction: 'Satisfaction',
        };
        for (const [jsonKey, propKey] of Object.entries(map)) {
            if (obj[jsonKey] !== undefined) this[propKey] = +obj[jsonKey];
        }
    }

    /** Apply an intervention (do-operator). Any variable can be set. */
    applyIntervention(obj) {
        // Actuator setpoints (normal policy actions)
        if (obj.hvacPower     !== undefined) this.HVACPower     = clamp(+obj.hvacPower, 0, 100);
        if (obj.lightingPower !== undefined) this.LightingPower  = clamp(+obj.lightingPower, 0, 100);

        // do-operator: force any variable (for causal edge testing)
        if (obj.temperature     !== undefined) this.Temperature     = +obj.temperature;
        if (obj.humidity        !== undefined) this.Humidity         = +obj.humidity;
        if (obj.co2             !== undefined) this.CO2              = +obj.co2;
        if (obj.lightLevel      !== undefined) this.LightLevel       = +obj.lightLevel;
        if (obj.noiseDB         !== undefined) this.NoiseDB          = +obj.noiseDB;
        if (obj.airQuality      !== undefined) this.AirQuality       = +obj.airQuality;
        if (obj.occupancy       !== undefined) this.Occupancy        = clamp(Math.round(+obj.occupancy), 0, 8);
        if (obj.windowPosition  !== undefined) this.WindowPosition   = clamp(+obj.windowPosition, 0, 1);
        if (obj.outdoorTemp     !== undefined) this.OutdoorTemp      = +obj.outdoorTemp;
        if (obj.solarRadiation  !== undefined) this.SolarRadiation   = clamp(+obj.solarRadiation, 0, 800);
    }
}

// ── CLI Interface ────────────────────────────────────────────────────────────

if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.includes('--single-step')) {
        // ── Single-step mode (policy evaluation / intervention testing) ──────
        const getArg = (flag) => {
            const idx = args.indexOf(flag);
            return (idx !== -1 && idx + 1 < args.length) ? args[idx + 1] : null;
        };

        const elapsedMs       = parseInt(getArg('--elapsed-ms') || '0', 10);
        const stateJson       = getArg('--state');
        const interventionJson = getArg('--intervention');
        const seedArg         = getArg('--seed');

        const outdoorOffset = parseFloat(getArg('--outdoor-offset') || '0');

        const room = new SmartBuildingRich(seedArg ? parseInt(seedArg, 10) : undefined);
        room.outdoorTempOffset = outdoorOffset;
        room.elapsedMs = elapsedMs;

        // Restore previous state
        if (stateJson) {
            try { room.restoreState(JSON.parse(stateJson)); } catch (_) {}
        }

        // Always run full dynamics (including regime evolution)
        room.step();
        // Then apply intervention (actuator setpoints) on top
        if (interventionJson) {
            try { room.applyIntervention(JSON.parse(interventionJson)); } catch (_) {}
            // One endogenous step to let intervention take effect
            room.stepEndogenousOnly();
        }

        // Output
        process.stdout.write('RESULT:' + JSON.stringify(room.getState()) + '\n');
        process.exit(0);

    } else if (args.includes('--generate')) {
        // ── Batch data generation mode ───────────────────────────────────────
        const fs = require('fs');
        const seedArg = args.includes('--seed')
            ? parseInt(args[args.indexOf('--seed') + 1], 10) : 42;
        const nDays = args.includes('--days')
            ? parseInt(args[args.indexOf('--days') + 1], 10) : 8;

        const nSteps = nDays * 1440; // 1 step per minute
        const room = new SmartBuildingRich(seedArg);

        // Per-day outdoor offset schedule: varied weather ensures all 4 regime
        // combinations (base, occ-only, win-only, full) appear in training data.
        // Day 6-7: mild evenings with forced window-open during off-hours
        // creates window-only regime samples for monitor training.
        const dayOffsets = [0, -3, 6, 13, 0, -3, 6, 6];

        const header = [
            'OutdoorTemp', 'SolarRadiation', 'Occupancy', 'WindowPosition',
            'Temperature', 'Humidity', 'CO2', 'LightLevel', 'NoiseDB',
            'AirQuality', 'PMV', 'HVACPower', 'LightingPower',
            'EnergyConsumption', 'Satisfaction'
        ];
        const rows = [header.join(',')];

        for (let i = 0; i < nSteps; i++) {
            room.elapsedMs = i * 60000; // 1-minute steps

            const hour = room.elapsedMs / 3600000;
            const day = Math.floor(hour / 24) % nDays;
            const hourOfDay = hour % 24;

            // Vary weather per day
            room.outdoorTempOffset = dayOffsets[day] || 0;

            // Simple proportional control loop for HVAC/Lighting
            room.HVACPower = clamp(
                50 + 10 * (22 - room.Temperature) + 5 * (45 - room.Humidity),
                0, 100);
            room.LightingPower = clamp(
                room.Occupancy > 0
                    ? Math.max(20, 80 - room.LightLevel * 0.1)
                    : 5,
                0, 100);

            room.step();

            // Days 6-7 (mild): force window open during off-hours to create
            // window-only regime (open window + unoccupied).  Simulates a
            // realistic scenario: window left open overnight on a mild evening.
            if ((day === 6 || day === 7) && (hourOfDay >= 19 || hourOfDay < 7)) {
                room.WindowPosition = clamp(0.6 + (room.rng ? (room.rng.next() - 0.5) * 0.1 : 0), 0, 1);
            }

            const s = room.getState();
            rows.push([
                s.outdoorTemp, s.solarRadiation, s.occupancy, s.windowPosition,
                s.temperature, s.humidity, s.co2, s.lightLevel, s.noiseDB,
                s.airQuality, s.pmv, s.hvacPower, s.lightingPower,
                s.energyConsumption, s.satisfaction
            ].join(','));
        }

        const outFile = args.includes('--output')
            ? args[args.indexOf('--output') + 1]
            : 'data_regen/smart_building_rich_data_10k.csv';

        fs.mkdirSync(require('path').dirname(outFile), { recursive: true });
        fs.writeFileSync(outFile, rows.join('\n') + '\n');
        console.log(`Generated ${nSteps} rows → ${outFile}`);
        process.exit(0);

    } else {
        console.log(`Usage:
  Single step:  node js/smart_building_rich.js --single-step --elapsed-ms 36000000 --state '{"temperature":22}' --intervention '{"hvacPower":80}'
  Generate:     node js/smart_building_rich.js --generate [--seed 42] [--days 7] [--output path.csv]`);
        process.exit(0);
    }
}

// ── simulateAndGetLatestData (tester compatibility) ───────────────────────────

async function simulateAndGetLatestData(duration, interventions, interventionTime, seed) {
    const room = new SmartBuildingRich(seed);
    const stepMs = 60000; // 1-minute steps

    // Warm-up to pre-intervention time
    const preSteps = Math.max(1, Math.floor(interventionTime / stepMs));
    for (let i = 0; i < preSteps; i++) {
        room.elapsedMs += stepMs;
        room.step();
    }

    // Capture pre-intervention state (use PascalCase keys for tester compat)
    const preState = room.getState();
    const preInterventionData = {
        Temperature:          preState.temperature,
        Humidity:             preState.humidity,
        AirQuality:           preState.airQuality,
        EnergyConsumption:    preState.energyConsumption,
        Satisfaction:         preState.satisfaction,
        OverallSatisfaction:  preState.satisfaction,
        CO2:                  preState.co2,
        LightLevel:           preState.lightLevel,
        NoiseDB:              preState.noiseDB,
        HVACPower:            preState.hvacPower,
        LightingPower:        preState.lightingPower,
        Occupancy:            preState.occupancy,
        OutdoorTemp:          preState.outdoorTemp,
        SolarRadiation:       preState.solarRadiation,
        WindowPosition:       preState.windowPosition,
        PMV:                  preState.pmv,
    };

    // Apply interventions
    for (const intervention of interventions) {
        const { variable, value } = intervention;
        const obj = {};
        // Map variable name (case-insensitive) to camelCase key
        const lcVar = variable.toLowerCase();
        const varMap = {
            temperature: 'temperature', humidity: 'humidity', co2: 'co2',
            lightlevel: 'lightLevel', noisedb: 'noiseDB', airquality: 'airQuality',
            hvacpower: 'hvacPower', lightingpower: 'lightingPower',
            occupancy: 'occupancy', windowposition: 'windowPosition',
            outdoortemp: 'outdoorTemp', solarradiation: 'solarRadiation',
        };
        const key = varMap[lcVar] || variable;
        obj[key] = value;
        room.applyIntervention(obj);
    }

    // Stabilization: run endogenous-only steps to let the system respond to
    // the intervention without confounding from regime variable changes.
    // Real-world analogy: after setting HVAC to 80%, you wait a few minutes
    // for temperature to stabilize — occupancy doesn't randomly change
    // during that measurement window.
    const STABILIZATION_STEPS = 5; // 5 minutes of settling
    for (let i = 0; i < STABILIZATION_STEPS; i++) {
        room.stepEndogenousOnly();
    }

    // Then run remaining steps with full dynamics (regime can change again)
    const totalPostSteps = Math.max(1, Math.floor((duration - interventionTime) / stepMs));
    const remainingSteps = Math.max(0, totalPostSteps - STABILIZATION_STEPS);
    for (let i = 0; i < remainingSteps; i++) {
        room.elapsedMs += stepMs;
        room.step();
    }

    // Capture post-intervention state
    const postState = room.getState();
    const postInterventionData = {
        Temperature:          postState.temperature,
        Humidity:             postState.humidity,
        AirQuality:           postState.airQuality,
        EnergyConsumption:    postState.energyConsumption,
        Satisfaction:         postState.satisfaction,
        OverallSatisfaction:  postState.satisfaction,
        CO2:                  postState.co2,
        LightLevel:           postState.lightLevel,
        NoiseDB:              postState.noiseDB,
        HVACPower:            postState.hvacPower,
        LightingPower:        postState.lightingPower,
        Occupancy:            postState.occupancy,
        OutdoorTemp:          postState.outdoorTemp,
        SolarRadiation:       postState.solarRadiation,
        WindowPosition:       postState.windowPosition,
        PMV:                  postState.pmv,
    };

    return {
        preInterventionData,
        postInterventionData,
        pre_state: preInterventionData,
        post_state: postInterventionData,
    };
}

async function pairedCounterfactualTest(interventions, seed) {
    const WARMUP_STEPS = 5;
    const HOLD_STEPS = 30;
    const stepMs = 60000;
    const varMap = {
        temperature:'temperature', humidity:'humidity', co2:'co2',
        lightlevel:'lightLevel', noisedb:'noiseDB', airquality:'airQuality',
        hvacpower:'hvacPower', lightingpower:'lightingPower',
        occupancy:'occupancy', windowposition:'windowPosition',
        outdoortemp:'outdoorTemp', solarradiation:'solarRadiation',
    };
    const roomHigh = new SmartBuildingRich(seed);
    const roomLow  = new SmartBuildingRich(seed);
    for (let i = 0; i < WARMUP_STEPS; i++) {
        roomHigh.elapsedMs += stepMs; roomHigh.step();
        roomLow.elapsedMs += stepMs; roomLow.step();
    }
    for (const intv of interventions) {
        const key = varMap[intv.variable.toLowerCase()] || intv.variable;
        const objH = {}; objH[key] = intv.value; roomHigh.applyIntervention(objH);
        const objL = {}; objL[key] = 0; roomLow.applyIntervention(objL);
    }
    for (let i = 0; i < HOLD_STEPS; i++) {
        for (const intv of interventions) {
            const key = varMap[intv.variable.toLowerCase()] || intv.variable;
            const objH = {}; objH[key] = intv.value; roomHigh.applyIntervention(objH);
            const objL = {}; objL[key] = 0; roomLow.applyIntervention(objL);
        }
        roomHigh.stepEndogenousOnly(); roomLow.stepEndogenousOnly();
    }
    const postHigh = roomHigh.getState(); const postLow = roomLow.getState();
    const toData = (s) => ({
        Temperature:s.temperature, Humidity:s.humidity, AirQuality:s.airQuality,
        EnergyConsumption:s.energyConsumption, Satisfaction:s.satisfaction,
        OverallSatisfaction:s.satisfaction, CO2:s.co2, LightLevel:s.lightLevel,
        NoiseDB:s.noiseDB, HVACPower:s.hvacPower, LightingPower:s.lightingPower,
        Occupancy:s.occupancy, OutdoorTemp:s.outdoorTemp,
        SolarRadiation:s.solarRadiation, WindowPosition:s.windowPosition, PMV:s.pmv,
    });
    return {
        preInterventionData: toData(postLow), postInterventionData: toData(postHigh),
        pre_state: toData(postLow), post_state: toData(postHigh),
    };
}

module.exports = { SmartBuildingRich, clamp, poissonSample, simulateAndGetLatestData, pairedCounterfactualTest };
