// Challenge 1: Open Window Anomaly Detection
// Based on smart_room.js with modifications for window state tracking

const fs = require('fs');
const path = require('path');
const { spawn } = require('child_process');
const PMVCache = require('./pmv_cache');
const PsychrometricCalculator = require('./python_interface');
const os = require('os');
const tmpDir = os.tmpdir();
const Papa = require('papaparse');

let globalPreInterventionState = null;
let globalPostInterventionState = null;

// EnergyLookup class - handles EnergyPlus data
class EnergyLookup {
    constructor() {
        this.data = null;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            const fs = require('fs').promises;
            const path = require('path');
            const Papa = require('papaparse');
            
            const possiblePaths = [
                'eplus_test_results/epluszsz.csv',
                'epluszsz.csv',
                path.join(__dirname, 'eplus_test_results/epluszsz.csv'),
                path.join(__dirname, 'epluszsz.csv'),
                path.join(process.cwd(), 'epluszsz.csv'),
                path.join(process.cwd(), 'data/epluszsz.csv'),
                path.resolve(__dirname, '..', 'epluszsz.csv')
            ];
            
            let csvData;
            let loadedPath = '';
            
            for (const csvPath of possiblePaths) {
                try {
                    console.log(`Attempting to load CSV from: ${csvPath}`);
                    const stats = await fs.stat(csvPath);
                    if (stats.isFile()) {
                        csvData = await fs.readFile(csvPath, { encoding: 'utf8' });
                        loadedPath = csvPath;
                        console.log(`Found file at ${csvPath} with size ${stats.size} bytes`);
                        break;
                    }
                } catch (err) {
                    console.log(`File not found at ${csvPath}`);
                }
            }
            
            if (!csvData) {
                const embeddedData = await this._loadEmbeddedData();
                if (embeddedData) {
                    csvData = embeddedData;
                    loadedPath = "embedded data";
                } else {
                    throw new Error('Could not find epluszsz.csv in any expected location');
                }
            }
            
            console.log(`Successfully loaded CSV from: ${loadedPath}`);
            this.data = Papa.parse(csvData, {
                header: true,
                dynamicTyping: true,
                skipEmptyLines: true
            }).data;
            
            if (!this.data || this.data.length === 0) {
                throw new Error('CSV file loaded but no data found');
            }
            
            console.log(`Loaded ${this.data.length} rows of energy data`);
            
            this.initialized = true;
            return true;
        } catch (error) {
            console.error('Failed to load energy data:', error);
            this.initialized = false;
            return false;
        }
    }

    async _loadEmbeddedData() {
        try {
            const minimalData = [
                {
                    "Time": "01/01 00:00:00",
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Des Heat Load [W]": 5000,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Des Sens Cool Load [W]": 8000,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Temperature [C]": 22,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Relative Humidity [%]": 50
                },
                {
                    "Time": "01/01 00:00:00",
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL HEATING 99% DESIGN CONDITIONS DB:Des Heat Load [W]": 10000,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Des Sens Cool Load [W]": 12000,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Temperature [C]": 26,
                    "NORTH_ZONE:CHICAGO_IL_USA ANNUAL COOLING 1% DESIGN CONDITIONS DB/MCWB:Cooling Zone Relative Humidity [%]": 60
                }
            ];
            
            console.log("Using embedded minimal dataset as fallback");
            return Papa.unparse(minimalData);
        } catch (error) {
            console.error("Failed to load embedded data:", error);
            return null;
        }
    }

    calculateEnergy(temperature, humidity, airQuality = 300, windowOpen = false) {
        console.log(`Energy calculation: temp=${temperature}, humidity=${humidity}, windowOpen=${windowOpen}`);

        // Challenge 1: When window is open, energy consumption increases dramatically
        // due to heat loss to outdoor environment
        const nearest = this._findNearestDataPoint(temperature, humidity);

        if (nearest) {
            let coolLoadKey = Object.keys(nearest).find(k =>
                k.includes('Cool Load') && k.includes('CHICAGO') && !k.includes('No DOAS'));

            let heatLoadKey = Object.keys(nearest).find(k =>
                k.includes('Heat Load') && k.includes('CHICAGO') && !k.includes('No DOAS'));

            if (!coolLoadKey || !heatLoadKey || nearest[coolLoadKey] === 0 || nearest[heatLoadKey] === 0) {
                coolLoadKey = Object.keys(nearest).find(k => k.includes('Des Sens Cool Load'));
                heatLoadKey = Object.keys(nearest).find(k => k.includes('Des Heat Load'));
            }

            if (coolLoadKey && heatLoadKey &&
                typeof nearest[coolLoadKey] === 'number' &&
                typeof nearest[heatLoadKey] === 'number') {

                const coolLoad = nearest[coolLoadKey];
                const heatLoad = nearest[heatLoadKey];

                console.log(`Loads: coolLoad=${coolLoad}, heatLoad=${heatLoad}`);

                const totalLoad = coolLoad + heatLoad;
                const maxLoad = 31645.00;

                const loadRatio = Math.max(0, Math.min(1, totalLoad / maxLoad));
                let baseValue = loadRatio * 100;

                // Challenge 1: Window open multiplier - dramatically increases heating load
                if (windowOpen) {
                    baseValue *= 2.5;  // Window open causes 2.5x energy consumption
                    console.log(`Window open! Energy multiplied: ${baseValue.toFixed(2)}`);
                }

                const aqInfluence = Math.min(0.03, airQuality / 10000);
                const aqFactor = 1.0 + aqInfluence;

                const remainingHeadroom = 100 - baseValue * aqFactor;
                const randomVariation = Math.min(remainingHeadroom, 2) * Math.random();

                const energy = Math.min(100, baseValue * aqFactor + randomVariation);

                console.log(`Energy result: ${energy.toFixed(2)}%`);
                return energy;
            } else {
                console.error("Invalid load values in nearest data point");
            }
        } else {
            console.error("Could not find nearest data point");
        }

        console.log("Using fallback energy calculation");
        return this._calculateFallbackEnergy(temperature, humidity, airQuality, windowOpen);
    }

    _findNearestDataPoint(temperature, humidity) {
        if (!this.initialized || !this.data || !this.data.length) {
            console.error("Energy data not initialized");
            return null;
        }

        let minDistance = Infinity;
        let nearestPoint = null;

        for (const point of this.data) {
            const tempKey = Object.keys(point).find(k => k.includes('Temperature') && k.includes('CHICAGO'));
            const humidityKey = Object.keys(point).find(k => k.includes('Humidity') && k.includes('CHICAGO'));
            
            if (tempKey && humidityKey && typeof point[tempKey] === 'number' && typeof point[humidityKey] === 'number') {
                const tempDiff = Math.abs(point[tempKey] - temperature);
                const humidityDiff = Math.abs(point[humidityKey] - humidity);
                const distance = Math.sqrt(tempDiff * tempDiff + humidityDiff * humidityDiff);
                
                if (distance < minDistance) {
                    minDistance = distance;
                    nearestPoint = point;
                }
            }
        }
        
        return nearestPoint;
    }

    _calculateFallbackEnergy(temperature, humidity, airQuality, windowOpen = false) {
        const tempFactor = Math.abs(22 - temperature) / 10;
        const humidityFactor = Math.abs(50 - humidity) / 50;
        const aqFactor = airQuality / 1000;
        
        let baseEnergy = 30 + (tempFactor * 20) + (humidityFactor * 15) + (aqFactor * 10);
        
        // Challenge 1: Window open effect in fallback
        if (windowOpen) {
            baseEnergy *= 2.5;
        }
        
        return Math.min(100, Math.max(0, baseEnergy + (Math.random() * 10 - 5)));
    }
}

// SmartRoom class with Challenge 1 window state
class SmartRoom {
    constructor() {
        // Challenge 1: Window state configuration (must be before reset())
        // Window opens for 30 min every 2 hours → ~25% open time.
        // Schedule covers 0-48h so it works regardless of elapsed-ms offset.
        this.windowConfig = {
            schedule: (() => {
                const periods = [];
                for (let h = 0; h < 48; h += 2) {
                    const base = h * 3600 * 1000;
                    periods.push({
                        openTime:  base + 30 * 60 * 1000,   // 30 min into each 2h block
                        closeTime: base + 60 * 60 * 1000    // 60 min into each 2h block
                    });
                }
                return periods;
            })(),
            isOpen: false
        };
        
        // this._windowStateSet = false;
        
        // Challenge 1: Outdoor conditions
        this.outdoorTemperature = 15.0;  // Mild spring day baseline
        
        // Track simulation start time for window opening
        this.simulationStartTime = null;
        
        this.reset();
        this.isRunning = false;
        this.dataLog = [];
        this.lastDataLogTime = 0;
        this.dataLogInterval = 1000;

        process.on('SIGTERM', () => {
            console.log('Process termination signal received, cleaning up...');
            this.cleanup();
        });
        
        process.on('SIGINT', () => {
            console.log('Process interrupt signal received, cleaning up...');
            this.cleanup();
        });

        this.energyLookup = new EnergyLookup();
        this.isEnergyInitialized = false;
        this.initialized = false;
        
        this.initPromise = this.energyLookup.initialize().then(() => {
            this.isEnergyInitialized = true;
            console.log("Energy lookup successfully initialized");
        }).catch(err => {
            console.error("Failed to initialize energy lookup:", err);
        });

        this.psychroCalc = new PsychrometricCalculator();
        this.airSpeed = 0.1;
        this.meanRadiantTemp = 22;
        this.activePromises = new Set();

        this.eplusProcess = null;
        this.eplusQueue = Promise.resolve();
        
        // Track simulation start time for window opening
        this.simulationStartTime = null;
    }

    reset() {
        this.temperature = 22;
        this.humidity = 50;
        this.airQuality = 300;
        this.airSpeed = 0.1;
        this.energyConsumption = 50;
        this.satisfaction = 75;
        this.timestamp = new Date();
        this.dataLog = [];
        
        // Challenge 1: Reset window state
        this.windowConfig.isOpen = false;
        // this._windowStateSet = false;
        this.simulationStartTime = null;
    }

    async _initialize() {
        try {
            console.log("Starting SmartRoom initialization...");
            await this.energyLookup.initialize();
            this.initialized = true;
            console.log("SmartRoom initialization complete");
            return true;
        } catch (error) {
            console.error("SmartRoom initialization failed:", error);
            return false;
        }
    }

    // Challenge 1: Check and update window state based on elapsed time
    updateWindowState() {
        if (this.simulationStartTime === null) {
            return;
        }
        
        const elapsedTime = Date.now() - this.simulationStartTime;
        const hour = (elapsedTime / 3600000) % 24;
        // Diurnal cycle: cool morning (~12°C at 6am), warm afternoon (~25°C at 2pm), cool evening
        this.outdoorTemperature = 18 + 7 * Math.sin((hour - 6) * Math.PI / 12);
        
        let shouldBeOpen = false;
        for (const period of this.windowConfig.schedule) {
            if (elapsedTime >= period.openTime && elapsedTime < period.closeTime) {
                shouldBeOpen = true;
                break;
            }
        }
        
        if (shouldBeOpen && !this.windowConfig.isOpen) {
            this.windowConfig.isOpen = true;
            console.log(`\n=== WINDOW OPENED at ${(elapsedTime/60000).toFixed(1)}min ===`);
        } else if (!shouldBeOpen && this.windowConfig.isOpen) {
            this.windowConfig.isOpen = false;
            console.log(`\n=== WINDOW CLOSED at ${(elapsedTime/60000).toFixed(1)}min ===`);
        }
    }

    // Realistic temperature dynamics with natural drift
    updateTemperatureDynamics() {
        const targetTemp = 22; // Thermostat setpoint
        
        // Natural drift even when window closed (building heat loss, occupancy, sun, etc)
        const naturalDrift = (Math.random() - 0.5) * 0.15;
        
        if (this.windowConfig.isOpen) {
            // Strong heat loss to outdoor environment
            const driftRate = 0.08;
            const tempDiff = this.outdoorTemperature - this.temperature;
            const heatLoss = tempDiff * driftRate;
            
            // Thermostat heating (partial compensation)
            const heatingRate = 0.05;
            const heatingError = targetTemp - this.temperature;
            const heatingGain = heatingError * heatingRate;
            
            this.temperature += heatLoss + heatingGain + naturalDrift;
        } else {
            // Normal thermostat control with natural variations
            const error = targetTemp - this.temperature;
            const controlGain = error * 0.08;
            this.temperature += controlGain + naturalDrift;
        }
        
        // Humidity drifts naturally
        const humidityDrift = (Math.random() - 0.5) * 0.3;
        this.humidity += humidityDrift;
        if (this.windowConfig.isOpen) {
            // Window open affects humidity (outdoor = drier in winter)
            const outdoorHumidity = 35; // Winter outdoor humidity
            const humidityPull = (outdoorHumidity - this.humidity) * 0.02;
            this.humidity += humidityPull;
        }
        
        // Air quality varies with occupancy patterns and ventilation
        const aqDrift = (Math.random() - 0.5) * 5;
        this.airQuality += aqDrift;
        if (this.windowConfig.isOpen) {
            // Window open improves air quality
            this.airQuality -= 2;
        } else {
            // Closed room - AQ slowly degrades
            this.airQuality += 0.5;
        }
    }

    async updateDependentVariables() {
        console.log('\n=== Updating Dependent Variables ===');
        
        try {
            if (!this.initialized) {
                console.log("Waiting for initialization to complete...");
                await this.initPromise;
            }
            
            // Challenge 1: Update window state
            this.updateWindowState();
            
            // Ensure values are within bounds
            this.temperature = Math.max(18, Math.min(30, Number(this.temperature)));
            this.humidity = Math.max(30, Math.min(70, Number(this.humidity)));
            this.airQuality = Math.max(200, Math.min(1000, Number(this.airQuality)));
            
            console.log(`Input values: temp=${this.temperature.toFixed(2)}, humidity=${this.humidity.toFixed(2)}, airQuality=${this.airQuality.toFixed(2)}, windowOpen=${this.windowConfig.isOpen}`);
            
            // Challenge 1: Temperature dynamics when window is open
            this.updateTemperatureDynamics();
            
            // Calculate PMV
            const pmvPromise = this.psychroCalc.calculatePMV(
                this.temperature,
                this.humidity,
                this.airSpeed,
                this.meanRadiantTemp
            ).then(result => {
                console.log(`PMV calculation result:`, result);
                if (result && typeof result.pmv === 'number' && !isNaN(result.pmv)) {
                    this.pmv = result.pmv;
                    console.log(`Updated PMV: ${this.pmv.toFixed(2)}`);
                } else {
                    console.error('Invalid PMV result, using default 0');
                    this.pmv = 0;
                }
            }).catch(error => {
                console.error('PMV calculation error:', error);
                this.pmv = 0;
            });

            this.activePromises.add(pmvPromise);
            pmvPromise.finally(() => this.activePromises.delete(pmvPromise));

            // Challenge 1: Energy calculation with window state
            this.energyConsumption = this.energyLookup.calculateEnergy(
                this.temperature,
                this.humidity,
                this.airQuality,
                this.windowConfig.isOpen
            );
            
            console.log(`Updated energy consumption: ${this.energyConsumption.toFixed(2)}%`);
            
            // Wait for PMV before calculating satisfaction
            await pmvPromise;
            
            // Calculate overall satisfaction (after PMV is ready)
            const pmvSatisfaction = 100 - Math.abs(this.pmv) * 30;
            const humiditySatisfaction = Math.max(0, 100 - Math.abs(this.humidity - 50) * 2.5);
            const aqSatisfaction = Math.max(0, 100 - (this.airQuality - 400) / 6);
            const energySatisfaction = 100 - this.energyConsumption;

            this.satisfaction = 0.30 * pmvSatisfaction + 0.25 * humiditySatisfaction +
                                0.25 * aqSatisfaction + 0.20 * energySatisfaction;
            
            console.log(`Overall satisfaction: ${this.satisfaction.toFixed(2)}%`);
            console.log('=== Update Complete ===\n');
            
        } catch (error) {
            console.error('Error in updateDependentVariables:', error);
            this.pmv = 0;
            this.energyConsumption = 50;
            this.satisfaction = 50;
        }
    }

    start() {
        if (this.isRunning) {
            console.log("Room simulation already running");
            return;
        }
        
        // Challenge 1: Mark simulation start time
        this.simulationStartTime = Date.now();
        
        this.isRunning = true;
        console.log("Starting room simulation...");
        
        const update = async () => {
            if (!this.isRunning) return;
            
            try {
                await this.updateDependentVariables();
                
                // Log data
                const currentTime = Date.now();
                if (currentTime - this.lastDataLogTime >= this.dataLogInterval) {
                    this.logData();
                    this.lastDataLogTime = currentTime;
                }
            } catch (error) {
                console.error('Error in update loop:', error);
            }
            
            if (this.isRunning) {
                setTimeout(update, 1000);
            }
        };
        
        update();
    }

    stop() {
        this.isRunning = false;
        console.log("Stopping room simulation...");
    }

    getState() {
        const elapsed = this.simulationStartTime ? (Date.now() - this.simulationStartTime) / 1000 : 0;
        console.log(`DEBUG: simulationStartTime=${this.simulationStartTime}, now=${Date.now()}, elapsed=${elapsed}`);
        
        return {
            timestamp: this.timestamp || new Date().toISOString(),
            temperature: this.temperature,
            humidity: this.humidity,
            airQuality: this.airQuality,
            pmv: this.pmv,
            energyConsumption: this.energyConsumption,
            satisfaction: this.satisfaction,
            windowOpen: this.windowConfig.isOpen,
            outdoorTemperature: this.outdoorTemperature,
            elapsedTime: this.simulationStartTime ? (Date.now() - this.simulationStartTime) / 1000 : 0
        };
    }

    setState(newState) {
        if (newState.temperature !== undefined) this.temperature = newState.temperature;
        if (newState.humidity !== undefined) this.humidity = newState.humidity;
        if (newState.airQuality !== undefined) this.airQuality = newState.airQuality;
        
        // Challenge 1: Allow manual window control if needed
        if (newState.windowOpen !== undefined) {
            this.windowConfig.isOpen = newState.windowOpen;
        }
        
        this.updateDependentVariables();
    }

    logData() {
        const state = this.getState();
        this.dataLog.push({
            timestamp: state.timestamp,
            temperature: state.temperature,
            humidity: state.humidity,
            airQuality: state.airQuality,
            pmv: state.pmv,
            energyConsumption: state.energyConsumption,
            satisfaction: state.satisfaction,
            // Challenge 1: Log window state
            windowOpen: state.windowOpen,
            outdoorTemperature: state.outdoorTemperature,
            elapsedTime: state.elapsedTime
        });
        
        // Keep log size manageable
        if (this.dataLog.length > 10000) {
            this.dataLog = this.dataLog.slice(-5000);
        }
    }

    getDataLog() {
        return this.dataLog;
    }

    async cleanup() {
        console.log("Cleaning up SmartRoom resources...");
        this.stop();
        
        try {
            await Promise.all(Array.from(this.activePromises));
        } catch (error) {
            console.error('Error waiting for active promises:', error);
        }
        
        if (this.psychroCalc && typeof this.psychroCalc.cleanup === 'function') {
            await this.psychroCalc.cleanup();
        }
        
        if (this.eplusProcess) {
            this.eplusProcess.kill();
            this.eplusProcess = null;
        }
        
        console.log("Cleanup complete");
    }

    // Intervention support for causal discovery
    async applyInterventions(interventions) {
        console.log('\n=== Applying Interventions ===');
        console.log('Interventions:', interventions);
        
        // Ensure PMV is calculated before capturing pre-state
        await this.updateDependentVariables();

        // Save pre-intervention state
        const preState = this.getState();
        
        // Apply each intervention
        for (const intervention of interventions) {
            console.log(`Applying intervention: ${intervention.variable} = ${intervention.value}`);
            
            const varLower = intervention.variable.toLowerCase();
            if (varLower === 'temperature') {
                this.temperature = intervention.value;
            } else if (varLower === 'humidity') {
                this.humidity = intervention.value;
            } else if (varLower === 'airquality') {
                this.airQuality = intervention.value;
            } else if (varLower === 'windowopen') {
                this.windowConfig.isOpen = intervention.value;
            }
        }
        
        // Update dependent variables
        await this.updateDependentVariables();
        
        // Wait for system to stabilize
        await new Promise(resolve => setTimeout(resolve, 1000));
        
        // Get post-intervention state
        const postState = this.getState();
        
        // Convert to capitalized format for tester.py compatibility
        const toCapitalized = (state) => ({
            Timestamp: state.timestamp,
            Temperature: state.temperature,
            Humidity: state.humidity,
            AirQuality: state.airQuality,
            EnergyConsumption: state.energyConsumption,
            Satisfaction: state.satisfaction,
            PMV: state.pmv,
            InterventionApplied: false,
            WindowOpen: state.windowOpen,
            OutdoorTemperature: state.outdoorTemperature,
            ElapsedTime: state.elapsedTime
        });
        
        const preData = toCapitalized(preState);
        const postData = toCapitalized(postState);
        postData.InterventionApplied = true;
        
        globalPreInterventionState = preData;
        globalPostInterventionState = postData;
        
        const result = {
            preInterventionData: preData,
            postInterventionData: postData,
            allData: this.dataLog,
            success: true
        };
        
        console.log('Intervention result:', JSON.stringify(result, null, 2));
        console.log('=== Interventions Complete ===\n');
        
        return result;
    }
}

// Initialize and export
async function initializeSmartRoom() {
    const room = new SmartRoom();
    await room._initialize();
    return room;
}

/**
 * Simulate the smart room for a given duration and apply interventions
 * @param {number} duration - Duration of the simulation in milliseconds
 * @param {Array} interventions - Array of interventions to apply during the simulation
 * @param {number} interventionTime - Time at which to apply the interventions
 * @returns {Object} Object containing simulation results
 */
function simulateAndGetLatestData(duration, interventions, interventionTime) {
    return new Promise((resolve, reject) => {
        let room = new SmartRoom();
        let simulationResult;
        let completed = false;
        
        const masterTimeout = setTimeout(() => {
            if (!completed) {
                console.error('Master simulation timeout reached');
                cleanup();
                reject(new Error('Simulation master timeout'));
            }
        }, Math.min(duration + 10000, 30000));

        let intervals = [];

        function cleanup() {
            clearTimeout(masterTimeout);
            intervals.forEach(interval => clearInterval(interval));
            if (room) {
                room.stop();
                room = null;
            }
            completed = true;
        }

        try {
            room._initialize()
                .then(() => {
                    room.start();
                    
                    const interventionTimeout = setTimeout(() => {
                        try {
                            console.log("Starting intervention at:", new Date().toISOString());
                            
                            if (interventions && interventions.length > 0) {
                                const interventionCompletionTimeout = setTimeout(() => {
                                    if (!completed) {
                                        console.error('Intervention timed out');
                                        cleanup();
                                        reject(new Error('Intervention timeout'));
                                    }
                                }, 15000);
                                
                                intervals.push(interventionCompletionTimeout);
                                
                                room.applyInterventions(interventions)
                                    .then(result => {
                                        clearTimeout(interventionCompletionTimeout);
                                        
                                        if (!result || !result.preInterventionData || !result.postInterventionData) {
                                            throw new Error('Invalid simulation result structure');
                                        }
                                        
                                        simulationResult = result;
                                        resolve(simulationResult);
                                    })
                                    .catch(error => {
                                        console.error("Intervention error:", error);
                                        reject(error);
                                    })
                                    .finally(() => {
                                        cleanup();
                                    });
                            } else {
                                cleanup();
                                reject(new Error('No interventions provided'));
                            }
                        } catch (error) {
                            console.error("Simulation error:", error);
                            cleanup();
                            reject(error);
                        }
                    }, interventionTime);

                    intervals.push(interventionTimeout);
                })
                .catch(error => {
                    console.error("Initialization error:", error);
                    cleanup();
                    reject(error);
                });
        } catch (error) {
            console.error("Simulation setup error:", error);
            cleanup();
            reject(error);
        }

        process.on('SIGTERM', () => {
            cleanup();
            reject(new Error('Process terminated'));
        });

        process.on('SIGINT', () => {
            cleanup();
            reject(new Error('Process interrupted'));
        });
    });
}

// Export for multirun_main.py
module.exports = { 
    SmartRoom, 
    initializeSmartRoom,
    simulateAndGetLatestData
};

// Standalone execution
if (require.main === module) {
    const args = process.argv.slice(2);

    // --single-step mode: create room, set elapsed time, apply state + intervention, output JSON
    if (args.includes('--single-step')) {
        const getArg = (flag) => {
            const idx = args.indexOf(flag);
            return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : null;
        };
        const elapsedMs = parseInt(getArg('--elapsed-ms') || '0', 10);
        const stateJson = getArg('--state');
        const interventionJson = getArg('--intervention');

        const room = new SmartRoom();
        room._initialize().then(async () => {
            // Position window schedule by pretending simulation started elapsedMs ago
            room.simulationStartTime = Date.now() - elapsedMs;

            // Restore previous state if provided
            if (stateJson) {
                try {
                    const prev = JSON.parse(stateJson);
                    if (prev.temperature !== undefined) room.temperature = prev.temperature;
                    if (prev.humidity !== undefined) room.humidity = prev.humidity;
                    if (prev.airQuality !== undefined) room.airQuality = prev.airQuality;
                } catch (_) {}
            }

            // Apply intervention (action setpoints)
            if (interventionJson) {
                try {
                    const action = JSON.parse(interventionJson);
                    if (action.temperature !== undefined) room.temperature = action.temperature;
                    if (action.humidity !== undefined) room.humidity = action.humidity;
                    if (action.airQuality !== undefined) room.airQuality = action.airQuality;
                } catch (_) {}
            }

            // Run physics update (window state, temperature dynamics, energy, satisfaction)
            await room.updateDependentVariables();

            const s = room.getState();
            // Output result as a single JSON line on stdout
            const output = JSON.stringify({
                temperature: s.temperature,
                humidity: s.humidity,
                airQuality: s.airQuality,
                energyConsumption: s.energyConsumption,
                overallSatisfaction: s.satisfaction,
                pmv: s.pmv,
                windowOpen: s.windowOpen,
                outdoorTemperature: s.outdoorTemperature,
                elapsedTime: s.elapsedTime
            });
            process.stdout.write('RESULT:' + output + '\n');
            await room.cleanup();
            process.exit(0);
        }).catch(err => {
            process.stderr.write('INIT_FAIL:' + err.message + '\n');
            process.exit(1);
        });
    } else {
        // Default: long-running real-time simulation
        console.log('=== Challenge 1: Open Window Simulation ===');
        console.log('Window opens at 30 minutes');
        console.log('This creates TOA -> T1 causal link strengthening\n');

        const room = new SmartRoom();

        room._initialize().then(() => {
            room.start();

            // Run for 90 minutes
            setTimeout(() => {
                room.stop();
                console.log('\n=== Simulation Complete ===');
                console.log('Final state:', room.getState());
                console.log(`Data points collected: ${room.dataLog.length}`);

                // Save data to CSV
                const csvRows = [
                    'timestamp,temperature,humidity,airQuality,pmv,energyConsumption,satisfaction,windowOpen,outdoorTemp,elapsedTime'
                ];

                for (const log of room.dataLog) {
                    csvRows.push([
                        log.timestamp,
                        log.temperature.toFixed(2),
                        log.humidity.toFixed(2),
                        log.airQuality.toFixed(2),
                        log.pmv.toFixed(2),
                        log.energyConsumption.toFixed(2),
                        log.satisfaction.toFixed(2),
                        log.windowOpen ? 1 : 0,
                        log.outdoorTemperature.toFixed(2),
                        log.elapsedTime.toFixed(2)
                    ].join(','));
                }

                const filename = `challenge1_window_${Date.now()}.csv`;
                fs.writeFileSync(filename, csvRows.join('\n'));
                console.log(`Data saved to: ${filename}`);

                room.cleanup().then(() => {
                    process.exit(0);
                });
            }, 90 * 60 * 1000); // 90 minutes
        });
    }
}