// Require Node.js modules
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

// EnergyLookup class with improved data handling
class EnergyLookup {
    constructor() {
        this.data = null;
        this.initialized = false;
    }

    async initialize() {
        if (this.initialized) return;
        
        try {
            // Use promises with fs instead of callbacks
            const fs = require('fs').promises;
            const path = require('path');
            const Papa = require('papaparse');
            
            // Expanded list of possible file locations
            const possiblePaths = [
                'eplus_test_results/epluszsz.csv',
                'epluszsz.csv',
                path.join(__dirname, 'eplus_test_results/epluszsz.csv'),
                path.join(__dirname, 'epluszsz.csv'),
                path.join(process.cwd(), 'epluszsz.csv'),
                path.join(process.cwd(), 'data/epluszsz.csv'),
                path.resolve(__dirname, '..', 'epluszsz.csv')
            ];

            // console.log("Current directory:", process.cwd());
            // console.log("Module directory:", __dirname);
            // console.log("Searching for CSV in:", possiblePaths);
            
            let csvData;
            let loadedPath = '';
            
            // Try each path until one works
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
                    // Continue to next path
                }
            }
            
            // If all paths fail, try loading embedded data
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
            
            // Validate data has proper contents
            if (!this.data || this.data.length === 0) {
                throw new Error('CSV file loaded but no data found');
            }
            
            // Show sample of loaded data
            // console.log("Sample data:", this.data[0]);
            console.log(`Loaded ${this.data.length} rows of energy data`);
            
            this.initialized = true;
            return true;
        } catch (error) {
            console.error('Failed to load energy data:', error);
            // Continue with fallback
            this.initialized = false;
            return false;
        }
    }

    // Add a method to load embedded data as last resort
    async _loadEmbeddedData() {
        try {
            // A minimal set of data points to enable basic operation
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

    calculateEnergy(temperature, humidity, airQuality = 300) {
        console.log(`Energy calculation with data: initialized=${this.initialized}, dataLength=${this.data ? this.data.length : 0}`);
        
        // Find nearest point with better variation
        const nearest = this._findNearestDataPoint(temperature, humidity);
        
        if (nearest) {
            // Debug the data point
            // console.log(`Found nearest data point:`, nearest);
            
            // First try the primary load columns
            let coolLoadKey = Object.keys(nearest).find(k => 
                k.includes('Cool Load') && k.includes('CHICAGO') && !k.includes('No DOAS'));
            
            let heatLoadKey = Object.keys(nearest).find(k => 
                k.includes('Heat Load') && k.includes('CHICAGO') && !k.includes('No DOAS'));
            
            // If not found, try alternative columns
            if (!coolLoadKey || !heatLoadKey || nearest[coolLoadKey] === 0 || nearest[heatLoadKey] === 0) {
                coolLoadKey = Object.keys(nearest).find(k => k.includes('Des Sens Cool Load'));
                heatLoadKey = Object.keys(nearest).find(k => k.includes('Des Heat Load'));
            }
            
            // console.log(`Found load keys: coolLoadKey=${coolLoadKey}, heatLoadKey=${heatLoadKey}`);
            
            if (coolLoadKey && heatLoadKey && 
                typeof nearest[coolLoadKey] === 'number' && 
                typeof nearest[heatLoadKey] === 'number') {

                const coolLoad = nearest[coolLoadKey];
                const heatLoad = nearest[heatLoadKey];
                                
                console.log(`Loads: coolLoad=${coolLoad}, heatLoad=${heatLoad}`);
                                
                const totalLoad = coolLoad + heatLoad;
                // Use actual maximum from data
                const maxLoad = 31645.00; 

                // Calculate ratio, ensuring it's between 0-1 before multiplying by 100
                const loadRatio = Math.max(0, Math.min(1, totalLoad / maxLoad));
                const baseValue = loadRatio * 100;
                                
                // Add air quality influence with capped scaling
                const aqInfluence = Math.min(0.03, airQuality / 10000);
                const aqFactor = 1.0 + aqInfluence;
                                
                // Add small random variation that won't push over 100%
                const remainingHeadroom = 100 - baseValue * aqFactor;
                const randomVariation = Math.min(remainingHeadroom, 2) * Math.random();

                // Calculate final value without needing Math.min
                const energy = baseValue * aqFactor + randomVariation;
                                
                console.log(`Energy calculation: totalLoad=${totalLoad}, baseValue=${baseValue.toFixed(2)}, aqFactor=${aqFactor.toFixed(3)}, randomVariation=${randomVariation.toFixed(2)}, result=${energy.toFixed(2)}`);
                return energy;
            } else {
                console.error("Invalid load values in nearest data point");
            }
        } else {
            console.error("Could not find nearest data point");
        }
        
        // Fallback calculation with detailed logging
        console.log("Using fallback energy calculation");
        return this._calculateFallbackEnergy(temperature, humidity, airQuality);
    }

    _findNearestDataPoint(temperature, humidity) {
        if (!this.initialized || !this.data || !this.data.length) {
            console.error("Energy data not initialized, data source missing");
            return null;
        }
        
        // Log a sample data point to check structure
        // console.log("Sample data point:", this.data[0]);
        
        // Find temperature and humidity columns - handle column name variations
        const tempKeys = Object.keys(this.data[0]).filter(k => 
            k.includes('Temperature') || k.includes('temperature') || k.includes('TEMP'));
        
        const humidKeys = Object.keys(this.data[0]).filter(k => 
            k.includes('Humidity') || k.includes('humidity') || k.includes('HUMID'));
        
        // console.log("Potential temp keys:", tempKeys);
        // console.log("Potential humidity keys:", humidKeys);
        
        const tempKey = tempKeys.find(k => k.includes('Cooling')) || tempKeys[0];
        const humidKey = humidKeys.find(k => k.includes('Cooling')) || humidKeys[0];
        
        if (!tempKey || !humidKey) {
            console.error("Cannot find temperature/humidity columns in data");
            return null;
        }
        
        // console.log(`Using columns: tempKey=${tempKey}, humidKey=${humidKey}`);
        
        // Sort by distance to target conditions
        const sortedByMatch = [...this.data].sort((a, b) => {
            const aDiff = Math.abs(temperature - a[tempKey]) + Math.abs(humidity - a[humidKey]);
            const bDiff = Math.abs(temperature - b[tempKey]) + Math.abs(humidity - b[humidKey]);
            return aDiff - bDiff;
        });
        
        // Get the closest match
        const nearest = sortedByMatch[0];
        // console.log(`Nearest match: T=${nearest[tempKey]}, H=${nearest[humidKey]}`);
        
        return nearest;
    }

    // Replace the fallback energy calculation
    _calculateFallbackEnergy(temperature, humidity, airQuality) {
        // Non-linear components for more realistic response
        const tempDiff = Math.abs(temperature - 22);
        const humidityDiff = Math.abs(humidity - 50);
        const airQualityFactor = airQuality / 500;
        
        // Add time-based variation (simulate daily cycle)
        const hourOfDay = new Date().getHours();
        const timeComponent = Math.sin((hourOfDay - 12) * (Math.PI / 12)) * 5;
        
        // Add small random component for natural variation
        const randomComponent = (Math.random() * 10) - 5;  // ±5%
        
        // Calculate base energy
        const baseEnergy = 25 + 
                        (tempDiff * 3) + 
                        (Math.pow(tempDiff, 1.5) * 0.5) + 
                        (humidityDiff * 0.7) + 
                        (airQualityFactor * 15) +
                        timeComponent + 
                        randomComponent;
        
        // Ensure result is within reasonable bounds
        return Math.min(100, Math.max(10, baseEnergy));
    }
}

/**
 * SmartRoom class representing a simulated smart room environment
 */
class SmartRoom {
    constructor() {
        this.reset();
        this.isRunning = false;
        this.dataLog = [];
        this.lastDataLogTime = 0;
        this.dataLogInterval = 1000;

        // Add process termination handling here
        process.on('SIGTERM', () => {
            console.log('Process termination signal received, cleaning up...');
            this.cleanup();
        });
        
        process.on('SIGINT', () => {
            console.log('Process interrupt signal received, cleaning up...');
            this.cleanup();
        });

        // Initialize energy lookup
        this.energyLookup = new EnergyLookup();
        this.isEnergyInitialized = false;
        this.initialized = false;
        
        // Start initialization immediately and store the promise
        this.initPromise = this.energyLookup.initialize().then(() => {
            this.isEnergyInitialized = true;
            console.log("Energy lookup successfully initialized");
        }).catch(err => {
            console.error("Failed to initialize energy lookup:", err);
        });

        // for psychrometric calculator
        this.psychroCalc = new PsychrometricCalculator();
        this.airSpeed = 0.1; // m/s
        this.meanRadiantTemp = 22; // °C
        this.activePromises = new Set(); // Initialize the activePromises Set

        // for energyplus integration
        this.eplusProcess = null;
        this.eplusQueue = Promise.resolve();
    }

    reset() {
        this.temperature = 22;       // °C
        this.humidity = 50;          // %
        this.airQuality = 300;        // AQI
        this.airSpeed = 0.1;         // m/s
        this.energyConsumption = 50; // %
        this.overallSatisfaction = 75; // %
        this.timestamp = new Date();
        this.dataLog = [];
    }

    // Add initialization method
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

    async updateDependentVariables() {
        console.log('\n=== Updating Dependent Variables ===');
        
        try {
            // Wait for initialization to complete if not already done
            if (!this.initialized) {
                console.log("Waiting for initialization to complete...");
                await this.initPromise;
            }
            // Ensure values are within bounds
            this.temperature = Math.max(18, Math.min(30, Number(this.temperature)));
            this.humidity = Math.max(30, Math.min(70, Number(this.humidity)));
            this.airQuality = Math.max(0, Math.min(500, Number(this.airQuality)));
    
            // // Use proper Promise.all with error handling
            // const results = await Promise.allSettled([
            //     this.updateEnergyConsumption(),
            //     this.updateOverallSatisfaction()
            // ]);
            // Create timeout-protected promises
            const energyPromise = this._withTimeout(
                this.updateEnergyConsumption(),
                10000,
                'Energy calculation timed out'
            );
            
            const satisfactionPromise = this._withTimeout(
                this.updateOverallSatisfaction(),
                10000,
                'Satisfaction calculation timed out'
            );
            
            // Use Promise.allSettled to handle partial failures
            const results = await Promise.allSettled([
                energyPromise,
                satisfactionPromise
            ]);
            
            // Process results with error handling
            results.forEach((result, index) => {
                if (result.status === 'rejected') {
                    console.error(`Calculation ${index} failed:`, result.reason);
                    // Apply fallback values if needed
                    if (index === 0) this._updateFallbackEnergy();
                    if (index === 1) this.overallSatisfaction = this._updateFallbackSatisfaction();
                }
            });
    
            const invalidState = this.validateState();
            if (invalidState) {
                console.error('Invalid state detected:', invalidState);
                this.repairState();
            }
    
            return {
                temperature: this.temperature,
                humidity: this.humidity,
                airQuality: this.airQuality,
                energyConsumption: this.energyConsumption,
                overallSatisfaction: this.overallSatisfaction.toFixed(2)
            };
        } catch (error) {
            console.error('Error updating variables:', error);
            this._updateFallbackValues();
            return {
                temperature: this.temperature,
                humidity: this.humidity,
                airQuality: this.airQuality,
                energyConsumption: this.energyConsumption,
                overallSatisfaction: this.overallSatisfaction
            };
        }
    }

    _withTimeout(promise, timeoutMs, errorMessage) {
        // Create a promise that rejects after the timeout
        const timeoutPromise = new Promise((_, reject) => {
            const id = setTimeout(() => {
                clearTimeout(id);
                reject(new Error(errorMessage));
            }, timeoutMs);
        });
        
        // Return a promise that resolves to the result of the original promise
        // or rejects with the timeout error if the original promise takes too long
        return Promise.race([promise, timeoutPromise]);
    }

    async updateEnergyConsumption() {
        // let energyLookup = new EnergyLookup();
        console.log('Updating Energy'); 
        try {
            // Ensure method returns a promise
            return new Promise((resolve, reject) => {
                try {
                    console.log(`Using temp ${this.temperature}, air ${this.airQuality} and humidity ${this.humidity}`);

                    this.energyConsumption = this.energyLookup.calculateEnergy(
                        this.temperature,
                        this.humidity,
                        this.airQuality
                    );
                    console.log(`Updated Energy: ${this.energyConsumption}`);
                    resolve(this.energyConsumption);
                } catch (error) {
                    reject(error);
                }
            });
        } catch (error) {
            console.error('Energy consumption calculation failed:', error);
            throw error;
        }
    }

    _updateFallbackEnergy() {
        console.log('Using fallback energy calculation');
        this.energyConsumption = this.energyLookup._calculateFallbackEnergy(
            this.temperature,
            this.humidity,
            this.airQuality
        );
    }

    _updateFallbackValues() {
        this._updateFallbackEnergy();
        this._updateFallbackSatisfaction();
    }

    cleanup() {
        if (this.psychroCalc) {
            this.psychroCalc.cleanup();
        }
        
        // Reset energy lookup
        if (this.energyLookup) {
            this.energyLookup = new EnergyLookup();
            this.energyInitPromise = this.energyLookup.initialize();
        }
        
        // Cancel any in-progress calculations
        for (const promise of this.activePromises) {
            if (promise.cancel) {
                promise.cancel();
            }
        }
        this.activePromises.clear();
    }

    /**
     * Update overall satisfaction based on current state
     */
    updateOverallSatisfaction() {
        console.log('Updating Satisfaction...'); 
        // Only calculate PMV if conditions have changed significantly
        const conditionsKey = `${this.temperature.toFixed(1)}_${this.humidity.toFixed(1)}_${this.airSpeed.toFixed(2)}`;
        if (this._lastConditionsKey === conditionsKey) {
            return;
        }
        
        this._lastConditionsKey = conditionsKey;   

        console.log('Calculating Satisfaction');
        let timeoutId;
        const promise = new Promise((resolve, reject) => {
            try {
                // Log initial calculation parameters
                console.log(`Calculating Overall Satisfaction:
                    Temperature: ${this.temperature}°C
                    Humidity: ${this.humidity}%
                    Air Speed: ${this.airSpeed} m/s
                    Air Quality: ${this.airQuality} AQI
                    Current Energy Consumption: ${this.energyConsumption}%`);

                // Set a timeout to prevent hanging calculations
                timeoutId = setTimeout(() => {
                    console.warn('Satisfaction calculation timed out, using fallback');
                    this._updateFallbackSatisfaction();
                    resolve(this.overallSatisfaction);
                }, 5000);

                // Ensure consistent promise handling
                this.psychroCalc.calculatePMV(
                    this.temperature, 
                    this.humidity, 
                    this.airSpeed
                )
                .then(comfortResults => {
                    clearTimeout(timeoutId);
                    
                    // Satisfaction calculation with direct humidity comfort
                    // (ASHRAE 55: optimal 40-60% RH, discomfort outside 30-70%)
                    const pmvSatisfaction = 100 - (Math.abs(comfortResults.pmv) * 16.67);
                    const ppdSatisfaction = 100 - comfortResults.ppd;
                    const aqiSatisfaction = Math.max(0, 100 - (this.airQuality / 5));
                    const humiditySatisfaction = Math.max(0, 100 - Math.abs(this.humidity - 50) * 2.5);
                    const energyPenalty = this.energyConsumption * 0.2;

                    console.log('Satisfaction Component Breakdown:', {
                        pmvSatisfaction: pmvSatisfaction.toFixed(2),
                        ppdSatisfaction: ppdSatisfaction.toFixed(2),
                        humiditySatisfaction: humiditySatisfaction.toFixed(2),
                        aqiSatisfaction: aqiSatisfaction.toFixed(2),
                        energyPenalty: energyPenalty.toFixed(2)
                    });

                    this.overallSatisfaction = Math.max(0, Math.min(100,
                        0.30 * pmvSatisfaction +
                        0.25 * ppdSatisfaction +
                        0.20 * humiditySatisfaction +
                        0.15 * aqiSatisfaction -
                        0.10 * energyPenalty
                    ));

                    console.log(`Calculated Overall Satisfaction: ${this.overallSatisfaction.toFixed(2)}%`);
                    resolve(this.overallSatisfaction);
                })
                .catch(error => {
                    clearTimeout(timeoutId);
                    
                    // Detailed error logging
                    console.error('Satisfaction Calculation Detailed Error:', {
                        message: error.message,
                        stack: error.stack,
                        inputParameters: {
                            temperature: this.temperature,
                            humidity: this.humidity,
                            airSpeed: this.airSpeed,
                            airQuality: this.airQuality,
                            energyConsumption: this.energyConsumption
                        }
                    });

                    // Fallback to default calculation
                    this._updateFallbackSatisfaction();
                    
                    console.warn(`Fallback Satisfaction Calculation Used: ${this.overallSatisfaction}%`);

                    reject(error);
                });
            } catch (error) {
                clearTimeout(timeoutId);
                this._updateFallbackSatisfaction();
                reject(error);
            }
        });
        
        // Add a cancel method to the promise
        promise.cancel = () => {
            if (timeoutId) {
                clearTimeout(timeoutId);
            }
        };
        
        // Track the promise
        this.activePromises.add(promise);
        
        // Clean up when the promise is settled
        promise.finally(() => {
            this.activePromises.delete(promise);
        });
        
        return promise;
    }

    _updateFallbackSatisfaction() {
        const tempComfort = Math.max(0, 100 - (Math.abs(22 - this.temperature) * 5));
        const humidityComfort = Math.max(0, 100 - (Math.abs(50 - this.humidity) * 2));
        const airQualityComfort = Math.max(0, 100 - (this.airQuality * 0.1));
        const energyPenalty = Math.min(20, this.energyConsumption * 0.2);

        this.overallSatisfaction = Math.max(0, Math.min(100,
            0.4 * tempComfort +
            0.3 * humidityComfort +
            0.2 * airQualityComfort -
            0.1 * energyPenalty
        ));
        
        console.log(`Fallback satisfaction: ${this.overallSatisfaction}`);
        return this.overallSatisfaction;
    }

    validateState() {
        // Define valid ranges for all variables
        const requiredRanges = {
            temperature: [18, 30],
            humidity: [30, 70],
            airQuality: [0, 500],
            airSpeed: [0, 0.3],
            energyConsumption: [0, 100],
            overallSatisfaction: [0, 100]
        };

        const invalidValues = {};
        
        // Check each variable against its valid range
        for (const [key, [min, max]] of Object.entries(requiredRanges)) {
            const value = this[key];
            
            // First check for NaN or invalid values
            if (value === null || value === undefined || isNaN(value)) {
                console.error(`Invalid ${key} value: ${value}`);
                invalidValues[key] = value;
                continue;
            }
            
            // Then check range
            const numValue = Number(value);
            if (numValue < min || numValue > max) {
                console.error(`${key} out of range: ${numValue} (valid: ${min}-${max})`);
                invalidValues[key] = numValue;
            }
        }

        return Object.keys(invalidValues).length > 0 ? invalidValues : null;
    }

    repairState() {
        console.log('Repairing invalid state...');
        
        // Default values for recovery
        const defaults = {
            temperature: 22,
            humidity: 50,
            airQuality: 300,
            airSpeed: 0.1,
            energyConsumption: 50,
            overallSatisfaction: 75
        };

        // Apply defaults to any invalid values
        for (const [key, defaultValue] of Object.entries(defaults)) {
            if (this[key] === null || this[key] === undefined || isNaN(this[key])) {
                console.log(`Repairing ${key} from ${this[key]} to ${defaultValue}`);
                this[key] = defaultValue;
            }
        }

        // Ensure values are within bounds
        this.temperature = Math.max(18, Math.min(30, this.temperature));
        this.humidity = Math.max(30, Math.min(70, this.humidity));
        this.airQuality = Math.max(0, Math.min(500, this.airQuality));
        this.airSpeed = Math.max(0, Math.min(0.3, this.airSpeed));
        
        // Update dependent variables after repair
        this.updateEnergyConsumption();
        this.updateOverallSatisfaction();
    }

    /**
     * Log current state of the smart room
     * @param {boolean} interventionApplied - Whether an intervention was just applied
     */
    logData(interventionApplied = false) {
        const entry = {
            Timestamp: new Date().toISOString(),
            Temperature: this.temperature,
            Humidity: this.humidity,
            AirQuality: this.airQuality,
            EnergyConsumption: this.energyConsumption,
            OverallSatisfaction: this.overallSatisfaction,
            InterventionApplied: interventionApplied
        };

        // Use circular buffer approach for data logging
        if (this.dataLog.length >= 1000) {
            this.dataLog.shift(); // Remove oldest entry
        }

        this.dataLog.push(entry);
    }

    getCurrentTemperature() { return this.temperature; }
    getCurrentHumidity() { return this.humidity; }
    getCurrentAQ() { return this.airQuality; }

    /**
     * Set temperature and update related states
     * @param {number} value - New temperature value
     */
    setTemperature(value) {
        this.temperature = value;
        this.updateDependentVariables();
    }

    /**
     * Set humidity and update related states
     * @param {number} value - New humidity value
     */
    setHumidity(value) {
        this.humidity = value;
        this.updateDependentVariables();
    }

    /**
     * Set air quality and update related states
     * @param {number} value - New air quality value
     */
    setAirQuality(value) {
        this.airQuality = value;
        this.updateDependentVariables();
    }

    /**
     * Apply multiple interventions to the smart room
     * @param {Array} interventions - Array of intervention details
     */
    async applyInterventions(interventions) {
        console.log('\n=== Starting Intervention Sequence ===');
        console.log(`Time: ${new Date().toISOString()}`);
        console.log(`Applying interventions: ${JSON.stringify(interventions)}`);

        // Ensure initialization is complete
        if (!this.initialized) {
            console.log("Waiting for initialization before intervention...");
            await this.initPromise;
        }

        // Apply all interventions first
        return new Promise(async (resolve, reject) => {
            // Set timeout for the entire intervention process
            const interventionTimeout = setTimeout(() => {
                reject(new Error('Intervention process timed out'));
            }, 10000); // 10 second timeout

            try {
                // Store pre-intervention state
                globalPreInterventionState = {
                    Timestamp: new Date().toISOString(),
                    Temperature: this.temperature,
                    Humidity: this.humidity,
                    AirQuality: this.airQuality,
                    EnergyConsumption: this.energyConsumption,
                    OverallSatisfaction: this.overallSatisfaction,
                    InterventionApplied: false
                };
                
                let latestLoggedValues = { ...globalPreInterventionState };
                console.log("Pre-intervention state captured:", JSON.stringify(globalPreInterventionState)); 

                // Apply all interventions
                interventions.forEach(intervention => {
                    if (!['Temperature', 'Humidity', 'AirQuality'].includes(intervention.variable)) {
                        throw new Error(`Invalid intervention variable: ${intervention.variable}`);
                    }
                    if (!['set', 'increase', 'decrease'].includes(intervention.action)) {
                        throw new Error(`Invalid intervention action: ${intervention.action}`);
                    }

                    // Get current value, ensuring it's a number
                    const varName = intervention.variable.toLowerCase();
                    const preValue = Number(this[varName === 'airquality' ? 'airQuality' : varName]);
                    
                    if (isNaN(preValue)) {
                        throw new Error(`Current ${varName} value is invalid: ${preValue}`);
                    }

                    // Calculate new value
                    let newValue;
                    if (intervention.action === 'set') {
                        newValue = Number(intervention.value);
                    } else {
                        const change = intervention.action === 'increase' ? 
                            Number(intervention.value) : -Number(intervention.value);
                        newValue = preValue + change;
                    }

                    // Validate new value
                    if (isNaN(newValue)) {
                        throw new Error(`Calculated ${varName} value is invalid: ${newValue}`);
                    }

                    // Apply bounds based on variable
                    if (varName === 'airquality' || varName === 'airQuality' || varName === 'AirQuality') {
                        newValue = Math.max(0, Math.min(500, newValue));
                        this.airQuality = newValue;
                    } else if (varName === 'temperature' || varName === 'Temperature') {
                        newValue = Math.max(18, Math.min(30, newValue));
                        this.temperature = newValue;
                    } else if (varName === 'humidity' || varName === 'Humidity') {
                        newValue = Math.max(30, Math.min(70, newValue));
                        this.humidity = newValue; 
                    }

                    console.log(`Applied ${intervention.variable}: ${preValue} -> ${newValue}`);
                });

                // Update dependent variables and capture post-intervention state
                await this.updateDependentVariables();
                
                const invalidState = this.validateState();
                if (invalidState) {
                    console.error('Invalid state after intervention:', invalidState);
                    this.repairState();
                }

                globalPostInterventionState = {
                    Timestamp: new Date().toISOString(),
                    Temperature: this.temperature,
                    Humidity: this.humidity,
                    AirQuality: this.airQuality,
                    EnergyConsumption: this.energyConsumption,
                    OverallSatisfaction: this.overallSatisfaction,
                    InterventionApplied: true
                };
    
                console.log("Post-intervention state captured:", JSON.stringify(globalPostInterventionState));
                this.logData(true);
                
                clearTimeout(interventionTimeout);
                resolve({
                    preInterventionData: globalPreInterventionState,
                    postInterventionData: globalPostInterventionState,
                    allData: this.dataLog
                });
            } catch (error) {
                clearTimeout(interventionTimeout);
                console.error('Error during intervention:', error);
                reject(error);
            }
        });
    }

    // Simulation control methods
    start() { this.isRunning = true; }
    stop() { this.isRunning = false; }

    getLatestData() { return this.dataLog[this.dataLog.length - 1]; }
    getAllData() { return this.dataLog; }
}

/**
 * Initialize a new SmartRoom instance
 * @returns {string} JSON string representation of a new SmartRoom instance
 */
function initializeSmartRoom() {
    return JSON.stringify(new SmartRoom());
}

function saveSimulationDataToCSV(data, filename) {
    const headers = ['Timestamp', 'Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction', 'InterventionApplied'];
    const csvContent = [
        headers.join(','),
        ...data.map(entry => 
            [
                entry.Timestamp,
                entry.Temperature,
                entry.Humidity,
                entry.AirQuality,
                entry.EnergyConsumption,
                entry.OverallSatisfaction,
                entry.InterventionApplied
            ].join(',')
        )
    ].join('\n');

    const filePath = path.join(__dirname, filename);
    fs.writeFileSync(filePath, csvContent);
    console.log(`Simulation data saved to ${filePath}`);
}

/**
 * Generate simulation data at regular intervals
 * @param {number} duration - Total duration in milliseconds
 * @param {number} interval - Interval between data points in milliseconds
 * @param {string} filename - Optional filename to save CSV (default: generated with timestamp)
 * @returns {Promise<string>} Promise resolving to the filename where data is saved
 */
function generateSimulationData(duration = 24 * 60 * 60 * 1000, interval = 15 * 60 * 1000, filename = null) {
    return new Promise(async (resolve, reject) => {
        try {
            console.log(`Generating simulation data for ${duration/3600000} hours at ${interval/60000} minute intervals`);
            
            const room = new SmartRoom();
            const data = [];
            const startTime = Date.now();
            const endTime = startTime + duration;
            let currentTime = startTime;
            
            // Initialize the simulation
            await room._initialize();
            console.log("Simulation initialized");
            
            // Generate data points at the specified interval
            while (currentTime < endTime) {
                // Small random changes to control variables
                room.temperature += (Math.random() * 2 - 1) * 0.5;
                room.temperature = Math.max(18, Math.min(30, room.temperature));
                
                room.humidity += (Math.random() * 2 - 1) * 2;
                room.humidity = Math.max(30, Math.min(70, room.humidity));
                
                room.airQuality += (Math.random() * 2 - 1) * 10;
                room.airQuality = Math.max(0, Math.min(500, room.airQuality));
                
                // Calculate dependent variables
                await room.updateDependentVariables();
                
                // Record data point
                data.push({
                    Timestamp: new Date(currentTime).toISOString(),
                    Temperature: room.temperature,
                    Humidity: room.humidity,
                    AirQuality: room.airQuality,
                    EnergyConsumption: room.energyConsumption,
                    OverallSatisfaction: room.overallSatisfaction
                });
                
                // Move to the next time interval
                currentTime += interval;
                if (data.length % 100 === 0) {
                    console.log(`Generated ${data.length} data points`);
                }
            }
            
            console.log(`Generated ${data.length} data points`);
            
            // Export to CSV
            // const csvContent = generateCsvContent(data);
            // const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            // const outputFilename = filename || `simulation_data_${timestamp}.csv`;
            
            // const fs = require('fs');
            // fs.writeFileSync(outputFilename, csvContent);
            // console.log(`Data exported to ${outputFilename}`);
            
            // Clean up
            room.cleanup();
            
            resolve(outputFilename);
        } catch (error) {
            console.error("Error generating simulation data:", error);
            reject(error);
        }
    });
}

function generateCsvContent(data) {
    if (!data || data.length === 0) return '';
    
    // Get headers from the first data point
    const headers = Object.keys(data[0]);
    
    // Create CSV content
    const headerRow = headers.join(',');
    const dataRows = data.map(row => 
        headers.map(header => {
            const value = row[header];
            // Quote string values that contain commas
            return typeof value === 'string' && value.includes(',') 
                ? `"${value}"` 
                : value;
        }).join(',')
    );
    
    return [headerRow, ...dataRows].join('\n');
}

// If this script is run directly (not imported as a module)
if (require.main === module) {
    const args = process.argv.slice(2);

    // --single-step mode: create room, apply state + intervention, output JSON
    if (args.includes('--single-step')) {
        const getArg = (flag) => {
            const idx = args.indexOf(flag);
            return idx !== -1 && idx + 1 < args.length ? args[idx + 1] : null;
        };
        const stateJson = getArg('--state');
        const interventionJson = getArg('--intervention');

        const room = new SmartRoom();
        room._initialize().then(async () => {
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

            // Run physics update (energy, satisfaction via PMV)
            await room.updateDependentVariables();

            const output = JSON.stringify({
                temperature: room.temperature,
                humidity: room.humidity,
                airQuality: room.airQuality,
                energyConsumption: room.energyConsumption,
                overallSatisfaction: room.overallSatisfaction,
            });
            process.stdout.write('RESULT:' + output + '\n');
            room.cleanup();
            process.exit(0);
        }).catch(err => {
            process.stderr.write('INIT_FAIL:' + err.message + '\n');
            process.exit(1);
        });
    } else {
        console.log("Running simulation data generation...");

        // Generate 24 hours of data at 15-minute intervals
        generateSimulationData(24 * 60 * 60 * 1000, 15 * 60 * 1000)
            .then(filename => {
                console.log(`Simulation completed successfully. Data saved to ${filename}`);
                process.exit(0);
            })
            .catch(error => {
                console.error("Simulation failed:", error);
                process.exit(1);
            });
    }
}

/**
 * Simulate the smart room for a given duration and apply multiple interventions
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
        
        // Add stricter timeout safety
        const masterTimeout = setTimeout(() => {
            if (!completed) {
                console.error('Master simulation timeout reached, forcing cleanup');
                cleanup();
                reject(new Error('Simulation master timeout'));
            }
        }, Math.min(duration + 10000, 30000)); // Max 30 seconds or duration+10s

        let intervals = [];

        function cleanup() {
            clearTimeout(masterTimeout);
            intervals.forEach(interval => clearInterval(interval));
            if (room) {
                room.cleanup();
                room.stop();
                room = null;
            }
            completed = true;
        }

        try {
            const startTime = Date.now();
            room.start();

            // Apply intervention with timeout
            const interventionTimeout = setTimeout(() => {
                try {
                    console.log("Starting intervention at:", new Date().toISOString());
                    
                    if (interventions && interventions.length > 0) {
                        // Set timeout for intervention completion
                        const interventionCompletionTimeout = setTimeout(() => {
                            if (!completed) {
                                console.error('Intervention timed out');
                                cleanup();
                                reject(new Error('Intervention timeout'));
                            }
                        }, 15000); // 15 second timeout for intervention
                        
                        intervals.push(interventionCompletionTimeout);
                        
                        // Use Promise from applyInterventions
                        room.applyInterventions(interventions)
                            .then(result => {
                                clearTimeout(interventionCompletionTimeout);
                                
                                // Validate the simulation result
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
        } catch (error) {
            console.error("Simulation setup error:", error);
            cleanup();
            reject(error);
        }

        // Handle process termination
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

// Add this to the module exports
function cleanupProcesses() {
    console.log("Cleaning up all processes...");
    try {
        // Find all Node.js child processes
        const { execSync } = require('child_process');
        if (process.platform === 'win32') {
            execSync('taskkill /F /IM node.exe /T', { stdio: 'ignore' });
        } else {
            // On Linux/Unix, we need a more targeted approach
            const psutil = require('psutil');
            const currentPid = process.pid;
            
            const killChildren = (pid) => {
                try {
                    const output = execSync(`pgrep -P ${pid}`).toString();
                    const childPids = output.split('\n').filter(Boolean).map(Number);
                    
                    childPids.forEach(childPid => {
                        killChildren(childPid);
                        process.kill(childPid, 'SIGKILL');
                    });
                } catch (e) {
                    // No children or error
                }
            };
            
            killChildren(currentPid);
        }
    } catch (error) {
        console.error("Error during process cleanup:", error);
    }
}

// Add to module exports
module.exports = { 
    SmartRoom, 
    initializeSmartRoom, 
    simulateAndGetLatestData,
    saveSimulationDataToCSV,
    cleanupProcesses,
    generateSimulationData
};