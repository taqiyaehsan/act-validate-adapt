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

        // For psychrometric calculator
        this.psychroCalc = new PsychrometricCalculator();
        this.airSpeed = 0.1; // m/s
        this.meanRadiantTemp = 22; // °C
        this.activePromises = new Set(); // Initialize the activePromises Set

        // For energyplus integration
        this.eplusProcess = null;
        this.eplusQueue = Promise.resolve();
        
        // Hidden variables with realistic defaults
        this.hvacEfficiency = 0.85;        // SEER rating converted to efficiency (13 SEER ≈ 0.85)
        this.buildingEnvelope = {
            wallRValue: 13,                // R-13 typical wall insulation (2x4 framing)
            ceilingRValue: 30,             // R-30 typical attic insulation
            windowUFactor: 0.35            // U-factor for standard double-pane windows
        };
        this.windowState = "closed";       // Window state (open/closed/partially)
        this.occupancy = {
            count: 2,                      // Number of occupants
            activity: 1.2                  // Metabolic rate (1.2 = seated, light activity)
        };
        this.outdoorConditions = {
            temperature: 20,               // °C
            humidity: 50,                  // %
            windSpeed: 2.5                 // m/s
        };
        this.timeOfDay = new Date().getHours();
        this.electricityRate = 0.15;       // $/kWh standard rate
        this.hvacMode = "auto";            // heating/cooling/auto
        this.lastUpdateTime = Date.now();
        
        // Add ASHRAE comfort standards
        this.comfortStandards = {
            winter: { 
                tempMin: 20.0, 
                tempMax: 23.5, 
                humidityMin: 30, 
                humidityMax: 60 
            },
            summer: { 
                tempMin: 23.0, 
                tempMax: 26.0, 
                humidityMin: 30, 
                humidityMax: 60 
            }
        };
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
    
            // Occasionally update hidden variables to create realistic variation
            if (Math.random() < 0.05) {  // 5% chance per update
                this._updateHiddenVariables();
            }
        
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
                    if (index === 0) this._calculateFallbackEnergy();
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

    _updateHiddenVariables() {
        const currentTime = Date.now();
        const elapsedMinutes = (currentTime - this.lastUpdateTime) / 60000;
        this.lastUpdateTime = currentTime;
        
        // Current date/time info
        const date = new Date();
        const hour = date.getHours();
        const dayOfWeek = date.getDay(); // 0 = Sunday, 6 = Saturday
        const isWeekend = (dayOfWeek === 0 || dayOfWeek === 6);
        const month = date.getMonth(); // 0 = January
        
        // Seasonal factors
        const isSummer = (month >= 4 && month <= 9);
        const isWinter = (month <= 1 || month >= 10);
        
        // Update outdoor conditions
        this._updateOutdoorConditions(hour, isSummer, isWinter);
        
        // HVAC efficiency - degrades slightly over time (0.01% per day)
        // Efficiency would be reset on maintenance
        const dailyDegradation = 0.0001 * (elapsedMinutes / 1440);
        this.hvacEfficiency = Math.max(0.6, this.hvacEfficiency - dailyDegradation);
        
        // Update window state based on outdoor conditions & time of day
        this._updateWindowState(hour, isWeekend);
        
        // Update occupancy based on realistic patterns
        this._updateOccupancy(hour, isWeekend);
        
        // Time of day is updated
        this.timeOfDay = hour;
        
        // Update electricity rates based on time of use
        this.electricityRate = this._calculateElectricityRate(hour);
    
        console.log('Updated hidden variables:', {
            outdoorTemp: this.outdoorConditions.temperature,
            hvacEfficiency: this.hvacEfficiency,
            windowState: this.windowState,
            occupancyCount: this.occupancy.count,
            timeOfDay: this.timeOfDay,
            electricityRate: this.electricityRate
        });
    }
    
    _updateOutdoorConditions(hour, isSummer, isWinter) {
        // Daily temperature cycle with season-appropriate baseline
        const baseTemp = isSummer ? 25 : (isWinter ? 5 : 15);
        const amplitude = isSummer ? 8 : (isWinter ? 5 : 6);
        const peakHour = 14; // 2 PM peak temperature
        const hourOffset = (hour - peakHour + 24) % 24;
        const hourFactor = Math.cos(hourOffset / 24 * 2 * Math.PI);
        
        // Temperature with small random variation
        this.outdoorConditions.temperature = baseTemp + amplitude * hourFactor + 
                                          (Math.random() * 2 - 1);
        
        // Update humidity (inversely related to temperature)
        const baseHumidity = isSummer ? 65 : (isWinter ? 40 : 55);
        this.outdoorConditions.humidity = baseHumidity - 10 * hourFactor + 
                                       (Math.random() * 10 - 5);
        this.outdoorConditions.humidity = Math.max(20, Math.min(95, this.outdoorConditions.humidity));
        
        // Update wind speed
        this.outdoorConditions.windSpeed = 2 + Math.random() * 3;
    }
    
    _updateWindowState(hour, isWeekend) {
        // Realistic window behavior:
        // 1. More likely to open windows in morning/evening when weather is nice
        // 2. Windows stay closed during extreme outdoor temperatures 
        // 3. More window opening on weekends
        
        const outdoorTemp = this.outdoorConditions.temperature;
        const isPleasantOutside = (outdoorTemp > 18 && outdoorTemp < 26);
        const isMorningOrEvening = (hour >= 7 && hour <= 10) || (hour >= 17 && hour <= 20);
        
        // Probabilities based on conditions
        let openProbability = 0;
        
        if (isPleasantOutside) {
            if (isMorningOrEvening) {
                openProbability = isWeekend ? 0.4 : 0.2;
            } else {
                openProbability = isWeekend ? 0.2 : 0.1;
            }
        } else {
            openProbability = 0.05; // Very unlikely when weather is bad
        }
        
        // Update window state
        const rand = Math.random();
        if (rand < openProbability) {
            this.windowState = "open";
        } else if (rand < openProbability + 0.1) {
            this.windowState = "partially";
        } else {
            this.windowState = "closed";
        }
    }
    
    _updateOccupancy(hour, isWeekend) {
        // Different occupancy patterns for weekdays vs weekends
        if (isWeekend) {
            // Weekend patterns
            if (hour >= 0 && hour < 8) {
                // Night time: sleeping
                this.occupancy.count = Math.floor(Math.random() * 2) + 1;
                this.occupancy.activity = 0.8; // Resting
            } else if (hour >= 8 && hour < 11) {
                // Morning: variable occupancy
                this.occupancy.count = Math.floor(Math.random() * 3) + 1;
                this.occupancy.activity = 1.2; // Light activity
            } else if (hour >= 11 && hour < 22) {
                // Day/evening: more people at home
                this.occupancy.count = Math.floor(Math.random() * 4) + 1;
                this.occupancy.activity = 1.4; // Active
            } else {
                // Late evening: winding down
                this.occupancy.count = Math.floor(Math.random() * 3) + 1;
                this.occupancy.activity = 1.0; // Seated
            }
        } else {
            // Weekday patterns
            if (hour >= 0 && hour < 6) {
                // Night time: sleeping
                this.occupancy.count = Math.floor(Math.random() * 2) + 1;
                this.occupancy.activity = 0.8; // Resting
            } else if (hour >= 6 && hour < 9) {
                // Morning rush: getting ready
                this.occupancy.count = Math.floor(Math.random() * 3) + 1;
                this.occupancy.activity = 1.6; // Moving around
            } else if (hour >= 9 && hour < 17) {
                // Work hours: fewer people
                this.occupancy.count = Math.floor(Math.random() * 2);
                this.occupancy.activity = 1.0; // Low activity
            } else if (hour >= 17 && hour < 22) {
                // Evening: home time
                this.occupancy.count = Math.floor(Math.random() * 3) + 1;
                this.occupancy.activity = 1.3; // Moderate activity
            } else {
                // Late evening: winding down
                this.occupancy.count = Math.floor(Math.random() * 2) + 1;
                this.occupancy.activity = 1.0; // Seated
            }
        }
    }
    
    _calculateElectricityRate(hour) {
        // Time of use electricity rates
        if (hour >= 0 && hour < 7) {
            return 0.08; // Off-peak
        } else if ((hour >= 7 && hour < 16) || (hour >= 21 && hour < 24)) {
            return 0.12; // Mid-peak
        } else {
            return 0.20; // On-peak (4pm-9pm)
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
        try {
            // Calculate thermal load based on building physics
            const indoorTemp = this.temperature;
            const outdoorTemp = this.outdoorConditions.temperature;
            const tempDiff = Math.abs(indoorTemp - outdoorTemp);
            
            // Calculate U-value based on R-values (conversion factor)
            const wallUValue = 1 / this.buildingEnvelope.wallRValue;
            const ceilingUValue = 1 / this.buildingEnvelope.ceilingRValue;
            const windowUValue = this.buildingEnvelope.windowUFactor;
            
            // Simplified building dimensions (m²)
            const wallArea = 80;     // 20m perimeter x 4m height
            const windowArea = 12;   // 20% of wall area
            const ceilingArea = 50;  // floor area
            
            // Calculate heat transfer (W) through each surface
            const wallTransfer = wallUValue * (wallArea - windowArea) * tempDiff;
            const windowTransfer = windowUValue * windowArea * tempDiff;
            const ceilingTransfer = ceilingUValue * ceilingArea * tempDiff;
            
            // Add infiltration load based on window state
            let infiltrationLoad = 0;
            if (this.windowState === "open") {
                // Air exchange calculation based on wind speed and opening size
                const airChangeRate = 3.0;  // air changes per hour
                infiltrationLoad = airChangeRate * 50 * 1.2 * 1000 * tempDiff / 3600;
            } else if (this.windowState === "partially") {
                infiltrationLoad = 1.0 * 50 * 1.2 * 1000 * tempDiff / 3600;
            } else {
                // Closed window still has some infiltration
                infiltrationLoad = 0.2 * 50 * 1.2 * 1000 * tempDiff / 3600;
            }
            
            // Add latent load from humidity difference
            const humidityDiff = Math.abs(this.humidity - this.outdoorConditions.humidity);
            const latentLoad = 0.83 * humidityDiff * 10;
            
            // Add ventilation load based on air quality requirements
            // ASHRAE 62.1 ventilation standard - 5 CFM/person + 0.06 CFM/ft²
            const ventilationLoad = (5 * this.occupancy.count + 0.06 * 538) * tempDiff * 0.3;
            
            // Add occupant heat gain (W)
            const occupantLoad = this.occupancy.count * 75 * this.occupancy.activity;
            
            // Calculate total thermal load
            const totalThermalLoad = wallTransfer + windowTransfer + ceilingTransfer + 
                                  infiltrationLoad + latentLoad + ventilationLoad - occupantLoad;
            
            // Convert thermal load to energy consumption considering HVAC efficiency
            const hvacLoad = Math.abs(totalThermalLoad) / this.hvacEfficiency;
            
            // Add air quality control energy (proportional to AQI)
            const aqLoad = this.airQuality * 0.5;
            
            // Scale to percentage (0-100) for output
            const maxLoad = 10000;  // Maximum expected load in W
            let energyPercentage = (hvacLoad + aqLoad) / maxLoad * 100;
            
            // Apply time-of-use factor
            const isPeakHour = (this.timeOfDay >= 14 && this.timeOfDay <= 19);
            const timeOfUseFactor = isPeakHour ? 1.25 : 1.0;
            
            // Apply final energy calculation
            this.energyConsumption = Math.max(0, Math.min(100, energyPercentage * timeOfUseFactor));
            console.log(`Energy calculation: ${energyPercentage.toFixed(2)}% * ${timeOfUseFactor} = ${this.energyConsumption.toFixed(2)}%`);
            
            return this.energyConsumption;
        } catch (error) {
            console.error('Energy consumption calculation failed:', error);
            this._updateFallbackEnergy();
            return this.energyConsumption;
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
                    
                    // PMV/PPD to satisfaction conversion
                    const pmvSatisfaction = 100 - (Math.abs(comfortResults.pmv) * 16.67);
                    const ppdSatisfaction = 100 - comfortResults.ppd;
                    
                    // Air quality satisfaction (EPA AQI based)
                    let aqiSatisfaction = 100;
                    if (this.airQuality < 50) {
                        aqiSatisfaction = 100; // Good
                    } else if (this.airQuality < 100) {
                        aqiSatisfaction = 80; // Moderate
                    } else if (this.airQuality < 150) {
                        aqiSatisfaction = 60; // Unhealthy for sensitive groups
                    } else if (this.airQuality < 200) {
                        aqiSatisfaction = 40; // Unhealthy
                    } else if (this.airQuality < 300) {
                        aqiSatisfaction = 20; // Very unhealthy
                    } else {
                        aqiSatisfaction = 0; // Hazardous
                    }
                    
                    // Energy cost satisfaction
                    const energyPenalty = this.energyConsumption * 0.2;
                    
                    // Season-specific comfort expectations
                    const month = new Date().getMonth();
                    const isSummer = month >= 4 && month <= 9;
                    const season = isSummer ? 'summer' : 'winter';
                    
                    // Check if conditions are within ASHRAE comfort zone
                    const tempInRange = this.temperature >= this.comfortStandards[season].tempMin && 
                                     this.temperature <= this.comfortStandards[season].tempMax;
                    const humidityInRange = this.humidity >= this.comfortStandards[season].humidityMin && 
                                         this.humidity <= this.comfortStandards[season].humidityMax;
                    
                    const standardsMultiplier = (tempInRange && humidityInRange) ? 1.1 : 0.9;
                    
                    // Occupancy impact - crowding reduces satisfaction
                    const occupancyFactor = Math.max(0.8, 1 - 0.05 * Math.max(0, this.occupancy.count - 2));
                    
                    // Window state factor - fresh air increases satisfaction when AQI is good
                    let windowFactor = 1.0;
                    if (this.windowState === "open" && this.airQuality < 50) {
                        windowFactor = 1.15; // People like fresh air when it's clean
                    } else if (this.windowState === "open" && this.airQuality > 100) {
                        windowFactor = 0.85; // Unhealthy air coming in
                    }
                    
                    // Adaptive comfort model - people adapt to seasonal expectations
                    const adaptiveComfort = 0.9 + 0.1 * Math.min(1, Math.max(0, 
                        isSummer ? (this.temperature - 20) / 6 : (26 - this.temperature) / 6));
                    
                    // Direct humidity comfort (ASHRAE 55: 30-70% RH optimal)
                    const humiditySatisfaction = Math.max(0, 100 - Math.abs(this.humidity - 50) * 2.5);

                    // Final satisfaction calculation with all factors
                    this.overallSatisfaction = Math.max(0, Math.min(100,
                        (0.30 * pmvSatisfaction * standardsMultiplier +
                         0.25 * ppdSatisfaction * adaptiveComfort * occupancyFactor +
                         0.20 * humiditySatisfaction +
                         0.15 * aqiSatisfaction * windowFactor -
                         0.10 * energyPenalty)
                    ));
                    
                    console.log(`Satisfaction: PMV=${pmvSatisfaction.toFixed(1)}, PPD=${ppdSatisfaction.toFixed(1)}, AQI=${aqiSatisfaction.toFixed(1)}`);
                    console.log(`Factors: standards=${standardsMultiplier}, adaptive=${adaptiveComfort.toFixed(2)}, occupancy=${occupancyFactor.toFixed(2)}, window=${windowFactor.toFixed(2)}`);
                    console.log(`Final satisfaction: ${this.overallSatisfaction.toFixed(1)}%`);
                    
                    resolve(this.overallSatisfaction);
                })
                .catch(error => {
                    clearTimeout(timeoutId);
                    console.error('Satisfaction Calculation Error:', error);
                    this._updateFallbackSatisfaction();
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

    // Add to SmartRoom class
    getHvacEfficiency() { return this.hvacEfficiency; }
    getInsulationQuality() { return this.buildingEnvelope.wallRValue / 30; } // Normalized 0-1
    getWindowState() { return this.windowState; }
    getOccupancyLevel() { return this.occupancy.count; }
    getOccupancyActivity() { return this.occupancy.activity; }
    getTimeOfDay() { return this.timeOfDay; }
    getOutdoorTemperature() { return this.outdoorConditions.temperature; }
    getOutdoorHumidity() { return this.outdoorConditions.humidity; }

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

// Default Settings: The code creates simulated data points at 15-minute intervals across a 24-hour period; 
// it's generating 96 data points (24 hours ÷ 15 minutes = 96) with various calculations for each
function generateSimulationData(duration = 24 * 60 * 60 * 1000, interval = 15 * 60 * 1000) {
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
                // Update time-dependent variables
                const timeOffset = (currentTime - startTime) / 60000; // minutes
                room.lastUpdateTime = currentTime - interval;
                room._updateHiddenVariables();
                
                // Small random changes to control variables
                room.temperature += (Math.random() * 2 - 1) * 0.5;
                room.temperature = Math.max(18, Math.min(30, room.temperature));
                
                room.humidity += (Math.random() * 2 - 1) * 2;
                room.humidity = Math.max(30, Math.min(70, room.humidity));
                
                room.airQuality += (Math.random() * 2 - 1) * 10;
                room.airQuality = Math.max(0, Math.min(500, room.airQuality));
                
                // Calculate dependent variables
                await room.updateDependentVariables();
                
                // Record all variables
                data.push({
                    Timestamp: new Date(currentTime).toISOString(),
                    // Observable variables
                    Temperature: room.temperature,
                    Humidity: room.humidity,
                    AirQuality: room.airQuality,
                    EnergyConsumption: room.energyConsumption,
                    OverallSatisfaction: room.overallSatisfaction,
                    // Hidden variables
                    HVACEfficiency: room.hvacEfficiency,
                    WindowState: room.windowState,
                    OccupancyCount: room.occupancy.count,
                    OccupancyActivity: room.occupancy.activity,
                    OutdoorTemperature: room.outdoorConditions.temperature,
                    OutdoorHumidity: room.outdoorConditions.humidity,
                    OutdoorWindSpeed: room.outdoorConditions.windSpeed,
                    TimeOfDay: room.timeOfDay,
                    ElectricityRate: room.electricityRate,
                    WallRValue: room.buildingEnvelope.wallRValue,
                    WindowUFactor: room.buildingEnvelope.windowUFactor
                });
                
                // Move to the next time interval
                currentTime += interval;
                console.log(`Generated data point for time ${new Date(currentTime).toLocaleTimeString()}`);
            }
            
            console.log(`Generated ${data.length} data points`);
            
            // Export to CSV
            const csvContent = generateCsvContent(data);
            const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
            const filename = `simulation_data_${timestamp}.csv`;
            
            const fs = require('fs');
            fs.writeFileSync(filename, csvContent);
            console.log(`Data exported to ${filename}`);
            
            // Clean up
            room.cleanup();
            
            resolve(filename);
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

            // Run physics update (energy, satisfaction, hidden variable dynamics)
            await room.updateDependentVariables();

            const output = JSON.stringify({
                temperature: room.temperature,
                humidity: room.humidity,
                airQuality: room.airQuality,
                energyConsumption: room.energyConsumption,
                overallSatisfaction: room.overallSatisfaction,
                // Hidden variables (for analysis only)
                hvacEfficiency: room.hvacEfficiency,
                windowState: room.windowState,
                outdoorTemperature: room.outdoorConditions.temperature,
                occupancy: room.occupancy.count,
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
// Make sure the existing exports are maintained
module.exports = { 
    SmartRoom, 
    initializeSmartRoom, 
    simulateAndGetLatestData,
    saveSimulationDataToCSV,
    cleanupProcesses,
    generateSimulationData 
};