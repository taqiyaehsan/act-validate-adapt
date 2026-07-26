#!/usr/bin/env node

/**
 * Smart Building Simulation with exact smart_room.js compatibility
 * Complete version with all functions and proper CSV generation
 */

const fs = require('fs');
const path = require('path');

// Global state tracking for causal framework
let globalPreInterventionState = null;
let globalPostInterventionState = null;

/**
 * Psychrometric Calculator - exact copy from smart_room.js
 */
class PsychrometricCalculator {
    constructor() {
        this.clothingInsulation = 0.61; // clo
        this.metabolicRate = 1.2; // met
        this.externalWork = 0; // W/m²
    }

    async calculatePMV(temperature, humidity, airSpeed = 0.1, meanRadiantTemp = null) {
        return new Promise((resolve, reject) => {
            try {
                const ta = temperature;
                const rh = humidity;
                const vel = airSpeed;
                const tr = meanRadiantTemp || temperature;
                const met = this.metabolicRate;
                const clo = this.clothingInsulation;
                const wme = this.externalWork;

                // Vapor pressure
                const pa = (rh * this.saturatedVaporPressure(ta)) / 100;

                // Clothing surface temperature calculation
                const icl = 0.155 * clo;
                const m = met * 58.15;
                const w = wme;

                const fcl = clo <= 0.078 ? 1.0 + 1.290 * clo : 1.05 + 0.645 * clo;

                let tcl = ta;
                for (let i = 0; i < 150; i++) {
                    const hcf = 12.1 * Math.sqrt(vel);
                    const hc = vel > 0.1 ? hcf : 2.38 * Math.pow(Math.abs(tcl - ta), 0.25);
                    const hr = 3.96e-8 * fcl * (Math.pow(tcl + 273, 4) - Math.pow(tr + 273, 4)) / (tcl - tr);
                    
                    const tcl_new = 35.7 - 0.028 * (m - w) - icl * fcl * (hr + hc) * (tcl - ta);
                    
                    if (Math.abs(tcl_new - tcl) <= 0.01) break;
                    tcl = tcl_new;
                }

                const hcf = 12.1 * Math.sqrt(vel);
                const hc = vel > 0.1 ? hcf : 2.38 * Math.pow(Math.abs(tcl - ta), 0.25);
                const hr = 3.96e-8 * fcl * (Math.pow(tcl + 273, 4) - Math.pow(tr + 273, 4)) / (tcl - tr);

                const pmv = (0.303 * Math.exp(-0.036 * m) + 0.028) * 
                           ((m - w) - 3.05e-3 * (5733 - 6.99 * (m - w) - pa) - 
                            0.42 * ((m - w) - 58.15) - 1.7e-5 * m * (5867 - pa) - 
                            0.0014 * m * (34 - ta) - 3.96e-8 * fcl * 
                            (Math.pow(tcl + 273, 4) - Math.pow(tr + 273, 4)) - 
                            fcl * hc * (tcl - ta));

                const ppd = 100 - 95 * Math.exp(-0.03353 * Math.pow(pmv, 4) - 0.2179 * Math.pow(pmv, 2));

                resolve({
                    pmv: Math.max(-3, Math.min(3, pmv)),
                    ppd: Math.max(0, Math.min(100, ppd))
                });

            } catch (error) {
                reject(error);
            }
        });
    }

    saturatedVaporPressure(temperature) {
        return Math.exp(16.6536 - 4030.183 / (temperature + 235));
    }

    cleanup() {
        // Cleanup method for compatibility
    }
}

/**
 * Energy Lookup - exact copy from smart_room.js
 */
class EnergyLookup {
    constructor() {
        this.energyData = new Map();
        this.isInitialized = false;
    }

    async initialize() {
        try {
            console.log("Initializing energy lookup table...");
            await this._generateEnergyLookupTable();
            this.isInitialized = true;
            console.log("Energy lookup table initialized successfully");
        } catch (error) {
            console.error("Energy lookup initialization failed:", error);
            this.isInitialized = true; // Continue with fallback
        }
    }

    async _generateEnergyLookupTable() {
        // Generate comprehensive lookup table
        const tempRange = Array.from({length: 21}, (_, i) => 15 + i); // 15-35°C
        const humidityRange = Array.from({length: 11}, (_, i) => 30 + i * 4); // 30-70%
        const aqRange = Array.from({length: 11}, (_, i) => 50 + i * 45); // 50-500 AQI

        for (const temp of tempRange) {
            for (const humidity of humidityRange) {
                for (const aqi of aqRange) {
                    const key = `${temp}_${humidity}_${aqi}`;
                    const energy = this._calculateEnergyForConditions(temp, humidity, aqi);
                    this.energyData.set(key, energy);
                }
            }
        }
    }

    _calculateEnergyForConditions(temperature, humidity, airQuality) {
        const tempDiff = Math.abs(temperature - 22);
        const humidityDiff = Math.abs(humidity - 50);
        const airQualityFactor = airQuality / 500;
        
        const hourOfDay = new Date().getHours();
        const timeComponent = Math.sin((hourOfDay - 12) * (Math.PI / 12)) * 5;
        const randomComponent = (Math.random() * 10) - 5;
        
        const baseEnergy = 25 + 
                        (tempDiff * 3) + 
                        (Math.pow(tempDiff, 1.5) * 0.5) + 
                        (humidityDiff * 0.7) + 
                        (airQualityFactor * 15) +
                        timeComponent + 
                        randomComponent;
        
        return Math.min(100, Math.max(10, baseEnergy));
    }

    async getEnergyConsumption(temperature, humidity, airQuality) {
        if (!this.isInitialized) {
            return this._calculateFallbackEnergy(temperature, humidity, airQuality);
        }

        try {
            const tempKey = Math.round(temperature);
            const humidityKey = Math.round(humidity / 4) * 4;
            const aqiKey = Math.round(airQuality / 45) * 45;
            
            const key = `${tempKey}_${humidityKey}_${aqiKey}`;
            
            if (this.energyData.has(key)) {
                return this.energyData.get(key);
            } else {
                return this._calculateFallbackEnergy(temperature, humidity, airQuality);
            }
        } catch (error) {
            console.error('Energy lookup failed:', error);
            return this._calculateFallbackEnergy(temperature, humidity, airQuality);
        }
    }

    _calculateFallbackEnergy(temperature, humidity, airQuality) {
        const tempDiff = Math.abs(temperature - 22);
        const humidityDiff = Math.abs(humidity - 50);
        const airQualityFactor = airQuality / 500;
        
        const hourOfDay = new Date().getHours();
        const timeComponent = Math.sin((hourOfDay - 12) * (Math.PI / 12)) * 5;
        const randomComponent = (Math.random() * 10) - 5;
        
        const baseEnergy = 25 + 
                        (tempDiff * 3) + 
                        (Math.pow(tempDiff, 1.5) * 0.5) + 
                        (humidityDiff * 0.7) + 
                        (airQualityFactor * 15) +
                        timeComponent + 
                        randomComponent;
        
        return Math.min(100, Math.max(10, baseEnergy));
    }
}

/**
 * Building Zone
 */
class BuildingZone {
    constructor(zoneId, zoneType, floorArea = 100) {
        this.zoneId = zoneId;
        this.zoneType = zoneType;
        this.floorArea = floorArea;
        
        // Environmental state
        this.temperature = 22 + (Math.random() - 0.5) * 4;
        this.humidity = 45 + (Math.random() - 0.5) * 20;
        this.airQuality = 300 + Math.random() * 200;
        this.airSpeed = 0.1;
        
        // Control states
        this.hvacSetpoint = 22;
        this.hvacMode = 'auto';
        this.lightingLevel = 0.8;
        
        // Occupancy
        this.occupantCount = this.getTypicalOccupancy();
        this.occupancySchedule = this.generateOccupancySchedule();
        
        // Energy (as percentage like smart_room.js)
        this.energyConsumption = 50;
        this.overallSatisfaction = 75;
        
        // Components from smart_room.js
        this.psychroCalc = new PsychrometricCalculator();
        this.energyLookup = new EnergyLookup();
        this.activePromises = new Set();
        this.initialized = false;
        
        // Initialize
        this.initPromise = this._initialize();
    }

    async _initialize() {
        try {
            await this.energyLookup.initialize();
            this.initialized = true;
            return true;
        } catch (error) {
            console.error(`Zone ${this.zoneId} initialization failed:`, error);
            return false;
        }
    }

    async update() {
        // Update occupancy
        this.updateOccupancy();
        
        // Update dependent variables
        await this.updateDependentVariables();
        
        // Apply any thermal dynamics or environmental changes
        this.applyThermalDynamics();
    }

    applyThermalDynamics() {
        // Simple thermal dynamics - temperature moves toward HVAC setpoint
        const tempDiff = this.hvacSetpoint - this.temperature;
        this.temperature += tempDiff * 0.1; // 10% adjustment per update
        
        // Humidity is affected by temperature changes
        if (Math.abs(tempDiff) > 0.5) {
            this.humidity += (Math.random() - 0.5) * 2;
            this.humidity = Math.max(20, Math.min(80, this.humidity));
        }
        
        // Air quality slowly improves over time (HVAC filtering)
        if (this.airQuality > 100) {
            this.airQuality -= 5;
        }
    }

    getTypicalOccupancy() {
        const occupancyByType = {
            'office': 2 + Math.floor(Math.random() * 4),
            'meeting': Math.floor(Math.random() * 12),
            'lobby': 5 + Math.floor(Math.random() * 15),
            'server': 0,
            'residential': 1 + Math.floor(Math.random() * 3)
        };
        return occupancyByType[this.zoneType] || 2;
    }

    generateOccupancySchedule() {
        const schedules = {
            'office': [0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.5, 0.8, 0.9, 0.9, 0.9, 0.9, 0.7, 0.9, 0.9, 0.9, 0.8, 0.6, 0.3, 0.2, 0.1, 0.1, 0.1, 0.1],
            'meeting': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.3, 0.6, 0.8, 0.9, 0.8, 0.7, 0.5, 0.8, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1],
            'lobby': [0.2, 0.1, 0.1, 0.1, 0.1, 0.3, 0.6, 0.8, 0.9, 0.8, 0.7, 0.8, 0.9, 0.8, 0.8, 0.8, 0.9, 0.8, 0.6, 0.4, 0.3, 0.2, 0.2, 0.2],
            'server': [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
            'residential': [0.9, 0.9, 0.9, 0.9, 0.9, 0.8, 0.6, 0.4, 0.2, 0.1, 0.1, 0.2, 0.3, 0.2, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.9, 0.9, 0.9, 0.9]
        };
        return schedules[this.zoneType] || schedules['office'];
    }

    updateOccupancy() {
        const hour = new Date().getHours();
        const probability = this.occupancySchedule[hour];
        const maxOccupancy = this.getTypicalOccupancy();
        const variance = 0.3;
        const actualProbability = Math.max(0, Math.min(1, probability + (Math.random() - 0.5) * variance));
        this.occupantCount = Math.floor(maxOccupancy * actualProbability);
    }

    async updateDependentVariables() {
        if (!this.initialized) {
            await this.initPromise;
        }

        this.temperature = Math.max(18, Math.min(30, Number(this.temperature)));
        this.humidity = Math.max(30, Math.min(70, Number(this.humidity)));
        this.airQuality = Math.max(0, Math.min(500, Number(this.airQuality)));

        try {
            // Update energy and satisfaction
            this.energyConsumption = await this.energyLookup.getEnergyConsumption(
                this.temperature, this.humidity, this.airQuality
            );
            
            await this.updateOverallSatisfaction();
        } catch (error) {
            console.error(`Zone ${this.zoneId} update failed:`, error);
            this._updateFallbackValues();
        }
    }

    async updateOverallSatisfaction() {
        try {
            const comfortResults = await this.psychroCalc.calculatePMV(
                this.temperature, this.humidity, this.airSpeed
            );
            
            const pmvSatisfaction = 100 - (Math.abs(comfortResults.pmv || 0) * 16.67);
            const ppdSatisfaction = 100 - (comfortResults.ppd || 0);
            const aqiSatisfaction = Math.max(0, 100 - (this.airQuality / 5));
            const energyPenalty = this.energyConsumption * 0.2;

            this.overallSatisfaction = Math.max(0, Math.min(100,
                0.4 * pmvSatisfaction +
                0.3 * ppdSatisfaction +
                0.2 * aqiSatisfaction -
                0.1 * energyPenalty
            ));

            if (isNaN(this.overallSatisfaction)) {
                this._updateFallbackSatisfaction();
            }
        } catch (error) {
            this._updateFallbackSatisfaction();
        }
    }

    _updateFallbackValues() {
        this.energyConsumption = this.energyLookup._calculateFallbackEnergy(
            this.temperature, this.humidity, this.airQuality
        );
        this._updateFallbackSatisfaction();
    }

    _updateFallbackSatisfaction() {
        const tempComfort = Math.max(0, 100 - Math.abs(this.temperature - 22) * 4);
        const humidityComfort = Math.max(0, 100 - Math.abs(this.humidity - 50) * 2);
        const aqiComfort = Math.max(0, 100 - (this.airQuality / 5));
        
        this.overallSatisfaction = (tempComfort + humidityComfort + aqiComfort) / 3;
        
        if (isNaN(this.overallSatisfaction)) {
            this.overallSatisfaction = 75;
        }
    }

    _calculateThermalComfort() {
        const optimalTemp = this.zoneType === 'server' ? 20 : 22;
        return Math.max(0, 100 - Math.abs(this.temperature - optimalTemp) * 4);
    }

    _calculateVisualComfort() {
        return this.lightingLevel * 100;
    }

    getState() {
        // Ensure all calculations are complete
        if (isNaN(this.overallSatisfaction)) {
            this._updateFallbackSatisfaction();
        }
        if (isNaN(this.energyConsumption)) {
            this.energyConsumption = 50;
        }
        
        return {
            zoneId: this.zoneId,
            zoneType: this.zoneType,
            
            // Interventional Variables (can be controlled)
            Temperature: Number(this.temperature) || 22,
            Humidity: Number(this.humidity) || 50,
            AirQuality: Number(this.airQuality) || 300,
            HVACSetpoint: Number(this.hvacSetpoint) || 22,
            LightingLevel: Number(this.lightingLevel) || 0.8,
            OccupantCount: Number(this.occupantCount) || 0,
            
            // Outcome Variables (dependent responses)
            EnergyConsumption: Number(this.energyConsumption) || 50,
            OverallSatisfaction: Number(this.overallSatisfaction) || 75,
            ThermalComfort: Number(this._calculateThermalComfort()) || 75,
            VisualComfort: Number(this._calculateVisualComfort()) || 80,
            AirQualityIndex: Number(Math.max(0, 100 - (this.airQuality || 300) / 5)) || 40,
            HVACPower: Number((this.energyConsumption || 50) * 0.6) || 30,
            LightingPower: Number((this.lightingLevel || 0.8) * 15) || 12
        };
    }

    applyIntervention(variable, action, value) {
        console.log(`Applying ${action} ${variable} = ${value} to zone ${this.zoneId}`);
        
        switch (variable) {
            case 'Temperature':
                if (action === 'set') this.temperature = Number(value);
                else if (action === 'increase') this.temperature += Number(value);
                else if (action === 'decrease') this.temperature -= Number(value);
                // Keep within reasonable bounds
                this.temperature = Math.max(16, Math.min(32, this.temperature));
                break;
                
            case 'Humidity':
                if (action === 'set') this.humidity = Number(value);
                else if (action === 'increase') this.humidity += Number(value);
                else if (action === 'decrease') this.humidity -= Number(value);
                // Keep within reasonable bounds
                this.humidity = Math.max(20, Math.min(80, this.humidity));
                break;
                
            case 'AirQuality':
                if (action === 'set') this.airQuality = Number(value);
                else if (action === 'increase') this.airQuality += Number(value);
                else if (action === 'decrease') this.airQuality -= Number(value);
                // Keep within reasonable bounds
                this.airQuality = Math.max(0, Math.min(500, this.airQuality));
                break;
                
            case 'HVACSetpoint':
                if (action === 'set') this.hvacSetpoint = Number(value);
                else if (action === 'increase') this.hvacSetpoint += Number(value);
                else if (action === 'decrease') this.hvacSetpoint -= Number(value);
                // Keep within reasonable bounds
                this.hvacSetpoint = Math.max(16, Math.min(32, this.hvacSetpoint));
                break;
                
            case 'LightingLevel':
                if (action === 'set') this.lightingLevel = Math.max(0, Math.min(1, Number(value)));
                else if (action === 'increase') this.lightingLevel = Math.min(1, this.lightingLevel + Number(value));
                else if (action === 'decrease') this.lightingLevel = Math.max(0, this.lightingLevel - Number(value));
                break;

            case 'OccupantCount':
                if (action === 'set') this.occupantCount = Math.max(0, Number(value));
                else if (action === 'increase') this.occupantCount += Number(value);
                else if (action === 'decrease') this.occupantCount = Math.max(0, this.occupantCount - Number(value));
                break;
                
            default:
                console.warn(`Unknown intervention variable: ${variable}`);
        }
    }

    async waitForStabilization(duration = 3000) {
        return new Promise((resolve) => {
            setTimeout(async () => {
                // Only call update if it exists
                if (typeof this.update === 'function') {
                    await this.update();
                }
                resolve();
            }, duration);
        });
    }

    cleanup() {
        if (this.psychroCalc) {
            this.psychroCalc.cleanup();
        }
        
        for (const promise of this.activePromises) {
            if (promise.cancel) {
                promise.cancel();
            }
        }
        this.activePromises.clear();
    }
}

/**
 * Smart Building - exact interface as smart_room.js
 */
class SmartBuilding {
    constructor() {
        this.zones = new Map();
        
        // Initialize zones
        this.initializeZones();
        
        // Simulation control
        this.isRunning = false;
        this.dataLog = [];
        this.lastDataLogTime = 0;
        this.dataLogInterval = 1000;
        this.initialized = false;
        
        // Initialize all zones
        this.initPromise = this.initializeAllZones();
        
        // Process termination handling
        process.on('SIGTERM', () => this.cleanup());
        process.on('SIGINT', () => this.cleanup());
    }

    initializeZones() {
        const zoneConfigs = [
            { id: 'Office_1', type: 'office', area: 120 },
            { id: 'Office_2', type: 'office', area: 100 },
            { id: 'Meeting_1', type: 'meeting', area: 60 },
            { id: 'Lobby', type: 'lobby', area: 200 },
            { id: 'Server_Room', type: 'server', area: 50 }
        ];
        
        zoneConfigs.forEach(config => {
            this.zones.set(config.id, new BuildingZone(config.id, config.type, config.area));
        });
        
        console.log(`Initialized ${this.zones.size} building zones`);
    }

    async initializeAllZones() {
        const initPromises = Array.from(this.zones.values()).map(zone => zone.initPromise);
        await Promise.all(initPromises);
        this.initialized = true;
    }

    getAggregatedState() {
        const timestamp = new Date().toISOString();
        
        // Initialize aggregation variables with safe defaults
        let totalTemp = 0, totalHumidity = 0, totalAirQuality = 0;
        let totalHVACSetpoint = 0, totalLightingLevel = 0, totalOccupants = 0;
        let totalEnergy = 0, totalSatisfaction = 0, totalThermalComfort = 0;
        let totalVisualComfort = 0, totalAirQualityIndex = 0;
        let totalHVACPower = 0, totalLightingPower = 0;
        
        const zoneCount = this.zones.size;
        
        this.zones.forEach(zone => {
            const state = zone.getState();
            
            // Safely add all values with defaults
            totalTemp += Number(state.Temperature) || 0;
            totalHumidity += Number(state.Humidity) || 0;
            totalAirQuality += Number(state.AirQuality) || 0;
            totalHVACSetpoint += Number(state.HVACSetpoint) || 0;
            totalLightingLevel += Number(state.LightingLevel) || 0;
            totalOccupants += Number(state.OccupantCount) || 0;
            totalEnergy += Number(state.EnergyConsumption) || 0;
            totalSatisfaction += Number(state.OverallSatisfaction) || 0;
            totalThermalComfort += Number(state.ThermalComfort) || 0;
            totalVisualComfort += Number(state.VisualComfort) || 0;
            totalAirQualityIndex += Number(state.AirQualityIndex) || 0;
            totalHVACPower += Number(state.HVACPower) || 0;
            totalLightingPower += Number(state.LightingPower) || 0;
        });
        
        // Return complete state with all 13 variables + timestamp
        return {
            Timestamp: timestamp,
            Temperature: Math.round((totalTemp / zoneCount) * 100) / 100,
            Humidity: Math.round((totalHumidity / zoneCount) * 100) / 100,
            AirQuality: Math.round(totalAirQuality / zoneCount),
            HVACSetpoint: Math.round((totalHVACSetpoint / zoneCount) * 100) / 100,
            LightingLevel: Math.round((totalLightingLevel / zoneCount) * 100) / 100,
            OccupantCount: totalOccupants,
            EnergyConsumption: Math.round((totalEnergy / zoneCount) * 100) / 100,
            OverallSatisfaction: Math.round((totalSatisfaction / zoneCount) * 100) / 100,
            ThermalComfort: Math.round((totalThermalComfort / zoneCount) * 100) / 100,
            VisualComfort: Math.round((totalVisualComfort / zoneCount) * 100) / 100,
            AirQualityIndex: Math.round(totalAirQualityIndex / zoneCount),
            HVACPower: Math.round(totalHVACPower * 100) / 100,
            LightingPower: Math.round(totalLightingPower * 100) / 100
        };
    }

    getCurrentState() {
        return this.getAggregatedState();
    }

    async update() {
        const updatePromises = Array.from(this.zones.values()).map(zone => {
            zone.updateOccupancy();
            return zone.updateDependentVariables();
        });
        
        await Promise.allSettled(updatePromises);
        
        // Log data if interval has passed
        const now = Date.now();
        if (now - this.lastDataLogTime >= this.dataLogInterval) {
            this.dataLog.push(this.getAggregatedState());
            this.lastDataLogTime = now;
        }
    }

    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.updateInterval = setInterval(async () => {
            await this.update();
        }, 1000);
        
        console.log("Smart building simulation started");
    }

    stop() {
        if (!this.isRunning) return;
        
        this.isRunning = false;
        if (this.updateInterval) {
            clearInterval(this.updateInterval);
        }
        
        console.log("Smart building simulation stopped");
    }

    cleanup() {
        this.stop();
        this.zones.forEach(zone => zone.cleanup());
        console.log("Smart building simulation cleaned up");
    }

    exportData(filename) {
        if (this.dataLog.length === 0) {
            console.log("No data to export");
            return null;
        }
        
        const csvData = this.convertToCSV(this.dataLog);
        const fullPath = path.join(process.cwd(), filename);
        
        fs.writeFileSync(fullPath, csvData);
        console.log(`Data exported to ${fullPath}`);
        return fullPath;
    }

    convertToCSV(data) {
        if (data.length === 0) return '';
        
        const headers = Object.keys(data[0]);
        const headerRow = headers.join(',');
        
        const dataRows = data.map(row => 
            headers.map(header => {
                const value = row[header];
                return typeof value === 'string' && value.includes(',') ? 
                    `"${value}"` : value;
            }).join(',')
        );
        
        return [headerRow, ...dataRows].join('\n');
    }

// Intervention methods for compatibility
async applyInterventions(interventions) {
    console.log(`Time: ${new Date().toISOString()}`);
    console.log(`Applying interventions: ${JSON.stringify(interventions)}`);

    if (!this.initialized) {
        await this.initPromise;
    }

    // Validate interventions format
    if (!Array.isArray(interventions)) {
        throw new Error('Interventions must be an array');
    }

    for (const intervention of interventions) {
        if (!intervention.variable || typeof intervention.variable !== 'string') {
            throw new Error(`Invalid or missing variable: ${JSON.stringify(intervention)}`);
        }
        if (!intervention.action || typeof intervention.action !== 'string') {
            throw new Error(`Invalid or missing action: ${JSON.stringify(intervention)}`);
        }
        if (intervention.value === undefined || typeof intervention.value !== 'number') {
            throw new Error(`Invalid or missing value: ${JSON.stringify(intervention)}`);
        }
        if (!['set', 'increase', 'decrease'].includes(intervention.action)) {
            throw new Error(`Invalid action '${intervention.action}'. Must be 'set', 'increase', or 'decrease'`);
        }
    }

    return new Promise(async (resolve, reject) => {
        const timeout = setTimeout(() => {
            reject(new Error('Intervention timeout'));
        }, 10000);

        try {
            globalPreInterventionState = this.getAggregatedState();
            globalPreInterventionState.InterventionApplied = false;

            for (const intervention of interventions) {
                await this.applyIntervention(intervention);
            }

            await this.waitForStabilization(3000);

            globalPostInterventionState = this.getAggregatedState();
            globalPostInterventionState.InterventionApplied = true;

            // Validate result states
            if (!globalPreInterventionState || !globalPostInterventionState) {
                throw new Error('Invalid simulation result: null state');
            }

            clearTimeout(timeout);
            resolve({
                preState: globalPreInterventionState,
                postState: globalPostInterventionState
            });

        } catch (error) {
            clearTimeout(timeout);
            reject(error);
        }
    });
}

async applyIntervention(intervention) {
    const { variable, action, value, zoneId } = intervention;
    
    // Additional validation for supported variables
    const supportedVars = ['Temperature', 'Humidity', 'AirQuality', 'HVACSetpoint', 'LightingLevel', 'OccupantCount'];
    const normalizedVar = supportedVars.find(v => v.toLowerCase() === variable.toLowerCase());
    
    if (!normalizedVar) {
        throw new Error(`Unsupported variable: ${variable}`);
    }
    
    const normalizedIntervention = { ...intervention, variable: normalizedVar };
    
    if (zoneId && zoneId !== 'all' && this.zones.has(zoneId)) {
        const zone = this.zones.get(zoneId);
        this.applyZoneIntervention(zone, normalizedIntervention);
    } else if (zoneId === 'all' || !zoneId) {
        this.zones.forEach(zone => {
            this.applyZoneIntervention(zone, normalizedIntervention);
        });
    }
}

    applyZoneIntervention(zone, intervention) {
        const { variable, action, value } = intervention;
        
        switch (variable) {
            case 'Temperature':
                if (action === 'set') zone.temperature = Number(value);
                else if (action === 'increase') zone.temperature += Number(value);
                else if (action === 'decrease') zone.temperature -= Number(value);
                break;
                
            case 'Humidity':
                if (action === 'set') zone.humidity = Number(value);
                else if (action === 'increase') zone.humidity += Number(value);
                else if (action === 'decrease') zone.humidity -= Number(value);
                break;
                
            case 'AirQuality':
                if (action === 'set') zone.airQuality = Number(value);
                else if (action === 'increase') zone.airQuality += Number(value);
                else if (action === 'decrease') zone.airQuality -= Number(value);
                break;
                
            case 'HVACSetpoint':
                if (action === 'set') zone.hvacSetpoint = Number(value);
                else if (action === 'increase') zone.hvacSetpoint += Number(value);
                else if (action === 'decrease') zone.hvacSetpoint -= Number(value);
                break;
                
            case 'LightingLevel':
                if (action === 'set') zone.lightingLevel = Math.max(0, Math.min(1, Number(value)));
                else if (action === 'increase') zone.lightingLevel = Math.min(1, zone.lightingLevel + Number(value));
                else if (action === 'decrease') zone.lightingLevel = Math.max(0, zone.lightingLevel - Number(value));
                break;
        }
    }

    async waitForStabilization(duration = 3000) {
        return new Promise((resolve) => {
            setTimeout(async () => {
                await this.update();
                resolve();
            }, duration);
        });
    }
}

// Main data generation function
async function generateSimulationData(duration, interval) {
    console.log(`Generating ${duration/1000/60} minutes of building simulation data...`);
    console.log(`Expected data points: ${Math.floor(duration/interval)}`);
    
    const building = new SmartBuilding();
    const filename = `building_simulation_${new Date().toISOString().replace(/[:.]/g, '-')}.csv`;
    
    try {
        // Wait for building initialization
        await building.initPromise;
        console.log("Building simulation initialized");
        
        // Initialize data collection array
        const simulationData = [];
        const startTime = Date.now();
        const endTime = startTime + duration;
        let currentTime = startTime;
        
        // Generate data points in a loop (like smart_room.js)
        while (currentTime < endTime) {
            // Apply small random environmental changes
            building.zones.forEach(zone => {
                zone.temperature += (Math.random() * 2 - 1) * 0.5;
                zone.temperature = Math.max(18, Math.min(30, zone.temperature));
                
                zone.humidity += (Math.random() * 2 - 1) * 2;
                zone.humidity = Math.max(30, Math.min(70, zone.humidity));
                
                zone.airQuality += (Math.random() * 2 - 1) * 10;
                zone.airQuality = Math.max(0, Math.min(500, zone.airQuality));
            });
            
            // Update all zones
            const updatePromises = Array.from(building.zones.values()).map(zone => {
                zone.updateOccupancy();
                return zone.updateDependentVariables();
            });
            await Promise.allSettled(updatePromises);
            
            // Get the current state
            const state = building.getAggregatedState();
            
            // Create data point with explicit values (not references)
            const dataPoint = {
                Timestamp: new Date(currentTime).toISOString(),
                Temperature: state.Temperature,
                Humidity: state.Humidity,
                AirQuality: state.AirQuality,
                HVACSetpoint: state.HVACSetpoint,
                LightingLevel: state.LightingLevel,
                OccupantCount: state.OccupantCount,
                EnergyConsumption: state.EnergyConsumption,
                OverallSatisfaction: state.OverallSatisfaction,
                ThermalComfort: state.ThermalComfort,
                VisualComfort: state.VisualComfort,
                AirQualityIndex: state.AirQualityIndex,
                HVACPower: state.HVACPower,
                LightingPower: state.LightingPower
            };
            
            // Add to simulation data
            simulationData.push(dataPoint);
            
            // Move to next time step
            currentTime += interval;
            
            // Progress logging
            if (simulationData.length % 100 === 0) {
                console.log(`Generated ${simulationData.length} data points`);
            }
        }
        
        console.log(`Generated ${simulationData.length} data points`);
        
        // Convert to CSV using the exact same method as exportData
        const csvContent = building.convertToCSV(simulationData);
        
        // Write to file
        fs.writeFileSync(filename, csvContent);
        console.log(`Data exported to ${filename}`);
        
        // Cleanup
        building.cleanup();
        
        return filename;
        
    } catch (error) {
        console.error("Error generating simulation data:", error);
        building.cleanup();
        throw error;
    }
}

// Export functions for compatibility
// Add this function to smart_building.js before the module.exports section

async function simulateAndGetLatestData(duration, interventions, interventionTime) {
    console.log(`Starting simulation: duration=${duration}ms, interventionTime=${interventionTime}ms`);
    console.log(`Interventions:`, JSON.stringify(interventions));
    
    return new Promise(async (resolve, reject) => {
        const building = new SmartBuilding();
        let preInterventionData = null;
        let postInterventionData = null;
        
        const cleanup = () => {
            building.cleanup();
        };
        
        try {
            // Wait for building initialization
            await building.initPromise;
            console.log("Building initialized");
            
            // Start the simulation
            building.start();
            
            // Capture pre-intervention state
            setTimeout(() => {
                try {
                    preInterventionData = building.getAggregatedState();
                    console.log("Pre-intervention state captured:", preInterventionData);
                } catch (error) {
                    cleanup();
                    reject(new Error(`Failed to capture pre-intervention state: ${error.message}`));
                    return;
                }
            }, Math.max(50, interventionTime - 50)); // Capture just before intervention
            
            // Apply interventions
            setTimeout(async () => {
                try {
                    console.log("Applying interventions...");
                    
                    // Apply each intervention to all zones
                    for (const intervention of interventions) {
                        const { variable, action, value } = intervention;
                        console.log(`Applying ${action} ${variable} = ${value}`);
                        
                        building.zones.forEach((zone) => {
                            zone.applyIntervention(variable, action, value);
                        });
                    }
                    
                    // Wait for all zones to stabilize
                    const stabilizationPromises = Array.from(building.zones.values()).map(zone => 
                        zone.waitForStabilization(1000)
                    );
                    await Promise.all(stabilizationPromises);
                    
                    console.log("Interventions applied successfully");
                    
                } catch (error) {
                    cleanup();
                    reject(new Error(`Failed to apply interventions: ${error.message}`));
                }
            }, interventionTime);
            
            // Capture post-intervention state and finish
            setTimeout(() => {
                try {
                    postInterventionData = building.getAggregatedState();
                    console.log("Post-intervention state captured:", postInterventionData);
                    
                    cleanup();
                    
                    // Validate both states exist
                    if (!preInterventionData || !postInterventionData) {
                        reject(new Error('Failed to capture complete intervention data'));
                        return;
                    }
                    
                    const result = {
                        preInterventionData,
                        postInterventionData,
                        // Legacy compatibility
                        pre_state: preInterventionData,
                        post_state: postInterventionData
                    };
                    
                    console.log("Simulation completed successfully");
                    resolve(result);
                    
                } catch (error) {
                    cleanup();
                    reject(new Error(`Failed to capture post-intervention state: ${error.message}`));
                }
            }, duration);

        } catch (error) {
            cleanup();
            reject(new Error(`Simulation initialization failed: ${error.message}`));
        }
    });
}

// Module exports
module.exports = {
    SmartBuilding,
    BuildingZone,
    PsychrometricCalculator,
    EnergyLookup,
    simulateAndGetLatestData,
    generateSimulationData
};

// CLI interface
if (require.main === module) {
    const args = process.argv.slice(2);

    if (args.length === 0) {
        console.log("Smart Building Simulation");
        console.log("Usage:");
        console.log("  node smart_building.js generate [duration_minutes] [interval_seconds]");
        console.log("  node smart_building.js test");
        console.log("  node smart_building.js --single-step [--state JSON] [--intervention JSON]");
        process.exit(0);
    }

    const command = args[0];

    // ── --single-step mode: one step, output RESULT JSON, exit ──
    if (command === '--single-step') {
        (async () => {
            const building = new SmartBuilding();
            await building.initPromise;

            // Parse optional --state to restore previous state
            const stateIdx = args.indexOf('--state');
            if (stateIdx !== -1 && args[stateIdx + 1]) {
                try {
                    const prev = JSON.parse(args[stateIdx + 1]);
                    // Apply previous state to all zones (averaged values)
                    building.zones.forEach(zone => {
                        if (prev.Temperature !== undefined) zone.temperature = prev.Temperature;
                        if (prev.Humidity !== undefined) zone.humidity = prev.Humidity;
                        if (prev.AirQuality !== undefined) zone.airQuality = prev.AirQuality;
                        if (prev.HVACSetpoint !== undefined) zone.hvacSetpoint = prev.HVACSetpoint;
                        if (prev.LightingLevel !== undefined) zone.lightingLevel = prev.LightingLevel;
                    });
                } catch (e) { /* ignore parse errors */ }
            }

            // Parse optional --intervention to apply setpoints
            const intIdx = args.indexOf('--intervention');
            if (intIdx !== -1 && args[intIdx + 1]) {
                try {
                    const iv = JSON.parse(args[intIdx + 1]);
                    building.zones.forEach(zone => {
                        if (iv.Temperature !== undefined) zone.temperature = iv.Temperature;
                        if (iv.temperature !== undefined) zone.temperature = iv.temperature;
                        if (iv.Humidity !== undefined) zone.humidity = iv.Humidity;
                        if (iv.humidity !== undefined) zone.humidity = iv.humidity;
                        if (iv.AirQuality !== undefined) zone.airQuality = iv.AirQuality;
                        if (iv.airQuality !== undefined) zone.airQuality = iv.airQuality;
                        if (iv.HVACSetpoint !== undefined) zone.hvacSetpoint = iv.HVACSetpoint;
                        if (iv.hvacSetpoint !== undefined) zone.hvacSetpoint = iv.hvacSetpoint;
                        if (iv.LightingLevel !== undefined) zone.lightingLevel = iv.LightingLevel;
                        if (iv.lightingLevel !== undefined) zone.lightingLevel = iv.lightingLevel;
                    });
                } catch (e) { /* ignore parse errors */ }
            }

            // Advance physics one step
            await building.update();

            // Output aggregated state
            const state = building.getAggregatedState();
            delete state.Timestamp;
            console.log('RESULT:' + JSON.stringify(state));

            building.cleanup();
            process.exit(0);
        })();
        return; // prevent falling through to other commands
    }
    
    if (command === 'generate') {
        const durationMinutes = parseInt(args[1]) || 60;
        const intervalSeconds = parseInt(args[2]) || 60;
        
        generateSimulationData(durationMinutes * 60 * 1000, intervalSeconds * 1000)
            .then(filename => {
                console.log(`✅ Simulation completed. Data saved to ${filename}`);
                process.exit(0);
            })
            .catch(error => {
                console.error(`❌ Simulation failed: ${error.message}`);
                process.exit(1);
            });
            
    } else if (command === 'test') {
        console.log("Running smart building test...");
        
        async function runTest() {
            const building = new SmartBuilding();
            
            try {
                await building.initPromise;
                const state = building.getAggregatedState();
                console.log("✅ Test completed successfully");
                console.log("Sample state:", state);
                building.cleanup();
            } catch (error) {
                console.error("❌ Test failed:", error.message);
                building.cleanup();
                process.exit(1);
            }
        }
        
        runTest();
    }
}