#!/usr/bin/env node
const path = require('path');
const fs = require('fs');
const PsychrometricCalculator = require('./python_interface');

// Inline SmartRoom with modified schedule
class SmartRoom {
    constructor() {
        this.pmvCalculator = new PsychrometricCalculator();
        // SHORT schedule for data generation (5-minute cycles)
        this.windowConfig = {
            schedule: [
                {openTime: 60000, closeTime: 120000},      // 1-2 min
                {openTime: 180000, closeTime: 240000},     // 3-4 min
                {openTime: 300000, closeTime: 360000},     // 5-6 min
                {openTime: 480000, closeTime: 540000},     // 8-9 min
            ],
            isOpen: false
        };
        
        this.outdoorTemperature = 5.0;
        this.simulationStartTime = null;
        this.temperature = 22;
        this.humidity = 50;
        this.airQuality = 300;
        this.pmv = 0;
        this.energyConsumption = 50;
        this.satisfaction = 75;
    }

    async _initialize() {
        this.simulationStartTime = Date.now();
        return true;
    }

    updateWindowState() {
        const elapsedTime = Date.now() - this.simulationStartTime;
        const timeOfDay = elapsedTime % (24*3600*1000);
        const hour = timeOfDay / 3600000;
        
        this.outdoorTemperature = 5 + 3 * Math.sin((hour - 6) * Math.PI / 12);

        const schedule = [
            {openTime: 2*3600*1000, closeTime: 3*3600*1000},      // 2-3hr
            {openTime: 6*3600*1000, closeTime: 7*3600*1000},      // 6-7hr
            {openTime: 10*3600*1000, closeTime: 11*3600*1000},    // 10-11hr
            {openTime: 14*3600*1000, closeTime: 15*3600*1000},    // 14-15hr
            {openTime: 18*3600*1000, closeTime: 19*3600*1000},    // 18-19hr
            {openTime: 22*3600*1000, closeTime: 23*3600*1000},    // 22-23hr
        ];
        
        this.windowConfig.isOpen = schedule.some(p => 
            timeOfDay >= p.openTime && timeOfDay < p.closeTime
        );
    }

    updateTemperatureDynamics() {
        const targetTemp = 22;
        const naturalDrift = (Math.random() - 0.5) * 0.15;
        
        if (this.windowConfig.isOpen) {
            const heatLoss = (this.outdoorTemperature - this.temperature) * 0.08;
            const heatingGain = (targetTemp - this.temperature) * 0.05;
            this.temperature += heatLoss + heatingGain + naturalDrift;
        } else {
            this.temperature += (targetTemp - this.temperature) * 0.08 + naturalDrift;
        }
        
        const humidityDrift = (Math.random() - 0.5) * 0.3;
        this.humidity += humidityDrift;
        if (this.windowConfig.isOpen) {
            this.humidity += (35 - this.humidity) * 0.02;
        }
        
        const aqDrift = (Math.random() - 0.5) * 5;
        this.airQuality += aqDrift + (this.windowConfig.isOpen ? -2 : 0.5);
        
        this.temperature = Math.max(5, Math.min(30, this.temperature));
        this.humidity = Math.max(30, Math.min(70, this.humidity));
        this.airQuality = Math.max(200, Math.min(500, this.airQuality));
    }

    async updateDependentVariables() {
        this.updateWindowState();
        this.updateTemperatureDynamics();
        
        // Energy calculation
        const baseEnergy = 40 + (Math.random()-0.5)*5;
        this.energyConsumption = this.windowConfig.isOpen ? Math.min(100, baseEnergy * 2.5) : baseEnergy;
        
        // PMV from real pythermalcomfort
        const pmvResult = await this.pmvCalculator.calculatePMV(this.temperature, this.humidity);
        this.pmv = pmvResult.pmv;
        const pmvSat = 100 - Math.abs(this.pmv) * 30;
        const aqSat = Math.max(0, 100 - (this.airQuality - 400) / 6);
        const energySat = 100 - this.energyConsumption;
        this.satisfaction = (pmvSat + aqSat + energySat) / 3;
    }

    getState() {
        const elapsed = this.simulationStartTime ? (Date.now() - this.simulationStartTime) / 1000 : 0;
        return {
            timestamp: new Date().toISOString(),
            temperature: this.temperature,
            humidity: this.humidity,
            airQuality: this.airQuality,
            pmv: this.pmv,
            energyConsumption: this.energyConsumption,
            satisfaction: this.satisfaction,
            windowOpen: this.windowConfig.isOpen,
            outdoorTemperature: this.outdoorTemperature,
            elapsedTime: elapsed
        };
    }

    async cleanup() {
        if (this.pmvCalculator) await this.pmvCalculator.cleanup();
    }
}

async function generateData() {
    console.log('Generating 10,080 points (7 days)...\n');
    
    const room = new SmartRoom();
    await room._initialize();
    
    const dataPoints = [];
    const totalPoints = 10080;
    const intervalMs = 60000;
    
    const startTime = Date.now();
    room.simulationStartTime = startTime;

    const originalDateNow = Date.now;
    Date.now = () => currentSimTime;

    let currentSimTime = startTime;
    
    for (let i = 0; i < totalPoints; i++) {
        currentSimTime = startTime + (i * intervalMs);
        room.simulationStartTime = startTime;
        
        await room.updateDependentVariables();
        const state = room.getState();
        
        dataPoints.push({
            timestamp: new Date(currentSimTime).toISOString(),
            temperature: state.temperature.toFixed(3),
            humidity: state.humidity.toFixed(3),
            airQuality: state.airQuality.toFixed(3),
            pmv: state.pmv.toFixed(3),
            energyConsumption: state.energyConsumption.toFixed(3),
            satisfaction: state.satisfaction.toFixed(3),
            windowOpen: state.windowOpen ? 1 : 0,
            outdoorTemperature: state.outdoorTemperature.toFixed(3),
            elapsedTime: state.elapsedTime.toFixed(3)
        });
        
        if ((i + 1) % 1440 === 0) {
            const days = Math.floor(i / 1440) + 1;
            console.log(`Day ${days} complete (${i+1}/${totalPoints})`);
        }
        // console.log(`i=${i}, elapsed=${Date.now() - room.simulationStartTime}, window=${room.windowConfig.isOpen}`);
    }
    
    Date.now = originalDateNow;
    
    const headers = Object.keys(dataPoints[0]).join(',');
    const rows = dataPoints.map(p => Object.values(p).join(','));
    fs.writeFileSync('challenge1_data_10k.csv', [headers, ...rows].join('\n'));
    
    console.log(`\n✓ Generated ${totalPoints} points`);
    
    const windowOpen = dataPoints.filter(p => p.windowOpen === 1);
    console.log(`✓ ${(windowOpen.length/totalPoints*100).toFixed(1)}% window open`);
    
    await room.cleanup();
    process.exit(0);
}

generateData().catch(err => {
    console.error('Error:', err);
    process.exit(1);
});