const fs = require('fs');

class Room {
    constructor() {
        this.temperature = 22;
        this.humidity = 50;
        this.airQuality = 300;
        this.outdoorTemperature = 5.0;
        this.windowOpen = false;
        
        // More frequent window cycles for richer signal
        this.windowSchedule = [
            {open: 2*3600, close: 2.5*3600},
            {open: 5*3600, close: 6*3600},
            {open: 9*3600, close: 10*3600},
            {open: 13*3600, close: 14*3600},
            {open: 17*3600, close: 18.5*3600},
            {open: 21*3600, close: 22*3600}
        ];
        
        this.interventions = {};
    }
    
    intervene(variable, value) {
        this.interventions[variable] = value;
    }
    
    clearInterventions() {
        this.interventions = {};
    }
    
    update(t) {
        const hour = (t / 3600) % 24;
        this.outdoorTemperature = 5 + 3 * Math.sin((hour - 6) * Math.PI / 12);
        
        let shouldBeOpen = false;
        for (const period of this.windowSchedule) {
            if (t >= period.open && t < period.close) {
                shouldBeOpen = true;
                break;
            }
        }
        this.windowOpen = shouldBeOpen;
        
        if ('windowOpen' in this.interventions) {
            this.windowOpen = this.interventions.windowOpen;
        }
        
        const targetTemp = 22;
        const naturalDrift = (Math.random() - 0.5) * 0.15;
        
        if (this.windowOpen) {
            const heatLoss = (this.outdoorTemperature - this.temperature) * 0.08;
            const heatingGain = (targetTemp - this.temperature) * 0.05;
            this.temperature += heatLoss + heatingGain + naturalDrift;
        } else {
            this.temperature += (targetTemp - this.temperature) * 0.08 + naturalDrift;
        }
        
        this.humidity += (Math.random() - 0.5) * 0.3;
        if (this.windowOpen) {
            this.humidity += (35 - this.humidity) * 0.02;
        }
        
        this.airQuality += (Math.random() - 0.5) * 5;
        this.airQuality += this.windowOpen ? -2 : 0.5;
        
        if ('temperature' in this.interventions) {
            this.temperature = this.interventions.temperature;
        }
        if ('outdoorTemperature' in this.interventions) {
            this.outdoorTemperature = this.interventions.outdoorTemperature;
        }
        
        this.temperature = Math.max(5, Math.min(30, this.temperature));
        this.humidity = Math.max(30, Math.min(70, this.humidity));
        this.airQuality = Math.max(200, Math.min(500, this.airQuality));
        
        const baseEnergy = 40 + (Math.random()-0.5)*5;
        const energy = this.windowOpen ? Math.min(100, baseEnergy * 2.5) : baseEnergy;
        
        const pmv = (this.temperature - 22) / 6;
        const pmvSat = 100 - Math.abs(pmv) * 30;
        const aqSat = Math.max(0, 100 - (this.airQuality - 400) / 6);
        const energySat = 100 - energy;
        const satisfaction = (pmvSat + aqSat + energySat) / 3;
        
        return {
            timestamp: new Date().toISOString(),
            temperature: this.temperature.toFixed(3),
            humidity: this.humidity.toFixed(3),
            airQuality: this.airQuality.toFixed(3),
            pmv: pmv.toFixed(3),
            energyConsumption: energy.toFixed(3),
            satisfaction: satisfaction.toFixed(3),
            windowOpen: this.windowOpen ? 1 : 0,
            outdoorTemperature: this.outdoorTemperature.toFixed(3),
            elapsedTime: t.toFixed(3),
            intervention: Object.keys(this.interventions).length > 0 ? 1 : 0
        };
    }
}

console.log('Generating 7 days data (10080 points)...');
const room = new Room();
const data = [];

const DAYS = 7;  // Changed from 24hr to 7 days
const INTERVAL = 60;
const POINTS = DAYS * 24 * 60;  // 10,080 points

for (let i = 0; i < POINTS; i++) {
    const t = i * INTERVAL;
    
    // Intervention on day 4 at hour 12
    const dayHour = (t / 3600) % (24 * DAYS);
    if (dayHour >= (3*24 + 12) && dayHour < (3*24 + 12.5)) {
        room.intervene('windowOpen', true);
    } else {
        room.clearInterventions();
    }
    
    data.push(room.update(t));
}

const headers = Object.keys(data[0]).join(',');
const rows = data.map(d => Object.values(d).join(','));
fs.writeFileSync('challenge1_data.csv', [headers, ...rows].join('\n'));

console.log(`✓ Generated ${data.length} points (${DAYS} days)`);
console.log(`✓ ${(data.filter(d => d.windowOpen == 1).length / data.length * 100).toFixed(1)}% window open`);