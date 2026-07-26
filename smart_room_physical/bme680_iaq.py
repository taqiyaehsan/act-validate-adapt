# bme680_reader.py

import numpy as np
import pandas as pd
from scipy import signal
from datetime import datetime, timedelta

class AdvancedIAQ:
    def __init__(self):
        self.gas_baseline = None
        self.humidity_baseline = 40.0
        self.readings = []
        self.calibrated = False
        self.burn_in_time = timedelta(minutes=5)  # 5 min burn-in
        self.start_time = datetime.now()
        
    def add_reading(self, temperature, humidity, pressure, gas_resistance):
        """Add new sensor reading and calculate IAQ"""
        
        reading = {
            'timestamp': datetime.now(),
            'temperature': temperature,
            'humidity': humidity,
            'pressure': pressure,
            'gas_resistance': gas_resistance
        }
        
        self.readings.append(reading)
        
        # Keep only last 100 readings
        if len(self.readings) > 100:
            self.readings = self.readings[-100:]
        
        return self.calculate_iaq()
    
    def calculate_iaq(self):
        """Calculate comprehensive IAQ metrics"""
        
        if len(self.readings) < 10:
            return self._initial_response()
        
        # Get recent data
        recent_data = pd.DataFrame(self.readings[-50:])
        current = self.readings[-1]
        
        # Establish gas baseline (running average of clean readings)
        if not self.calibrated:
            self._establish_baseline(recent_data)
        
        # Calculate individual component scores
        gas_score = self._calculate_gas_score(current['gas_resistance'])
        humidity_score = self._calculate_humidity_score(current['humidity'])
        temperature_score = self._calculate_temperature_score(current['temperature'])
        
        # Stability factor
        stability_score = self._calculate_stability(recent_data)
        
        # Combined IAQ (0-500 scale)
        iaq = self._combine_scores(gas_score, humidity_score, temperature_score, stability_score)
        
        # Additional metrics
        co2_equivalent = self._estimate_co2(current['gas_resistance'])
        voc_equivalent = self._estimate_voc(current['gas_resistance'])
        
        return {
            'iaq': round(iaq, 1),
            'iaq_accuracy': self._get_accuracy(),
            'gas_resistance': current['gas_resistance'],
            'co2_equivalent': round(co2_equivalent, 1),
            'voc_equivalent': round(voc_equivalent, 3),
            'air_quality_text': self._iaq_to_text(iaq),
            'calibration_status': 'Calibrated' if self.calibrated else 'Calibrating',
            'stability': stability_score
        }
    
    def _establish_baseline(self, data):
        """Establish gas resistance baseline"""
        if len(data) >= 20:
            # Use median of stable readings
            gas_values = data['gas_resistance'].values
            stable_readings = gas_values[gas_values > np.percentile(gas_values, 25)]
            self.gas_baseline = np.median(stable_readings)
            
            # Mark as calibrated after burn-in period
            if datetime.now() - self.start_time > self.burn_in_time:
                self.calibrated = True
                print(f"IAQ calibrated! Baseline: {self.gas_baseline:.0f} Ohms")
    
    def _calculate_gas_score(self, gas_resistance):
        """Calculate gas quality score (0-100)"""
        if self.gas_baseline is None:
            return 50
        
        ratio = gas_resistance / self.gas_baseline
        if ratio > 1:
            # Better than baseline
            score = 50 - (ratio - 1) * 50
        else:
            # Worse than baseline
            score = 50 + (1 - ratio) * 50
        
        return max(0, min(100, score))
    
    def _calculate_humidity_score(self, humidity):
        """Calculate humidity comfort score"""
        optimal = 40.0
        if 30 <= humidity <= 50:
            score = 100 - abs(humidity - optimal) * 2
        else:
            score = 100 - abs(humidity - optimal) * 3
        
        return max(0, min(100, score))
    
    def _calculate_temperature_score(self, temperature):
        """Temperature comfort contribution"""
        # Optimal range 20-25°C
        if 20 <= temperature <= 25:
            return 100
        elif 18 <= temperature <= 28:
            return 80
        else:
            return max(0, 100 - abs(temperature - 22.5) * 5)
    
    def _calculate_stability(self, data):
        """Calculate measurement stability"""
        if len(data) < 10:
            return 50
        
        gas_std = data['gas_resistance'].std()
        gas_mean = data['gas_resistance'].mean()
        
        stability = 100 - min(50, (gas_std / gas_mean) * 1000)
        return max(0, stability)
    
    def _combine_scores(self, gas_score, humidity_score, temp_score, stability):
        """Combine all scores into final IAQ"""
        # Weighted combination
        iaq = (
            gas_score * 0.6 +          # Gas is most important
            humidity_score * 0.2 +      # Humidity secondary
            temp_score * 0.1 +          # Temperature minor
            stability * 0.1             # Stability factor
        )
        
        # Convert to 0-500 scale (inverted - lower is better)
        iaq = (100 - iaq) * 5
        return max(0, min(500, iaq))
    
    def _estimate_co2(self, gas_resistance):
        """Estimate CO2 equivalent (ppm)"""
        if self.gas_baseline is None:
            return 400
        
        ratio = self.gas_baseline / gas_resistance
        co2_eq = 400 + (ratio - 1) * 1000
        return max(400, min(5000, co2_eq))
    
    def _estimate_voc(self, gas_resistance):
        """Estimate VOC equivalent (mg/m³)"""
        if self.gas_baseline is None:
            return 0.5
        
        ratio = self.gas_baseline / gas_resistance
        voc_eq = (ratio - 1) * 2
        return max(0, min(10, voc_eq))
    
    def _get_accuracy(self):
        """Get calibration accuracy (0-3 like BSEC)"""
        if not self.calibrated:
            return 0
        elif len(self.readings) < 50:
            return 1
        elif len(self.readings) < 100:
            return 2
        else:
            return 3
    
    def _iaq_to_text(self, iaq):
        """Convert IAQ number to text"""
        if iaq <= 50:
            return "Excellent"
        elif iaq <= 100:
            return "Good"
        elif iaq <= 150:
            return "Lightly Polluted"
        elif iaq <= 200:
            return "Moderately Polluted"
        elif iaq <= 300:
            return "Heavily Polluted"
        else:
            return "Severely Polluted"
    
    def _initial_response(self):
        """Response during initial readings"""
        return {
            'iaq': 25,
            'iaq_accuracy': 0,
            'gas_resistance': self.readings[-1]['gas_resistance'] if self.readings else 0,
            'co2_equivalent': 400,
            'voc_equivalent': 0.5,
            'air_quality_text': 'Initializing',
            'calibration_status': 'Starting up',
            'stability': 0
        }