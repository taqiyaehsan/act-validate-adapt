from flask import Flask, request, jsonify
from waitress import serve
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import math

app = Flask(__name__)

# the latest data
latest_data = {
    'timestamp': None,
    'temperature': None,
    'pressure': None,
    'humidity': None,
    'gas_resistance': None,
    'iaq': None,
    'iaq_accuracy': None,
    'static_iaq': None,
    'co2_equivalent': None,
    'breath_voc_equivalent': None,
    'gas_percentage': None,
    'calibration_status': None,
    'baseline_gas': None,
}

class BME680_IAQ:
    def __init__(self):
        self.readings = deque(maxlen=300)  # 15 minutes at 3s intervals
        self.gas_baseline = None
        self.gas_upper_limit = None
        self.gas_lower_limit = None
        self.humidity_baseline = 40.0
        self.humidity_weighting = 0.25
        self.gas_weighting = 0.75
        self.calibration_data = []
        self.burn_in_cycles = 150  # ~7.5 minutes
        self.gas_reference = 250000
        self.humidity_reference = 40.0
        
    def add_reading(self, temperature, pressure, humidity, gas_resistance):
        """Process new sensor reading using BME680-IAQ algorithm"""
        reading = {
            'timestamp': datetime.now(),
            'temperature': temperature,
            'pressure': pressure,
            'humidity': humidity,
            'gas_resistance': gas_resistance
        }
        
        self.readings.append(reading)
        self.calibration_data.append(gas_resistance)
        
        # Keep calibration data manageable
        if len(self.calibration_data) > 300:
            self.calibration_data = self.calibration_data[-300:]
        
        return self.calculate_iaq()
    
    def calculate_iaq(self):
        """BME680-IAQ calculation algorithm"""
        if len(self.readings) < 10:
            return self._initial_response()
        
        current = self.readings[-1]
        
        # Establish baseline during burn-in
        if len(self.calibration_data) >= self.burn_in_cycles and self.gas_baseline is None:
            self._establish_baseline()
        
        # Calculate IAQ components
        gas_score = self._calculate_gas_score(current['gas_resistance'])
        humidity_score = self._calculate_humidity_score(current['humidity'])
        
        # Combine scores (BME680-IAQ weighting)
        air_quality_score = (self.humidity_weighting * humidity_score) + (self.gas_weighting * gas_score)
        
        # Convert to 0-500 IAQ scale
        iaq = self._score_to_iaq(air_quality_score)
        
        # Additional metrics
        co2_equivalent = self._calculate_co2_equivalent(current['gas_resistance'])
        breath_voc = self._calculate_breath_voc(current['gas_resistance'])
        gas_percentage = self._calculate_gas_percentage(current['gas_resistance'])
        
        return {
            'iaq': round(iaq, 1),
            'iaq_accuracy': self._get_accuracy(),
            'static_iaq': round(iaq, 1),  # Simplified - same as IAQ
            'co2_equivalent': round(co2_equivalent, 1),
            'breath_voc_equivalent': round(breath_voc, 2),
            'gas_percentage': round(gas_percentage, 2),
            'gas_resistance': current['gas_resistance'],
            'humidity_score': round(humidity_score, 1),
            'gas_score': round(gas_score, 1),
            'air_quality_text': self._iaq_to_text(iaq),
            'calibration_status': 'Calibrated' if self.gas_baseline else 'Calibrating',
            'baseline_gas': self.gas_baseline,
            'run_in_status': min(100, (len(self.calibration_data) / self.burn_in_cycles) * 100)
        }
    
    def _establish_baseline(self):
        """Establish gas baseline using statistical analysis"""
        gas_data = np.array(self.calibration_data)
        
        # Remove outliers (beyond 2 standard deviations)
        mean = np.mean(gas_data)
        std = np.std(gas_data)
        filtered_data = gas_data[np.abs(gas_data - mean) <= 2 * std]
        
        # Set baseline as median of filtered data
        self.gas_baseline = np.median(filtered_data)
        self.gas_upper_limit = np.percentile(filtered_data, 75)
        self.gas_lower_limit = np.percentile(filtered_data, 25)
        
        print(f"🎯 Baseline established: {self.gas_baseline:.0f} Ohms")
        print(f"   Range: {self.gas_lower_limit:.0f} - {self.gas_upper_limit:.0f} Ohms")
    
    def _calculate_gas_score(self, gas_resistance):
        """Calculate gas quality score (0-100)"""
        if self.gas_baseline is None:
            return 50  # Neutral during calibration
        
        # BME680-IAQ approach: logarithmic scaling
        if gas_resistance > self.gas_baseline:
            # Better than baseline (cleaner air)
            ratio = gas_resistance / self.gas_baseline
            gas_score = 100 - (math.log(ratio) * 25)
        else:
            # Worse than baseline (polluted air)
            ratio = self.gas_baseline / gas_resistance
            gas_score = 100 - (math.log(ratio) * 25)
        
        return max(0, min(100, gas_score))
    
    def _calculate_humidity_score(self, humidity):
        """Calculate humidity comfort score"""
        if humidity >= self.humidity_reference:
            humidity_score = (100 - self.humidity_reference) / (100 - self.humidity_reference) * (humidity - self.humidity_reference)
        else:
            humidity_score = (self.humidity_reference - 0) / (self.humidity_reference - 0) * (self.humidity_reference - humidity)
        
        return 100 - humidity_score
    
    def _score_to_iaq(self, air_quality_score):
        """Convert air quality score to IAQ (0-500 scale)"""
        # Invert scale (lower IAQ = better air quality)
        iaq = (100 - air_quality_score) * 5
        return max(0, min(500, iaq))
    
    def _calculate_co2_equivalent(self, gas_resistance):
        """Estimate CO2 equivalent (ppm)"""
        if self.gas_baseline is None:
            return 400
        
        # Logarithmic relationship based on gas resistance ratio
        ratio = self.gas_baseline / gas_resistance
        co2_eq = 400 + (math.log(max(0.1, ratio)) * 200)
        return max(400, min(8192, co2_eq))
    
    def _calculate_breath_voc(self, gas_resistance):
        """Calculate breath VOC equivalent (mg/m³)"""
        if self.gas_baseline is None:
            return 0.5
        
        ratio = self.gas_baseline / gas_resistance
        breath_voc = max(0.5, ratio - 1) * 2
        return max(0.5, min(50, breath_voc))
    
    def _calculate_gas_percentage(self, gas_resistance):
        """Calculate gas percentage relative to baseline"""
        if self.gas_baseline is None:
            return 50
        
        percentage = (gas_resistance / self.gas_baseline) * 100
        return max(0, min(100, percentage))
    
    def _get_accuracy(self):
        """Get calibration accuracy (0-3 scale like BSEC)"""
        if self.gas_baseline is None:
            return 0
        elif len(self.readings) < 50:
            return 1
        elif len(self.readings) < 150:
            return 2
        else:
            return 3
    
    def _iaq_to_text(self, iaq):
        """Convert IAQ to descriptive text"""
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
        """Response during startup"""
        return {
            'iaq': 25,
            'iaq_accuracy': 0,
            'static_iaq': 25,
            'co2_equivalent': 400,
            'breath_voc_equivalent': 0.5,
            'gas_percentage': 50,
            'gas_resistance': self.readings[-1]['gas_resistance'] if self.readings else 0,
            'humidity_score': 50,
            'gas_score': 50,
            'air_quality_text': 'Initializing',
            'calibration_status': 'Starting up',
            'baseline_gas': None,
            'run_in_status': 0
        }

# Initialize BME680-IAQ calculator
iaq_calculator = BME680_IAQ()

@app.route('/sensor-data', methods=['POST'])
def receive_sensor_data():
    try:
        data = request.json
        server_timestamp = datetime.now().isoformat()
        
        # Calculate BME680-IAQ metrics
        iaq_result = iaq_calculator.add_reading(
            temperature=data.get('temperature', 20),
            humidity=data.get('humidity', 40),
            pressure=data.get('pressure', 1013),
            gas_resistance=data.get('gas_resistance', 50000)
        )
        
        # Update latest_data for real-time endpoint
        global latest_data
        latest_data.update({
            'timestamp': server_timestamp,
            'temperature': data.get('temperature'),
            'pressure': data.get('pressure'),
            'humidity': data.get('humidity'),
            'gas_resistance': data.get('gas_resistance'),
            'iaq': iaq_result['iaq'],
            'iaq_accuracy': iaq_result['iaq_accuracy'],
            'static_iaq': iaq_result['static_iaq'],
            'co2_equivalent': iaq_result['co2_equivalent'],
            'breath_voc_equivalent': iaq_result['breath_voc_equivalent'],
            'gas_percentage': iaq_result['gas_percentage'],
            'calibration_status': iaq_result['calibration_status'],
            'baseline_gas': iaq_result.get('baseline_gas')
        })
        
        # Enhanced logging
        print(f"[{datetime.now().strftime('%H:%M:%S')}] "
              f"IAQ: {iaq_result['iaq']:.1f} ({iaq_result['air_quality_text']}) | "
              f"CO2: {iaq_result['co2_equivalent']:.0f}ppm | "
              f"VOC: {iaq_result['breath_voc_equivalent']:.2f}mg/m³ | "
              f"Acc: {iaq_result['iaq_accuracy']}/3 | "
              f"Run-in: {iaq_result['run_in_status']:.0f}%")
        
        return jsonify({
            "status": "success",
            "iaq_data": iaq_result
        })
        
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/status', methods=['GET'])
def get_status():
    return jsonify({
        "status": "running",
        "algorithm": "BME680-IAQ Open Source",
        "total_readings": len(iaq_calculator.readings),
        "calibrated": iaq_calculator.gas_baseline is not None,
        "baseline_gas": iaq_calculator.gas_baseline,
        "burn_in_progress": f"{min(100, (len(iaq_calculator.calibration_data) / iaq_calculator.burn_in_cycles) * 100):.1f}%"
    })

@app.route('/reset', methods=['POST'])
def reset_calibration():
    """Reset calibration for new environment"""
    global iaq_calculator
    iaq_calculator = BME680_IAQ()
    return jsonify({"status": "calibration reset"})



@app.route('/latest_data', methods=['GET'])
def get_latest_data():
    global latest_data
    return jsonify({
        "status": "success",
        "data": latest_data
    })


if __name__ == '__main__':
    print("Starting BME680-IAQ Open Source Server...")
    print("Status: http://192.168.0.208:5001/status")
    print("Reset: POST to http://192.168.0.208:5001/reset")
    print("Latest data: GET to http://192.168.0.208:5001/latest_data")
    serve(app, host="192.168.0.208", port=5001)