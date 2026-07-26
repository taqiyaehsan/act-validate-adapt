from flask import Flask, request, jsonify
import csv
import os
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from collections import deque
import math

app = Flask(__name__)

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
        self.current_iaq = 25.0  # Default IAQ value
        self.is_calibrated = False
        
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
        
        result = self.calculate_iaq()
        self.current_iaq = result['iaq']  # Store current IAQ
        
        # Update calibration status
        if self.gas_baseline is not None:
            self.is_calibrated = True
        
        return result
    
    def get_current_iaq(self):
        """Get the current IAQ value directly"""
        return {
            'iaq': self.current_iaq,
            'calibrated': self.is_calibrated,
            'burn_in_complete': len(self.calibration_data) >= self.burn_in_cycles,
            'burn_in_progress': min(100, (len(self.calibration_data) / self.burn_in_cycles) * 100),
            'total_readings': len(self.readings),
            'baseline_established': self.gas_baseline is not None,
            'gas_baseline': self.gas_baseline
        }
    
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

CSV_FILE = 'bme680_iaq_data.csv'
CSV_HEADERS = [
    'server_timestamp', 'esp32_timestamp_ms', 'temperature', 'pressure', 
    'humidity', 'gas_resistance', 'iaq', 'iaq_accuracy', 'static_iaq',
    'co2_equivalent', 'breath_voc_equivalent', 'gas_percentage',
    'humidity_score', 'gas_score', 'air_quality_text', 'calibration_status',
    'run_in_status'
]

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
        
        # Prepare CSV row
        row_data = {
            'server_timestamp': server_timestamp,
            'esp32_timestamp_ms': data.get('timestamp_ms', ''),
            'temperature': data.get('temperature', ''),
            'pressure': data.get('pressure', ''),
            'humidity': data.get('humidity', ''),
            'gas_resistance': data.get('gas_resistance', ''),
            'iaq': iaq_result['iaq'],
            'iaq_accuracy': iaq_result['iaq_accuracy'],
            'static_iaq': iaq_result['static_iaq'],
            'co2_equivalent': iaq_result['co2_equivalent'],
            'breath_voc_equivalent': iaq_result['breath_voc_equivalent'],
            'gas_percentage': iaq_result['gas_percentage'],
            'humidity_score': iaq_result['humidity_score'],
            'gas_score': iaq_result['gas_score'],
            'air_quality_text': iaq_result['air_quality_text'],
            'calibration_status': iaq_result['calibration_status'],
            'run_in_status': iaq_result['run_in_status']
        }
        
        write_to_csv(row_data)
        
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

@app.route('/iaq', methods=['GET'])
def get_current_iaq():
    """NEW ENDPOINT: Get current IAQ value directly"""
    try:
        iaq_data = iaq_calculator.get_current_iaq()
        return jsonify(iaq_data)
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
        "burn_in_progress": f"{min(100, (len(iaq_calculator.calibration_data) / iaq_calculator.burn_in_cycles) * 100):.1f}%",
        "burn_in_complete": len(iaq_calculator.calibration_data) >= iaq_calculator.burn_in_cycles,
        "current_iaq": iaq_calculator.current_iaq,
        "csv_file": CSV_FILE
    })

@app.route('/reset', methods=['POST'])
def reset_calibration():
    """Reset calibration for new environment"""
    global iaq_calculator
    iaq_calculator = BME680_IAQ()
    return jsonify({"status": "calibration reset"})

def write_to_csv(row_data):
    file_exists = os.path.isfile(CSV_FILE)
    
    with open(CSV_FILE, 'a', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=CSV_HEADERS)
        
        if not file_exists:
            writer.writeheader()
        
        writer.writerow(row_data)

# NEW STANDALONE FUNCTIONS FOR DIRECT ACCESS
def get_iaq_value():
    """Get current IAQ value directly (for use by other modules)"""
    return iaq_calculator.get_current_iaq()

def is_calibrated():
    """Check if the sensor is calibrated AND burn-in is complete"""
    return (iaq_calculator.gas_baseline is not None and 
            len(iaq_calculator.calibration_data) >= iaq_calculator.burn_in_cycles)

def get_calibration_progress():
    """Get calibration progress percentage"""
    return min(100, (len(iaq_calculator.calibration_data) / iaq_calculator.burn_in_cycles) * 100)

if __name__ == '__main__':
    print("🌟 Starting BME680-IAQ Open Source Server...")
    print("📊 Algorithm: Advanced gas sensor analysis with BSEC-like features")
    print("🔗 Status: http://localhost:5001/status")
    print("🔗 Current IAQ: http://localhost:5001/iaq")
    print("🔄 Reset: POST to http://localhost:5001/reset")
    print("\n📝 Direct access functions available:")
    print("   - get_iaq_value(): Get current IAQ")
    print("   - is_calibrated(): Check calibration status")
    print("   - get_calibration_progress(): Get burn-in progress")
    app.run(host='0.0.0.0', port=5001, debug=True)