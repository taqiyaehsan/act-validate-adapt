// python_interface.js
const { spawn } = require('child_process');
const PMVCache = require('./pmv_cache');

class PsychrometricCalculator {
    constructor() {
        this.cache = new PMVCache();
        this.pythonProcess = null;
        this.pendingRequests = new Map();
        this.requestCounter = 0;
        this._initPythonProcess();
    }

    _initPythonProcess() {
        // Add process error handling and restart mechanism
        if (this.pythonProcess) {
            try {
                this.pythonProcess.kill('SIGKILL');
            } catch (e) {
                console.error('Error killing previous Python process:', e);
            }
        }

        const pythonScript = `
import sys
import json
import math
from pythermalcomfort.models import pmv_ppd_iso
from pythermalcomfort.utilities import v_relative

while True:
    try:
        input_data = input()
        request = json.loads(input_data)
        
        tdb = request['temperature']
        rh = request['humidity']
        v = request['airSpeed']
        request_id = request['id']
        
        v_r = v_relative(v=v, met=1.2)
        results = pmv_ppd_iso(
            tdb=tdb, 
            tr=tdb,
            vr=v_r, 
            rh=rh, 
            met=1.2,
            clo=0.7,
            model="7730-2005"
        )
        
        try:
            pmv_value = float(results.pmv)
            ppd_value = float(results.ppd)
            # Handle NaN values explicitly
            if math.isnan(pmv_value) or math.isnan(ppd_value):
                response = {
                    'id': request_id,
                    'pmv': 0,
                    'ppd': 0,
                    'error': 'Calculation returned NaN'
                }
            else:
                response = {
                    'id': request_id,
                    'pmv': pmv_value,
                    'ppd': ppd_value
                }
        except Exception as e:
            response = {
                'id': request_id,
                'pmv': 0,
                'ppd': 0,
                'error': str(e)
            }
        
        print(json.dumps(response))
        sys.stdout.flush()
        
    except EOFError:
        break
    except Exception as e:
        error_response = {
            'id': request_id,
            'error': str(e)
        }
        print(json.dumps(error_response))
        sys.stdout.flush()
`;
        
        this.pythonProcess = spawn('python', ['-c', pythonScript]);

        // Set up a restart timer in case process hangs
        if (this.restartTimer) clearTimeout(this.restartTimer);
        this.restartTimer = setTimeout(() => {
            console.log('Restarting Python process due to inactivity');
            this._initPythonProcess();
        }, 60000); // Restart after 60 seconds of inactivity
            
        let buffer = '';
        this.pythonProcess.stdout.on('data', (data) => {
            buffer += data.toString();
            let lines = buffer.split('\n');
            buffer = lines.pop();
            
            for (const line of lines) {
                if (!line.trim()) continue;
                try {
                    const response = JSON.parse(line.trim());
                    const resolve = this.pendingRequests.get(response.id);
                    if (resolve) {
                        this.pendingRequests.delete(response.id);
                        if (response.error) {
                            resolve({ pmv: 0, ppd: 0 });
                        } else {
                            resolve(response);
                        }
                    }
                } catch (error) {
                    console.error('Error parsing line:', line);
                    console.error('Parse error:', error);
                }
            }
        });

        this.pythonProcess.stderr.on('data', (data) => {
            console.error(`Python Error: ${data}`);
        });

        this.pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            return; 
        });

        this.pythonProcess.on('error', (error) => {
            console.error(`Python process error: ${error}`);
            this._restartPythonProcess();
        });
        
        this.pythonProcess.on('close', (code) => {
            console.log(`Python process exited with code ${code}`);
            return; 
        });
    
    }

    _restartPythonProcess() {
        console.log("Restarting Python process...");
        if (this.pythonProcess) {
            try {
                this.pythonProcess.kill();
            } catch (e) {
                console.error("Error killing process:", e);
            }
        }
        
        setTimeout(() => {
            this._initPythonProcess();
            console.log("Python process restarted");
        }, 1000);
    }

    async calculatePMV(temperature, humidity, airSpeed = 0.1, meanRadiantTemp = 22) {
        const cachedResult = this.cache.get(temperature, humidity, airSpeed);
        if (cachedResult) return cachedResult;

        // Use stub if Python failed
        if (this.useStub) {
            return this._calculateStub(temperature, humidity);
        }

        const requestId = this.requestCounter++;
        const request = { id: requestId, temperature, humidity, airSpeed };

        return new Promise((resolve) => {
            const timeout = setTimeout(() => {
                this.pendingRequests.delete(requestId);
                // Fall back to stub on timeout
                resolve(this._calculateStub(temperature, humidity));
            }, 2000);
            
            this.pendingRequests.set(requestId, (result) => {
                clearTimeout(timeout);
                this.cache.set(temperature, humidity, airSpeed, result);
                resolve(result);
            });
            
            try {
                this.pythonProcess.stdin.write(JSON.stringify(request) + '\n');
            } catch (e) {
                clearTimeout(timeout);
                this.useStub = true;
                resolve(this._calculateStub(temperature, humidity));
            }
        });
    }

    _calculateStub(temperature, humidity) {
        const pmv = ((temperature - 22) * 0.3) + ((humidity - 50) * 0.01);
        const clampedPMV = Math.max(-3, Math.min(3, pmv));
        const ppd = 100 - 95 * Math.exp(-0.03353 * Math.pow(clampedPMV, 4) - 0.2179 * Math.pow(clampedPMV, 2));
        
        return {
            pmv: clampedPMV,
            ppd: Math.max(5, Math.min(100, ppd))
        };
    }

    async cleanup() {
        if (this.pythonProcess) {
            this.pythonProcess.kill();
        }
        this.cache.clear();
    }
}

module.exports = PsychrometricCalculator;