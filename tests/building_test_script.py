import os
import json
import subprocess
import traceback

def run_enhanced_building_test():
    print("=== Enhanced Smart Building Diagnostic Test ===\n")

    test_script_path = "enhanced_building_test.js"

    with open(test_script_path, "w") as f:
        f.write("""
const { SmartBuilding } = require('./smart_building.js');

function validateState(state, testName) {
    const errors = [];
    
    if (!state || typeof state !== 'object') {
        errors.push(`${testName}: State is not an object`);
        return errors;
    }
    
    const requiredVars = ['Temperature', 'Humidity', 'AirQuality', 'EnergyConsumption', 'OverallSatisfaction'];
    const optionalVars = ['HVACSetpoint', 'LightingLevel', 'OccupantCount', 'ThermalComfort', 'VisualComfort'];
    
    for (const varName of requiredVars) {
        if (!(varName in state)) {
            errors.push(`${testName}: Missing required variable ${varName}`);
        } else {
            const value = state[varName];
            if (typeof value !== 'number') {
                errors.push(`${testName}: ${varName} is not a number (${typeof value})`);
            } else if (isNaN(value)) {
                errors.push(`${testName}: ${varName} is NaN`);
            } else if (!isFinite(value)) {
                errors.push(`${testName}: ${varName} is not finite`);
            }
        }
    }
    
    return errors;
}

function testInterventionFormat(intervention, testName) {
    const errors = [];
    
    if (!Array.isArray(intervention)) {
        errors.push(`${testName}: Intervention must be an array`);
        return errors;
    }
    
    intervention.forEach((item, i) => {
        if (!item.variable) errors.push(`${testName}[${i}]: Missing variable`);
        if (!item.action) errors.push(`${testName}[${i}]: Missing action`);
        if (item.value === undefined) errors.push(`${testName}[${i}]: Missing value`);
        
        if (!['set', 'increase', 'decrease'].includes(item.action)) {
            errors.push(`${testName}[${i}]: Invalid action ${item.action}`);
        }
    });
    
    return errors;
}

async function runDiagnosticTests() {
    let building;
    const testResults = {
        passed: 0,
        failed: 0,
        errors: []
    };
    
    try {
        console.log('=== INITIALIZATION TEST ===');
        building = new SmartBuilding();
        await building.initPromise;
        console.log('✓ Building initialized successfully');
        testResults.passed++;
        
        console.log('\\n=== BASELINE STATE TEST ===');
        const initialState = building.getAggregatedState();
        const stateErrors = validateState(initialState, 'Baseline');
        
        if (stateErrors.length > 0) {
            stateErrors.forEach(err => console.log('❌', err));
            testResults.errors.push(...stateErrors);
            testResults.failed++;
        } else {
            console.log('✓ Baseline state validation passed');
            testResults.passed++;
        }
        
        console.log('Baseline state:');
        console.log(JSON.stringify(initialState, null, 2));
        
        console.log('\\n=== VARIABLE NAME CASE SENSITIVITY TEST ===');
        const caseTests = [
            [{ variable: "Temperature", action: "set", value: 24 }],
            [{ variable: "temperature", action: "set", value: 24 }],
            [{ variable: "TEMPERATURE", action: "set", value: 24 }]
        ];
        
        for (let i = 0; i < caseTests.length; i++) {
            try {
                const result = await building.applyInterventions(caseTests[i]);
                if (result && result.postState) {
                    console.log(`✓ Case test ${i+1} (${caseTests[i][0].variable}) passed`);
                    testResults.passed++;
                } else {
                    console.log(`❌ Case test ${i+1} returned invalid result`);
                    testResults.failed++;
                }
            } catch (err) {
                console.log(`❌ Case test ${i+1} failed: ${err.message}`);
                testResults.failed++;
                testResults.errors.push(`Case sensitivity test ${i+1}: ${err.message}`);
            }
        }
        
        console.log('\\n=== INTERVENTION FORMAT VALIDATION TEST ===');
        const formatTests = [
            { name: "Valid basic", intervention: [{ variable: "Temperature", action: "set", value: 25 }] },
            { name: "Missing variable", intervention: [{ action: "set", value: 25 }] },
            { name: "Missing action", intervention: [{ variable: "Temperature", value: 25 }] },
            { name: "Missing value", intervention: [{ variable: "Temperature", action: "set" }] },
            { name: "Invalid action", intervention: [{ variable: "Temperature", action: "invalid", value: 25 }] },
            { name: "Non-array", intervention: { variable: "Temperature", action: "set", value: 25 } },
        ];
        
        for (const test of formatTests) {
            const formatErrors = testInterventionFormat(test.intervention, test.name);
            
            try {
                const result = await building.applyInterventions(test.intervention);
                if (formatErrors.length > 0) {
                    console.log(`❌ ${test.name}: Should have failed validation but didn't`);
                    testResults.failed++;
                } else {
                    console.log(`✓ ${test.name}: Passed as expected`);
                    testResults.passed++;
                }
            } catch (err) {
                if (formatErrors.length > 0) {
                    console.log(`✓ ${test.name}: Failed as expected (${err.message})`);
                    testResults.passed++;
                } else {
                    console.log(`❌ ${test.name}: Unexpectedly failed (${err.message})`);
                    testResults.failed++;
                    testResults.errors.push(`${test.name}: ${err.message}`);
                }
            }
        }
        
        console.log('\\n=== BOUNDARY VALUE TEST ===');
        const boundaryTests = [
            { name: "Temperature high", intervention: [{ variable: "Temperature", action: "set", value: 50 }] },
            { name: "Temperature low", intervention: [{ variable: "Temperature", action: "set", value: -10 }] },
            { name: "Humidity high", intervention: [{ variable: "Humidity", action: "set", value: 150 }] },
            { name: "Humidity low", intervention: [{ variable: "Humidity", action: "set", value: -20 }] },
            { name: "Lighting high", intervention: [{ variable: "LightingLevel", action: "set", value: 5.0 }] },
            { name: "Lighting negative", intervention: [{ variable: "LightingLevel", action: "set", value: -0.5 }] }
        ];
        
        for (const test of boundaryTests) {
            try {
                const result = await building.applyInterventions(test.intervention);
                if (result && result.postState) {
                    const postStateErrors = validateState(result.postState, test.name);
                    if (postStateErrors.length > 0) {
                        console.log(`❌ ${test.name}: State validation failed after intervention`);
                        postStateErrors.forEach(err => console.log('  ', err));
                        testResults.failed++;
                    } else {
                        console.log(`✓ ${test.name}: Handled gracefully`);
                        testResults.passed++;
                    }
                } else {
                    console.log(`❌ ${test.name}: Returned invalid result structure`);
                    testResults.failed++;
                }
            } catch (err) {
                console.log(`❌ ${test.name}: Crashed with error: ${err.message}`);
                testResults.failed++;
                testResults.errors.push(`${test.name}: ${err.message}`);
            }
        }
        
        console.log('\\n=== CAUSAL FRAMEWORK COMPATIBILITY TEST ===');
        const causalTests = [
            {
                name: "Temperature->Energy",
                intervention: [{ variable: "Temperature", action: "increase", value: 5 }],
                expectedEffects: ["EnergyConsumption"]
            },
            {
                name: "Humidity effect",
                intervention: [{ variable: "Humidity", action: "increase", value: 10 }],
                expectedEffects: ["ThermalComfort", "OverallSatisfaction"]
            },
            {
                name: "Lighting->Energy",
                intervention: [{ variable: "LightingLevel", action: "set", value: 1.0 }],
                expectedEffects: ["EnergyConsumption"]
            }
        ];
        
        for (const test of causalTests) {
            try {
                const result = await building.applyInterventions(test.intervention);
                
                if (!result || !result.preState || !result.postState) {
                    console.log(`❌ ${test.name}: Invalid result structure`);
                    testResults.failed++;
                    continue;
                }
                
                const preStateErrors = validateState(result.preState, `${test.name}-pre`);
                const postStateErrors = validateState(result.postState, `${test.name}-post`);
                
                if (preStateErrors.length > 0 || postStateErrors.length > 0) {
                    console.log(`❌ ${test.name}: State validation failed`);
                    [...preStateErrors, ...postStateErrors].forEach(err => console.log('  ', err));
                    testResults.failed++;
                    continue;
                }
                
                // Check for measurable effects
                let hasEffect = false;
                for (const effectVar of test.expectedEffects) {
                    if (effectVar in result.preState && effectVar in result.postState) {
                        const change = Math.abs(result.postState[effectVar] - result.preState[effectVar]);
                        if (change > 0.01) {  // Threshold for detectable change
                            hasEffect = true;
                            console.log(`✓ ${test.name}: Detected ${effectVar} change (${change.toFixed(3)})`);
                        }
                    }
                }
                
                if (hasEffect) {
                    testResults.passed++;
                } else {
                    console.log(`⚠️  ${test.name}: No measurable effects detected`);
                    testResults.failed++;
                }
                
            } catch (err) {
                console.log(`❌ ${test.name}: Failed with error: ${err.message}`);
                testResults.failed++;
                testResults.errors.push(`${test.name}: ${err.message}`);
            }
        }
        
        console.log('\\n=== SIMULATION STABILITY TEST ===');
        try {
            for (let i = 0; i < 5; i++) {
                await building.update();
                const state = building.getAggregatedState();
                const stateErrors = validateState(state, `Update ${i+1}`);
                
                if (stateErrors.length > 0) {
                    console.log(`❌ Update ${i+1}: State validation failed`);
                    stateErrors.forEach(err => console.log('  ', err));
                    testResults.failed++;
                } else {
                    console.log(`✓ Update ${i+1}: State valid`);
                    testResults.passed++;
                }
            }
        } catch (err) {
            console.log(`❌ Simulation updates failed: ${err.message}`);
            testResults.failed++;
            testResults.errors.push(`Simulation updates: ${err.message}`);
        }
        
    } catch (err) {
        console.log(`❌ Critical error: ${err.message}`);
        console.log(`Stack trace: ${err.stack}`);
        testResults.failed++;
        testResults.errors.push(`Critical: ${err.message}`);
    } finally {
        if (building) {
            try {
                building.cleanup();
                console.log('✓ Cleanup completed');
            } catch (err) {
                console.log(`❌ Cleanup failed: ${err.message}`);
            }
        }
    }
    
    console.log('\\n=== TEST SUMMARY ===');
    console.log(`Passed: ${testResults.passed}`);
    console.log(`Failed: ${testResults.failed}`);
    console.log(`Total: ${testResults.passed + testResults.failed}`);
    
    if (testResults.errors.length > 0) {
        console.log('\\nErrors encountered:');
        testResults.errors.forEach(err => console.log(`  - ${err}`));
    }
    
    const successRate = testResults.passed / (testResults.passed + testResults.failed) * 100;
    console.log(`\\nSuccess rate: ${successRate.toFixed(1)}%`);
    
    if (successRate >= 90) {
        console.log('✅ SIMULATION READY FOR CAUSAL FRAMEWORK');
    } else if (successRate >= 70) {
        console.log('⚠️ SIMULATION NEEDS MINOR FIXES');
    } else {
        console.log('❌ SIMULATION NEEDS MAJOR FIXES');
    }
    
    process.exit(successRate >= 90 ? 0 : 1);
}

runDiagnosticTests().catch(err => {
    console.error("Diagnostic test crashed:", err);
    process.exit(1);
});
        """)

    try:
        print("Running enhanced diagnostic tests...")
        process = subprocess.Popen(
            ["node", test_script_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        stdout, stderr = process.communicate(timeout=120) # 2 minute timeout
        
        print("=== STDOUT ===")
        print(stdout)
        
        if stderr:
            print("\n=== STDERR ===")
            print(stderr)
            
        print(f"\n=== EXIT CODE: {process.returncode} ===")
        
        if process.returncode != 0:
            print("❌ Tests failed - simulation needs fixes before causal framework use")
        else:
            print("✅ Tests passed - simulation ready for causal framework")
            
    except subprocess.TimeoutExpired:
        print("❌ Test timed out - simulation may be hanging")
        process.kill()
    except Exception as e:
        print(f"❌ Test execution failed: {e}")
        traceback.print_exc()
    finally:
        if os.path.exists(test_script_path):
            os.remove(test_script_path)

if __name__ == "__main__":
    run_enhanced_building_test()