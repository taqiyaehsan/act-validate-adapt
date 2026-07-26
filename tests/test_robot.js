const { RobotArm, simulateAndGetLatestData } = require('../robot_arm.js');

// close port: kill -9 $(lsof -ti:5001)

async function test() {
    // Test using direct RobotArm class
    console.log('===== TESTING DIRECT ROBOTARM CLASS =====');
    const robotArm = new RobotArm();
    
    // Test baseline
    console.log('Baseline State:', robotArm.getCurrentState());
    
    // Test with modifications - move hand to ball position
    console.log('\nApplying intervention: moving hand to ball position');
    
    const interventions = [
        {
            variable: 'handX',
            action: 'set',
            value: 100  // Move handX to ballX position
        },
        {
            variable: 'handY',
            action: 'set',
            value: 100  // Move handY to ballY position
        }
    ];
    
    // Apply interventions and get results
    const result = await robotArm.applyInterventions(interventions);
    
    console.log('\nPre-intervention State:', result.preInterventionData);
    console.log('\nPost-intervention State:', result.postInterventionData);
    
    // Check if collision occurred
    console.log('\nIntervention result:');
    console.log(`- Hand moved to (${result.postInterventionData.handX}, ${result.postInterventionData.handY})`);
    console.log(`- Ball at (${result.postInterventionData.ballX}, ${result.postInterventionData.ballY})`);
    console.log(`- Collision detected: ${result.postInterventionData.collision}`);
    
    // Test another intervention - move hand away from ball
    console.log('\nApplying second intervention: moving hand away from ball');
    
    const interventions2 = [
        {
            variable: 'handX',
            action: 'set',
            value: 300  // Move handX far from ball
        },
        {
            variable: 'handY',
            action: 'set',
            value: 200  // Move handY far from ball
        }
    ];
    
    // Apply second intervention
    const result2 = await robotArm.applyInterventions(interventions2);
    
    console.log('\nSecond intervention result:');
    console.log(`- Hand moved to (${result2.postInterventionData.handX}, ${result2.postInterventionData.handY})`);
    console.log(`- Ball at (${result2.postInterventionData.ballX}, ${result2.postInterventionData.ballY})`);
    console.log(`- Collision detected: ${result2.postInterventionData.collision}`);
    
    // Test using the simulateAndGetLatestData function (used by Python framework)
    console.log('\n\n===== TESTING SIMULATE AND GET LATEST DATA =====');
    
    // Test the same interventions using the framework interface
    console.log('\nTesting collision case through simulateAndGetLatestData');
    const simResult = await simulateAndGetLatestData(
        5000,  // duration
        [      // interventions
            {
                variable: 'handX',
                action: 'set',
                value: 100
            },
            {
                variable: 'handY',
                action: 'set', 
                value: 100
            }
        ],
        1000   // intervention time
    );
    
    console.log('\nSimulation result with collision:');
    console.log(`- Hand at (${simResult.postInterventionData.handX}, ${simResult.postInterventionData.handY})`);
    console.log(`- Ball at (${simResult.postInterventionData.ballX}, ${simResult.postInterventionData.ballY})`);
    console.log(`- Collision: ${simResult.postInterventionData.collision}`);
    
    // Test no-collision case
    console.log('\nTesting no-collision case through simulateAndGetLatestData');
    const simResult2 = await simulateAndGetLatestData(
        5000,  // duration
        [      // interventions
            {
                variable: 'handX',
                action: 'set',
                value: 300
            },
            {
                variable: 'handY',
                action: 'set',
                value: 200
            }
        ],
        1000   // intervention time
    );
    
    console.log('\nSimulation result without collision:');
    console.log(`- Hand at (${simResult2.postInterventionData.handX}, ${simResult2.postInterventionData.handY})`);
    console.log(`- Ball at (${simResult2.postInterventionData.ballX}, ${simResult2.postInterventionData.ballY})`);
    console.log(`- Collision: ${simResult2.postInterventionData.collision}`);
    
    // Calculate distance in the non-collision case
    const dx = simResult2.postInterventionData.handX - simResult2.postInterventionData.ballX;
    const dy = simResult2.postInterventionData.handY - simResult2.postInterventionData.ballY;
    const distance = Math.sqrt(dx*dx + dy*dy);
    console.log(`- Distance between hand and ball: ${distance.toFixed(2)} units`);
    console.log(`- Collision threshold: 20 units`);
    
    robotArm.cleanup();
    console.log('\nTest completed successfully!');
}

test().catch(err => {
    console.error('Test failed:', err);
    process.exit(1);
});