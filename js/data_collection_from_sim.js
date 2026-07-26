// change the filename as needed
const { generateSimulationData } = require('./smart_building.js');

// Generate 24 hours (24 * 60 * 60 * 1000ms) of data with 15-minute (15 * 60 * 1000ms) intervals
// Parameters: duration in ms, interval in ms
generateSimulationData(300000000, 300000) // 5-minute intervals over 3.5 days
  .then(filename => {
    console.log(`Data collection complete! Data saved to ${filename}`);
  })
  .catch(error => {
    console.error("Data collection failed:", error);
  });