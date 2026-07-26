"""
Simulator Configuration Registry
=================================
Each simulator gets a config dict with all domain-specific info.
The framework reads from config instead of hardcoded globals.
"""

SMART_BUILDING_RICH = {
    'name': 'smart_building_rich',
    'domain_description': (
        'A 15-variable smart building with HVAC and lighting control. '
        'The building has hidden regime variables: occupancy (0-8 people, '
        'office schedule) and window position (0-1, temperature-driven). '
        'Sensors measure temperature, humidity, CO2, light, noise, and air quality.'
    ),

    # Paths
    'sim_path': 'js/smart_building_rich.js',
    'data_path': 'data_regen/smart_building_rich_processed.csv',
    'scaling_path': 'data_regen/smart_building_rich_processed_scaling.csv',
    'scaling_format': 'minmax',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'smart_building_rich',

    # Variables
    'all_vars': [
        'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
        'temperature', 'humidity', 'co2', 'lightlevel', 'noisedb',
        'airquality', 'pmv', 'hvacpower', 'lightingpower',
        'energyconsumption', 'satisfaction',
    ],
    'latent_vars': {'occupancy', 'windowposition'},
    'sensor_vars': [
        'temperature', 'humidity', 'co2', 'lightlevel', 'noisedb', 'airquality',
    ],
    'actuator_vars': ['hvacpower', 'lightingpower'],
    'non_intervenable_vars': [
        'outdoortemp', 'solarradiation', 'occupancy', 'windowposition',
        'pmv', 'energyconsumption', 'satisfaction',
    ],

    # Regime detection
    'regime_vars': {
        'occupancy': {'threshold': 0.1, 'label': 'occ'},
        'windowposition': {'threshold': 0.15, 'label': 'win'},
    },
    'regime_names': ['base', 'occ', 'win', 'full'],

    # Test data generation
    'test_duration_steps': 5760,  # 4 days
    'day_offsets': {0: -3.0, 1: 6.0, 2: 13.0, 3: 6.0},
    'forced_regimes': {
        # Day 3 evening: force window open for win-only regime
        'day': 3,
        'hours': (19, 7),  # hour_of_day >= 19 or < 7
        'intervention': {'windowPosition': 0.6},
    },

    # Physical constants
    'hvac_max_kw': 5.0,
    'light_max_kw': 0.5,
    'comfort_ref_c': 22.0,
    'comfort_deadband_c': 1.0,

}


OPEN_WINDOW = {
    'name': 'open_window',
    'domain_description': (
        'An 8-variable smart room with a window that opens/closes on a schedule. '
        'The room has no direct actuator control — only passive sensors. '
        'The hidden regime variable is window state (open/closed). '
        'Sensors measure temperature, humidity, air quality, PMV, energy, and satisfaction.'
    ),

    # Paths
    'sim_path': 'js/open_window.js',
    'data_path': 'data/challenge1_data_10k_processed.csv',
    'scaling_path': 'data/challenge1_data_10k_processed_scaling.csv',
    'scaling_format': 'minmax',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'open_window',

    # Variables
    'all_vars': [
        'temperature', 'humidity', 'airquality', 'pmv',
        'energyconsumption', 'satisfaction', 'windowopen', 'outdoortemperature',
    ],
    'latent_vars': {'windowopen', 'outdoortemperature'},
    'sensor_vars': [
        'temperature', 'humidity', 'airquality', 'pmv',
    ],
    'actuator_vars': ['temperature', 'humidity', 'airquality'],
    'non_intervenable_vars': [
        'outdoortemperature', 'pmv',
        'energyconsumption', 'satisfaction',
    ],

    # Regime detection
    'regime_vars': {
        'windowopen': {'threshold': 0.5, 'label': 'win'},
    },
    'regime_names': ['closed', 'open'],

    # Test data generation
    'test_duration_steps': 4320,  # 3 days
    'day_offsets': {0: 0.0, 1: 0.0, 2: 0.0},
    'forced_regimes': None,  # No forcing needed, window cycles naturally

    # Physical constants (open_window has simpler physics)
    'hvac_max_kw': None,  # No HVAC
    'light_max_kw': None,
    'comfort_ref_c': 22.0,
    'comfort_deadband_c': 1.0,
    
}


SMART_ROOM = {
    'name': 'smart_room',
    'domain_description': (
        'A 5-variable smart room simulation with temperature, humidity, and '
        'air quality as input variables driving energy consumption and occupant '
        'satisfaction. No hidden confounders, fully observable.'
    ),

    # Paths
    'sim_path': 'js/smart_room.js',
    'data_path': 'data_regen/smart_room_processed.csv',
    'scaling_path': 'data_regen/smart_room_processed_scaling.csv',
    'scaling_format': 'minmax',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'smart_room',

    # Variables
    'all_vars': [
        'temperature', 'humidity', 'airquality',
        'energyconsumption', 'satisfaction',
    ],
    'latent_vars': set(),
    'sensor_vars': ['temperature', 'humidity', 'airquality'],
    'actuator_vars': ['temperature', 'humidity', 'airquality'],
    'non_intervenable_vars': [
        'energyconsumption', 'satisfaction',
    ],

    # Regime detection
    'regime_vars': {},
    'regime_names': [],

    # Test data generation
    'test_duration_steps': 0,
    'day_offsets': {},
    'forced_regimes': None,

    # Physical constants
    'hvac_max_kw': None,
    'light_max_kw': None,
    'comfort_ref_c': 22.0,
    'comfort_deadband_c': 1.0,
    
}


SMART_ROOM_NOISE = {
    'name': 'smart_room_noise',
    'domain_description': (
        'A 5-variable smart room simulation with Gaussian measurement noise '
        'added to sensor readings. Tests robustness of causal discovery under '
        'realistic sensor uncertainty (±0.2°C temp, ±2% humidity, ±15 AQI).'
    ),

    # Paths
    'sim_path': 'js/smart_room_noise.js',
    'data_path': 'data_regen/smart_room_noise_processed.csv',
    'scaling_path': 'data_regen/smart_room_noise_processed_scaling.csv',
    'scaling_format': 'minmax',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'smart_room_noise',

    # Variables
    'all_vars': [
        'temperature', 'humidity', 'airquality',
        'energyconsumption', 'satisfaction',
    ],
    'latent_vars': set(),
    'sensor_vars': ['temperature', 'humidity', 'airquality'],
    'actuator_vars': ['temperature', 'humidity', 'airquality'],
    'non_intervenable_vars': [
        'energyconsumption', 'satisfaction',
    ],

    # Regime detection
    'regime_vars': {},
    'regime_names': [],

    # Test data generation
    'test_duration_steps': 0,
    'day_offsets': {},
    'forced_regimes': None,

    # Physical constants
    'hvac_max_kw': None,
    'light_max_kw': None,
    'comfort_ref_c': 22.0,
    'comfort_deadband_c': 1.0,
    
}


SMART_ROOM_HIDDEN_VARS = {
    'name': 'smart_room_hidden_vars',
    'domain_description': (
        'A 6-variable smart room with hidden confounders. OutdoorTemperature '
        'is a latent variable that influences observed sensors but is not '
        'directly measured. Tests causal discovery under partial observability.'
    ),

    # Paths
    'sim_path': 'js/smart_room_hidden_vars.js',
    'data_path': 'data_regen/hidden_vars_processed.csv',
    'scaling_path': 'data_regen/hidden_vars_processed_scaling.csv',
    'scaling_format': 'minmax',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'hidden_vars',

    # Variables
    'all_vars': [
        'temperature', 'humidity', 'airquality',
        'energyconsumption', 'satisfaction', 'outdoortemperature',
    ],
    'latent_vars': {'outdoortemperature'},
    'sensor_vars': ['temperature', 'humidity', 'airquality'],
    'actuator_vars': ['temperature', 'humidity', 'airquality'],
    'non_intervenable_vars': [
        'energyconsumption', 'satisfaction', 'outdoortemperature',
    ],

    # Regime detection
    'regime_vars': {},
    'regime_names': [],

    # Test data generation
    'test_duration_steps': 0,
    'day_offsets': {},
    'forced_regimes': None,

    # Physical constants
    'hvac_max_kw': None,
    'light_max_kw': None,
    'comfort_ref_c': 22.0,
    'comfort_deadband_c': 1.0,
    
}


ASHRAE = {
    'name': 'ashrae',
    'domain_description': (
        'Real-world building energy dataset from ASHRAE Great Energy Predictor III. '
        '6 variables: air temperature, dew point, sea-level pressure (weather), '
        'square footage, year built (building properties), and meter reading (energy). '
        'Causal structure: weather and building properties drive energy consumption.'
    ),

    # Paths
    'sim_path': None,  # No JS simulator — uses RF surrogate for interventions
    'data_path': 'data/ashrae_data_processed.csv',
    'scaling_path': 'data/ashrae_data_processed_scaling_params.csv',
    'scaling_format': 'zscore',
    'gt_path': 'ground_truth_graphs.json',
    'gt_key': 'ashrae',

    # Variables
    'all_vars': [
        'air_temperature', 'dew_temperature', 'sea_level_pressure',
        'meter_reading', 'square_feet', 'year_built',
    ],
    'latent_vars': set(),
    'sensor_vars': [
        'air_temperature', 'dew_temperature', 'sea_level_pressure',
    ],
    'actuator_vars': ['air_temperature', 'dew_temperature', 'sea_level_pressure',
                      'square_feet', 'year_built'],
    'non_intervenable_vars': [],  # RF surrogate handles all interventions
    'drop_columns': ['weight'],  # Extra column to drop from data

    # Regime detection
    'regime_vars': {},
    'regime_names': [],

    # Test data generation
    'test_duration_steps': 0,
    'day_offsets': {},
    'forced_regimes': None,

    # Physical constants
    'hvac_max_kw': None,
    'light_max_kw': None,
    'comfort_ref_c': None,
    'comfort_deadband_c': None,
    
}


# Registry
SIM_CONFIGS = {
    'smart_building_rich': SMART_BUILDING_RICH,
    'open_window': OPEN_WINDOW,
    'smart_room': SMART_ROOM,
    'smart_room_noise': SMART_ROOM_NOISE,
    'smart_room_hidden_vars': SMART_ROOM_HIDDEN_VARS,
    'ashrae': ASHRAE,
}


def get_config(sim_name):
    """Get simulator configuration by name."""
    if sim_name not in SIM_CONFIGS:
        raise ValueError(f"Unknown simulator: {sim_name}. "
                         f"Available: {list(SIM_CONFIGS.keys())}")
    config = SIM_CONFIGS[sim_name]
    # Derive observable vars
    config['observable_vars'] = [v for v in config['all_vars']
                                  if v not in config['latent_vars']]
    return config
