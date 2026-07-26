"""
W2: Generate PID-driven training data for smart_building_rich.

The existing training CSV (data_regen/smart_building_rich_processed.csv) uses
a 90/10 split: 90% steady-state actuator at 50% + 5% min + 5% max. This
engineered actuator-context independence is itself a form of intervention
that operationalizes Proposition 1's deconfounding condition.

To test how much of Row B's performance comes from this engineered split,
this script generates a parallel CSV using a *realistic* PID controller that
responds to room temperature, with schedule-driven lighting. The resulting
actuator distribution is naturally correlated with regime context (occupancy
elevates body heat → reduces HVAC demand; window-open elevates heat loss →
raises HVAC demand). This is the "truly observational" baseline a real
building deployment would have.

Output:
  data_regen/smart_building_rich_pid_data.csv     (raw)
  data_regen/smart_building_rich_pid_processed.csv (normalized, same schema as
                                                    smart_building_rich_processed.csv)
  data_regen/smart_building_rich_pid_processed_scaling.csv

Usage:
  python generate_pid_training_data.py --steps 11520 --seed 42
"""
import os, sys, json, subprocess, argparse, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, '.')

import numpy as np
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s',
                    datefmt='%H:%M:%S')
logger = logging.getLogger(__name__)

# ── CLI ──
parser = argparse.ArgumentParser()
parser.add_argument('--steps', type=int, default=11520,
                    help='Total steps to generate (default 11520, matches existing CSV).')
parser.add_argument('--seed', type=int, default=42)
parser.add_argument('--outdoor-offset', type=float, default=0.0,
                    help='Outdoor temperature offset (used only if --single-weather).')
parser.add_argument('--single-weather', action='store_true',
                    help='Run a single fixed weather rather than cycling.')
parser.add_argument('--smoke', action='store_true',
                    help='Smoke test: 200 steps.')
parser.add_argument('--out-prefix', type=str, default='smart_building_rich_pid')
args = parser.parse_args()

if args.smoke:
    args.steps = 200

SIM_PATH = os.path.abspath('js/smart_building_rich.js')
OUT_RAW = f'data_regen/{args.out_prefix}_data.csv'
OUT_PROC = f'data_regen/{args.out_prefix}_processed.csv'
OUT_SCALE = f'data_regen/{args.out_prefix}_processed_scaling.csv'

np.random.seed(args.seed)


# ── PID controller (realistic thermostat + schedule lighting) ────────────
def pid_action(state_phys, t):
    """Realistic confounded controller.

    HVAC: PID-style heating, ramps with temperature deviation from 22 C
    setpoint within a 21-23 deadband, with mild proportional gain. Output
    in [0,1] HVAC fraction (mapped to 0..5kW by sim).

    Lighting: schedule-driven (high during 6-22h, low overnight). This
    introduces a time-of-day pattern that correlates with the simulator's
    natural occupancy schedule.

    Adds small Gaussian jitter (sigma=0.05) so SEM has a non-degenerate
    design matrix at fitting time — without jitter, lighting becomes a
    deterministic function of hour-of-day.
    """
    temp = state_phys.get('temperature', 22.0) if state_phys else 22.0
    setpoint = 22.0
    deadband = 1.0  # +/- 1 C
    Kp = 0.25       # proportional gain

    err = setpoint - temp
    if err > deadband:
        hvac = 0.6 + Kp * (err - deadband)
    elif err < -deadband:
        hvac = 0.4 - Kp * (-err - deadband)  # symmetric for cooling
    else:
        hvac = 0.5  # nominal mid-power within deadband
    hvac = float(np.clip(hvac + np.random.normal(0, 0.05), 0.0, 1.0))

    hour_of_day = (t * 60 / 3600) % 24
    if 6 <= hour_of_day < 22:
        light = 0.85 + np.random.normal(0, 0.05)
    else:
        light = 0.10 + np.random.normal(0, 0.03)
    light = float(np.clip(light, 0.0, 1.0))

    return {'hvacpower': hvac, 'lightingpower': light}


# ── Single sim step ──
def step_sim(state_phys, action, elapsed_ms, outdoor_offset=0.0,
             force_intervention=None):
    """One simulator step. action is normalized [0,1]; we map to physical
    via the sim's expected scale (HVACPower in % 0-100, LightingPower 0-100).
    force_intervention overrides specific keys (e.g. {'windowPosition': 0.6})."""
    intervention = {
        'hvacPower': action['hvacpower'] * 100.0,
        'lightingPower': action['lightingpower'] * 100.0,
    }
    if force_intervention:
        intervention.update(force_intervention)
    cmd = ['node', SIM_PATH, '--single-step',
           '--elapsed-ms', str(int(elapsed_ms))]
    if outdoor_offset:
        cmd += ['--outdoor-offset', str(outdoor_offset)]
    if state_phys:
        cmd += ['--state', json.dumps(state_phys)]
    cmd += ['--intervention', json.dumps(intervention)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
    except Exception as e:
        return None
    for line in result.stdout.split('\n'):
        if 'RESULT:' in line:
            return json.loads(line.split('RESULT:', 1)[1].strip())
    return None


# ── Generate trajectory ──
# Match the existing CSV column order exactly so downstream code is plug-and-play
COL_ORDER = [
    'OutdoorTemp', 'SolarRadiation', 'Occupancy', 'WindowPosition',
    'Temperature', 'Humidity', 'CO2', 'LightLevel', 'NoiseDB', 'AirQuality',
    'PMV', 'HVACPower', 'LightingPower', 'EnergyConsumption', 'Satisfaction',
]
JS_KEY_MAP = {  # CSV column → JS sim key
    'OutdoorTemp': 'outdoorTemp', 'SolarRadiation': 'solarRadiation',
    'Occupancy': 'occupancy', 'WindowPosition': 'windowPosition',
    'Temperature': 'temperature', 'Humidity': 'humidity',
    'CO2': 'co2', 'LightLevel': 'lightLevel', 'NoiseDB': 'noiseDB',
    'AirQuality': 'airQuality', 'PMV': 'pmv',
    'HVACPower': 'hvacPower', 'LightingPower': 'lightingPower',
    'EnergyConsumption': 'energyConsumption', 'Satisfaction': 'satisfaction',
}


def extract_row(state_phys):
    """Pull each CSV column out of the sim's physical state dict."""
    row = {}
    for col in COL_ORDER:
        js_key = JS_KEY_MAP[col]
        # Try multiple casings since the JS sim is occasionally inconsistent
        val = state_phys.get(js_key)
        if val is None:
            val = state_phys.get(col)
        if val is None:
            val = state_phys.get(col[0].lower() + col[1:])
        if val is None:
            return None  # missing column → drop row
        row[col] = float(val)
    return row


# Weather cycling: mirror run_closedloop_server.WEATHER mapping
# (cold offset -3, warm offset +6, mild offset +6 with forced window overnight)
# Cycling produces natural variation in OutdoorTemp + window-regime data.
STEPS_PER_DAY = 1440
WEATHER_CYCLE = [
    {'name': 'cold', 'offset': -3.0, 'force_window': False},
    {'name': 'warm', 'offset':  6.0, 'force_window': False},
    {'name': 'mild', 'offset':  6.0, 'force_window': True},
]

def get_weather_for_step(t):
    if args.single_weather:
        return {'name': 'fixed', 'offset': args.outdoor_offset, 'force_window': False}
    day_idx = (t // STEPS_PER_DAY) % len(WEATHER_CYCLE)
    return WEATHER_CYCLE[day_idx]


logger.info(f"Generating PID-driven trajectory: {args.steps} steps "
            f"(weather cycle: {[w['name'] for w in WEATHER_CYCLE]})")
logger.info(f"  Sim: {SIM_PATH}")
t0 = time.time()
state_phys = None
rows = []
last_action = {'hvacpower': 0.5, 'lightingpower': 0.5}

for t in range(args.steps):
    weather = get_weather_for_step(t)
    elapsed_ms = (t * 60) * 1000  # 1 minute per step
    hour_of_day = (t * 60 / 3600) % 24

    # Compute PID action from current state (or default if first step)
    action = pid_action(state_phys, t)

    # Force window open overnight on 'mild' days (mirrors closed-loop)
    force_intv = None
    if weather['force_window'] and (hour_of_day >= 19 or hour_of_day < 7):
        force_intv = {'windowPosition': 0.6}

    new_state = step_sim(state_phys, action, elapsed_ms, weather['offset'],
                         force_intv)
    if new_state is None:
        continue
    state_phys = new_state
    last_action = action

    row = extract_row(state_phys)
    if row is None:
        continue

    # Override HVACPower/LightingPower from sim with our applied action
    # (sim should already report these, but set explicitly to be safe)
    row['HVACPower'] = action['hvacpower'] * 100.0
    row['LightingPower'] = action['lightingpower'] * 100.0
    rows.append(row)

    if (t + 1) % 1000 == 0:
        rate = (t + 1) / (time.time() - t0)
        eta = (args.steps - t - 1) / rate
        logger.info(f"  step {t+1}/{args.steps} ({rate:.1f}/s, eta {eta:.0f}s)")

logger.info(f"Generated {len(rows)} rows in {time.time()-t0:.0f}s")

# ── Save raw + processed CSVs ──
df = pd.DataFrame(rows, columns=COL_ORDER)
df.to_csv(OUT_RAW, index=False)
logger.info(f"  raw -> {OUT_RAW}")

# Normalize using same approach as existing scaling CSV (min/max per column)
scaling = pd.DataFrame({
    'feature': df.columns,
    'data_min': df.min().values,
    'data_max': df.max().values,
})
scaling.to_csv(OUT_SCALE, index=False)
logger.info(f"  scaling -> {OUT_SCALE}")

denom = df.max() - df.min()
denom[denom == 0] = 1e-12
df_norm = (df - df.min()) / denom
df_norm.to_csv(OUT_PROC, index=False)
logger.info(f"  processed -> {OUT_PROC}")

# Print summary
logger.info("\nActuator distribution (HVACPower phys units):")
hv = df['HVACPower']
for q in [0.05, 0.25, 0.5, 0.75, 0.95]:
    logger.info(f"  q{q*100:.0f}: {hv.quantile(q):.1f}")
logger.info(f"  mean ± std: {hv.mean():.1f} ± {hv.std():.1f}")

logger.info("\nRegime breakdown (occ_active = Occupancy>0.1, win_active = WinPos>0.15):")
occ_on = df['Occupancy'] >= 0.1
win_on = df['WindowPosition'] >= 0.15
logger.info(f"  base: {((~occ_on)&(~win_on)).sum()}")
logger.info(f"  occ:  {(occ_on & ~win_on).sum()}")
logger.info(f"  win:  {(~occ_on & win_on).sum()}")
logger.info(f"  full: {(occ_on & win_on).sum()}")

# Confounding check: HVAC vs Occupancy correlation
hvac_norm = df_norm['HVACPower']
occ_norm = df_norm['Occupancy']
logger.info(f"\nConfounding check: corr(HVACPower, Occupancy) = "
            f"{np.corrcoef(hvac_norm, occ_norm)[0,1]:.3f}")
logger.info(f"  vs. obs CSV's actuator-context correlation: "
            f"should be 0 by 90/10 design")
