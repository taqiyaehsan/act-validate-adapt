#!/usr/bin/env python3
"""
Cache observational-only edges (PolicyGRID-O) for each dataset.

Runs PC + SAM + LLM discovery methods on each dataset (no intervention
validation), takes the union of all proposed edges, and saves to JSON.
These cached edges are loaded by run_b1_prelim.py for the GRID_ObsOnly arm.

Usage:
  python cache_obs_only_edges.py                    # all datasets
  python cache_obs_only_edges.py --dataset open_window
  python cache_obs_only_edges.py --no-llm           # skip LLM (no API calls)
"""
import argparse, json, os, sys, logging
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.WARNING,
                    format='%(asctime)s %(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('cache_obs_only')
logger.setLevel(logging.INFO)

from src.generators import PCGenerator, SAMGenerator, LLMGenerator

# ── Dataset configs ──────────────────────────────────────────────────────────

DATASET_CONFIGS = {
    'open_window': {
        'data_path': 'data_regen/challenge1_data_10k_processed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality',
                     'PMV', 'EnergyConsumption', 'Satisfaction'],
    },
    'smart_room': {
        'data_path': 'data_regen/smart_room_processed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality',
                     'EnergyConsumption', 'Satisfaction'],
    },
    'smart_room_noise': {
        'data_path': 'data_regen/smart_room_noise_processed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality',
                     'EnergyConsumption', 'Satisfaction'],
    },
    'hidden_vars': {
        'data_path': 'data_regen/hidden_vars_processed.csv',
        'columns': ['Temperature', 'Humidity', 'AirQuality',
                     'EnergyConsumption', 'Satisfaction'],
    },
    'ashrae': {
        'data_path': 'data/ashrae_data_processed.csv',
        'columns': ['air_temperature', 'dew_temperature',
                     'sea_level_pressure', 'meter_reading',
                     'square_feet', 'year_built'],
    },
}

CACHE_DIR = 'data/obs_only_edges'


def run_discovery(dataset_name, cfg, use_llm=True, api_key=None):
    """Run PC + SAM (+ optionally LLM) on one dataset, return union of edges."""
    df = pd.read_csv(cfg['data_path'])
    columns = cfg['columns']
    logger.info(f"[{dataset_name}] Loaded {len(df)} rows, "
                f"using columns: {columns}")

    all_edges = set()
    method_edges = {}

    # PC
    try:
        pc_gen = PCGenerator(relevant_columns=columns)
        result = pc_gen.generate(df)
        edges = {tuple(e) for e in result.get('edges', [])}
        all_edges |= edges
        method_edges['PC'] = sorted(edges)
        logger.info(f"  PC: {len(edges)} edges")
    except Exception as e:
        logger.warning(f"  PC failed: {e}")
        method_edges['PC'] = []

    # SAM
    try:
        sam_gen = SAMGenerator(relevant_columns=columns)
        result = sam_gen.generate(df)
        edges = {tuple(e) for e in result.get('edges', [])}
        all_edges |= edges
        method_edges['SAM'] = sorted(edges)
        logger.info(f"  SAM: {len(edges)} edges")
    except Exception as e:
        logger.warning(f"  SAM failed: {e}")
        method_edges['SAM'] = []

    # LLM (optional)
    if use_llm and api_key:
        try:
            llm_gen = LLMGenerator(api_key=api_key, relevant_columns=columns)
            result = llm_gen.generate(df)
            edges = {tuple(e) for e in result.get('edges', [])}
            all_edges |= edges
            method_edges['LLM'] = sorted(edges)
            logger.info(f"  LLM: {len(edges)} edges")
        except Exception as e:
            logger.warning(f"  LLM failed: {e}")
            method_edges['LLM'] = []
    else:
        logger.info("  LLM: skipped")

    union = sorted(all_edges)
    logger.info(f"  Union: {len(union)} edges")
    return union, method_edges


def cache_one(dataset_name, cfg, use_llm=True, api_key=None):
    """Run discovery and save to JSON cache."""
    union, method_edges = run_discovery(dataset_name, cfg, use_llm, api_key)

    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_file = os.path.join(CACHE_DIR, f'{dataset_name}.json')

    cache = {
        'dataset': dataset_name,
        'data_path': cfg['data_path'],
        'columns': cfg['columns'],
        'union_edges': union,
        'method_edges': method_edges,
        'n_union': len(union),
    }

    with open(cache_file, 'w') as f:
        json.dump(cache, f, indent=2)
    logger.info(f"  Saved to {cache_file}")
    return cache


def load_cached_obs_edges(dataset_name):
    """Load cached obs-only edges for a dataset. Returns set of tuples or None."""
    cache_file = os.path.join(CACHE_DIR, f'{dataset_name}.json')
    if not os.path.exists(cache_file):
        return None
    with open(cache_file) as f:
        cache = json.load(f)
    return {tuple(e) for e in cache['union_edges']}


def main():
    parser = argparse.ArgumentParser(
        description='Cache obs-only edges (PolicyGRID-O) per dataset')
    parser.add_argument('--dataset', type=str, default=None,
                        choices=list(DATASET_CONFIGS.keys()),
                        help='Run for one dataset (default: all)')
    parser.add_argument('--no-llm', action='store_true',
                        help='Skip LLM generator (no API calls)')
    args = parser.parse_args()

    api_key = os.environ.get('OPENAI_API_KEY')
    use_llm = not args.no_llm and api_key is not None
    if not use_llm:
        logger.info("LLM discovery disabled (--no-llm or no OPENAI_API_KEY)")

    datasets = [args.dataset] if args.dataset else list(DATASET_CONFIGS.keys())

    for name in datasets:
        cfg = DATASET_CONFIGS[name]
        print(f"\n{'='*60}")
        print(f"  Caching obs-only edges: {name}")
        print(f"  Data: {cfg['data_path']}")
        print(f"{'='*60}")
        cache = cache_one(name, cfg, use_llm, api_key)
        print(f"  -> {cache['n_union']} union edges saved to "
              f"{CACHE_DIR}/{name}.json")

    print(f"\nDone. Cached edges in {CACHE_DIR}/")


if __name__ == '__main__':
    main()
