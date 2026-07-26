#!/usr/bin/env python3
"""
Regime-Aware Validation System — RF Classifier + BMA Policy Switching
======================================================================

Detects hidden regime changes (e.g. window open/closed) from observable
physical features, then selects the appropriate causal policy.

Components:
  1. **WindowStatePredictor**: Random Forest classifier trained on physical
     features only (temperature, humidity, airQuality, outdoorTemperature)
     plus derived signals (temp_drift, outdoor-indoor correlation).
     Excludes outcome variables (energy, satisfaction) to avoid circularity.

  2. **RegimeAwareValidator**: Integrates the RF classifier with the
     CWM Bayesian monitoring to provide a regime-consensus signal.

DOMAIN-AGNOSTIC KNOBS — update for a new domain:
  - ``WindowStatePredictor.extract_features()``:
      Change the feature columns to match your domain's observable physical
      state variables.  Add domain-relevant derived signals.
  - ``n_estimators``, ``max_depth``: RF hyperparameters (default 100 trees, depth 10).
  - ``train()``:  Expects a DataFrame with a binary 'windowopen' column as
      the regime label.  Rename to match your domain's regime indicator.
  - ``RegimeAwareValidator.regime_threshold``: P(regime) > threshold → switch.

Paper reference: Section 4.4 (Regime-Aware Policy Selection).
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import argparse
import sys
import os

# ============================================================================
# PART 1: RF WINDOW STATE PREDICTOR
# ============================================================================

class WindowStatePredictor:
    """RF classifier to infer window state from observables"""
    
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42
        )
        self.is_trained = False
    
    def extract_features(self, data):
        """Extract physical-state features only (no outcome variables).

        Uses temperature, humidity, airQuality, outdoorTemperature plus
        derived causal signals: temp_drift, toa_t_corr.

        Excludes energyConsumption and satisfaction — these are outcome
        variables in the causal graph.  Using them would create circularity:
        the regime classifier would rely on the very signals the causal policy
        is trying to optimise, and would trivially separate regimes via the
        simulator's deterministic output mapping rather than genuine physical
        state inference.
        """
        df = pd.DataFrame(data) if not isinstance(data, pd.DataFrame) else data

        features = pd.DataFrame({
            'temperature':        df['temperature'],
            'humidity':           df['humidity'],
            'airQuality':         df['airQuality'],
            'outdoorTemperature': df['outdoorTemperature'],
        })

        # Derived physical signals
        if len(df) > 10:
            features['temp_drift'] = df['temperature'].diff().rolling(5).mean().fillna(0)
            features['toa_t_corr'] = df['outdoorTemperature'].rolling(10).corr(
                df['temperature']).fillna(0)
        else:
            features['temp_drift'] = 0
            features['toa_t_corr'] = 0

        return features
    
    def train(self, obs_data):
        """Train on observational data"""
        features = self.extract_features(obs_data)
        labels = obs_data['windowOpen'].values
        
        X_train, X_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42
        )
        
        self.model.fit(X_train, y_train)
        self.is_trained = True
        
        # Evaluate
        y_pred = self.model.predict(X_test)
        print("\n=== Window State Predictor Performance ===")
        print(classification_report(y_test, y_pred, target_names=['Closed', 'Open']))
        print("\nConfusion Matrix:")
        print(confusion_matrix(y_test, y_pred))
        
        accuracy = self.model.score(X_test, y_test)
        print(f"\nAccuracy: {accuracy:.3f}")
        
        return accuracy
    
    def predict_window_state(self, observation):
        """Predict window state from single observation"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        features = pd.DataFrame([{
            'temperature':        observation['temperature'],
            'humidity':           observation['humidity'],
            'airQuality':         observation['airQuality'],
            'outdoorTemperature': observation['outdoorTemperature'],
            'temp_drift':         observation.get('temp_drift', 0),
            'toa_t_corr':         observation.get('toa_t_corr', 0),
        }])
        
        prediction = self.model.predict(features)[0]
        probability = self.model.predict_proba(features)[0]
        
        return {
            'predicted_open': bool(prediction),
            'confidence': float(max(probability)),
            'prob_open': float(probability[1]) if len(probability) > 1 else 0
        }
    
    def save(self, filepath):
        """Save trained model"""
        joblib.dump(self.model, filepath)
        print(f"Model saved to {filepath}")
    
    def load(self, filepath):
        """Load trained model"""
        self.model = joblib.load(filepath)
        self.is_trained = True
        print(f"Model loaded from {filepath}")

# ============================================================================
# PART 2: REGIME-AWARE VALIDATOR
# ============================================================================

class RegimeAwareValidator:
    """Validates interventions with regime inference"""
    
    def __init__(self, rf_predictor):
        self.rf_predictor = rf_predictor
        self.regime_history = []
        
    def validate_intervention(self, intervention_result, graph_h0=None, graph_h1=None):
        # Normalize keys: any case/underscore → camelCase for RF
        key_map = {
            'temperature': 'temperature',
            'humidity': 'humidity',
            'airquality': 'airQuality',
            'energyconsumption': 'energyConsumption',
            'pmv': 'pmv',
            'outdoortemperature': 'outdoorTemperature',
            'outdoortemp': 'outdoorTemperature',
            'windowopen': 'windowOpen'
        }
        
        normalized_data = {}
        for key, value in intervention_result.items():
            clean_key = key.lower().replace(' ', '').replace('_', '')
            mapped_key = key_map.get(clean_key, key)
            normalized_data[mapped_key] = value
        
        # Predict using normalized data
        regime_prediction = self.rf_predictor.predict_window_state(normalized_data)
        
        inferred_regime = 'open' if regime_prediction['predicted_open'] else 'closed'
        confidence = regime_prediction['confidence']
        
        self.regime_history.append({
            'regime': inferred_regime,
            'confidence': confidence,
            'prob_open': regime_prediction['prob_open'],
            'observation': intervention_result
        })
        
        primary_hypothesis = 'H1' if inferred_regime == 'open' else 'H0'
        
        return {
            'inferred_regime': inferred_regime,
            'confidence': confidence,
            'prob_open': regime_prediction['prob_open'],
            'primary_hypothesis': primary_hypothesis,
            'match_quality': confidence
        }
    
    def get_regime_distribution(self):
        """Get distribution of inferred regimes"""
        if not self.regime_history:
            return None
        
        total = len(self.regime_history)
        open_count = sum(1 for r in self.regime_history if r['regime'] == 'open')
        
        return {
            'total_interventions': total,
            'open': open_count,
            'closed': total - open_count,
            'pct_open': 100 * open_count / total if total > 0 else 0,
            'avg_confidence': np.mean([r['confidence'] for r in self.regime_history])
        }

# ============================================================================
# PART 3: HYPOTHESIS WEIGHT MANAGER
# ============================================================================

class HypothesisWeightManager:
    """Track and update H0/H1 weights based on regime-matched validations"""
    
    def __init__(self):
        self.h0_weight = 1.0
        self.h1_weight = 1.0
        self.validation_history = []
    
    def update_weights(self, validation_result, effect_size):
        """
        Update weights using Bayesian-style likelihood matching
        
        Args:
            validation_result: Output from RegimeAwareValidator
            effect_size: Observed effect magnitude
        """
        regime = validation_result['inferred_regime']
        confidence = validation_result['confidence']
        
        # Define expected effects for each hypothesis
        # H0 (closed): weak effects (~0.2)
        # H1 (open): strong effects (~0.6)
        
        if regime == 'closed':
            # H0 should match better
            h0_likelihood = np.exp(-abs(effect_size - 0.2) / 0.1)
            h1_likelihood = np.exp(-abs(effect_size - 0.2) / 0.3)
        else:  # regime == 'open'
            # H1 should match better
            h0_likelihood = np.exp(-abs(effect_size - 0.2) / 0.3)
            h1_likelihood = np.exp(-abs(effect_size - 0.6) / 0.1)
        
        # Weight by confidence
        self.h0_weight *= (1 - confidence) + confidence * h0_likelihood
        self.h1_weight *= (1 - confidence) + confidence * h1_likelihood
        
        # Normalize
        total = self.h0_weight + self.h1_weight
        self.h0_weight /= total
        self.h1_weight /= total
        
        self.validation_history.append({
            'regime': regime,
            'confidence': confidence,
            'effect_size': effect_size,
            'h0_weight': self.h0_weight,
            'h1_weight': self.h1_weight
        })
        
        return {
            'h0_weight': self.h0_weight,
            'h1_weight': self.h1_weight,
            'primary_hypothesis': 'H1' if self.h1_weight > self.h0_weight else 'H0'
        }
    
    def should_switch_hypothesis(self, threshold=0.8):
        """Check if one hypothesis dominates"""
        if self.h1_weight > threshold:
            return 'H1', self.h1_weight
        elif self.h0_weight > threshold:
            return 'H0', self.h0_weight
        return None, max(self.h0_weight, self.h1_weight)

# ============================================================================
# PART 4: INTEGRATION UTILITIES
# ============================================================================

def validate_edge_with_regime_inference(edge, intervention_result, validator, weight_manager):
    """
    Validate edge with regime inference
    
    Args:
        edge: tuple (source, target)
        intervention_result: dict with pre/post intervention states
        validator: RegimeAwareValidator instance
        weight_manager: HypothesisWeightManager instance
        
    Returns:
        dict with validation results
    """
    source, target = edge
    
    # Get states
    pre_state = intervention_result['preInterventionData']
    post_state = intervention_result['postInterventionData']
    
    # Calculate effect size
    target_change = abs(post_state.get(target, 0) - pre_state.get(target, 0))
    source_change = abs(post_state.get(source, 0) - pre_state.get(source, 0))
    effect_size = target_change / source_change if source_change > 0.001 else 0
    
    # Infer regime from intervention outcome
    regime_info = validator.validate_intervention(post_state)
    
    # Update hypothesis weights
    weights = weight_manager.update_weights(regime_info, effect_size)
    
    print(f"\nEdge: {source} → {target}")
    print(f"  Regime: {regime_info['inferred_regime']} (confidence: {regime_info['confidence']:.2f})")
    print(f"  Effect: {effect_size:.3f}")
    print(f"  Weights: H0={weights['h0_weight']:.3f}, H1={weights['h1_weight']:.3f}")
    
    # Check for hypothesis switch
    switch, switch_confidence = weight_manager.should_switch_hypothesis(threshold=0.8)
    if switch:
        print(f"  *** HYPOTHESIS SWITCH TO {switch} (confidence: {switch_confidence:.3f}) ***")
    
    return {
        'edge': edge,
        'effect_size': effect_size,
        'regime': regime_info['inferred_regime'],
        'confidence': regime_info['confidence'],
        'hypothesis_weights': weights,
        'switch': switch
    }

# ============================================================================
# PART 5: CLI INTERFACE
# ============================================================================

def train_from_csv(csv_path, model_output='window_predictor_model.pkl'):
    """Train RF model from observational CSV"""
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)
    
    print(f"Dataset: {len(df)} observations")
    print(f"Window open: {df['windowOpen'].sum()} / {len(df)} ({100*df['windowOpen'].mean():.1f}%)")
    
    predictor = WindowStatePredictor()
    accuracy = predictor.train(df)
    
    predictor.save(model_output)
    
    return predictor, accuracy

def test_predictor(predictor):
    """Test predictor on sample cases"""
    test_cases = [
        {
            'description': 'Window closed - stable conditions',
            'observation': {
                'temperature': 22.0,
                'humidity': 50,
                'airQuality': 300,
                'outdoorTemperature': 5.0,
                'energyConsumption': 30,
                'satisfaction': 75
            },
            'expected': 'CLOSED'
        },
        {
            'description': 'Window open - high energy, temp drift',
            'observation': {
                'temperature': 18.5,
                'humidity': 45,
                'airQuality': 280,
                'outdoorTemperature': 5.0,
                'energyConsumption': 75,
                'satisfaction': 65
            },
            'expected': 'OPEN'
        }
    ]
    
    print("\n=== Testing Predictor ===")
    for i, case in enumerate(test_cases):
        result = predictor.predict_window_state(case['observation'])
        match = '✓' if (result['predicted_open'] and case['expected'] == 'OPEN') or \
                      (not result['predicted_open'] and case['expected'] == 'CLOSED') else '✗'
        print(f"\n{match} Case {i+1}: {case['description']}")
        print(f"  Predicted: {'OPEN' if result['predicted_open'] else 'CLOSED'} "
              f"(confidence: {result['confidence']:.2f})")
        print(f"  Expected: {case['expected']}")

def main():
    parser = argparse.ArgumentParser(description='Regime-Aware Validation System')
    parser.add_argument('--train', type=str, help='Train from observational CSV', default='data/challenge1_data_10k.csv')
    parser.add_argument('--test', action='store_true', help='Test trained model')
    parser.add_argument('--model', type=str, default='window_predictor_model.pkl', 
                        help='Model file path')
    
    args = parser.parse_args()
    
    if args.train:
        predictor, accuracy = train_from_csv(args.train, args.model)
        
        if args.test:
            test_predictor(predictor)
        
        print(f"\n{'='*60}")
        print("Integration example:")
        print("  from regime_aware_validation import WindowStatePredictor, RegimeAwareValidator")
        print("  predictor = WindowStatePredictor()")
        print(f"  predictor.load('{args.model}')")
        print("  validator = RegimeAwareValidator(predictor)")
        print(f"{'='*60}")
    
    elif args.test:
        if not os.path.exists(args.model):
            print(f"Error: {args.model} not found. Train first with --train")
            sys.exit(1)
        
        predictor = WindowStatePredictor()
        predictor.load(args.model)
        test_predictor(predictor)
    
    else:
        parser.print_help()

if __name__ == "__main__":
    main()