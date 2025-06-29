# src/data_manager.py - MODERATE CHANGES: KLa-Focused Data Pipeline

import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

@dataclass
class NormalizationParams:
    """Store physics-aware normalization parameters for each variable."""
    # Time normalization
    time_scale: float = 24.0        # Convert hours to days
    
    # Physical parameters with engineering ranges
    do_scale: float = 10.0          # DO typically 0-10 mg/L
    our_scale: float = 1000.0       # OUR typically 0-1000 mg/L/h
    do_sat_scale: float = 15.0      # DO saturation typically 8-15 mg/L
    temperature_scale: float = 40.0  # Temperature range 0-40°C
    biomass_scale: float = 5000.0   # MLSS typically 0-5000 mg/L
    flow_scale: float = 10000.0     # Flow rate (site-specific scaling)
    volume_scale: float = 10000.0   # Reactor volume (site-specific scaling)
    
    # Power normalization (critical for KLa prediction!)
    power_scale: float = 1000.0     # Power in kW


class TruePINNDataManager:
    """
    Data manager for true PINN - focuses on KLa prediction
    
    Key changes from original:
    1. Target is KLa (not DO)
    2. Features are KLa-determining variables (not just influent)
    3. Physics-aware normalization preserving relationships
    4. Proper train/validation separation without data leakage
    """
    
    def __init__(self, config):
        self.config = config
        self.normalization_params = NormalizationParams()
        
        # Define KLa-determining features (physics-based selection)
        self.feature_names = self._get_kla_determining_features()
        
        # Primary target is KLa (not DO!)
        self.target_name = 'Sumo__Plant__CSTR3__kLaGO2'
        
        # Secondary targets for physics validation
        self.secondary_targets = {
            'do_measured': 'Sumo__Plant__CSTR3__SO2',
            'our_measured': 'Sumo__Plant__CSTR3__OUR',
            'temperature': 'Sumo__Plant__Influent__T'
        }
        
        logger.info("True PINN Data Manager initialized")
        logger.info(f"Primary target: {self.target_name}")
        logger.info(f"KLa-determining features: {len(self.feature_names)}")
        
    def _get_kla_determining_features(self):
        """
        Features that actually determine KLa (physics-based selection)
        
        These are REACTOR STATE variables that affect mass transfer,
        NOT influent composition parameters!
        """
        return [
            # Basic operating conditions
            'Sumo__Time',                                    # Time (temporal patterns)
            'Sumo__Plant__Influent__T',                      # Temperature (affects kinetics)
            'Sumo__Plant__Influent__Q',                      # Flow rate (affects mixing)
            
            # CRITICAL: Aeration system parameters  
            'Sumo__Plant__EnergyCenter__Hel_aeration',       # Aeration power (PRIMARY KLa driver!)
            
            # Current reactor state
            'Sumo__Plant__CSTR3__SO2',                       # Current DO concentration
            'Sumo__Plant__CSTR3__OUR',                       # Oxygen uptake rate
            'Sumo__Plant__CSTR3__XTSS',                      # Biomass concentration (affects rheology)
            
            # Physical parameters
            'reactor_volume'                                 # Reactor volume (for specific power)
        ]
    
    def prepare_data_for_true_pinn(self, df: pd.DataFrame) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Prepare data for KLa prediction (not DO prediction!)
        
        Args:
            df: Raw SUMO simulation data
            
        Returns:
            Tuple of (features, targets) for True PINN training
        """
        logger.info("Preparing data for True PINN (KLa prediction)")
        
        # Validate required columns
        missing_features = [f for f in self.feature_names if f not in df.columns and f != 'reactor_volume']
        if missing_features:
            logger.error(f"Missing required features: {missing_features}")
            raise ValueError(f"Missing required features for KLa prediction: {missing_features}")
        
        # Check if primary target exists
        if self.target_name not in df.columns:
            logger.error(f"Primary target '{self.target_name}' not found in data")
            raise ValueError(f"KLa target '{self.target_name}' not found")
        
        # Extract features with proper handling of missing columns
        features_data = []
        for feature in self.feature_names:
            if feature == 'reactor_volume':
                # Add reactor volume as constant (site-specific parameter)
                features_data.append(np.full(len(df), 1000.0))  # Default 1000 m³
            elif feature in df.columns:
                features_data.append(df[feature].values)
            else:
                logger.warning(f"Feature '{feature}' not found, using zeros")
                features_data.append(np.zeros(len(df)))
        
        X = np.column_stack(features_data)
        
        # Extract primary target (KLa)
        y_kla = df[self.target_name].values
        
        # Extract secondary targets for physics validation
        targets = {'kla_actual': y_kla}
        
        for target_key, column_name in self.secondary_targets.items():
            if column_name in df.columns:
                targets[target_key] = df[column_name].values
            else:
                logger.warning(f"Secondary target '{column_name}' not found")
                targets[target_key] = np.zeros(len(df))
        
        # Apply physics-aware normalization
        X_normalized = self._physics_aware_normalize_features(X)
        targets_normalized = self._normalize_targets(targets)
        
        logger.info(f"Data prepared: {X_normalized.shape[0]} samples, {X_normalized.shape[1]} features")
        logger.info(f"KLa range: {np.min(y_kla):.2f} - {np.max(y_kla):.2f} 1/h")
        
        return X_normalized, targets_normalized
    
    def _physics_aware_normalize_features(self, X: np.ndarray) -> np.ndarray:
        """
        Normalize features while preserving physics relationships
        
        Critical: Don't break physical relationships like P/V ratio!
        """
        X_norm = X.copy().astype(float)
        params = self.normalization_params
        
        for i, feature in enumerate(self.feature_names):
            if feature == 'Sumo__Time':
                # Convert time from milliseconds to days
                X_norm[:, i] = X[:, i] / (1000 * 3600 * params.time_scale)
                
            elif feature == 'Sumo__Plant__Influent__T':
                # Temperature normalization
                X_norm[:, i] = X[:, i] / params.temperature_scale
                
            elif feature == 'Sumo__Plant__Influent__Q':
                # Flow rate normalization
                X_norm[:, i] = X[:, i] / params.flow_scale
                
            elif feature == 'Sumo__Plant__EnergyCenter__Hel_aeration':
                # CRITICAL: Aeration power - preserve P/V relationship
                volume_idx = self.feature_names.index('reactor_volume')
                # Calculate specific power (P/V) then normalize
                specific_power = X[:, i] / (X[:, volume_idx] + 1e-6)
                X_norm[:, i] = specific_power / params.power_scale
                
            elif feature == 'Sumo__Plant__CSTR3__SO2':
                # DO concentration normalization
                X_norm[:, i] = X[:, i] / params.do_scale
                
            elif feature == 'Sumo__Plant__CSTR3__OUR':
                # OUR normalization
                X_norm[:, i] = X[:, i] / params.our_scale
                
            elif feature == 'Sumo__Plant__CSTR3__XTSS':
                # Biomass concentration normalization
                X_norm[:, i] = X[:, i] / params.biomass_scale
                
            elif feature == 'reactor_volume':
                # Volume normalization
                X_norm[:, i] = X[:, i] / params.volume_scale
                
            else:
                # Default min-max normalization for unknown features
                x_min, x_max = np.min(X[:, i]), np.max(X[:, i])
                if x_max > x_min:
                    X_norm[:, i] = (X[:, i] - x_min) / (x_max - x_min)
                else:
                    X_norm[:, i] = 0.0
        
        return X_norm
    
    def _normalize_targets(self, targets: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Normalize target values with physics-aware scaling
        """
        normalized_targets = {}
        params = self.normalization_params
        
        for key, values in targets.items():
            if key == 'kla_actual':
                # KLa normalization - keep meaningful scale
                # Literature range: 0.1 - 200 1/h, normalize to 0-1 range
                normalized_targets[key] = values / 100.0  # Max expected KLa
                
            elif key == 'do_measured':
                # DO normalization
                normalized_targets[key] = values / params.do_scale
                
            elif key == 'our_measured':
                # OUR normalization
                normalized_targets[key] = values / params.our_scale
                
            elif key == 'temperature':
                # Temperature normalization
                normalized_targets[key] = values / params.temperature_scale
                
            else:
                # Default normalization
                normalized_targets[key] = values
        
        return normalized_targets
    
    def create_train_validation_split(self, X: np.ndarray, targets: Dict[str, np.ndarray], 
                                    validation_fraction: float = 0.2) -> Tuple[Dict, Dict]:
        """
        Create proper train/validation split without data leakage
        
        Uses time-based split to ensure no future information in training
        """
        n_samples = X.shape[0]
        n_train = int(n_samples * (1 - validation_fraction))
        
        logger.info(f"Creating time-based train/validation split")
        logger.info(f"Training samples: {n_train}, Validation samples: {n_samples - n_train}")
        
        # Time-based split (no future information in training!)
        train_indices = np.arange(0, n_train)
        val_indices = np.arange(n_train, n_samples)
        
        # Create training data dictionary
        train_data = {
            'inputs': X[train_indices],
            'targets': {key: values[train_indices] for key, values in targets.items()}
        }
        
        # Create validation data dictionary
        val_data = {
            'inputs': X[val_indices], 
            'targets': {key: values[val_indices] for key, values in targets.items()}
        }
        
        # Log data quality metrics
        self._log_data_quality_metrics(train_data, val_data)
        
        return train_data, val_data
    
    def _log_data_quality_metrics(self, train_data: Dict, val_data: Dict):
        """
        Log data quality metrics for True PINN
        """
        logger.info("=== Data Quality Metrics ===")
        
        # Training data metrics
        train_kla = train_data['targets']['kla_actual']
        logger.info(f"Training KLa range: {np.min(train_kla):.2f} - {np.max(train_kla):.2f}")
        logger.info(f"Training KLa mean: {np.mean(train_kla):.2f} ± {np.std(train_kla):.2f}")
        
        # Validation data metrics
        val_kla = val_data['targets']['kla_actual']
        logger.info(f"Validation KLa range: {np.min(val_kla):.2f} - {np.max(val_kla):.2f}")
        logger.info(f"Validation KLa mean: {np.mean(val_kla):.2f} ± {np.std(val_kla):.2f}")
        
        # Check for realistic KLa values
        realistic_kla_train = np.sum((train_kla >= 0.1) & (train_kla <= 200.0)) / len(train_kla)
        realistic_kla_val = np.sum((val_kla >= 0.1) & (val_kla <= 200.0)) / len(val_kla)
        
        logger.info(f"Realistic KLa values - Train: {realistic_kla_train:.1%}, Val: {realistic_kla_val:.1%}")
        
        # Feature analysis
        train_features = train_data['inputs']
        aeration_power_idx = 3  # Index of aeration power in features
        
        if train_features.shape[1] > aeration_power_idx:
            power_range = f"{np.min(train_features[:, aeration_power_idx]):.3f} - {np.max(train_features[:, aeration_power_idx]):.3f}"
            logger.info(f"Normalized aeration power range: {power_range}")
        
        logger.info("=" * 40)
    
    def denormalize_kla_predictions(self, kla_normalized: np.ndarray) -> np.ndarray:
        """
        Denormalize KLa predictions back to engineering units
        """
        return kla_normalized * 100.0  # Convert back from 0-1 to 0-200 1/h range
    
    def denormalize_all_predictions(self, predictions_dict: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Denormalize all predictions back to engineering units
        """
        denormalized = {}
        params = self.normalization_params
        
        for key, values in predictions_dict.items():
            if key == 'kla_predicted':
                denormalized[key] = values * 200.0  # KLa denormalization
            elif key == 'do_calculated' or key == 'do_measured':
                denormalized[key] = values * params.do_scale  # DO denormalization
            elif key == 'our_measured':
                denormalized[key] = values * params.our_scale  # OUR denormalization
            elif key == 'do_saturation':
                denormalized[key] = values * params.do_sat_scale  # DO sat denormalization
            else:
                denormalized[key] = values  # Keep as-is for other variables
        
        return denormalized
    
    def validate_data_for_physics(self, X: np.ndarray, targets: Dict[str, np.ndarray]) -> Dict[str, bool]:
        """
        Validate that data is suitable for physics-informed training
        """
        validation_results = {}
        
        # Check feature completeness
        validation_results['features_complete'] = not np.any(np.isnan(X))
        validation_results['features_finite'] = np.all(np.isfinite(X))
        
        # Check KLa target quality
        kla_values = targets.get('kla_actual', np.array([]))
        if len(kla_values) > 0:
            validation_results['kla_realistic'] = np.all((kla_values >= 0.1) & (kla_values <= 200.0))
            validation_results['kla_complete'] = not np.any(np.isnan(kla_values))
        else:
            validation_results['kla_realistic'] = False
            validation_results['kla_complete'] = False
        
        # Check aeration power presence (critical for KLa prediction)
        aeration_power_idx = 3  # Index of aeration power
        if X.shape[1] > aeration_power_idx:
            aeration_power = X[:, aeration_power_idx]
            validation_results['aeration_power_present'] = np.any(aeration_power > 0)
            validation_results['aeration_power_variable'] = np.std(aeration_power) > 1e-6
        else:
            validation_results['aeration_power_present'] = False
            validation_results['aeration_power_variable'] = False
        
        # Overall validation
        all_checks = [
            validation_results['features_complete'],
            validation_results['features_finite'],
            validation_results['kla_realistic'],
            validation_results['kla_complete'],
            validation_results['aeration_power_present']
        ]
        validation_results['overall_valid'] = all(all_checks)
        
        # Log validation results
        logger.info("=== Data Validation for Physics-Informed Training ===")
        for check, result in validation_results.items():
            status = "✅ PASS" if result else "❌ FAIL"
            logger.info(f"{check}: {status}")
        
        if validation_results['overall_valid']:
            logger.info("🟢 Data is suitable for True PINN training")
        else:
            logger.warning("🟡 Data quality issues detected - may affect physics learning")
        
        logger.info("=" * 60)
        
        return validation_results


# Helper function for backward compatibility
def create_true_pinn_data_manager(config):
    """
    Factory function to create True PINN Data Manager
    """
    return TruePINNDataManager(config)