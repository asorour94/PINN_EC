# src/data_normalizer.py

import numpy as np
from dataclasses import dataclass
from typing import Dict, Optional

@dataclass
class NormalizationParams:
    """Store normalization parameters for each variable."""
    do_scale: float = 10.0          # DO typically 0-10 mg/L
    our_scale: float = 1000.0       # OUR typically 0-1000 mg/L/h
    do_sat_scale: float = 15.0      # DO saturation typically 8-15 mg/L
    power_scale: float = 1000.0     # Power in kW (assuming current values are in W)
    time_scale: float = 24.0        # Time in hours to days
    
class DataNormalizer:
    """Normalize and denormalize data for PINN training."""
    
    def __init__(self, params: Optional[NormalizationParams] = None):
        self.params = params or NormalizationParams()
        
    def normalize(self, data: Dict[str, float]) -> Dict[str, float]:
        """Normalize input data."""
        normalized = {}
        
        # Time normalization (hours to days)
        if 't' in data:
            normalized['t'] = data['t'] / self.params.time_scale
            
        # DO normalization
        if 'do' in data:
            normalized['do'] = data['do'] / self.params.do_scale
            
        # OUR normalization
        if 'our' in data:
            normalized['our'] = data['our'] / self.params.our_scale
            
        # DO saturation normalization
        if 'do_sat' in data:
            normalized['do_sat'] = data['do_sat'] / self.params.do_sat_scale
            
        # Power normalization (convert W to kW then normalize)
        if 'power' in data:
            # First convert to kW if in Watts
            power_kw = data['power'] / 1000.0
            normalized['power'] = power_kw / self.params.power_scale
            
        # Pass through other variables
        for key, value in data.items():
            if key not in normalized:
                normalized[key] = value
                
        return normalized
    
    def denormalize_kla(self, kla_normalized: float) -> float:
        """Denormalize KLa (1/h). KLa doesn't need scaling typically."""
        return kla_normalized
    
    def denormalize_predictions(self, predictions: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Denormalize model predictions."""
        denormalized = {}
        
        if 'do' in predictions:
            denormalized['do'] = predictions['do'] * self.params.do_scale
            
        if 'power' in predictions:
            denormalized['power'] = predictions['power'] * self.params.power_scale
            
        return denormalized