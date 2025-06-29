# src/config_manager.py - MINOR CHANGES: Physics-Based Configuration

import os
import json
from dataclasses import dataclass
from typing import List, Dict, Optional

@dataclass
class SumoConfig:
    """SUMO simulation configuration"""
    project_path: str
    data_comm_interval: int
    parallel_jobs: int
    variables: List[str]
    simulation_duration_days: int = 28  # Default to 4 weeks

@dataclass 
class TruePINNConfig:
    """
    Physics-based configuration for True PINN
    
    Key changes from original PinnConfig:
    1. Physics-based parameter bounds (from literature)
    2. Physics hierarchy loss weights (physics dominates)
    3. Single optimizer configuration (no dual optimizers)
    4. Physics compliance parameters
    """
    # Training parameters (physics-informed)
    learning_rate: float = 0.001           # Conservative for stable physics learning
    batch_size: int = 32                   # Small batches for stable convergence
    
    # Physics-hierarchy loss weights (CRITICAL CHANGE!)
    loss_weights: Dict[str, float] = None
    
    # Buffer and training limits
    buffer_size: int = 5000
    max_training_steps: Optional[int] = None
    post_simulation_epochs: Optional[int] = 10
    
    # Model architecture
    model_config: Optional[Dict] = None
    
    # Physics-based parameter bounds (from literature, not arbitrary!)
    physics_bounds: Optional[Dict] = None
    
    # Physics compliance parameters  
    physics_tolerance: float = 1e-6        # Strict physics compliance
    physics_compliance_threshold: float = 0.95  # 95% of predictions must satisfy physics
    
    def __post_init__(self):
        """Set default values for complex fields"""
        if self.loss_weights is None:
            # Physics-hierarchy weights (physics DOMINATES!)
            self.loss_weights = {
                'physics': 1000.0,    # Physics violations heavily penalized
                'data': 1.0,          # Data accuracy secondary
                'power': 0.5          # Power prediction tertiary
            }
        
        if self.model_config is None:
            self.model_config = {
                'hidden_layers': [64, 64, 32],   # Moderate complexity
                'activation': 'tanh',            # Good for physics problems
                'use_power_head': False          # Focus on KLa prediction
            }
        
        if self.physics_bounds is None:
            # Physics-based bounds from activated sludge literature
            self.physics_bounds = {
                'kla_min': 0.1,       # Minimum realistic KLa (1/h)
                'kla_max': 200.0,     # Maximum realistic KLa (1/h) - from literature
                'do_min': 0.1,        # Minimum DO for biological activity (mg/L)
                'do_max': 15.0,       # Maximum DO at typical conditions (mg/L)
                'our_max': 2000.0,    # Maximum OUR for activated sludge (mg/L/h)
                'temperature_min': 5.0,   # Minimum operating temperature (°C)
                'temperature_max': 40.0   # Maximum operating temperature (°C)
            }

@dataclass
class PathsConfig:
    """File paths configuration"""
    model_cache: str
    checkpoints: str  
    results: str = "data"

@dataclass
class LoggingConfig:
    """Logging configuration"""
    level: str
    file: str

@dataclass 
class TruePINNConfigContainer:
    """
    Complete configuration container for True PINN system
    """
    sumo: SumoConfig
    pinn: TruePINNConfig  # Updated to use TruePINNConfig
    paths: PathsConfig
    logging: LoggingConfig

class ConfigManager:
    """
    Configuration manager with True PINN support
    """
    
    @staticmethod
    def load() -> TruePINNConfigContainer:
        """Load configuration from JSON file"""
        # Find config.json relative to current file
        root = os.path.dirname(os.path.dirname(__file__))
        config_path = os.path.join(root, "config.json")
        
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
        with open(config_path, "r", encoding="utf-8") as f:
            config_data = json.load(f)
        
        # Create configuration objects
        sumo_config = SumoConfig(**config_data["sumo"])
        
        # Handle PINN config with defaults
        pinn_data = config_data.get("pinn", {})
        pinn_config = TruePINNConfig(**pinn_data)
        
        paths_config = PathsConfig(**config_data["paths"])
        logging_config = LoggingConfig(**config_data["logging"])
        
        return TruePINNConfigContainer(
            sumo=sumo_config,
            pinn=pinn_config,
            paths=paths_config,
            logging=logging_config
        )
    
    @staticmethod
    def create_default_config() -> Dict:
        """
        Create default configuration for True PINN
        
        This provides a template for config.json with physics-based defaults
        """
        return {
            "sumo": {
                "project_path": "path/to/your/model.sumo",
                "data_comm_interval": 600,  # 10 minutes in seconds
                "parallel_jobs": 1,
                "simulation_duration_days": 28,  # 4 weeks
                "variables": [
                    # Time and basic conditions
                    "Sumo__Time",
                    "Sumo__Plant__Influent__T",
                    "Sumo__Plant__Influent__Q",
                    
                    # CRITICAL: Aeration system (KLa drivers)
                    "Sumo__Plant__EnergyCenter__Hel_aeration",
                    
                    # Reactor state variables  
                    "Sumo__Plant__CSTR3__SO2",      # Current DO
                    "Sumo__Plant__CSTR3__OUR",      # Oxygen uptake rate
                    "Sumo__Plant__CSTR3__XTSS",     # Biomass concentration
                    "Sumo__Plant__CSTR3__kLaGO2"    # KLa target (if available)
                ]
            },
            "pinn": {
                "learning_rate": 0.001,
                "batch_size": 32,
                "buffer_size": 5000,
                "max_training_steps": 10000,
                "post_simulation_epochs": 10,
                
                # Physics-hierarchy loss weights
                "loss_weights": {
                    "physics": 1000.0,  # Physics dominates!
                    "data": 1.0,
                    "power": 0.5
                },
                
                # Model architecture
                "model_config": {
                    "hidden_layers": [64, 64, 32],
                    "activation": "tanh",
                    "use_power_head": False
                },
                
                # Physics bounds (literature-based)
                "physics_bounds": {
                    "kla_min": 0.1,
                    "kla_max": 200.0,
                    "do_min": 0.1,
                    "do_max": 15.0,
                    "our_max": 2000.0,
                    "temperature_min": 5.0,
                    "temperature_max": 40.0
                },
                
                # Physics compliance
                "physics_tolerance": 1e-6,
                "physics_compliance_threshold": 0.95
            },
            "paths": {
                "model_cache": "models",
                "checkpoints": "checkpoints",
                "results": "data"
            },
            "logging": {
                "level": "INFO",
                "file": "true_pinn.log"
            }
        }
    
    @staticmethod
    def save_default_config(path: str = "config.json"):
        """
        Save default True PINN configuration to file
        """
        default_config = ConfigManager.create_default_config()
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, indent=2)
        
        print(f"Default True PINN configuration saved to: {path}")
        print("Please update the SUMO project path and other site-specific parameters.")
    
    @staticmethod
    def validate_config(config: TruePINNConfigContainer) -> List[str]:
        """
        Validate True PINN configuration and return any issues
        """
        issues = []
        
        # Validate SUMO configuration
        if not os.path.exists(config.sumo.project_path):
            issues.append(f"SUMO project file not found: {config.sumo.project_path}")
        
        if config.sumo.data_comm_interval < 60:
            issues.append("Data communication interval too small (< 1 minute)")
        
        # Validate required variables for KLa prediction
        required_vars = [
            "Sumo__Time",
            "Sumo__Plant__EnergyCenter__Hel_aeration",  # Critical for KLa
            "Sumo__Plant__CSTR3__SO2",                  # Current DO
            "Sumo__Plant__CSTR3__OUR"                   # Oxygen demand
        ]
        
        missing_vars = [var for var in required_vars if var not in config.sumo.variables]
        if missing_vars:
            issues.append(f"Missing critical variables for KLa prediction: {missing_vars}")
        
        # Validate PINN configuration
        if config.pinn.loss_weights['physics'] < config.pinn.loss_weights['data']:
            issues.append("Physics weight should be higher than data weight for True PINN")
        
        if config.pinn.physics_compliance_threshold < 0.8:
            issues.append("Physics compliance threshold too low (< 80%)")
        
        # Validate physics bounds
        bounds = config.pinn.physics_bounds
        if bounds['kla_min'] >= bounds['kla_max']:
            issues.append("Invalid KLa bounds: min >= max")
        
        if bounds['kla_max'] > 500.0:
            issues.append("KLa max bound unrealistically high (> 500 1/h)")
        
        # Validate paths
        for path_name, path_value in [('model_cache', config.paths.model_cache), 
                                    ('results', config.paths.results)]:
            try:
                os.makedirs(path_value, exist_ok=True)
            except Exception as e:
                issues.append(f"Cannot create {path_name} directory: {e}")
        
        return issues
    
    @staticmethod
    def print_config_summary(config: TruePINNConfigContainer):
        """
        Print a summary of True PINN configuration
        """
        print("=" * 60)
        print("TRUE PINN CONFIGURATION SUMMARY")
        print("=" * 60)
        
        print(f"🏭 SUMO Simulation:")
        print(f"   Project: {os.path.basename(config.sumo.project_path)}")
        print(f"   Duration: {config.sumo.simulation_duration_days} days")
        print(f"   Data interval: {config.sumo.data_comm_interval / 60} minutes")
        print(f"   Variables: {len(config.sumo.variables)}")
        
        print(f"\n🧠 True PINN Model:")
        print(f"   Architecture: {config.pinn.model_config['hidden_layers']}")
        print(f"   Activation: {config.pinn.model_config['activation']}")
        print(f"   Learning rate: {config.pinn.learning_rate}")
        print(f"   Batch size: {config.pinn.batch_size}")
        
        print(f"\n⚖️ Physics Hierarchy:")
        print(f"   Physics weight: {config.pinn.loss_weights['physics']}")
        print(f"   Data weight: {config.pinn.loss_weights['data']}")
        print(f"   Compliance target: {config.pinn.physics_compliance_threshold:.1%}")
        
        print(f"\n📏 Physics Bounds:")
        bounds = config.pinn.physics_bounds
        print(f"   KLa range: {bounds['kla_min']} - {bounds['kla_max']} 1/h")
        print(f"   DO range: {bounds['do_min']} - {bounds['do_max']} mg/L")
        print(f"   Temperature: {bounds['temperature_min']} - {bounds['temperature_max']} °C")
        
        print(f"\n📁 Paths:")
        print(f"   Models: {config.paths.model_cache}")
        print(f"   Results: {config.paths.results}")
        
        print("=" * 60)

# Backward compatibility alias
Config = TruePINNConfigContainer