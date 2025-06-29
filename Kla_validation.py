# src/kla_validation.py

import logging
import numpy as np
import pandas as pd
from collections import deque
from typing import Dict, List, Optional
import os
import json
from datetime import datetime

logger = logging.getLogger(__name__)

class KLaValidationLogger:
    """
    Logs and validates KLa values from SUMO simulation against PINN model predictions.
    Provides real-time comparison and statistical analysis.
    """
    
    def __init__(self, buffer_size: int = 5000):
        """
        Initialize the KLa validation logger.
        
        Args:
            buffer_size: Maximum number of KLa comparisons to store in memory
        """
        self.buffer_size = buffer_size
        
        # Storage for KLa values
        self.sumo_kla_values = deque(maxlen=buffer_size)
        self.pinn_kla_values = deque(maxlen=buffer_size)
        self.timestamps = deque(maxlen=buffer_size)
        
        # Validation metrics
        self.validation_metrics = {
            'mean_absolute_error': deque(maxlen=1000),
            'root_mean_square_error': deque(maxlen=1000),
            'relative_error': deque(maxlen=1000),
            'correlation_coefficient': deque(maxlen=1000),
            'validation_points': 0
        }
        
        # Logging intervals
        self.last_log_time = None
        self.log_interval_points = 50  # Log every 50 data points
        
        logger.info("KLa Validation Logger initialized")
    
    def log_kla_comparison(self, sumo_time: float, sumo_kla: float, pinn_kla: float):
        """
        Log a single KLa comparison between SUMO and PINN values.
        
        Args:
            sumo_time: Simulation time in milliseconds
            sumo_kla: KLa value from SUMO simulation (1/h)
            pinn_kla: KLa value from PINN model (1/h)
        """
        try:
            # Convert time to hours for consistency
            time_hours = sumo_time / (1000 * 3600)  # ms to hours
            
            # Store values
            self.sumo_kla_values.append(sumo_kla)
            self.pinn_kla_values.append(pinn_kla)
            self.timestamps.append(time_hours)
            
            # Update validation point counter
            self.validation_metrics['validation_points'] += 1
            
            # Calculate real-time metrics if we have enough data
            if len(self.sumo_kla_values) >= 2:
                self._update_validation_metrics()
            
            # Periodic logging
            if (self.validation_metrics['validation_points'] % self.log_interval_points == 0):
                self._log_validation_summary()
                
        except Exception as e:
            logger.error(f"Error in KLa comparison logging: {e}", exc_info=True)
    
    def _update_validation_metrics(self):
        """Update validation metrics with latest data."""
        try:
            # Convert to numpy arrays for calculations
            sumo_arr = np.array(list(self.sumo_kla_values))
            pinn_arr = np.array(list(self.pinn_kla_values))
            
            # Calculate metrics
            mae = np.mean(np.abs(sumo_arr - pinn_arr))
            rmse = np.sqrt(np.mean((sumo_arr - pinn_arr) ** 2))
            
            # Relative error (avoid division by zero)
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_error = np.mean(np.abs((sumo_arr - pinn_arr) / sumo_arr)) * 100
                rel_error = rel_error if np.isfinite(rel_error) else 0.0
            
            # Correlation coefficient
            if len(sumo_arr) > 1 and np.std(sumo_arr) > 0 and np.std(pinn_arr) > 0:
                corr_coef = np.corrcoef(sumo_arr, pinn_arr)[0, 1]
                corr_coef = corr_coef if np.isfinite(corr_coef) else 0.0
            else:
                corr_coef = 0.0
            
            # Store metrics
            self.validation_metrics['mean_absolute_error'].append(mae)
            self.validation_metrics['root_mean_square_error'].append(rmse)
            self.validation_metrics['relative_error'].append(rel_error)
            self.validation_metrics['correlation_coefficient'].append(corr_coef)
            
        except Exception as e:
            logger.error(f"Error updating validation metrics: {e}", exc_info=True)
    
    def _log_validation_summary(self):
        """Log current validation statistics."""
        try:
            if not self.validation_metrics['mean_absolute_error']:
                return
            
            # Get recent metrics (last 100 points)
            recent_mae = list(self.validation_metrics['mean_absolute_error'])[-100:]
            recent_rmse = list(self.validation_metrics['root_mean_square_error'])[-100:]
            recent_rel_err = list(self.validation_metrics['relative_error'])[-100:]
            recent_corr = list(self.validation_metrics['correlation_coefficient'])[-100:]
            
            # Get recent KLa values for comparison
            recent_sumo = list(self.sumo_kla_values)[-10:]
            recent_pinn = list(self.pinn_kla_values)[-10:]
            
            logger.info(f"=== KLa Validation Summary (Point {self.validation_metrics['validation_points']}) ===")
            logger.info(f"Recent SUMO KLa: {np.mean(recent_sumo):.3f} ± {np.std(recent_sumo):.3f} 1/h")
            logger.info(f"Recent PINN KLa: {np.mean(recent_pinn):.3f} ± {np.std(recent_pinn):.3f} 1/h")
            logger.info(f"Mean Absolute Error: {np.mean(recent_mae):.4f} 1/h")
            logger.info(f"Root Mean Square Error: {np.mean(recent_rmse):.4f} 1/h")
            logger.info(f"Relative Error: {np.mean(recent_rel_err):.2f}%")
            logger.info(f"Correlation Coefficient: {np.mean(recent_corr):.4f}")
            logger.info("=" * 60)
            
        except Exception as e:
            logger.error(f"Error logging validation summary: {e}", exc_info=True)
    
    def get_validation_report(self) -> Dict:
        """
        Generate comprehensive validation report.
        
        Returns:
            Dictionary containing validation statistics and data
        """
        try:
            if len(self.sumo_kla_values) == 0:
                return {"error": "No validation data available"}
            
            # Convert to numpy arrays
            sumo_arr = np.array(list(self.sumo_kla_values))
            pinn_arr = np.array(list(self.pinn_kla_values))
            times_arr = np.array(list(self.timestamps))
            
            # Calculate final statistics
            mae = np.mean(np.abs(sumo_arr - pinn_arr))
            rmse = np.sqrt(np.mean((sumo_arr - pinn_arr) ** 2))
            
            with np.errstate(divide='ignore', invalid='ignore'):
                mape = np.mean(np.abs((sumo_arr - pinn_arr) / sumo_arr)) * 100
                mape = mape if np.isfinite(mape) else 0.0
            
            # Correlation
            if len(sumo_arr) > 1:
                correlation = np.corrcoef(sumo_arr, pinn_arr)[0, 1]
                correlation = correlation if np.isfinite(correlation) else 0.0
            else:
                correlation = 0.0
            
            # Bias analysis
            bias = np.mean(pinn_arr - sumo_arr)
            
            report = {
                'validation_summary': {
                    'total_comparisons': len(sumo_arr),
                    'time_span_hours': float(times_arr[-1] - times_arr[0]) if len(times_arr) > 1 else 0.0,
                    'mean_absolute_error': float(mae),
                    'root_mean_square_error': float(rmse),
                    'mean_absolute_percentage_error': float(mape),
                    'correlation_coefficient': float(correlation),
                    'bias': float(bias)
                },
                'sumo_kla_stats': {
                    'mean': float(np.mean(sumo_arr)),
                    'std': float(np.std(sumo_arr)),
                    'min': float(np.min(sumo_arr)),
                    'max': float(np.max(sumo_arr)),
                    'median': float(np.median(sumo_arr))
                },
                'pinn_kla_stats': {
                    'mean': float(np.mean(pinn_arr)),
                    'std': float(np.std(pinn_arr)),
                    'min': float(np.min(pinn_arr)),
                    'max': float(np.max(pinn_arr)),
                    'median': float(np.median(pinn_arr))
                },
                'timestamp': datetime.now().isoformat()
            }
            
            return report
            
        except Exception as e:
            logger.error(f"Error generating validation report: {e}", exc_info=True)
            return {"error": f"Error generating report: {str(e)}"}
    
    def save_validation_data(self, filepath: str):
        """
        Save validation data to CSV file.
        
        Args:
            filepath: Path to save the CSV file
        """
        try:
            if len(self.sumo_kla_values) == 0:
                logger.warning("No validation data to save")
                return
            
            # Create DataFrame
            df = pd.DataFrame({
                'time_hours': list(self.timestamps),
                'sumo_kla': list(self.sumo_kla_values),
                'pinn_kla': list(self.pinn_kla_values),
                'absolute_error': [abs(s - p) for s, p in zip(self.sumo_kla_values, self.pinn_kla_values)],
                'relative_error_percent': [abs((s - p) / s) * 100 if s != 0 else 0 
                                         for s, p in zip(self.sumo_kla_values, self.pinn_kla_values)]
            })
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save to CSV
            df.to_csv(filepath, index=False)
            logger.info(f"Validation data saved to {filepath}")
            
            # Also save the validation report as JSON
            report_path = filepath.replace('.csv', '_report.json')
            report = self.get_validation_report()
            with open(report_path, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Validation report saved to {report_path}")
            
        except Exception as e:
            logger.error(f"Error saving validation data: {e}", exc_info=True)
    
    def get_current_kla_difference(self) -> Optional[float]:
        """
        Get the most recent KLa difference between SUMO and PINN.
        
        Returns:
            Most recent absolute difference, or None if no data
        """
        if len(self.sumo_kla_values) == 0:
            return None
        
        return abs(self.sumo_kla_values[-1] - self.pinn_kla_values[-1])


# Integration function for use in callback_manager.py
def integrate_kla_validation(validation_logger: KLaValidationLogger, 
                           job_data: Dict, 
                           current_time: float):
    """
    Integration function to be called from callback_manager.py data_callback.
    
    Args:
        validation_logger: Instance of KLaValidationLogger
        job_data: Job data dictionary containing training engine
        current_time: Current simulation time in milliseconds
    """
    try:
        # Get SUMO KLa value from the most recent data point
        # This assumes 'SumoPlant__CSTR3__kLaGO2' is in the monitored variables
        sumo_kla_data = job_data.get('SumoPlant__CSTR3__kLaGO2', [])
        
        if not sumo_kla_data:
            return  # No SUMO KLa data available yet
        
        sumo_kla = sumo_kla_data[-1]  # Most recent value
        
        # Get PINN KLa value from training engine
        training_engine = job_data.get('training_engine')
        if training_engine and hasattr(training_engine, 'model'):
            pinn_kla = float(training_engine.model.KLa.numpy())
        else:
            return  # No PINN model available
        
        # Log the comparison
        validation_logger.log_kla_comparison(current_time, sumo_kla, pinn_kla)
        
    except Exception as e:
        logger.error(f"Error in KLa validation integration: {e}", exc_info=True)