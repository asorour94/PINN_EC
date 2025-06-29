import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

class EnhancedPINNAnalyzer:
    """Enhanced analysis metrics for PINN training results"""
    
    def __init__(self, training_data, process_data):
        self.training_data = training_data
        self.process_data = process_data
        
    def analyze_comprehensive(self):
        """Comprehensive analysis with enhanced metrics"""
        results = {}
        
        # 1. TRAINING QUALITY METRICS
        results['training_quality'] = self._analyze_training_quality()
        
        # 2. PARAMETER ESTIMATION QUALITY
        results['parameter_quality'] = self._analyze_parameter_estimation()
        
        # 3. PHYSICS COMPLIANCE
        results['physics_compliance'] = self._analyze_physics_compliance()
        
        # 4. PROCESS ENGINEERING VALIDATION
        results['process_validation'] = self._analyze_process_engineering()
        
        # 5. STATISTICAL ROBUSTNESS
        results['statistical_robustness'] = self._analyze_statistical_robustness()
        
        # 6. PREDICTIVE PERFORMANCE
        results['predictive_performance'] = self._analyze_predictive_performance()
        
        # 7. UNCERTAINTY QUANTIFICATION
        results['uncertainty_analysis'] = self._analyze_uncertainty()
        
        # 8. CONVERGENCE DIAGNOSTICS
        results['convergence_diagnostics'] = self._analyze_convergence()
        
        return results
    
    def _analyze_training_quality(self):
        """Enhanced training quality metrics"""
        losses = self.training_data['losses']
        
        return {
            # Loss convergence metrics
            'final_data_loss': losses['data'][-1],
            'final_physics_loss': losses['physics'][-1],
            'loss_reduction_ratio': losses['data'][0] / losses['data'][-1],
            'physics_loss_reduction': losses['physics'][0] / losses['physics'][-1],
            
            # Convergence stability
            'loss_volatility': np.std(losses['data'][-20:]) / np.mean(losses['data'][-20:]),
            'convergence_rate': self._calculate_convergence_rate(losses['data']),
            'plateau_detection': self._detect_plateau(losses['data']),
            
            # Training efficiency
            'steps_to_90_percent': self._steps_to_percent_reduction(losses['data'], 0.9),
            'training_efficiency': len(losses['data']) / np.log10(losses['data'][0] / losses['data'][-1])
        }
    
    def _analyze_parameter_estimation(self):
        """Enhanced KLa parameter analysis"""
        kla_values = self.training_data['kla_history']
        
        return {
            # Basic statistics
            'final_kla': kla_values[-1],
            'mean_kla': np.mean(kla_values),
            'std_kla': np.std(kla_values),
            'cv_kla': np.std(kla_values) / np.mean(kla_values),  # Coefficient of variation
            
            # Convergence quality
            'kla_convergence_stability': np.std(kla_values[-20:]) / np.mean(kla_values[-20:]),
            'kla_drift': (kla_values[-1] - kla_values[-20]) / kla_values[-20],
            'kla_oscillation': self._calculate_oscillation_index(kla_values),
            
            # Engineering validity
            'kla_realistic_range': 0.5 <= kla_values[-1] <= 5.0,  # Typical range for activated sludge
            'kla_confidence_interval': self._calculate_bootstrap_ci(kla_values[-50:]),
            
            # Parameter sensitivity
            'gradient_magnitude_final': self.training_data.get('kla_gradients', [0])[-1],
            'parameter_sensitivity': self._calculate_parameter_sensitivity()
        }
    
    def _analyze_physics_compliance(self):
        """Physics constraint satisfaction analysis"""
        residuals = self.training_data.get('physics_residuals', [])
        
        return {
            # Residual statistics
            'mean_absolute_residual': np.mean(np.abs(residuals)),
            'max_residual': np.max(np.abs(residuals)),
            'residual_std': np.std(residuals),
            'residual_bias': np.mean(residuals),
            
            # Physics compliance score
            'physics_compliance_score': np.exp(-np.mean(np.abs(residuals))),
            'mass_balance_error': self._calculate_mass_balance_error(),
            
            # Residual patterns
            'residual_autocorrelation': self._calculate_autocorrelation(residuals),
            'residual_normality_test': stats.shapiro(residuals[-100:]) if len(residuals) >= 100 else None
        }
    
    def _analyze_process_engineering(self):
        """Process engineering validation metrics"""
        do_data = self.process_data['DO']
        our_data = self.process_data['OUR'] 
        power_data = self.process_data['Power']
        
        return {
            # DO analysis
            'do_range_validity': (0 <= np.min(do_data)) and (np.max(do_data) <= 10),
            'do_growth_rate': np.polyfit(range(len(do_data)), do_data, 1)[0],
            'do_saturation_approach': self._analyze_saturation_approach(do_data),
            
            # OUR analysis  
            'our_max': np.max(our_data),
            'our_growth_pattern': self._classify_growth_pattern(our_data),
            'our_do_correlation': np.corrcoef(do_data, our_data)[0,1],
            
            # Power consumption
            'specific_power': power_data[-1] / do_data[-1] if do_data[-1] > 0 else np.inf,
            'power_efficiency': self._calculate_power_efficiency(power_data, do_data),
            
            # Process ratios
            'kla_our_ratio': self.training_data['kla_history'][-1] / (our_data[-1] / 1000),
            'oxygen_transfer_efficiency': self._calculate_ote()
        }
    
    def _analyze_statistical_robustness(self):
        """Statistical robustness analysis"""
        kla_values = self.training_data['kla_history']
        
        return {
            # Distribution analysis
            'kla_skewness': stats.skew(kla_values),
            'kla_kurtosis': stats.kurtosis(kla_values),
            'normality_test': stats.jarque_bera(kla_values),
            
            # Outlier detection
            'outlier_percentage': self._calculate_outlier_percentage(kla_values),
            'robust_mean': np.median(kla_values),  # Less sensitive to outliers
            'interquartile_range': np.percentile(kla_values, 75) - np.percentile(kla_values, 25),
            
            # Confidence intervals
            'confidence_95': stats.t.interval(0.95, len(kla_values)-1, 
                                            loc=np.mean(kla_values), 
                                            scale=stats.sem(kla_values)),
            
            # Stability metrics
            'moving_average_stability': self._calculate_moving_stability(kla_values),
            'trend_significance': self._test_trend_significance(kla_values)
        }
    
    def _analyze_predictive_performance(self):
        """Predictive performance metrics"""
        predicted_do = self.training_data.get('predicted_do', [])
        observed_do = self.training_data.get('observed_do', [])
        
        if len(predicted_do) == 0 or len(observed_do) == 0:
            return {'status': 'No prediction data available'}
        
        return {
            # Accuracy metrics
            'r2_score': r2_score(observed_do, predicted_do),
            'mae': mean_absolute_error(observed_do, predicted_do),
            'rmse': np.sqrt(mean_squared_error(observed_do, predicted_do)),
            'mape': np.mean(np.abs((observed_do - predicted_do) / observed_do)) * 100,
            
            # Bias analysis
            'prediction_bias': np.mean(predicted_do - observed_do),
            'systematic_error': self._calculate_systematic_error(predicted_do, observed_do),
            
            # Prediction quality
            'nash_sutcliffe': 1 - np.sum((observed_do - predicted_do)**2) / np.sum((observed_do - np.mean(observed_do))**2),
            'index_of_agreement': self._calculate_index_of_agreement(predicted_do, observed_do)
        }
    
    def _analyze_uncertainty(self):
        """Uncertainty quantification analysis"""
        kla_values = self.training_data['kla_history']
        
        return {
            # Epistemic uncertainty (model uncertainty)
            'model_uncertainty': np.std(kla_values[-50:]),
            'parameter_uncertainty': self._estimate_parameter_uncertainty(),
            
            # Aleatoric uncertainty (data uncertainty)
            'measurement_noise_estimate': self._estimate_measurement_noise(),
            
            # Total uncertainty
            'total_uncertainty': self._calculate_total_uncertainty(),
            'uncertainty_propagation': self._propagate_uncertainty(),
            
            # Confidence bounds
            'prediction_intervals': self._calculate_prediction_intervals(),
            'credible_intervals': self._calculate_credible_intervals()
        }
    
    def _analyze_convergence(self):
        """Advanced convergence diagnostics"""
        losses = self.training_data['losses']['data']
        kla_values = self.training_data['kla_history']
        
        return {
            # Convergence tests
            'geweke_test': self._geweke_convergence_test(kla_values),
            'heidelberger_welch': self._heidelberger_welch_test(kla_values),
            'effective_sample_size': self._calculate_effective_sample_size(kla_values),
            
            # Convergence rates
            'linear_convergence_rate': self._estimate_linear_convergence_rate(losses),
            'exponential_decay_constant': self._estimate_exponential_decay(losses),
            
            # Multi-chain diagnostics (if applicable)
            'potential_scale_reduction': self._calculate_psrf() if hasattr(self, 'multiple_chains') else None,
            'between_chain_variance': self._calculate_between_chain_variance() if hasattr(self, 'multiple_chains') else None
        }
    
    # Helper methods (implementation details)
    def _calculate_convergence_rate(self, losses):
        """Calculate loss convergence rate"""
        if len(losses) < 10:
            return np.nan
        log_losses = np.log(losses[-50:])
        return -np.polyfit(range(len(log_losses)), log_losses, 1)[0]
    
    def _detect_plateau(self, losses, window=20, threshold=0.01):
        """Detect if training has plateaued"""
        if len(losses) < window:
            return False
        recent_losses = losses[-window:]
        return (np.max(recent_losses) - np.min(recent_losses)) / np.mean(recent_losses) < threshold
    
    def _steps_to_percent_reduction(self, losses, percent):
        """Find steps needed for percentage loss reduction"""
        target = losses[0] * (1 - percent)
        for i, loss in enumerate(losses):
            if loss <= target:
                return i
        return len(losses)
    
    def _calculate_oscillation_index(self, values):
        """Calculate parameter oscillation index"""
        if len(values) < 3:
            return 0
        direction_changes = 0
        for i in range(1, len(values)-1):
            if (values[i] > values[i-1] and values[i] > values[i+1]) or \
               (values[i] < values[i-1] and values[i] < values[i+1]):
                direction_changes += 1
        return direction_changes / (len(values) - 2)
    
    def _calculate_bootstrap_ci(self, values, n_bootstrap=1000, alpha=0.05):
        """Calculate bootstrap confidence interval"""
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = np.random.choice(values, size=len(values), replace=True)
            bootstrap_means.append(np.mean(sample))
        
        lower = np.percentile(bootstrap_means, 100 * alpha/2)
        upper = np.percentile(bootstrap_means, 100 * (1 - alpha/2))
        return (lower, upper)
    
    def _calculate_parameter_sensitivity(self):
        """Calculate parameter sensitivity analysis"""
        # Placeholder - would need gradient information
        return {'status': 'Requires gradient history for full analysis'}
    
    def _calculate_mass_balance_error(self):
        """Calculate mass balance closure error"""
        # Would implement based on DO mass balance equation
        return {'status': 'Requires detailed mass balance implementation'}
    
    def _calculate_autocorrelation(self, residuals, max_lag=20):
        """Calculate residual autocorrelation"""
        if len(residuals) < max_lag * 2:
            return np.nan
        autocorr = np.correlate(residuals, residuals, mode='full')
        autocorr = autocorr[autocorr.size // 2:]
        autocorr = autocorr / autocorr[0]
        return autocorr[:max_lag]
    
    def _analyze_saturation_approach(self, do_data):
        """Analyze approach to DO saturation"""
        # Fit exponential approach curve
        if len(do_data) < 50:
            return {'status': 'Insufficient data'}
        
        # Estimate saturation value and approach rate
        do_max = np.max(do_data)
        time_to_90_percent = None
        target = 0.9 * do_max
        
        for i, do in enumerate(do_data):
            if do >= target:
                time_to_90_percent = i
                break
                
        return {
            'estimated_saturation': do_max,
            'time_to_90_percent': time_to_90_percent,
            'approach_rate': self._fit_exponential_approach(do_data)
        }
    
    def _classify_growth_pattern(self, our_data):
        """Classify OUR growth pattern"""
        if len(our_data) < 10:
            return 'insufficient_data'
        
        # Fit different growth models
        x = np.arange(len(our_data))
        
        # Linear fit
        linear_r2 = r2_score(our_data, np.polyval(np.polyfit(x, our_data, 1), x))
        
        # Exponential fit (log-transform)
        try:
            log_our = np.log(our_data + 1e-10)
            exp_r2 = r2_score(log_our, np.polyval(np.polyfit(x, log_our, 1), x))
        except:
            exp_r2 = 0
        
        # Logistic fit would require more complex fitting
        
        if exp_r2 > linear_r2 and exp_r2 > 0.8:
            return 'exponential'
        elif linear_r2 > 0.8:
            return 'linear'
        else:
            return 'complex'
    
    def _calculate_power_efficiency(self, power_data, do_data):
        """Calculate aeration power efficiency"""
        if len(power_data) == 0 or len(do_data) == 0:
            return np.nan
        
        # Energy per unit DO increase
        power_avg = np.mean(power_data[-50:])  # kW
        do_increase = do_data[-1] - do_data[0]  # mg/L
        
        if do_increase <= 0:
            return np.inf
            
        return power_avg / do_increase  # kW per mg/L DO increase

    def generate_enhanced_report(self, results):
        """Generate comprehensive analysis report"""
        report = {
            'executive_summary': self._generate_executive_summary(results),
            'detailed_metrics': results,
            'recommendations': self._generate_recommendations(results),
            'quality_score': self._calculate_overall_quality_score(results)
        }
        return report
    
    def _generate_executive_summary(self, results):
        """Generate executive summary of results"""
        summary = {
            'training_success': results['training_quality']['final_data_loss'] < 1e-2,
            'parameter_reliability': results['parameter_quality']['cv_kla'] < 0.5,
            'physics_compliance': results['physics_compliance']['physics_compliance_score'] > 0.9,
            'engineering_validity': results['process_validation']['do_range_validity'],
            'overall_assessment': 'EXCELLENT' if all([
                results['training_quality']['final_data_loss'] < 1e-2,
                results['parameter_quality']['kla_realistic_range'],
                results['physics_compliance']['physics_compliance_score'] > 0.8
            ]) else 'NEEDS_IMPROVEMENT'
        }
        return summary
    
    def _generate_recommendations(self, results):
        """Generate improvement recommendations"""
        recommendations = []
        
        if results['training_quality']['loss_volatility'] > 0.1:
            recommendations.append("Consider reducing learning rate for more stable convergence")
        
        if results['parameter_quality']['cv_kla'] > 0.5:
            recommendations.append("Parameter uncertainty is high - consider longer training or regularization")
        
        if not results['parameter_quality']['kla_realistic_range']:
            recommendations.append("KLa value outside typical range - check model physics or data quality")
        
        return recommendations
    
    def _calculate_overall_quality_score(self, results):
        """Calculate overall model quality score (0-100)"""
        scores = []
        
        # Training quality (30%)
        training_score = min(100, 100 * np.exp(-results['training_quality']['final_data_loss'] * 100))
        scores.append(('training', training_score, 0.3))
        
        # Parameter quality (25%)
        param_score = 100 if results['parameter_quality']['kla_realistic_range'] else 50
        param_score *= (1 - min(0.5, results['parameter_quality']['cv_kla']))
        scores.append(('parameter', param_score, 0.25))
        
        # Physics compliance (25%)
        physics_score = results['physics_compliance']['physics_compliance_score'] * 100
        scores.append(('physics', physics_score, 0.25))
        
        # Process validity (20%)
        process_score = 100 if results['process_validation']['do_range_validity'] else 0
        scores.append(('process', process_score, 0.2))
        
        weighted_score = sum(score * weight for _, score, weight in scores)
        
        return {
            'overall_score': weighted_score,
            'component_scores': {name: score for name, score, _ in scores},
            'grade': self._assign_grade(weighted_score)
        }
    
    def _assign_grade(self, score):
        """Assign letter grade based on score"""
        if score >= 90: return 'A'
        elif score >= 80: return 'B'
        elif score >= 70: return 'C'
        elif score >= 60: return 'D'
        else: return 'F'

# Usage example
def run_enhanced_analysis(training_log_file, process_data_file):
    """Run enhanced PINN analysis"""
    
    # Load data (pseudocode)
    training_data = load_training_data(training_log_file)
    process_data = load_process_data(process_data_file)
    
    # Run analysis
    analyzer = EnhancedPINNAnalyzer(training_data, process_data)
    results = analyzer.analyze_comprehensive()
    report = analyzer.generate_enhanced_report(results)
    
    # Display results
    print(f"Overall Quality Score: {report['quality_score']['overall_score']:.1f} ({report['quality_score']['grade']})")
    print(f"Assessment: {report['executive_summary']['overall_assessment']}")
    
    return report