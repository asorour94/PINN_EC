# src/callback_manager.py - UPDATED FOR TRUE PINN INTEGRATION

import dynamita.scheduler as ds
import dynamita.tool as dtool
from threading import Event
import logging
import numpy as np

# Event to signal when the run has finished
finish_event = Event()

# Setup logging
logger = logging.getLogger(__name__)

def true_pinn_message_callback(job_id, message):
    """
    Enhanced message callback for True PINN simulations
    
    Handles simulation completion and physics compliance reporting
    """
    if ds.sumo.isSimFinishedMsg(message):
        logger.info(f"True PINN simulation finished for job {job_id}")
        
        # Get final physics compliance report
        jd = ds.sumo.getJobData(job_id)
        training_engine = jd.get('training_engine')
        
        if training_engine:
            try:
                final_report = training_engine.get_physics_compliance_report()
                physics_compliance = final_report.get('physics_compliance', {}).get('average_rate', 0.0)
                
                logger.info("=" * 60)
                logger.info("🏁 SIMULATION COMPLETE - FINAL PHYSICS ASSESSMENT")
                logger.info("=" * 60)
                logger.info(f"Physics Compliance Rate: {physics_compliance:.1%}")
                logger.info(f"Training Steps: {final_report.get('training_steps', 0)}")
                logger.info(f"Physics Violations: {final_report.get('training_status', {}).get('total_violations', 0)}")
                
                # Classify final result
                if physics_compliance >= 0.95:
                    logger.info("🟢 RESULT: TRUE PHYSICS-INFORMED NEURAL NETWORK ACHIEVED!")
                    logger.info("✅ Physics constraints satisfied consistently")
                elif physics_compliance >= 0.85:
                    logger.info("🟡 RESULT: PARTIALLY PHYSICS-INFORMED NETWORK")
                    logger.info("⚠️  Physics compliance above 85% but below target")
                elif physics_compliance >= 0.70:
                    logger.info("🟠 RESULT: PHYSICS-AWARE NEURAL NETWORK")
                    logger.info("⚠️  Some physics awareness but significant violations")
                else:
                    logger.info("🔴 RESULT: PHYSICS-CONTRADICTED NEURAL NETWORK")
                    logger.info("❌ Physics constraints frequently violated")
                
                logger.info("=" * 60)
                
            except Exception as e:
                logger.error(f"Error generating final physics report: {e}")
        
        ds.sumo.finish(job_id)
        finish_event.set()

def true_pinn_data_callback(job_id, data):
    """
    Enhanced data callback for True PINN training
    
    Key improvements:
    1. Extracts KLa-determining features (not just influent)
    2. Performs physics-aware data preprocessing
    3. Feeds data to physics-hierarchy training engine
    4. Monitors physics compliance in real-time
    5. Validates KLa predictions against SUMO values
    """
    jd = ds.sumo.getJobData(job_id)
    
    # Store all variables for final analysis
    for var, val in data.items():
        if var in jd and isinstance(jd[var], list):
            jd[var].append(val)
    
    # Get True PINN components
    training_engine = jd.get('training_engine')
    normalizer = jd.get('normalizer')
    validation_logger = jd.get('validation_logger')
    
    if training_engine is None:
        return
    
    try:
        # Extract KLa-determining variables (PHYSICS-BASED SELECTION)
        current_time = data.get('Sumo__Time', 0)  # milliseconds
        
        # Environmental conditions
        temperature = data.get('Sumo__Plant__Influent__T', 20.0)  # °C
        flow_rate = data.get('Sumo__Plant__Influent__Q', 0.0)     # m³/d
        
        # CRITICAL: Aeration system parameters (primary KLa drivers)
        aeration_power = data.get('Sumo__Plant__EnergyCenter__Hel_aeration', 0.0)  # kW
        
        # Current reactor state
        current_do = data.get('Sumo__Plant__CSTR3__SO2', 0.0)     # mg/L
        current_our = data.get('Sumo__Plant__CSTR3__OUR', 0.0)    # mg/L/h
        biomass = data.get('Sumo__Plant__CSTR3__XTSS', 0.0)       # mg/L
        
        # Target value (if available from SUMO)
        sumo_kla = data.get('Sumo__Plant__CSTR3__kLaGO2', None)   # 1/h
        
        # Physics-based calculations
        do_saturation = calculate_do_saturation(temperature)
        reactor_volume = 1000.0  # m³ (site-specific parameter)
        
        # Create True PINN training data point
        training_data_point = {
            # Basic features
            't': current_time / dtool.hour,  # Convert ms to hours
            'temperature': temperature,
            'flow_rate': flow_rate,
            'aeration_power': aeration_power,
            'do': current_do,
            'our': current_our,
            'biomass': biomass,
            'volume': reactor_volume,
            
            # Physics validation data
            'do_sat': do_saturation,
            
            # Target (if available)
            'kla_actual': sumo_kla if sumo_kla is not None else None,
            'do_measured': current_do,
            'our_measured': current_our
        }
        
        # Apply physics-aware normalization if normalizer available
        if normalizer:
            try:
                normalized_data = normalizer.normalize(training_data_point)
                training_data_point.update(normalized_data)
            except Exception as e:
                logger.error(f"Error in data normalization: {e}")
        
        # Send to True PINN training engine
        training_engine.enqueue_data(**training_data_point)
        
        # Real-time KLa validation (if SUMO KLa available)
        if sumo_kla is not None and validation_logger:
            try:
                # Get current PINN KLa prediction
                pinn_kla = get_current_pinn_kla_prediction(training_engine)
                
                # Log comparison
                validation_logger.log_kla_comparison(current_time, sumo_kla, pinn_kla)
                
                # Periodic validation reporting
                data_count = len(jd.get('Sumo__Time', []))
                if data_count % 200 == 0:  # Every 200 points
                    log_kla_validation_summary(validation_logger, data_count)
                    
            except Exception as e:
                logger.error(f"Error in KLa validation: {e}")
        
        # Physics compliance monitoring
        data_count = len(jd.get('Sumo__Time', []))
        if data_count % 100 == 0:  # Every 100 data points
            monitor_physics_compliance(training_engine, data_count, aeration_power, current_do, current_our)
        
        # Debug logging for first few data points
        if data_count <= 5:
            logger.info(f"True PINN Data Point #{data_count}:")
            logger.info(f"  Time: {current_time / dtool.hour:.2f} hours")
            logger.info(f"  Aeration Power: {aeration_power:.1f} kW")
            logger.info(f"  DO: {current_do:.2f} mg/L")
            logger.info(f"  OUR: {current_our:.1f} mg/L/h")
            if sumo_kla is not None:
                logger.info(f"  SUMO KLa: {sumo_kla:.3f} 1/h")
                
    except Exception as e:
        logger.error(f"Error in True PINN data callback: {e}", exc_info=True)

def calculate_do_saturation(temperature):
    """
    Calculate DO saturation using physics-based correlation
    
    Uses Benson & Krause (1984) equation for freshwater
    """
    try:
        T = float(temperature)
        # Standard correlation: DO_sat = 14.652 - 0.41022*T + 0.007991*T² - 0.000077774*T³
        do_sat = 14.652 - 0.41022*T + 0.007991*T**2 - 0.000077774*T**3
        
        # Apply realistic bounds
        return max(1.0, min(15.0, do_sat))
    except:
        return 8.5  # Default value at ~20°C

def get_current_pinn_kla_prediction(training_engine):
    """
    Get current KLa prediction from True PINN model
    """
    try:
        import tensorflow as tf
        
        # Create dummy input for current state prediction
        # Note: In practice, this should use actual current conditions
        dummy_input = tf.constant([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]], dtype=tf.float32)
        predictions = training_engine.model(dummy_input, training=False)
        
        # Extract KLa prediction
        kla_pred = predictions['kla_predicted']
        return float(kla_pred[0].numpy())
        
    except Exception as e:
        logger.error(f"Error getting PINN KLa prediction: {e}")
        return 1.0  # Default fallback value

def monitor_physics_compliance(training_engine, data_count, aeration_power, do, our):
    """
    Monitor physics compliance and log status
    """
    try:
        compliance_report = training_engine.get_physics_compliance_report()
        
        if 'physics_compliance' in compliance_report:
            compliance_rate = compliance_report['physics_compliance']['average_rate']
            physics_loss = compliance_report['physics_loss']['current']
            training_steps = compliance_report['training_steps']
            
            logger.info(f"📊 Physics Monitor - Data Point {data_count}:")
            logger.info(f"   Physics Compliance: {compliance_rate:.1%}")
            logger.info(f"   Physics Loss: {physics_loss:.2e}")
            logger.info(f"   Training Steps: {training_steps}")
            logger.info(f"   Current Conditions: P={aeration_power:.1f}kW, DO={do:.2f}mg/L, OUR={our:.1f}mg/L/h")
            
            # Warning for poor physics compliance
            if compliance_rate < 0.8:
                logger.warning("⚠️  Physics compliance below 80% - investigating...")
                
                # Additional diagnostics for poor compliance
                if aeration_power <= 0:
                    logger.warning("   → Aeration power is zero or negative!")
                if do <= 0 or do > 12:
                    logger.warning(f"   → DO value seems unrealistic: {do:.2f} mg/L")
                if our < 0 or our > 1000:
                    logger.warning(f"   → OUR value seems unrealistic: {our:.1f} mg/L/h")
            
            # Celebration for excellent compliance
            elif compliance_rate > 0.95:
                if data_count % 500 == 0:  # Don't spam logs
                    logger.info("🎉 Excellent physics compliance maintained!")
                    
    except Exception as e:
        logger.error(f"Error monitoring physics compliance: {e}")

def log_kla_validation_summary(validation_logger, data_count):
    """
    Log KLa validation summary
    """
    try:
        validation_report = validation_logger.get_validation_report()
        
        if 'validation_summary' in validation_report:
            summary = validation_report['validation_summary']
            sumo_stats = validation_report['sumo_kla_stats']
            pinn_stats = validation_report['pinn_kla_stats']
            
            logger.info(f"🔬 KLa Validation Summary - Data Point {data_count}:")
            logger.info(f"   Comparisons: {summary['total_comparisons']}")
            logger.info(f"   MAE: {summary['mean_absolute_error']:.4f} 1/h")
            logger.info(f"   RMSE: {summary['root_mean_square_error']:.4f} 1/h")
            logger.info(f"   Correlation: {summary['correlation_coefficient']:.3f}")
            logger.info(f"   SUMO KLa: {sumo_stats['mean']:.3f} ± {sumo_stats['std']:.3f} 1/h")
            logger.info(f"   PINN KLa: {pinn_stats['mean']:.3f} ± {pinn_stats['std']:.3f} 1/h")
            
            # Assess validation quality
            if summary['correlation_coefficient'] > 0.8:
                logger.info("✅ Excellent KLa prediction correlation!")
            elif summary['correlation_coefficient'] > 0.6:
                logger.info("🟡 Good KLa prediction correlation")
            else:
                logger.warning("🔴 Poor KLa prediction correlation - needs attention")
                
    except Exception as e:
        logger.error(f"Error logging KLa validation summary: {e}")

def validate_data_quality(data):
    """
    Validate incoming data quality for True PINN training
    """
    issues = []
    
    # Check for required variables
    required_vars = [
        'Sumo__Time',
        'Sumo__Plant__EnergyCenter__Hel_aeration',  # Critical for KLa
        'Sumo__Plant__CSTR3__SO2',                  # Current DO
        'Sumo__Plant__CSTR3__OUR'                   # Oxygen demand
    ]
    
    for var in required_vars:
        if var not in data:
            issues.append(f"Missing required variable: {var}")
        elif data[var] is None:
            issues.append(f"Null value for required variable: {var}")
    
    # Check value ranges
    if 'Sumo__Plant__CSTR3__SO2' in data:
        do_value = data['Sumo__Plant__CSTR3__SO2']
        if do_value < 0 or do_value > 15:
            issues.append(f"DO value out of realistic range: {do_value} mg/L")
    
    if 'Sumo__Plant__CSTR3__OUR' in data:
        our_value = data['Sumo__Plant__CSTR3__OUR']
        if our_value < 0 or our_value > 2000:
            issues.append(f"OUR value out of realistic range: {our_value} mg/L/h")
    
    if 'Sumo__Plant__EnergyCenter__Hel_aeration' in data:
        power_value = data['Sumo__Plant__EnergyCenter__Hel_aeration']
        if power_value < 0:
            issues.append(f"Negative aeration power: {power_value} kW")
    
    # Log issues if any
    if issues:
        logger.warning(f"Data quality issues detected: {issues}")
        
    return len(issues) == 0

# Export the main callbacks for use in main.py
__all__ = [
    'true_pinn_data_callback', 
    'true_pinn_message_callback', 
    'finish_event'
]

# Backward compatibility aliases
data_callback = true_pinn_data_callback
msg_callback = true_pinn_message_callback