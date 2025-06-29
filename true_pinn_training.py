# src/training_engine.py - MAJOR RESTRUCTURING: True PINN Training with Physics Hierarchy

import threading
import queue
import tensorflow as tf
import numpy as np
from collections import deque
import logging
import time

logger = logging.getLogger(__name__)

class TruePINNTrainingEngine:
    """
    Training engine for true Physics-Informed Neural Network
    
    Key principles:
    1. Physics compliance is MANDATORY (not optional)
    2. Data fitting is SECONDARY (only after physics satisfied)
    3. Single optimizer respecting physics hierarchy
    4. No arbitrary gradient clipping - let physics guide learning
    """
    
    def __init__(self, model, config):
        self.model = model
        self.config = config
        
       # More aggressive learning for physics compliance
        base_lr = config.pinn.learning_rate
        self.optimizer = tf.keras.optimizers.Adam(
        learning_rate=base_lr * 2.0,  # Double the learning rate
        beta_1=0.9,
        beta_2=0.999
)
        
        # Physics-hierarchy loss weights (physics dominates!)
        self.physics_weight = 1000.0    # Physics violations heavily penalized
        self.data_weight = 1.0          # Data accuracy secondary
        
        # Streaming data management
        self._data_queue = queue.Queue(maxsize=10000)
        self._stop_event = threading.Event()
        
        # Replay buffer for experience replay (physics-aware)
        self.replay_buffer = deque(maxlen=config.pinn.buffer_size)
        
        # Training metrics with physics focus
        self.metrics = {
            'physics_loss': deque(maxlen=1000),
            'data_loss': deque(maxlen=1000),
            'physics_compliance_rate': deque(maxlen=1000),
            'kla_predictions': deque(maxlen=5000),
            'training_steps': 0,
            'physics_violations': 0,
            'data_points_processed': 0
        }
        
        # Physics compliance monitoring
        self.physics_tolerance = 1e-3
        self.physics_compliance_threshold = 0.8
        
        # Training limits
        self.max_training_steps = getattr(config.pinn, 'max_training_steps', None)
        
        # Performance tracking
        self.last_log_time = time.time()
        self.best_physics_loss = float('inf')
        
        # Start background training thread
        self.training_thread = threading.Thread(target=self._training_loop, daemon=True)
        self.training_thread.start()
        
        logger.info("True PINN Training Engine initialized with physics hierarchy")
    
    def enqueue_data(self, **data_point):
        """
        Add new data point for physics-informed training
        
        Expected data: t, do, our, do_sat, aeration_power, temperature, biomass, volume
        """
        try:
            self._data_queue.put(data_point, block=False)
            self.metrics['data_points_processed'] += 1
        except queue.Full:
            logger.warning("Training queue full, dropping data point")
    
    def _training_loop(self):
        """
        Main training loop with physics hierarchy enforcement
        """
        logger.info("Starting True PINN training loop with physics hierarchy")
        
        while not self._stop_event.is_set():
            # Check training limits
            if self.max_training_steps and self.metrics['training_steps'] >= self.max_training_steps:
                logger.info(f"Reached maximum training steps ({self.max_training_steps})")
                time.sleep(1)
                continue
            
            try:
                # Collect batch of new data
                batch_data = self._collect_training_batch()
                
                if len(batch_data) >= self.config.pinn.batch_size:
                    # Train on batch with physics hierarchy
                    loss_results = self._physics_hierarchy_training_step(batch_data)
                    
                    # Update metrics
                    self._update_training_metrics(loss_results)
                    
                    # Periodic logging
                    if self.metrics['training_steps'] % 10 == 0:
                        self._log_training_progress(loss_results)
                    
                    # Physics compliance check
                    self._monitor_physics_compliance(loss_results)
                    
                else:
                    time.sleep(0.01)  # Wait for more data
                    
            except Exception as e:
                logger.error(f"Error in training loop: {e}", exc_info=True)
                continue
    
    def _collect_training_batch(self):
        """
        Collect batch of training data with physics-aware sampling
        """
        batch_data = []
        deadline = time.time() + 0.1  # Collect for max 100ms
        
        # Collect new data points
        while len(batch_data) < self.config.pinn.batch_size and time.time() < deadline:
            try:
                data_point = self._data_queue.get(timeout=0.01)
                batch_data.append(data_point)
                self.replay_buffer.append(data_point)
            except queue.Empty:
                break
        
        # If insufficient new data, supplement with replay buffer
        if len(batch_data) < self.config.pinn.batch_size and len(self.replay_buffer) > 0:
            needed = self.config.pinn.batch_size - len(batch_data)
            if len(self.replay_buffer) >= needed:
                replay_indices = np.random.choice(len(self.replay_buffer), size=needed, replace=False)
                replay_data = [self.replay_buffer[i] for i in replay_indices]
                batch_data.extend(replay_data)
        
        return batch_data
    
    def _physics_hierarchy_training_step(self, batch_data):
        """
        Single training step with physics hierarchy
        
        Physics-hierarchy logic:
        1. If physics is violated → Focus ONLY on physics compliance
        2. If physics is satisfied → Optimize data accuracy
        """
        # Convert batch to tensors
        inputs, targets = self._prepare_batch_tensors(batch_data)
        
        with tf.GradientTape() as tape:
            # Forward pass through True PINN
            predictions = self.model(inputs, training=True)
            
            # Physics-hierarchy loss computation
            loss_components = self._compute_physics_hierarchy_loss(predictions, targets)
            total_loss = loss_components['total_loss']
        
        # SINGLE GRADIENT UPDATE (respecting physics hierarchy)
        gradients = tape.gradient(total_loss, self.model.trainable_variables)
        
        # Apply gradients (no arbitrary clipping - let physics guide learning)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        # Increment training steps
        self.metrics['training_steps'] += 1
        
        return loss_components
    
    def _compute_physics_hierarchy_loss(self, predictions, targets):
        """
        Physics-hierarchy loss function
        
        Priority 1: Physics compliance (MUST be satisfied)
        Priority 2: Data accuracy (optimized only after physics satisfied)
        """
        kla_pred = predictions['kla_predicted']
        do_calc = predictions['do_calculated'] 
        physics_residual = predictions['physics_residual']
        physics_satisfied = predictions['physics_satisfied']
        
        kla_actual = targets.get('kla_actual')
        do_measured = targets.get('do_measured')
        
        # PHYSICS COMPLIANCE LOSS (top priority)
        physics_loss = tf.reduce_mean(tf.square(physics_residual))
        
        # Calculate physics compliance rate
        compliance_rate = tf.reduce_mean(tf.cast(physics_satisfied, tf.float32))
        
        # DATA ACCURACY LOSSES (secondary priority)
        data_losses = {}
        
        if kla_actual is not None:
            data_losses['kla_loss'] = tf.reduce_mean(tf.square(kla_pred - kla_actual))
        else:
            data_losses['kla_loss'] = tf.constant(0.0, dtype=tf.float32)
        
        if do_measured is not None:
            data_losses['do_loss'] = tf.reduce_mean(tf.square(do_calc - do_measured))
        else:
            data_losses['do_loss'] = tf.constant(0.0, dtype=tf.float32)
        
        total_data_loss = data_losses['kla_loss'] + data_losses['do_loss']
        
        # PHYSICS-HIERARCHY DECISION
        # If physics compliance is below threshold, focus ENTIRELY on physics
        if compliance_rate < self.physics_compliance_threshold:
            total_loss = self.physics_weight * physics_loss
            loss_type = "physics_enforcement"
        else:
            # If physics is satisfied, focus on data accuracy
            total_loss = total_data_loss + 0.1 * physics_loss  # Keep small physics penalty
            loss_type = "data_optimization"
        
        return {
            'total_loss': total_loss,
            'physics_loss': physics_loss,
            'data_loss': total_data_loss,
            'kla_loss': data_losses['kla_loss'],
            'do_loss': data_losses['do_loss'],
            'compliance_rate': compliance_rate,
            'loss_type': loss_type,
            'physics_satisfied': compliance_rate >= self.physics_compliance_threshold
        }
    
    def _prepare_batch_tensors(self, batch_data):
        """
        Convert batch data to tensors for True PINN
        
        Extracts KLa-determining features (not influent parameters!)
        """
        # Extract features for KLa prediction
        batch_size = len(batch_data)
        
        # Initialize feature arrays
        features = np.zeros((batch_size, 8))  # 8 features for True PINN
        targets = {}
        
        for i, data_point in enumerate(batch_data):
            # KLa-determining features (reactor state, not influent!)
            features[i, 0] = data_point.get('t', 0.0)           # Time
            features[i, 1] = data_point.get('temperature', 20.0)  # Temperature
            features[i, 2] = data_point.get('flow_rate', 0.0)    # Flow rate
            features[i, 3] = data_point.get('aeration_power', 0.0)  # Aeration power (CRITICAL!)
            features[i, 4] = data_point.get('do', 0.0)           # Current DO
            features[i, 5] = data_point.get('our', 0.0)          # Oxygen uptake rate
            features[i, 6] = data_point.get('biomass', 0.0)      # Biomass concentration
            features[i, 7] = data_point.get('volume', 1000.0)    # Reactor volume
        
        # Convert to tensors
        inputs = tf.convert_to_tensor(features, dtype=tf.float32)
        
        # Prepare targets (if available)
        if 'kla_actual' in batch_data[0]:
            targets['kla_actual'] = tf.convert_to_tensor(
                [dp.get('kla_actual', 0.0) for dp in batch_data], dtype=tf.float32
            )
        
        if 'do_measured' in batch_data[0]:
            targets['do_measured'] = tf.convert_to_tensor(
                [dp.get('do_measured', 0.0) for dp in batch_data], dtype=tf.float32
            )
        
        return inputs, targets
    
    def _update_training_metrics(self, loss_results):
        """
        Update training metrics with physics focus
        """
        self.metrics['physics_loss'].append(float(loss_results['physics_loss']))
        self.metrics['data_loss'].append(float(loss_results['data_loss']))
        self.metrics['physics_compliance_rate'].append(float(loss_results['compliance_rate']))
        
        # Track physics violations
        if loss_results['compliance_rate'] < self.physics_compliance_threshold:
            self.metrics['physics_violations'] += 1
    
    def _monitor_physics_compliance(self, loss_results):
        """
        Monitor physics compliance and adjust training if needed
        """
        current_physics_loss = float(loss_results['physics_loss'])
        
        # Track best physics performance
        if current_physics_loss < self.best_physics_loss:
            self.best_physics_loss = current_physics_loss
        
        # Warn if physics compliance is consistently poor
        if len(self.metrics['physics_compliance_rate']) >= 100:
            recent_compliance = list(self.metrics['physics_compliance_rate'])[-100:]
            avg_compliance = np.mean(recent_compliance)
            
            if avg_compliance < 0.8:  # Less than 80% compliance
                logger.warning(f"Physics compliance low: {avg_compliance:.1%}. "
                             f"Current physics loss: {current_physics_loss:.2e}")
    
    def _log_training_progress(self, loss_results):
        """
        Log training progress with physics focus
        """
        if self.metrics['training_steps'] % 50 == 0:  # Log every 50 steps
            logger.info(f"=== True PINN Training Progress (Step {self.metrics['training_steps']}) ===")
            logger.info(f"Physics Loss: {loss_results['physics_loss']:.2e}")
            logger.info(f"Data Loss: {loss_results['data_loss']:.4f}")
            logger.info(f"Physics Compliance: {loss_results['compliance_rate']:.1%}")
            logger.info(f"Loss Type: {loss_results['loss_type']}")
            logger.info(f"Physics Violations: {self.metrics['physics_violations']}")
            
            # Log recent KLa predictions if available
            if len(self.metrics['kla_predictions']) > 0:
                recent_kla = list(self.metrics['kla_predictions'])[-10:]
                logger.info(f"Recent KLa range: {np.min(recent_kla):.2f} - {np.max(recent_kla):.2f} 1/h")
            
            logger.info("=" * 70)
    
    def get_physics_compliance_report(self):
        """
        Generate comprehensive physics compliance report
        """
        if len(self.metrics['physics_compliance_rate']) == 0:
            return {"error": "No training data available"}
        
        recent_compliance = list(self.metrics['physics_compliance_rate'])[-100:]
        recent_physics_loss = list(self.metrics['physics_loss'])[-100:]
        
        report = {
            'training_steps': self.metrics['training_steps'],
            'physics_compliance': {
                'current_rate': float(recent_compliance[-1]) if recent_compliance else 0.0,
                'average_rate': float(np.mean(recent_compliance)),
                'target_rate': self.physics_compliance_threshold,
                'is_satisfactory': np.mean(recent_compliance) >= self.physics_compliance_threshold
            },
            'physics_loss': {
                'current': float(recent_physics_loss[-1]) if recent_physics_loss else float('inf'),
                'best': float(self.best_physics_loss),
                'average': float(np.mean(recent_physics_loss))
            },
            'training_status': {
                'total_violations': self.metrics['physics_violations'],
                'violation_rate': self.metrics['physics_violations'] / max(1, self.metrics['training_steps']),
                'data_points_processed': self.metrics['data_points_processed']
            }
        }
        
        return report
    
    def stop(self):
        """
        Stop the training engine and generate final report
        """
        logger.info("Stopping True PINN Training Engine...")
        
        # Generate final physics compliance report
        final_report = self.get_physics_compliance_report()
        
        logger.info("=== FINAL PHYSICS COMPLIANCE REPORT ===")
        logger.info(f"Training Steps: {final_report['training_steps']}")
        logger.info(f"Physics Compliance Rate: {final_report['physics_compliance']['average_rate']:.1%}")
        logger.info(f"Final Physics Loss: {final_report['physics_loss']['current']:.2e}")
        logger.info(f"Best Physics Loss: {final_report['physics_loss']['best']:.2e}")
        logger.info(f"Total Physics Violations: {final_report['training_status']['total_violations']}")
        
        # Assess if true PINN was achieved
        if final_report['physics_compliance']['is_satisfactory']:
            logger.info("🟢 TRUE PHYSICS-INFORMED NEURAL NETWORK ACHIEVED!")
            logger.info("✅ Physics constraints satisfied consistently")
        else:
            logger.warning("🟡 PARTIALLY PHYSICS-INFORMED NETWORK")
            logger.warning("⚠️  Physics compliance below target threshold")
        
        logger.info("=" * 50)
        
        # Stop training thread
        self._stop_event.set()
        if self.training_thread.is_alive():
            self.training_thread.join(timeout=10)
        
        return final_report
    
    def get_metrics(self):
        """
        Return current training metrics for analysis
        """
        return {
            'physics_loss': list(self.metrics['physics_loss']),
            'data_loss': list(self.metrics['data_loss']),
            'physics_compliance_rate': list(self.metrics['physics_compliance_rate']),
            'kla_predictions': list(self.metrics['kla_predictions']),
            'training_steps': self.metrics['training_steps'],
            'physics_violations': self.metrics['physics_violations'],
            'data_points_processed': self.metrics['data_points_processed']
        }