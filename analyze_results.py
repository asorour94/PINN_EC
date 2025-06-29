# src/analyze_results.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json
import os
import sys

def analyze_training_run(results_path, metrics_path):
    """Analyze and visualize training results."""
    
    # Load data
    df = pd.read_csv(results_path)
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('PINN Training Analysis', fontsize=16)
    
    # 1. Loss curves
    ax = axes[0, 0]
    steps = range(len(metrics['data_loss']))
    ax.semilogy(steps, metrics['data_loss'], label='Data Loss', alpha=0.7)
    ax.semilogy(steps, metrics['physics_loss'], label='Physics Loss', alpha=0.7)
    ax.set_xlabel('Training Steps')
    ax.set_ylabel('Loss (log scale)')
    ax.set_title('Training Losses')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 2. KLa evolution
    ax = axes[0, 1]
    if metrics['kla_history']:
        ax.plot(metrics['kla_history'], 'b-', linewidth=2)
        ax.axhline(y=np.mean(metrics['kla_history'][-100:]), 
                   color='r', linestyle='--', 
                   label=f'Final avg: {np.mean(metrics["kla_history"][-100:]):.2f}')
        ax.set_xlabel('Training Steps')
        ax.set_ylabel('KLa (1/h)')
        ax.set_title('KLa Parameter Evolution')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # 3. DO measurements vs time
    ax = axes[0, 2]
    time_hours = df['Sumo__Time'] / (1000 * 3600)  # Convert ms to hours
    ax.plot(time_hours, df['Sumo__Plant__CSTR3__SO2'], 'g-', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('DO (mg/L)')
    ax.set_title('Dissolved Oxygen in CSTR3')
    ax.grid(True, alpha=0.3)
    
    # 4. OUR vs time
    ax = axes[1, 0]
    ax.plot(time_hours, df['Sumo__Plant__CSTR3__OUR'], 'r-', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('OUR (mg/L/h)')
    ax.set_title('Oxygen Uptake Rate')
    ax.grid(True, alpha=0.3)
    
    # 5. Power consumption
    ax = axes[1, 1]
    ax.plot(time_hours, df['Sumo__Plant__EnergyCenter__Hel_aeration'], 'b-', alpha=0.7)
    ax.set_xlabel('Time (hours)')
    ax.set_ylabel('Power (kW)')
    ax.set_title('Aeration Power Consumption')
    ax.grid(True, alpha=0.3)
    
    # 6. Statistics summary
    ax = axes[1, 2]
    ax.axis('off')
    
    # Calculate statistics
    stats_text = f"""Training Summary:
    
Total Steps: {metrics['training_steps']}
Data Points: {metrics['data_points_processed']}

Final Losses:
  Data: {np.mean(metrics['data_loss'][-100:]):.4f}
  Physics: {np.mean(metrics['physics_loss'][-100:]):.2e}

KLa Statistics:
  Final: {metrics['kla_history'][-1] if metrics['kla_history'] else 'N/A':.3f} 1/h
  Mean: {np.mean(metrics['kla_history']) if metrics['kla_history'] else 'N/A':.3f} 1/h
  Std: {np.std(metrics['kla_history']) if metrics['kla_history'] else 'N/A':.3f} 1/h

DO Statistics:
  Mean: {df['Sumo__Plant__CSTR3__SO2'].mean():.2f} mg/L
  Std: {df['Sumo__Plant__CSTR3__SO2'].std():.2f} mg/L
  Min: {df['Sumo__Plant__CSTR3__SO2'].min():.2f} mg/L
  Max: {df['Sumo__Plant__CSTR3__SO2'].max():.2f} mg/L
"""
    
    ax.text(0.1, 0.9, stats_text, transform=ax.transAxes, 
            fontsize=10, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    
    # Save figure
    output_dir = os.path.dirname(results_path)
    output_path = os.path.join(output_dir, 'training_analysis.png')
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Analysis saved to: {output_path}")
    
    plt.show()

if __name__ == "__main__":
    # Find the most recent results
    data_dir = "data"
    model_dir = "models"
    
    # Get most recent CSV file
    csv_files = [f for f in os.listdir(data_dir) if f.startswith('results_4weeks') and f.endswith('.csv')]
    if not csv_files:
        print("No results files found!")
        sys.exit(1)
    
    latest_csv = sorted(csv_files)[-1]
    results_path = os.path.join(data_dir, latest_csv)
    
    # Extract timestamp from filename
    timestamp = latest_csv.replace('results_4weeks_', '').replace('.csv', '')
    
    # Find corresponding metrics file
    model_folder = f"pinn_model_4weeks_{timestamp}"
    metrics_path = os.path.join(model_dir, model_folder, "metrics.json")
    
    if not os.path.exists(metrics_path):
        print(f"Metrics file not found: {metrics_path}")
        sys.exit(1)
    
    print(f"Analyzing results from: {results_path}")
    print(f"Using metrics from: {metrics_path}")
    
    analyze_training_run(results_path, metrics_path)