import matplotlib.pyplot as plt
import numpy as np
from math import pi

# Data from the table - converting metrics to percentage scores (0-100)
# Higher is better for all metrics after normalization

models = ['AutoEncoder', 'BasicUNet', 'DenseNet169', 'EfficientNetB4', 'ResNet']

# Normalize metrics to 0-100 scale where higher is better
# For MSE, MAE, RMSE: lower is better, so we invert them
# For R², EVS, samples ≥80%: higher is better

data = {
    'AutoEncoder': {
        'MSE': 0.0063,
        'MAE': 0.0513,
        'R²': 0.87,
        'EVS': 0.87,
        'RMSE': 0.0794,
        'Samples≥80%': 56.67
    },
    'BasicUNet': {
        'MSE': 0.0052,
        'MAE': 0.0447,
        'R²': 0.89,
        'EVS': 0.89,
        'RMSE': 0.0721,
        'Samples≥80%': 75.00
    },
    'DenseNet169': {
        'MSE': 0.0039,
        'MAE': 0.0397,
        'R²': 0.92,
        'EVS': 0.92,
        'RMSE': 0.0622,
        'Samples≥80%': 88.96
    },
    'EfficientNetB4': {
        'MSE': 0.0057,
        'MAE': 0.0520,
        'R²': 0.88,
        'EVS': 0.89,
        'RMSE': 0.0756,
        'Samples≥80%': 72.08
    },
    'ResNet': {
        'MSE': 0.0060,
        'MAE': 0.0424,
        'R²': 0.88,
        'EVS': 0.88,
        'RMSE': 0.0777,
        'Samples≥80%': 71.67
    }
}

# Normalize to 0-100 scale
def normalize_metrics(data):
    normalized = {}
    
    # Get min/max for each metric across all models
    all_mse = [d['MSE'] for d in data.values()]
    all_mae = [d['MAE'] for d in data.values()]
    all_rmse = [d['RMSE'] for d in data.values()]
    all_r2 = [d['R²'] for d in data.values()]
    all_evs = [d['EVS'] for d in data.values()]
    all_samples = [d['Samples≥80%'] for d in data.values()]
    
    for model, metrics in data.items():
        normalized[model] = {}
        
        # For error metrics (MSE, MAE, RMSE): invert so lower error = higher score
        # Score = 100 * (max - value) / (max - min)
        normalized[model]['MSE'] = 100 * (max(all_mse) - metrics['MSE']) / (max(all_mse) - min(all_mse))
        normalized[model]['MAE'] = 100 * (max(all_mae) - metrics['MAE']) / (max(all_mae) - min(all_mae))
        normalized[model]['RMSE'] = 100 * (max(all_rmse) - metrics['RMSE']) / (max(all_rmse) - min(all_rmse))
        
        # For R² and EVS: scale to 0-100 (they're already 0-1)
        normalized[model]['R²'] = metrics['R²'] * 100
        normalized[model]['EVS'] = metrics['EVS'] * 100
        
        # Samples≥80% is already a percentage
        normalized[model]['Samples≥80%'] = metrics['Samples≥80%']
    
    return normalized

normalized_data = normalize_metrics(data)

# Categories for the radar chart
categories = ['MSE\n(inverted)', 'MAE\n(inverted)', 'R²', 'EVS', 'RMSE\n(inverted)', 'Samples≥80%']
N = len(categories)

# Create the radar chart
fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

# Compute angle for each axis
angles = [n / float(N) * 2 * pi for n in range(N)]
angles += angles[:1]

# Colors for each model
colors = ['#90EE90', '#FFA07A', '#FF6B6B', '#DDA0DD', '#87CEEB']
model_colors = dict(zip(models, colors))

# Plot data for each model
for model in models:
    values = [
        normalized_data[model]['MSE'],
        normalized_data[model]['MAE'],
        normalized_data[model]['R²'],
        normalized_data[model]['EVS'],
        normalized_data[model]['RMSE'],
        normalized_data[model]['Samples≥80%']
    ]
    values += values[:1]
    
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=model_colors[model])
    ax.fill(angles, values, alpha=0.15, color=model_colors[model])

# Customize the plot
ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories, size=10)
ax.set_ylim(0, 100)
ax.set_yticks([20, 40, 60, 80, 100])
ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8, color='gray')
ax.grid(True, linestyle='--', alpha=0.3)

# Add title and legend
plt.title('Neural Network Performance Evaluation\n(All metrics normalized to 0-100 scale)', 
          size=16, weight='bold', pad=20)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

plt.tight_layout()
plt.savefig('neural_network_radar_chart.png', dpi=300, bbox_inches='tight')
plt.show()

# Print the normalized values for reference
print("\nNormalized Values (0-100 scale):")
print("-" * 80)
for model in models:
    print(f"\n{model}:")
    for metric, value in normalized_data[model].items():
        print(f"  {metric}: {value:.2f}")