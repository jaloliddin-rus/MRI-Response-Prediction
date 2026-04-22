import sys
import matplotlib.pyplot as plt
from math import pi

try:
    sys.stdout.reconfigure(encoding='utf-8')
except (AttributeError, OSError):
    pass


models = ['AutoEncoder', 'BasicUNet', 'DenseNet169', 'EfficientNetB4', 'ResNet']

data = {
    'AutoEncoder': {
        'MSE': 0.0063,
        'MAE': 0.0513,
        'R²': 0.87,
        'EVS': 0.87,
        'RMSE': 0.0794,
        'Samples≥80%': 56.67,
    },
    'BasicUNet': {
        'MSE': 0.0052,
        'MAE': 0.0447,
        'R²': 0.89,
        'EVS': 0.89,
        'RMSE': 0.0721,
        'Samples≥80%': 75.00,
    },
    'DenseNet169': {
        'MSE': 0.0039,
        'MAE': 0.0397,
        'R²': 0.92,
        'EVS': 0.92,
        'RMSE': 0.0622,
        'Samples≥80%': 88.96,
    },
    'EfficientNetB4': {
        'MSE': 0.0057,
        'MAE': 0.0520,
        'R²': 0.88,
        'EVS': 0.89,
        'RMSE': 0.0756,
        'Samples≥80%': 72.08,
    },
    'ResNet': {
        'MSE': 0.0060,
        'MAE': 0.0424,
        'R²': 0.88,
        'EVS': 0.88,
        'RMSE': 0.0777,
        'Samples≥80%': 71.67,
    },
}


def normalize_metrics(data):
    normalized = {}
    all_mse = [d['MSE'] for d in data.values()]
    all_mae = [d['MAE'] for d in data.values()]
    all_rmse = [d['RMSE'] for d in data.values()]

    for model, metrics in data.items():
        normalized[model] = {
            'MSE': 100 * (max(all_mse) - metrics['MSE']) / (max(all_mse) - min(all_mse)),
            'MAE': 100 * (max(all_mae) - metrics['MAE']) / (max(all_mae) - min(all_mae)),
            'RMSE': 100 * (max(all_rmse) - metrics['RMSE']) / (max(all_rmse) - min(all_rmse)),
            'R²': metrics['R²'] * 100,
            'EVS': metrics['EVS'] * 100,
            'Samples≥80%': metrics['Samples≥80%'],
        }
    return normalized


def main():
    normalized_data = normalize_metrics(data)

    categories = ['MSE\n(inverted)', 'MAE\n(inverted)', 'R²', 'EVS', 'RMSE\n(inverted)', 'Samples≥80%']
    N = len(categories)

    fig, ax = plt.subplots(figsize=(12, 10), subplot_kw=dict(projection='polar'))

    angles = [n / float(N) * 2 * pi for n in range(N)]
    angles += angles[:1]

    colors = ['#90EE90', '#FFA07A', '#FF6B6B', '#DDA0DD', '#87CEEB']
    model_colors = dict(zip(models, colors))

    for model in models:
        values = [
            normalized_data[model]['MSE'],
            normalized_data[model]['MAE'],
            normalized_data[model]['R²'],
            normalized_data[model]['EVS'],
            normalized_data[model]['RMSE'],
            normalized_data[model]['Samples≥80%'],
        ]
        values += values[:1]

        ax.plot(angles, values, 'o-', linewidth=2, label=model, color=model_colors[model])
        ax.fill(angles, values, alpha=0.15, color=model_colors[model])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 100)
    ax.set_yticks([20, 40, 60, 80, 100])
    ax.set_yticklabels(['20', '40', '60', '80', '100'], size=8, color='gray')
    ax.grid(True, linestyle='--', alpha=0.3)

    plt.title(
        'Neural Network Performance Evaluation\n(All metrics normalized to 0-100 scale)',
        size=16, weight='bold', pad=20,
    )
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)

    plt.tight_layout()
    plt.savefig('neural_network_radar_chart.png', dpi=300, bbox_inches='tight')
    plt.show()

    print("\nNormalized Values (0-100 scale):")
    print("-" * 80)
    for model in models:
        print(f"\n{model}:")
        for metric, value in normalized_data[model].items():
            print(f"  {metric}: {value:.2f}")


if __name__ == "__main__":
    main()
