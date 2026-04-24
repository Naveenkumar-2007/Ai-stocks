import sys
import os
sys.path.append(os.path.join(os.getcwd(), 'backend'))

# Suppress TF warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from mlops_v2.training import TrainerV2
import matplotlib.pyplot as plt
import json
import numpy as np

tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
trainer = TrainerV2()
metrics_data = {}

print("Starting training for Top 5 stocks...")
for t in tickers:
    print(f"Training {t}...")
    try:
        res = trainer.train_if_needed(t, force=True)
        if res.metrics:
            metrics_data[t] = res.metrics
        else:
            # Try to read from existing if force failed
            path = f'backend/mlops_v2/models/{t}/metadata.json'
            if os.path.exists(path):
                with open(path) as f:
                    metrics_data[t] = json.load(f).get('metrics', {})
    except Exception as e:
        print(f"Failed to train {t}: {e}")

print("Training complete. Generating chart...")
if not metrics_data:
    print("No metrics gathered.")
    sys.exit(1)

labels = list(metrics_data.keys())
accuracies = [metrics_data[t].get('xgb_accuracy', 0.5) * 100 for t in labels]
pnls = [metrics_data[t].get('simulated_pnl', 0.0) for t in labels]

x = np.arange(len(labels))
width = 0.35

plt.style.use('dark_background')
fig, ax1 = plt.subplots(figsize=(10, 6))

color = '#00E5FF' # Neon Cyan
ax1.set_ylabel('XGBoost Directional Accuracy (%)', color=color, fontweight='bold', fontsize=12)
bars1 = ax1.bar(x - width/2, accuracies, width, color=color, alpha=0.8, label='Accuracy')
ax1.tick_params(axis='y', labelcolor=color)
ax1.set_ylim([40, max(accuracies) + 10]) 

ax2 = ax1.twinx()
color = '#00E676' # Neon Green
ax2.set_ylabel('Simulated PnL (last 20 trades, %)', color=color, fontweight='bold', fontsize=12)
bars2 = ax2.bar(x + width/2, pnls, width, color=color, alpha=0.8, label='Simulated PnL')
ax2.tick_params(axis='y', labelcolor=color)

# Add values on top of bars
for bar in bars1:
    yval = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, yval + 0.5, f"{yval:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

for bar in bars2:
    yval = bar.get_height()
    offset = 0.5 if yval >= 0 else -2.0
    ax2.text(bar.get_x() + bar.get_width()/2, yval + offset, f"{yval:.1f}%", ha='center', va='bottom', color='white', fontweight='bold')

plt.title('Top 5 Tech Stocks: AI Engine Performance (Validation Backtest)', fontsize=16, fontweight='bold', color='white', pad=20)
ax1.set_xticks(x)
ax1.set_xticklabels(labels, fontsize=12, fontweight='bold')

# Grid lines
ax1.grid(True, linestyle='--', alpha=0.2)

fig.tight_layout()

os.makedirs('assets', exist_ok=True)
plt.savefig('assets/model_performance_comparison.png', dpi=300, bbox_inches='tight', facecolor='#111827')
print("Chart saved successfully to assets/model_performance_comparison.png!")
