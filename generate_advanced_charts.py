import os
import json
import numpy as np
import matplotlib.pyplot as plt

os.makedirs('assets', exist_ok=True)
tickers = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'AMZN']
accuracies = []
sharpes = []
mae = []
rmse = []

for t in tickers:
    path = f'backend/mlops_v2/models/{t}/metadata.json'
    if os.path.exists(path):
        with open(path) as f:
            metrics = json.load(f)['metrics']
            accuracies.append(metrics.get('xgb_accuracy', 0.5) * 100)
            sharpes.append(metrics.get('sharpe_ratio', 0.0))
            mae.append(metrics.get('price_mae', 0.0))
            rmse.append(metrics.get('price_rmse', 0.0))
    else:
        accuracies.append(50.0)
        sharpes.append(0.0)
        mae.append(0.0)
        rmse.append(0.0)

plt.style.use('dark_background')

# Chart 1: Sharpe vs Accuracy
fig, ax = plt.subplots(figsize=(9, 6))
scatter = ax.scatter(accuracies, sharpes, s=400, c='#00E676', alpha=0.8, edgecolors='white', linewidth=2)

for i, t in enumerate(tickers):
    ax.annotate(t, (accuracies[i], sharpes[i]), xytext=(10, 5), textcoords='offset points', color='white', fontweight='bold', fontsize=11)

ax.set_title('Risk-Adjusted Returns: Sharpe Ratio vs XGBoost Accuracy', fontsize=15, fontweight='bold', color='white', pad=20)
ax.set_xlabel('Directional Accuracy (%)', color='#00E5FF', fontweight='bold', fontsize=12)
ax.set_ylabel('Simulated Sharpe Ratio', color='#00E5FF', fontweight='bold', fontsize=12)
ax.grid(True, linestyle='--', alpha=0.2)

# Trendline
if len(set(accuracies)) > 1:
    z = np.polyfit(accuracies, sharpes, 1)
    p = np.poly1d(z)
    x_trend = np.linspace(min(accuracies)-1, max(accuracies)+1, 100)
    ax.plot(x_trend, p(x_trend), "#FF6F00", linestyle='--', alpha=0.6, label='Yield Trendline')
    ax.legend(loc='upper left')

fig.tight_layout()
plt.savefig('assets/sharpe_accuracy_scatter.png', dpi=300, bbox_inches='tight', facecolor='#111827')
print("Saved sharpe_accuracy_scatter.png")


# Chart 2: LSTM Error Margins
x = np.arange(len(tickers))
width = 0.35
fig, ax = plt.subplots(figsize=(10, 6))

bars1 = ax.bar(x - width/2, mae, width, label='MAE (Mean Absolute Error)', color='#0194E2', alpha=0.9)
bars2 = ax.bar(x + width/2, rmse, width, label='RMSE (Root Mean Square Error)', color='#F46800', alpha=0.9)

ax.set_title('LSTM Magnitude Engine: Price Forecasting Error Margins', fontsize=15, fontweight='bold', color='white', pad=20)
ax.set_ylabel('Error Margin (Normalized)', fontweight='bold', fontsize=12)
ax.set_xticks(x)
ax.set_xticklabels(tickers, fontweight='bold', fontsize=11)
ax.legend(loc='upper left')
ax.grid(True, linestyle='--', alpha=0.2)

fig.tight_layout()
plt.savefig('assets/lstm_error_margins.png', dpi=300, bbox_inches='tight', facecolor='#111827')
print("Saved lstm_error_margins.png")
