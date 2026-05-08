import os, json, sys
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

backend_dir = os.path.dirname(os.path.abspath(__file__))
json_path = os.path.join(backend_dir, "monitoring", "reports", "multimarket_summary.json")

if not os.path.exists(json_path):
    print(f"Error: {json_path} not found.")
    sys.exit(1)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

stocks = [s for s in data["stocks"] if s["success"]]
if not stocks:
    print("No successful stocks found.")
    sys.exit(1)

# Sort stocks from Best to Lowest based on Accuracy
stocks.sort(key=lambda x: x["metrics"].get("accuracy", 0), reverse=True)

# Extract real metrics only (walk-forward validation)
tickers = [s["ticker"] for s in stocks]
accs = [s["metrics"].get("accuracy", 0) for s in stocks]
f1s = [s["metrics"].get("f1", 0) for s in stocks]
aucs = [s["metrics"].get("auc", 0.5) for s in stocks]
pvals = [s["metrics"].get("binom_pvalue", 1.0) for s in stocks]
fold_means = [s["metrics"].get("fold_mean_accuracy", 0) for s in stocks]
fold_stds = [s["metrics"].get("fold_std_accuracy", 0) for s in stocks]
train_samples = [s["metrics"].get("training_samples", 0) for s in stocks]
calib_samples = [s["metrics"].get("calibration_samples", 0) for s in stocks]
fold_counts = [s["metrics"].get("n_folds", 0) for s in stocks]

# Plotting
plt.rcParams.update({
    "figure.facecolor": "#0b101e",
    "axes.facecolor": "#0b101e",
    "axes.edgecolor": "#475569",
    "text.color": "#f8fafc",
    "axes.labelcolor": "#94a3b8",
    "xtick.color": "#94a3b8",
    "ytick.color": "#94a3b8",
    "grid.color": "#1e293b",
    "font.family": "sans-serif",
})

fig = plt.figure(figsize=(18, 14))
fig.suptitle("AI Stock Forecasting Performance Monitor",
             fontsize=20, fontweight="bold", color="#ffffff", y=0.96)
fig.text(0.5, 0.93, "Metrics from walk-forward validation only (no synthetic backtest estimates)",
         ha="center", fontsize=11, color="#94a3b8")

gs = gridspec.GridSpec(3, 2, height_ratios=[1, 1, 0.6], hspace=0.4, wspace=0.25, top=0.88, bottom=0.05, left=0.05, right=0.95)

# Color maps based on performance
def get_color(val, threshold1=60, threshold2=55):
    if val >= threshold1: return "#22c55e" # Green
    if val >= threshold2: return "#0ea5e9" # Blue
    if val >= 50: return "#f59e0b" # Orange
    return "#ef4444" # Red

short_tickers = [t.replace(".NS", "") for t in tickers]

# 1. Accuracy
ax1 = fig.add_subplot(gs[0, 0])
colors1 = [get_color(a, 60, 55) for a in accs]
bars1 = ax1.bar(short_tickers, accs, color=colors1, width=0.7)
ax1.axhline(50, color="#475569", ls="--", lw=1)
ax1.set_ylim(0, 80)
ax1.set_title("Walk-Forward Directional Accuracy (%)", fontsize=12, fontweight="bold", color="#ffffff")
for bar, val in zip(bars1, accs):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha="center", fontsize=9)

# 2. F1 Score
ax2 = fig.add_subplot(gs[0, 1])
colors2 = [get_color(f, 75, 70) for f in f1s]
bars2 = ax2.bar(short_tickers, f1s, color=colors2, width=0.7)
ax2.set_ylim(0, 90)
ax2.set_title("F1 Score (%)", fontsize=12, fontweight="bold", color="#ffffff")
for bar, val in zip(bars2, f1s):
    ax2.text(bar.get_x() + bar.get_width()/2, val + 1, f"{val:.1f}%", ha="center", fontsize=9)

# 3. AUC-ROC
ax3 = fig.add_subplot(gs[1, 0])
colors3 = ["#22c55e" if a >= 0.55 else "#0ea5e9" if a >= 0.50 else "#ef4444" for a in aucs]
bars3 = ax3.bar(short_tickers, aucs, color=colors3, width=0.7)
ax3.axhline(0.5, color="#475569", ls="--", lw=1)
ax3.set_ylim(0, 1.0)
ax3.set_title("AUC-ROC", fontsize=12, fontweight="bold", color="#ffffff")
for bar, val in zip(bars3, aucs):
    ax3.text(bar.get_x() + bar.get_width()/2, val + 0.02, f"{val:.3f}", ha="center", fontsize=9)

# 4. Accuracy vs AUC scatter
ax4 = fig.add_subplot(gs[1, 1])
for i, t in enumerate(short_tickers):
    ax4.scatter(aucs[i], accs[i], s=180, color=colors1[i], edgecolor="#ffffff", lw=1.2, zorder=3)
    ax4.text(aucs[i] + 0.005, accs[i] + 0.6, t, fontsize=9, color="#e2e8f0")
ax4.axvline(0.5, color="#475569", ls="--", lw=1)
ax4.axhline(50, color="#475569", ls="--", lw=1)
ax4.set_xlim(0.4, 0.75)
ax4.set_ylim(40, 80)
ax4.grid(True, zorder=0)
ax4.set_xlabel("AUC-ROC")
ax4.set_ylabel("Accuracy (%)")
ax4.set_title("Accuracy vs AUC-ROC", fontsize=12, fontweight="bold", color="#ffffff")

# 5. Table
ax5 = fig.add_subplot(gs[2, :])
ax5.axis("off")

columns = ["Ticker", "Acc %", "F1 %", "AUC", "p-value", "Fold Mean %", "Fold Std %", "Train N", "Calib N", "Folds"]
cell_text = []
for i in range(len(tickers)):
    row = [
        short_tickers[i],
        f"{accs[i]:.2f}",
        f"{f1s[i]:.2f}",
        f"{aucs[i]:.3f}",
        f"{pvals[i]:.4f}",
        f"{fold_means[i]:.2f}",
        f"{fold_stds[i]:.2f}",
        f"{train_samples[i]}",
        f"{calib_samples[i]}",
        f"{fold_counts[i]}"
    ]
    cell_text.append(row)

table = ax5.table(cellText=cell_text, colLabels=columns, loc="center", cellLoc="center")
table.auto_set_font_size(False)
table.set_fontsize(9)
table.scale(1, 1.8)

for (row, col), cell in table.get_celld().items():
    if row == 0:
        cell.set_text_props(weight="bold", color="#f8fafc")
        cell.set_facecolor("#1e293b")
    else:
        cell.set_facecolor("#0f172a" if row % 2 == 0 else "#162032")
        cell.set_text_props(color="#e2e8f0")
    cell.set_edgecolor("#334155")

out_path = os.path.join(backend_dir, "monitoring", "reports", "ultimate_v52_dashboard.png")
fig.savefig(out_path, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
print(f"Chart generated successfully: {out_path}")

# Optional: signal/sentiment overview from evaluation report
signal_json = os.path.join(backend_dir, "monitoring", "reports", "top10_signal_evaluation.json")
if os.path.exists(signal_json):
    try:
        with open(signal_json, "r", encoding="utf-8") as f:
            signal_data = json.load(f)

        rows = [r for r in signal_data.get("results", []) if r.get("status") == "ok"]
        if rows:
            fig2 = plt.figure(figsize=(18, 10))
            fig2.suptitle("Signal + Sentiment Overview", fontsize=18, fontweight="bold", color="#ffffff", y=0.96)
            fig2.text(0.5, 0.92, "Live signal outputs from current model artifacts", ha="center", fontsize=11, color="#94a3b8")

            gs2 = gridspec.GridSpec(2, 2, hspace=0.35, wspace=0.25, top=0.88, bottom=0.08, left=0.06, right=0.95)
            tickers2 = [r.get("ticker", "") for r in rows]
            short2 = [t.replace(".NS", "") for t in tickers2]

            # Signal counts
            ax_s1 = fig2.add_subplot(gs2[0, 0])
            signals = [r.get("signal", "HOLD") for r in rows]
            unique_signals = ["STRONG BUY", "BUY", "HOLD", "SELL", "STRONG SELL"]
            counts = [signals.count(s) for s in unique_signals]
            ax_s1.bar(unique_signals, counts, color=["#22c55e", "#4ade80", "#f59e0b", "#fb7185", "#ef4444"], width=0.7)
            ax_s1.set_title("Signal Distribution", fontsize=12, fontweight="bold", color="#ffffff")
            ax_s1.set_ylabel("Count")
            ax_s1.grid(True, axis="y")
            ax_s1.tick_params(axis="x", rotation=20)

            # Confidence
            ax_s2 = fig2.add_subplot(gs2[0, 1])
            confs = [float(r.get("confidence", 0) or 0) * 100 for r in rows]
            ax_s2.bar(short2, confs, color="#38bdf8", width=0.7)
            ax_s2.set_title("Signal Confidence (%)", fontsize=12, fontweight="bold", color="#ffffff")
            ax_s2.set_ylim(0, 100)
            ax_s2.grid(True, axis="y")
            ax_s2.tick_params(axis="x", rotation=35)

            # Expected move
            ax_s3 = fig2.add_subplot(gs2[1, 0])
            moves = [float(r.get("expected_move_pct", 0) or 0) for r in rows]
            move_colors = ["#22c55e" if m > 0 else "#ef4444" if m < 0 else "#f59e0b" for m in moves]
            ax_s3.bar(short2, moves, color=move_colors, width=0.7)
            ax_s3.axhline(0, color="#475569", ls="--", lw=1)
            ax_s3.set_title("Expected Move (%)", fontsize=12, fontweight="bold", color="#ffffff")
            ax_s3.grid(True, axis="y")
            ax_s3.tick_params(axis="x", rotation=35)

            # Sentiment score
            ax_s4 = fig2.add_subplot(gs2[1, 1])
            sentiment = [float((r.get("sentiment") or {}).get("score", 0) or 0) for r in rows]
            sent_colors = ["#22c55e" if s > 0 else "#ef4444" if s < 0 else "#f59e0b" for s in sentiment]
            ax_s4.bar(short2, sentiment, color=sent_colors, width=0.7)
            ax_s4.axhline(0, color="#475569", ls="--", lw=1)
            ax_s4.set_title("Sentiment Score", fontsize=12, fontweight="bold", color="#ffffff")
            ax_s4.grid(True, axis="y")
            ax_s4.tick_params(axis="x", rotation=35)

            signal_out = os.path.join(backend_dir, "monitoring", "reports", "top10_signal_overview.png")
            fig2.savefig(signal_out, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
            plt.close(fig2)
            print(f"Signal overview chart generated: {signal_out}")
    except Exception as exc:
        print(f"Signal overview skipped: {exc}")
