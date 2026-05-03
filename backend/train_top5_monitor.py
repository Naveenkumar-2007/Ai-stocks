"""
=============================================================================
 COMPREHENSIVE MULTI-MARKET TRAINING + MONITORING v5.2
=============================================================================
 Trains on 10 diverse stocks across US, India, UK, and other markets,
 then generates detailed per-stock and combined monitoring charts.
=============================================================================
"""

import sys, os, json, warnings, time
from datetime import datetime

warnings.filterwarnings("ignore")

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

import numpy as np

try:
    from dotenv import load_dotenv
    load_dotenv(os.path.join(backend_dir, ".env"))
except ImportError:
    pass

# ===========================================================================
#  STOCKS: 5 US + 3 India + 2 UK/Global
# ===========================================================================
STOCKS = [
    # --- US Mega-Caps ---
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA",
    # --- India (NSE) ---
    "RELIANCE.NS", "TCS.NS", "INFY.NS",
    # --- UK / Global ---
    "TSLA", "META",
]

CHART_DIR = os.path.join(backend_dir, "monitoring", "reports")
os.makedirs(CHART_DIR, exist_ok=True)

# ===========================================================================
#  TRAIN ALL STOCKS
# ===========================================================================
from unified_engine.training import UnifiedTrainer, TrainResult

print("\n" + "=" * 80)
print("  MULTI-MARKET TRAINING — Unified Engine v5.2")
print("  " + "=" * 60)
print(f"  Stocks: {', '.join(STOCKS)}")
print("=" * 80)

results: dict[str, TrainResult] = {}
timings = {}
start_wall = time.time()

for ticker in STOCKS:
    t0 = time.time()
    try:
        result = UnifiedTrainer.train(ticker)
    except Exception as e:
        print(f"\n  ❌ {ticker}: EXCEPTION — {e}")
        result = TrainResult(
            ticker=ticker, success=False, metrics={}, fold_results=[],
            feature_importance={}, selected_features=[], model_version="",
            artifact_path="", reason=str(e),
        )
    elapsed = time.time() - t0
    results[ticker] = result
    timings[ticker] = elapsed
    status = "✅ SUCCESS" if result.success else f"❌ FAILED ({result.reason})"
    print(f"\n  [{ticker}] {status}  ({elapsed:.1f}s)")

total_elapsed = time.time() - start_wall
print(f"\n{'='*80}")
print(f"  ALL TRAINING COMPLETE — Total: {total_elapsed:.0f}s")
print(f"{'='*80}")

# ===========================================================================
#  GENERATE CHARTS
# ===========================================================================
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    HAS_MPL = True
except ImportError:
    HAS_MPL = False

if HAS_MPL:
    plt.rcParams.update({
        "figure.facecolor": "#0f172a",
        "axes.facecolor": "#1e293b",
        "axes.edgecolor": "#475569",
        "text.color": "#e2e8f0",
        "axes.labelcolor": "#e2e8f0",
        "xtick.color": "#94a3b8",
        "ytick.color": "#94a3b8",
        "grid.color": "#334155",
        "grid.alpha": 0.4,
        "font.family": "sans-serif",
        "font.size": 10,
    })

    # -- Per-stock monitoring charts --
    for ticker in STOCKS:
        r = results[ticker]
        if not r.success:
            continue

        fig = plt.figure(figsize=(16, 10))
        fig.suptitle(f"Model Monitoring — {ticker}  ({r.model_version})",
                     fontsize=16, fontweight="bold", color="#38bdf8", y=0.98)
        gs = gridspec.GridSpec(2, 2, hspace=0.38, wspace=0.32,
                               left=0.07, right=0.96, top=0.92, bottom=0.08)

        # Panel 1: Fold accuracy bars
        ax1 = fig.add_subplot(gs[0, 0])
        folds = r.fold_results
        fold_nums = [f["fold"] + 1 for f in folds]
        fold_accs = [f["accuracy"] * 100 for f in folds]
        colors1 = ["#22c55e" if a >= 50 else "#ef4444" for a in fold_accs]
        bars = ax1.bar(fold_nums, fold_accs, color=colors1, edgecolor="#475569",
                       width=0.7, zorder=3)
        ax1.axhline(50, color="#f59e0b", ls="--", lw=1.5, label="Random baseline (50%)")
        mean_acc = np.mean(fold_accs)
        ax1.axhline(mean_acc, color="#38bdf8", ls="-", lw=1.5, alpha=0.8, label=f"Mean ({mean_acc:.1f}%)")
        ax1.set_xlabel("Fold #")
        ax1.set_ylabel("Accuracy (%)")
        ax1.set_title("Walk-Forward Fold Accuracy", fontsize=12, color="#f8fafc")
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=8, loc="lower right", facecolor="#1e293b", edgecolor="#475569")
        ax1.grid(True, axis="y")

        # Panel 2: Feature importance
        ax2 = fig.add_subplot(gs[0, 1])
        top_features = list(r.feature_importance.items())[:10]
        feat_names = [f[0] for f in reversed(top_features)]
        feat_vals = [f[1] * 100 for f in reversed(top_features)]
        feat_colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(feat_names)))
        ax2.barh(feat_names, feat_vals, color=feat_colors, edgecolor="#475569", height=0.6)
        ax2.set_xlabel("Importance (%)")
        ax2.set_title("Top-10 Feature Importance", fontsize=12, color="#f8fafc")
        ax2.grid(True, axis="x")
        for i, v in enumerate(feat_vals):
            ax2.text(v + 0.15, i, f"{v:.1f}%", va="center", fontsize=8, color="#e2e8f0")

        # Panel 3: Scorecard
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.axis("off")
        m = r.metrics
        metrics_text = [
            ("Accuracy",  f"{m.get('accuracy',0):.1f}%"),
            ("AUC-ROC",   f"{m.get('auc',0):.3f}"),
            ("Precision",  f"{m.get('precision',0):.1f}%"),
            ("Recall",     f"{m.get('recall',0):.1f}%"),
            ("F1 Score",   f"{m.get('f1',0):.1f}%"),
            ("p-value",    f"{m.get('binom_pvalue',1):.4f}"),
            ("Samples",    f"{m.get('training_samples',0):,}"),
            ("Features",   f"{len(r.selected_features)}"),
            ("Folds",      f"{m.get('n_folds',0)}"),
        ]
        for i, (label, value) in enumerate(metrics_text):
            row = i // 3
            col = i % 3
            x = 0.05 + col * 0.33
            y = 0.85 - row * 0.30
            ax3.text(x, y, label, fontsize=9, color="#94a3b8", transform=ax3.transAxes)
            ax3.text(x, y - 0.10, value, fontsize=14, color="#38bdf8",
                     transform=ax3.transAxes, fontweight="bold")
        ax3.set_title("Model Scorecard", fontsize=12, color="#f8fafc", pad=12)

        # Panel 4: Fold accuracy trend
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(fold_nums, fold_accs, "o-", color="#38bdf8", lw=2,
                 markersize=6, markerfacecolor="#0ea5e9", markeredgecolor="#f8fafc",
                 markeredgewidth=1, zorder=3)
        ax4.fill_between(fold_nums, 50, fold_accs, alpha=0.15, color="#38bdf8")
        ax4.axhline(50, color="#f59e0b", ls="--", lw=1.5, alpha=0.6)
        ax4.set_xlabel("Fold #")
        ax4.set_ylabel("Accuracy (%)")
        ax4.set_title("Accuracy Trend Across Folds", fontsize=12, color="#f8fafc")
        ax4.set_ylim(0, 100)
        ax4.grid(True)

        chart_path = os.path.join(CHART_DIR, f"{ticker.replace('.', '_')}_monitoring.png")
        fig.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ {ticker} chart → {chart_path}")

    # -----------------------------------------------------------------
    #  COMBINED DASHBOARD
    # -----------------------------------------------------------------
    successful = [t for t in STOCKS if results[t].success]

    if len(successful) >= 2:
        fig = plt.figure(figsize=(20, 14))
        fig.suptitle("Combined Model Monitoring — Multi-Market v5.2",
                     fontsize=18, fontweight="bold", color="#38bdf8", y=0.98)
        gs = gridspec.GridSpec(2, 2, hspace=0.40, wspace=0.30,
                               left=0.06, right=0.97, top=0.92, bottom=0.06)

        # Panel 1: Accuracy comparison
        ax1 = fig.add_subplot(gs[0, 0])
        labels = [t.replace(".NS","(IN)") for t in successful]
        accs = [results[t].metrics.get("accuracy", 0) for t in successful]
        barcolors = ["#22c55e" if a >= 55 else "#f59e0b" if a >= 50 else "#ef4444" for a in accs]
        bars = ax1.bar(range(len(labels)), accs, color=barcolors, edgecolor="#475569", width=0.55, zorder=3)
        ax1.axhline(50, color="#f59e0b", ls="--", lw=1.5, label="Random baseline")
        ax1.set_xticks(range(len(labels)))
        ax1.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
        ax1.set_ylabel("Held-Out Accuracy (%)")
        ax1.set_title("Per-Stock Accuracy", fontsize=13, color="#f8fafc")
        ax1.set_ylim(0, 100)
        ax1.legend(fontsize=9, facecolor="#1e293b", edgecolor="#475569")
        ax1.grid(True, axis="y")
        for bar, acc in zip(bars, accs):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height()+1.2,
                     f"{acc:.1f}%", ha="center", va="bottom", fontsize=10,
                     color="#e2e8f0", fontweight="bold")

        # Panel 2: AUC-ROC comparison
        ax2 = fig.add_subplot(gs[0, 1])
        aucs = [results[t].metrics.get("auc", 0.5) for t in successful]
        auccolors = ["#22c55e" if a >= 0.55 else "#f59e0b" if a >= 0.50 else "#ef4444" for a in aucs]
        bars2 = ax2.bar(range(len(labels)), aucs, color=auccolors, edgecolor="#475569", width=0.55, zorder=3)
        ax2.axhline(0.5, color="#f59e0b", ls="--", lw=1.5, label="No-skill")
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, fontsize=9, rotation=35, ha="right")
        ax2.set_ylabel("AUC-ROC")
        ax2.set_title("Per-Stock AUC-ROC", fontsize=13, color="#f8fafc")
        ax2.set_ylim(0, 1.0)
        ax2.legend(fontsize=9, facecolor="#1e293b", edgecolor="#475569")
        ax2.grid(True, axis="y")
        for bar, auc in zip(bars2, aucs):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height()+0.012,
                     f"{auc:.3f}", ha="center", va="bottom", fontsize=10,
                     color="#e2e8f0", fontweight="bold")

        # Panel 3: Box plots
        ax3 = fig.add_subplot(gs[1, 0])
        fold_data = [[f["accuracy"]*100 for f in results[t].fold_results] for t in successful]
        bp = ax3.boxplot(fold_data, labels=labels, patch_artist=True,
                         medianprops=dict(color="#f8fafc", linewidth=2),
                         whiskerprops=dict(color="#94a3b8"),
                         capprops=dict(color="#94a3b8"),
                         flierprops=dict(marker="o", markerfacecolor="#ef4444", markersize=4))
        palette = ["#3b82f6","#8b5cf6","#06b6d4","#f59e0b","#22c55e","#ec4899","#f97316","#14b8a6","#6366f1","#a855f7"]
        for i, (patch, color) in enumerate(zip(bp["boxes"], palette[:len(fold_data)])):
            patch.set_facecolor(color)
            patch.set_alpha(0.55)
        ax3.axhline(50, color="#f59e0b", ls="--", lw=1.5)
        ax3.set_ylabel("Accuracy (%)")
        ax3.set_title("Fold Accuracy Distribution", fontsize=13, color="#f8fafc")
        ax3.grid(True, axis="y")
        ax3.tick_params(axis="x", rotation=35)

        # Panel 4: Overall scorecard
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.axis("off")
        avg_acc = np.mean(accs)
        avg_auc = np.mean(aucs)
        avg_f1 = np.mean([results[t].metrics.get("f1", 0) for t in successful])
        sig_count = sum(1 for t in successful if results[t].metrics.get("binom_pvalue", 1) < 0.05)
        us_acc = np.mean([results[t].metrics.get("accuracy",0) for t in successful if not t.endswith(".NS")])
        in_accs = [results[t].metrics.get("accuracy",0) for t in successful if t.endswith(".NS")]
        in_acc = np.mean(in_accs) if in_accs else 0

        if avg_acc >= 55 and avg_auc >= 0.53:
            verdict = "HEALTHY ✅"; verdict_color = "#22c55e"
        elif avg_acc >= 50:
            verdict = "MARGINAL ⚠️"; verdict_color = "#f59e0b"
        else:
            verdict = "DEGRADED ❌"; verdict_color = "#ef4444"

        lines = [
            ("Verdict", verdict, verdict_color, 18),
            ("Avg Accuracy", f"{avg_acc:.1f}%", "#38bdf8", 16),
            ("Avg AUC-ROC", f"{avg_auc:.3f}", "#38bdf8", 16),
            ("Avg F1", f"{avg_f1:.1f}%", "#38bdf8", 16),
            ("US Avg", f"{us_acc:.1f}%", "#e2e8f0", 14),
            ("India Avg", f"{in_acc:.1f}%" if in_accs else "N/A", "#e2e8f0", 14),
            ("Significant", f"{sig_count}/{len(successful)}", "#e2e8f0", 14),
            ("Trained", f"{len(successful)}/{len(STOCKS)}", "#e2e8f0", 14),
            ("Time", f"{total_elapsed:.0f}s", "#94a3b8", 12),
        ]
        y_pos = 0.95
        for label, value, color, size in lines:
            ax4.text(0.05, y_pos, f"{label}:", fontsize=10, color="#94a3b8", transform=ax4.transAxes)
            ax4.text(0.50, y_pos, value, fontsize=size, color=color, transform=ax4.transAxes, fontweight="bold")
            y_pos -= 0.105
        ax4.set_title("Portfolio Health Assessment", fontsize=13, color="#f8fafc", pad=12)

        combined_path = os.path.join(CHART_DIR, "combined_multimarket_monitoring.png")
        fig.savefig(combined_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"\n  ✅ Combined dashboard → {combined_path}")

    # -----------------------------------------------------------------
    #  METRICS HEATMAP
    # -----------------------------------------------------------------
    if len(successful) >= 2:
        fig, ax = plt.subplots(figsize=(14, max(4, len(successful) * 0.8 + 1)))
        metric_keys = ["accuracy", "precision", "recall", "f1", "auc"]
        metric_labels = ["Accuracy", "Precision", "Recall", "F1", "AUC"]
        hlabels = [t.replace(".NS","(IN)") for t in successful]
        data_matrix = np.array([
            [results[t].metrics.get(mk, 0) * (100 if mk == "auc" else 1) for mk in metric_keys]
            for t in successful
        ])
        im = ax.imshow(data_matrix, cmap="RdYlGn", aspect="auto", vmin=30, vmax=80)
        ax.set_xticks(range(len(metric_labels)))
        ax.set_xticklabels(metric_labels, fontsize=11)
        ax.set_yticks(range(len(hlabels)))
        ax.set_yticklabels(hlabels, fontsize=12, fontweight="bold")
        for i in range(len(hlabels)):
            for j in range(len(metric_labels)):
                val = data_matrix[i, j]
                txt = f"{val:.1f}%" if metric_keys[j] != "auc" else f"{val/100:.3f}"
                color = "#000" if val > 55 else "#fff"
                ax.text(j, i, txt, ha="center", va="center", fontsize=10, fontweight="bold", color=color)
        ax.set_title("Model Metrics Heatmap — Multi-Market v5.2", fontsize=14, color="#f8fafc", pad=12)
        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label("Score (%)", color="#e2e8f0")
        cbar.ax.yaxis.set_tick_params(color="#94a3b8")
        plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color="#94a3b8")
        hm_path = os.path.join(CHART_DIR, "multimarket_heatmap.png")
        fig.savefig(hm_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  ✅ Heatmap → {hm_path}")

# ===========================================================================
#  SAVE JSON
# ===========================================================================
summary = {
    "engine_version": "v5.2",
    "generated_at": datetime.utcnow().isoformat() + "Z",
    "training_seconds": round(total_elapsed, 1),
    "stocks": [{
        "ticker": t,
        "success": results[t].success,
        "version": results[t].model_version if results[t].success else None,
        "reason": results[t].reason,
        "metrics": results[t].metrics if results[t].success else {},
        "timing_seconds": round(timings.get(t, 0), 1),
        "n_features": len(results[t].selected_features) if results[t].success else 0,
    } for t in STOCKS],
}
json_path = os.path.join(CHART_DIR, "multimarket_summary.json")
with open(json_path, "w", encoding="utf-8") as f:
    json.dump(summary, f, indent=2)
print(f"\n  ✅ JSON → {json_path}")

# ===========================================================================
#  FINAL VERDICT
# ===========================================================================
print("\n\n" + "=" * 80)
print("  FINAL MULTI-MARKET VERDICT")
print("=" * 80)
successful = [t for t in STOCKS if results[t].success]
failed = [t for t in STOCKS if not results[t].success]
if successful:
    avg_acc = np.mean([results[t].metrics.get("accuracy", 0) for t in successful])
    avg_auc = np.mean([results[t].metrics.get("auc", 0.5) for t in successful])
    avg_f1 = np.mean([results[t].metrics.get("f1", 0) for t in successful])
    sig = sum(1 for t in successful if results[t].metrics.get("binom_pvalue", 1) < 0.05)
    print(f"\n  Trained:   {len(successful)}/{len(STOCKS)}")
    print(f"  Avg Acc:   {avg_acc:.2f}%")
    print(f"  Avg AUC:   {avg_auc:.3f}")
    print(f"  Avg F1:    {avg_f1:.2f}%")
    print(f"  Sig (p<.05): {sig}/{len(successful)}")
    print()
    for t in STOCKS:
        r = results[t]
        if r.success:
            m = r.metrics
            sig_mark = "✅" if m.get("binom_pvalue", 1) < 0.05 else "⚠️"
            print(f"    {t:15s}  acc={m.get('accuracy',0):5.1f}%  auc={m.get('auc',0):.3f}  "
                  f"f1={m.get('f1',0):5.1f}%  p={m.get('binom_pvalue',1):.4f} {sig_mark}  ({timings[t]:.0f}s)")
        else:
            print(f"    {t:15s}  ❌ {r.reason}")
if failed:
    print(f"\n  Failed: {', '.join(failed)}")
print(f"\n  Charts: {CHART_DIR}")
print("=" * 80 + "\n")
