"""
CIS432 Optional Project 2 — Task 5: Comparison and Reflection
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "grid.color": "#333",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "monospace",
})
GREEN="#00ff9f"; ORANGE="#ff9f00"; RED="#ff4444"; BLUE="#4499ff"; PURPLE="#cc88ff"

print("="*60)
print("TASK 5 — COMPARISON AND REFLECTION")
print("="*60)

# ── Summary data from Tasks 1–4 ──────────────────────────────────────────────
comparison = {
    "Task 1\nPoint":     {"RMSE": 30.39, "MAE": 22.77, "Coverage": None,  "Width": None,   "Interpretable": 4, "Actionable": 3},
    "Task 2\nInterval":  {"RMSE": None,  "MAE": None,  "Coverage": 0.899, "Width": 86.5,   "Interpretable": 4, "Actionable": 4},
    "Task 3\nQuantile":  {"RMSE": None,  "MAE": None,  "Coverage": None,  "Width": None,   "Interpretable": 5, "Actionable": 5},
    "Task 4\nDynamic":   {"RMSE": 27.65, "MAE": 20.58, "Coverage": None,  "Width": None,   "Interpretable": 3, "Actionable": 5},
}

# ── Plot 1: Multi-panel comparison ───────────────────────────────────────────
fig = plt.figure(figsize=(16, 10))
fig.suptitle("Task 5 — Approach Comparison", color=GREEN, fontsize=15, fontweight="bold")
gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

labels = list(comparison.keys())
colors = [GREEN, ORANGE, BLUE, PURPLE]

# Panel 1: RMSE (Tasks 1 & 4 only)
ax1 = fig.add_subplot(gs[0, 0])
rmse_tasks  = ["Task 1\nPoint", "Task 4\nDynamic"]
rmse_vals   = [30.39, 27.65]
rmse_colors = [GREEN, PURPLE]
bars = ax1.bar(rmse_tasks, rmse_vals, color=rmse_colors, alpha=0.85, edgecolor="#0f0f0f")
for bar, val in zip(bars, rmse_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, val + 0.3,
             f"{val:.2f}", ha="center", fontsize=11, color="white", fontweight="bold")
ax1.set_ylabel("RMSE (minutes)")
ax1.set_title("Point Accuracy\n(lower = better)")
ax1.grid(True, axis="y")
ax1.set_ylim(0, 38)

# Panel 2: Interval coverage vs width
ax2 = fig.add_subplot(gs[0, 1])
ax2.bar(["Task 2\nInterval"], [0.899], color=ORANGE, alpha=0.85, edgecolor="#0f0f0f", label="Coverage")
ax2.axhline(0.90, color=RED, linestyle="--", linewidth=1.5, label="Target 90%")
ax2.set_ylabel("Coverage")
ax2.set_ylim(0.85, 0.95)
ax2.set_title("Interval Coverage\n(target ≥ 0.90)")
ax2.legend(fontsize=8)
ax2.grid(True, axis="y")
ax2_r = ax2.twinx()
ax2_r.bar(["Task 2\nInterval"], [86.5], color=ORANGE, alpha=0.3, width=0.4)
ax2_r.set_ylabel("Avg Width (min)", color=ORANGE)
ax2_r.tick_params(axis="y", colors=ORANGE)

# Panel 3: Quantile coverage (calibration)
ax3 = fig.add_subplot(gs[0, 2])
qs      = [0.10, 0.25, 0.50, 0.75, 0.90]
actuals = [0.115, 0.254, 0.505, 0.747, 0.894]
q_colors= [BLUE, GREEN, ORANGE, PURPLE, RED]
ax3.plot([0,1],[0,1],"--",color="white",linewidth=1.2,label="Perfect")
ax3.scatter(qs, actuals, s=100, color=q_colors, zorder=5, edgecolors="white")
for q, a in zip(qs, actuals):
    ax3.annotate(f"q{int(q*100)}", (q, a), xytext=(6, 4),
                 textcoords="offset points", fontsize=8, color="white")
ax3.set_xlabel("Target Quantile")
ax3.set_ylabel("Actual Coverage")
ax3.set_title("Quantile Calibration\n(Task 3)")
ax3.legend(fontsize=8)
ax3.grid(True)

# Panel 4: Dynamic RMSE over time
ax4 = fig.add_subplot(gs[1, 0])
t_vals    = [0, 5, 10, 15, 20]
rmse_dyn  = [29.25, 28.86, 28.33, 27.93, 27.65]
ax4.plot(t_vals, rmse_dyn, color=PURPLE, marker="o", linewidth=2, markersize=8)
for t, r in zip(t_vals, rmse_dyn):
    ax4.text(t, r+0.1, f"{r:.1f}", ha="center", fontsize=8, color=PURPLE)
ax4.set_xlabel("Minutes Elapsed")
ax4.set_ylabel("RMSE (minutes)")
ax4.set_title("Dynamic Accuracy\nImproves Over Time")
ax4.set_xticks(t_vals)
ax4.grid(True)

# Panel 5: Interpretability vs Actionability radar-style bar
ax5 = fig.add_subplot(gs[1, 1])
dims   = ["Interpretable", "Actionable"]
task_names = ["Point","Interval","Quantile","Dynamic"]
task_cols  = [GREEN, ORANGE, BLUE, PURPLE]
x = np.arange(len(dims))
width = 0.2
for i, (tname, tcol) in enumerate(zip(task_names, task_cols)):
    key = f"Task {'1' if tname=='Point' else '2' if tname=='Interval' else '3' if tname=='Quantile' else '4'}\n{tname}"
    vals = [comparison[key]["Interpretable"], comparison[key]["Actionable"]]
    ax5.bar(x + i*width, vals, width, color=tcol, alpha=0.85, label=tname, edgecolor="#0f0f0f")
ax5.set_xticks(x + width*1.5)
ax5.set_xticklabels(dims)
ax5.set_ylabel("Score (1–5)")
ax5.set_title("Interpretability &\nActionability")
ax5.legend(fontsize=8, loc="lower right")
ax5.set_ylim(0, 6)
ax5.grid(True, axis="y")

# Panel 6: Recommendation summary text
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
rec_text = (
    "DEPLOYMENT RECOMMENDATION\n\n"
    "Primary: Task 3 (Quantile)\n"
    "→ q50: patient-facing ETA\n"
    "→ q90: staff SLA target\n\n"
    "Secondary: Task 4 (Dynamic)\n"
    "→ Updates prediction as\n"
    "  patient continues waiting\n\n"
    "Task 2 (Interval) useful for\n"
    "communicating uncertainty\n\n"
    "Task 1 as internal baseline"
)
ax6.text(0.05, 0.95, rec_text, transform=ax6.transAxes,
         fontsize=10, verticalalignment="top", color="white",
         bbox=dict(boxstyle="round", facecolor="#1a1a2e", edgecolor=GREEN, linewidth=1.5))

plt.savefig("task5_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: task5_comparison.png")

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"""
{"="*60}
TASK 5 SUMMARY
{"="*60}

COMPARISON OF APPROACHES
─────────────────────────────────────────────────────────────
Task 1 — Point Prediction
  • Output   : single predicted wait (e.g., "37 minutes")
  • Metric   : RMSE = 30.4 min, MAE = 22.8 min, R² = 0.311
  • Insight  : fast baseline; shows key drivers (provider load,
               queue position, arrival timing)
  • Limit    : conveys false precision — no sense of uncertainty

Task 2 — Interval Prediction
  • Output   : 90% prediction interval (e.g., "14 – 100 min")
  • Metric   : Coverage = 89.9%, Avg Width = 86.5 min
  • Insight  : honest about uncertainty; useful for risk-averse
               communication to patients and administrators
  • Limit    : one fixed coverage level; wide intervals may
               frustrate patients who want a precise answer

Task 3 — Quantile Prediction
  • Output   : multiple quantiles (q10, q25, q50, q75, q90)
  • Metric   : Pinball loss per quantile; all well-calibrated
  • Insight  : exposes full conditional distribution; different
               stakeholders use different quantiles (patients → q50,
               schedulers → q75, SLA monitoring → q90)
  • Limit    : slightly harder to explain to non-technical users

Task 4 — Dynamic Prediction
  • Output   : updated remaining-wait prediction every 5 minutes
  • Metric   : RMSE improves from 29.3 (t=0) to 27.7 (t=20), ↓5.5%
  • Insight  : captures real-time queue state; reduces anxiety by
               giving patients a live updated ETA
  • Limit    : requires real-time data pipeline; model complexity
               increases (one model per snapshot)

TRADEOFFS
─────────────────────────────────────────────────────────────
  Precision vs Honesty  : Task 1 (precise) vs Task 2/3 (honest)
  Complexity vs Utility : Task 4 (complex pipeline, highest value)
  Flexibility           : Task 3 lets each stakeholder choose level

DEPLOYMENT RECOMMENDATION
─────────────────────────────────────────────────────────────
  Recommended: Task 3 (Quantile) + Task 4 (Dynamic) combined

  Rationale:
  - Quantile model gives multiple output levels at check-in time.
    q50 shown to patients on a display screen as expected wait;
    q90 used internally for staffing targets and SLA tracking.
  - Dynamic model updates the q50 estimate every 5 minutes as the
    patient waits, reducing uncertainty and managing expectations.
  - Interval prediction (Task 2) folded in by showing q25–q75 as
    a visual band around the point estimate.
  - Task 1 retained as an internal benchmark only.

  Rollout plan:
  1. Pilot with one provider team for 4 weeks
  2. Collect feedback from patients and staff
  3. Monitor coverage and RMSE on live data weekly
  4. Retrain models monthly as new data accumulates
""")
print("✓ Task 5 complete.")
