"""
CIS432 Optional Project 2 — Task 2: Interval Prediction
Waiting Time Prediction in Healthcare Operations
"""

# ── Imports ───────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_pinball_loss

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333",      "axes.labelcolor": "white",
    "xtick.color": "white",        "ytick.color": "white",
    "text.color": "white",         "grid.color": "#333",
    "grid.linestyle": "--",        "grid.alpha": 0.5,
    "font.family": "monospace",
})
GREEN = "#00ff9f"; ORANGE = "#ff9f00"; RED = "#ff4444"; BLUE = "#4499ff"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING (same as Task 1)
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 2 — INTERVAL PREDICTION")
print("=" * 60)

rt   = pd.read_excel("data.xlsx", sheet_name="realtime")
appt = pd.read_excel("data.xlsx", sheet_name="appointments")

rt["datetime"]        = pd.to_datetime(rt["date"].astype(str)   + " " + rt["time"].astype(str))
appt["appt_datetime"] = pd.to_datetime(appt["date"].astype(str) + " " + appt["time"].astype(str))

pat = rt[rt["entity_type"] == "patient"].copy()
key_events = ["hospital_checkin","bloodwork_checkin","bloodwork_start",
              "bloodwork_end","examination_checkin","examination_start"]
df = pat[pat["event"].isin(key_events)].pivot_table(
    index=["entity_id","date"], columns="event",
    values="datetime", aggfunc="first"
).reset_index()
df.columns.name = None

df["wait_minutes"] = (df["examination_start"] - df["examination_checkin"]).dt.total_seconds() / 60
df = df.dropna(subset=["wait_minutes","examination_checkin"])

appt_m = appt.rename(columns={"patient_id":"entity_id"})
df = df.merge(appt_m[["entity_id","date","appt_datetime","provider_id"]],
              on=["entity_id","date"], how="left")
df = df.sort_values("examination_checkin").reset_index(drop=True)

df["checkin_hour"]             = df["examination_checkin"].dt.hour
df["checkin_minute"]           = df["examination_checkin"].dt.minute
df["day_of_week"]              = df["examination_checkin"].dt.dayofweek
df["week"]                     = df["examination_checkin"].dt.isocalendar().week.astype(int)
df["bloodwork_duration"]       = (df["bloodwork_end"] - df["bloodwork_start"]).dt.total_seconds() / 60
df["total_pre_exam_time"]      = (df["examination_checkin"] - df["hospital_checkin"]).dt.total_seconds() / 60
df["early_late_min"]           = (df["examination_checkin"] - df["appt_datetime"]).dt.total_seconds() / 60
df["queue_position"]           = df.groupby("date").cumcount()
df["provider_patients_so_far"] = df.groupby(["date","provider_id"]).cumcount()

FEATURES = ["checkin_hour","checkin_minute","day_of_week","week",
            "bloodwork_duration","total_pre_exam_time",
            "early_late_min","queue_position","provider_patients_so_far"]
X = df[FEATURES].copy()
y = df["wait_minutes"].copy()

print(f"\n✓ Dataset ready: {len(X):,} samples")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. METHOD A — RESIDUAL BOOTSTRAP INTERVALS
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("METHOD A: Residual Bootstrap")
print("─" * 60)

kf = KFold(n_splits=5, shuffle=True, random_state=42)
base_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42)

oof_pred  = np.zeros(len(y))
oof_resid = np.zeros(len(y))

for train_idx, val_idx in kf.split(X):
    base_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_pred[val_idx]  = base_model.predict(X.iloc[val_idx])
    oof_resid[val_idx] = y.iloc[val_idx].values - oof_pred[val_idx]

# Use residuals to build empirical intervals
alpha = 0.10   # 90% prediction interval
lo_q  = np.percentile(oof_resid, 100 * alpha / 2)
hi_q  = np.percentile(oof_resid, 100 * (1 - alpha / 2))

lower_boot = oof_pred + lo_q
upper_boot = oof_pred + hi_q

# Clip negatives
lower_boot = np.maximum(lower_boot, 0)

# Evaluate
coverage_boot = np.mean((y >= lower_boot) & (y <= upper_boot))
width_boot    = np.mean(upper_boot - lower_boot)

print(f"  90% Prediction Interval (Bootstrap)")
print(f"  Coverage : {coverage_boot:.3f}  (target ≥ 0.90)")
print(f"  Avg Width: {width_boot:.2f} minutes")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. METHOD B — CONFORMAL PREDICTION (Split Conformal, manual)
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("METHOD B: Split Conformal Prediction")
print("─" * 60)

# Split: 60% train, 20% calibration, 20% test
n = len(X)
idx_all = np.random.RandomState(42).permutation(n)
train_idx = idx_all[:int(0.6*n)]
calib_idx = idx_all[int(0.6*n):int(0.8*n)]
test_idx  = idx_all[int(0.8*n):]

conf_model = GradientBoostingRegressor(n_estimators=200, max_depth=4,
                                       learning_rate=0.05, random_state=42)
conf_model.fit(X.iloc[train_idx], y.iloc[train_idx])

# Calibration nonconformity scores = |y - y_hat|
calib_pred   = conf_model.predict(X.iloc[calib_idx])
calib_scores = np.abs(y.iloc[calib_idx].values - calib_pred)

# Conformal quantile
alpha_level = 0.10
q_level = np.ceil((1 - alpha_level) * (len(calib_idx) + 1)) / len(calib_idx)
q_level = min(q_level, 1.0)
conformal_q = np.quantile(calib_scores, q_level)

# Apply to test set
test_pred   = conf_model.predict(X.iloc[test_idx])
lower_mapie = np.maximum(test_pred - conformal_q, 0)
upper_mapie = test_pred + conformal_q
y_pred_mapie = test_pred
y_test_conf  = y.iloc[test_idx].values

coverage_mapie = np.mean((y_test_conf >= lower_mapie) & (y_test_conf <= upper_mapie))
width_mapie    = np.mean(upper_mapie - lower_mapie)

print(f"  90% Prediction Interval (Conformal)")
print(f"  Coverage : {coverage_mapie:.3f}  (target ≥ 0.90)")
print(f"  Avg Width: {width_mapie:.2f} minutes")

# ═══════════════════════════════════════════════════════════════════════════════
# 4. VISUALISATION
# ═══════════════════════════════════════════════════════════════════════════════

# --- Plot 1: Interval width vs actual wait ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("Task 2 — Interval Prediction (90% PI)", color=GREEN, fontsize=14, fontweight="bold")

# Sample 300 points sorted by actual wait for readability
sort_idx = np.argsort(y_test_conf)[:300]
y_s     = y_test_conf[sort_idx]
lo_s    = lower_mapie[sort_idx]
hi_s    = upper_mapie[sort_idx]
pred_s  = y_pred_mapie[sort_idx]

ax = axes[0]
ax.fill_between(range(len(y_s)), lo_s, hi_s, alpha=0.3, color=BLUE, label="90% PI")
ax.plot(range(len(y_s)), pred_s, color=GREEN,  linewidth=1.2, label="Predicted")
ax.scatter(range(len(y_s)), y_s,  color=ORANGE, s=8, alpha=0.6, label="Actual")
ax.set_xlabel("Sample (sorted by actual wait)")
ax.set_ylabel("Wait (minutes)")
ax.set_title("Conformal Prediction Intervals")
ax.legend()
ax.grid(True)

# --- Plot 2: Coverage comparison ---
ax = axes[1]
methods   = ["Bootstrap\n(Method A)", "Conformal\n(Method B)"]
coverages = [coverage_boot,  coverage_mapie]
widths    = [width_boot,     width_mapie]

x = np.arange(2)
bars = ax.bar(x - 0.2, coverages, width=0.35, color=GREEN,  alpha=0.8, label="Coverage")
ax.set_ylim(0, 1.15)
ax.axhline(0.90, color=RED, linestyle="--", linewidth=1.5, label="Target 90%")
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.set_ylabel("Coverage")
ax.set_title("Coverage vs Width Comparison")
ax.legend(loc="upper left")
ax.grid(True, axis="y")

ax2 = ax.twinx()
ax2.bar(x + 0.2, widths, width=0.35, color=ORANGE, alpha=0.8, label="Avg Width (min)")
ax2.set_ylabel("Avg Interval Width (min)", color=ORANGE)
ax2.tick_params(axis="y", colors=ORANGE)
ax2.legend(loc="upper right")

for bar, val in zip(bars, coverages):
    ax.text(bar.get_x() + bar.get_width()/2, val + 0.02,
            f"{val:.3f}", ha="center", fontsize=10, color=GREEN)

plt.tight_layout()
plt.savefig("task2_interval_prediction.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved: task2_interval_prediction.png")

# --- Plot 2: Interval width distribution ---
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Interval Width Analysis", color=GREEN, fontsize=13, fontweight="bold")

axes[0].hist(upper_mapie - lower_mapie, bins=40, color=BLUE, alpha=0.8, edgecolor="#0f0f0f")
axes[0].axvline(width_mapie, color=ORANGE, linewidth=2, linestyle="--",
                label=f"Mean {width_mapie:.1f} min")
axes[0].set_xlabel("Interval Width (minutes)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution of Interval Widths (Conformal)")
axes[0].legend()
axes[0].grid(True)

# Width vs predicted value — do wider intervals occur for longer waits?
axes[1].scatter(y_pred_mapie, upper_mapie - lower_mapie,
                alpha=0.1, s=6, color=GREEN)
axes[1].set_xlabel("Predicted Wait (minutes)")
axes[1].set_ylabel("Interval Width (minutes)")
axes[1].set_title("Width vs Predicted Value")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("task2_width_analysis.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: task2_width_analysis.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 2 SUMMARY")
print("=" * 60)
print(f"""
Two methods for 90% Prediction Intervals:

Method A — Residual Bootstrap
  Coverage : {coverage_boot:.3f}
  Avg Width: {width_boot:.1f} min

Method B — Conformal Prediction (MAPIE)
  Coverage : {coverage_mapie:.3f}
  Avg Width: {width_mapie:.1f} min

Metrics Justification:
  - Coverage: proportion of actual values falling inside the interval.
               Should be ≥ target level (90%). This is the primary metric.
  - Avg Width: narrower intervals are more actionable. A tradeoff exists
               between coverage and width — wider intervals are easier to
               cover but less useful to patients and staff.

Recommendation:
  Conformal prediction is preferred because it provides a formal
  coverage guarantee by construction, regardless of model assumptions.
  Bootstrap intervals rely on residuals being roughly symmetric,
  which may not hold for skewed waiting time distributions.

Clinical Interpretation:
  A 90% PI of width ~{width_mapie:.0f} min means the system would tell a
  patient "your wait will likely be between X and Y minutes" and
  be correct 9 out of 10 times. This is more honest and useful
  than a single point estimate alone.
""")
print("✓ Task 2 complete.")
