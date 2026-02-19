"""
CIS432 Optional Project 2 — Task 1: Point Prediction
Waiting Time Prediction in Healthcare Operations
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.dummy import DummyRegressor

# ── Style ─────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0f0f0f",
    "axes.facecolor":   "#1a1a2e",
    "axes.edgecolor":   "#333",
    "axes.labelcolor":  "white",
    "xtick.color":      "white",
    "ytick.color":      "white",
    "text.color":       "white",
    "grid.color":       "#333",
    "grid.linestyle":   "--",
    "grid.alpha":       0.5,
    "font.family":      "monospace",
})
GREEN  = "#00ff9f"
ORANGE = "#ff9f00"
RED    = "#ff4444"
BLUE   = "#4499ff"

# ═══════════════════════════════════════════════════════════════════════════════
# 1. DATA LOADING & FEATURE ENGINEERING
# ═══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("TASK 1 — POINT PREDICTION")
print("=" * 60)

rt   = pd.read_excel("data.xlsx", sheet_name="realtime")
appt = pd.read_excel("data.xlsx", sheet_name="appointments")

rt["datetime"]       = pd.to_datetime(rt["date"].astype(str)   + " " + rt["time"].astype(str))
appt["appt_datetime"] = pd.to_datetime(appt["date"].astype(str) + " " + appt["time"].astype(str))

# --- pivot key patient events ---
pat = rt[rt["entity_type"] == "patient"].copy()
key_events = ["hospital_checkin","bloodwork_checkin","bloodwork_start",
              "bloodwork_end","examination_checkin","examination_start"]
df = pat[pat["event"].isin(key_events)].pivot_table(
    index=["entity_id","date"], columns="event",
    values="datetime", aggfunc="first"
).reset_index()
df.columns.name = None

# --- target ---
df["wait_minutes"] = (df["examination_start"] - df["examination_checkin"]).dt.total_seconds() / 60
df = df.dropna(subset=["wait_minutes", "examination_checkin"])

# --- merge appointments (scheduled provider, scheduled time) ---
appt_m = appt.rename(columns={"patient_id": "entity_id"})
df = df.merge(appt_m[["entity_id","date","appt_datetime","provider_id"]],
              on=["entity_id","date"], how="left")

# --- feature engineering ---
df = df.sort_values("examination_checkin").reset_index(drop=True)

df["checkin_hour"]           = df["examination_checkin"].dt.hour
df["checkin_minute"]         = df["examination_checkin"].dt.minute
df["day_of_week"]            = df["examination_checkin"].dt.dayofweek     # 0=Mon
df["week"]                   = df["examination_checkin"].dt.isocalendar().week.astype(int)
df["bloodwork_duration"]     = (df["bloodwork_end"] - df["bloodwork_start"]).dt.total_seconds() / 60
df["total_pre_exam_time"]    = (df["examination_checkin"] - df["hospital_checkin"]).dt.total_seconds() / 60
df["early_late_min"]         = (df["examination_checkin"] - df["appt_datetime"]).dt.total_seconds() / 60
df["queue_position"]         = df.groupby("date").cumcount()
df["provider_patients_so_far"] = df.groupby(["date","provider_id"]).cumcount()

FEATURES = [
    "checkin_hour", "checkin_minute", "day_of_week", "week",
    "bloodwork_duration", "total_pre_exam_time",
    "early_late_min", "queue_position", "provider_patients_so_far"
]
TARGET = "wait_minutes"

X = df[FEATURES].copy()
y = df[TARGET].copy()

print(f"\n✓ Dataset ready: {X.shape[0]:,} samples, {X.shape[1]} features")
print(f"  Target — mean: {y.mean():.1f} min | median: {y.median():.1f} min | std: {y.std():.1f} min")

# ═══════════════════════════════════════════════════════════════════════════════
# 2. TARGET DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.suptitle("Waiting Time Distribution", color=GREEN, fontsize=14, fontweight="bold")

axes[0].hist(y, bins=50, color=GREEN, alpha=0.8, edgecolor="#0f0f0f")
axes[0].axvline(y.mean(),   color=ORANGE, linewidth=2, linestyle="--", label=f"Mean {y.mean():.1f}")
axes[0].axvline(y.median(), color=BLUE,   linewidth=2, linestyle=":",  label=f"Median {y.median():.1f}")
axes[0].set_xlabel("Wait (minutes)")
axes[0].set_ylabel("Count")
axes[0].set_title("Distribution")
axes[0].legend()
axes[0].grid(True)

axes[1].hist(np.log1p(y), bins=50, color=BLUE, alpha=0.8, edgecolor="#0f0f0f")
axes[1].set_xlabel("log(1 + Wait)")
axes[1].set_ylabel("Count")
axes[1].set_title("Log-Transformed Distribution")
axes[1].grid(True)

plt.tight_layout()
plt.savefig("task1_target_distribution.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved: task1_target_distribution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 3. MODEL TRAINING — 5-FOLD CROSS VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "─" * 60)
print("MODEL EVALUATION (5-Fold CV)")
print("─" * 60)

models = {
    "Baseline (Mean)":       DummyRegressor(strategy="mean"),
    "Linear Regression":     Pipeline([("scaler", StandardScaler()), ("model", LinearRegression())]),
    "Ridge Regression":      Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))]),
    "Decision Tree":         DecisionTreeRegressor(max_depth=6, random_state=42),
    "Random Forest":         RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    "Gradient Boosting":     GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42),
}

kf = KFold(n_splits=5, shuffle=True, random_state=42)

results = {}
for name, model in models.items():
    rmse_scores = np.sqrt(-cross_val_score(model, X, y, cv=kf, scoring="neg_mean_squared_error"))
    mae_scores  = -cross_val_score(model, X, y, cv=kf, scoring="neg_mean_absolute_error")
    r2_scores   =  cross_val_score(model, X, y, cv=kf, scoring="r2")
    results[name] = {
        "RMSE mean": rmse_scores.mean(), "RMSE std": rmse_scores.std(),
        "MAE mean":  mae_scores.mean(),  "MAE std":  mae_scores.std(),
        "R² mean":   r2_scores.mean(),   "R² std":   r2_scores.std(),
    }
    print(f"  {name:<25} RMSE={rmse_scores.mean():.2f}±{rmse_scores.std():.2f}  "
          f"MAE={mae_scores.mean():.2f}±{mae_scores.std():.2f}  "
          f"R²={r2_scores.mean():.3f}±{r2_scores.std():.3f}")

results_df = pd.DataFrame(results).T

# ═══════════════════════════════════════════════════════════════════════════════
# 4. MODEL COMPARISON CHART
# ═══════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Task 1 — Model Comparison (5-Fold CV)", color=GREEN, fontsize=14, fontweight="bold")

metrics = [("RMSE mean", "RMSE std", "RMSE (minutes)", RED),
           ("MAE mean",  "MAE std",  "MAE (minutes)",  ORANGE),
           ("R² mean",   "R² std",   "R²",             BLUE)]

model_names = [n.replace(" ", "\n") for n in results_df.index]

for ax, (mean_col, std_col, label, color) in zip(axes, metrics):
    bars = ax.barh(model_names, results_df[mean_col],
                   xerr=results_df[std_col], color=color, alpha=0.8,
                   error_kw={"ecolor": "white", "capsize": 4})
    ax.set_xlabel(label)
    ax.set_title(label)
    ax.grid(True, axis="x")
    for bar, val in zip(bars, results_df[mean_col]):
        ax.text(bar.get_width() + results_df[std_col].max() * 0.05,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.2f}", va="center", fontsize=8, color="white")

plt.tight_layout()
plt.savefig("task1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved: task1_model_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 5. BEST MODEL: ACTUAL vs PREDICTED + FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════════
best_model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.05, random_state=42)

# Collect OOF predictions
oof_pred = np.zeros(len(y))
for train_idx, val_idx in kf.split(X):
    best_model.fit(X.iloc[train_idx], y.iloc[train_idx])
    oof_pred[val_idx] = best_model.predict(X.iloc[val_idx])

# Refit on full data for feature importance
best_model.fit(X, y)

fig, axes = plt.subplots(1, 2, figsize=(13, 5))
fig.suptitle("Gradient Boosting — Best Model Analysis", color=GREEN, fontsize=14, fontweight="bold")

# Actual vs Predicted
ax = axes[0]
ax.scatter(y, oof_pred, alpha=0.15, s=8, color=GREEN)
lims = [0, max(y.max(), oof_pred.max())]
ax.plot(lims, lims, "--", color=RED, linewidth=1.5, label="Perfect prediction")
ax.set_xlabel("Actual Wait (min)")
ax.set_ylabel("Predicted Wait (min)")
ax.set_title("Actual vs Predicted (OOF)")
ax.legend()
ax.grid(True)

final_rmse = np.sqrt(mean_squared_error(y, oof_pred))
final_mae  = mean_absolute_error(y, oof_pred)
final_r2   = r2_score(y, oof_pred)
ax.text(0.05, 0.92, f"RMSE={final_rmse:.2f}  MAE={final_mae:.2f}  R²={final_r2:.3f}",
        transform=ax.transAxes, color=ORANGE, fontsize=9)

# Feature importance
ax = axes[1]
fi = pd.Series(best_model.feature_importances_, index=FEATURES).sort_values()
colors = [GREEN if v > fi.median() else BLUE for v in fi.values]
ax.barh(fi.index, fi.values, color=colors, alpha=0.85)
ax.set_xlabel("Feature Importance")
ax.set_title("Feature Importance")
ax.grid(True, axis="x")

plt.tight_layout()
plt.savefig("task1_best_model.png", dpi=150, bbox_inches="tight")
plt.close()
print("✓ Saved: task1_best_model.png")

# ═══════════════════════════════════════════════════════════════════════════════
# 6. SUMMARY
# ═══════════════════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("TASK 1 SUMMARY")
print("=" * 60)
print(f"""
Best Model:   Gradient Boosting
Validation:   5-Fold Cross-Validation (n=12,762)

Performance:
  RMSE  = {final_rmse:.2f} minutes
  MAE   = {final_mae:.2f} minutes
  R²    = {final_r2:.3f}

  Baseline RMSE (predict mean) = {results['Baseline (Mean)']['RMSE mean']:.2f} minutes
  Improvement over baseline    = {(1 - final_rmse / results['Baseline (Mean)']['RMSE mean'])*100:.1f}%

Metrics Justification:
  - RMSE: penalizes large errors more heavily (important in clinical setting)
  - MAE:  interpretable in original units (minutes)
  - R²:   proportion of variance explained

Top Features (by importance):
""")
for feat, imp in fi.sort_values(ascending=False).head(4).items():
    print(f"  {feat:<30} {imp:.4f}")

print("\n✓ Task 1 complete. Output files saved.")