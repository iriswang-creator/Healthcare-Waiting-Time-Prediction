import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

plt.rcParams.update({
    "figure.facecolor": "#0f0f0f", "axes.facecolor": "#1a1a2e",
    "axes.edgecolor": "#333", "axes.labelcolor": "white",
    "xtick.color": "white", "ytick.color": "white",
    "text.color": "white", "grid.color": "#333",
    "grid.linestyle": "--", "grid.alpha": 0.5,
    "font.family": "monospace",
})
GREEN="#00ff9f"; ORANGE="#ff9f00"; RED="#ff4444"; BLUE="#4499ff"

print("="*60)
print("TASK 4 — DYNAMIC PREDICTION OVER TIME")
print("="*60)

rt   = pd.read_excel("data.xlsx", sheet_name="realtime")
appt = pd.read_excel("data.xlsx", sheet_name="appointments")

rt["datetime"]        = pd.to_datetime(rt["date"].astype(str) + " " + rt["time"].astype(str))
appt["appt_datetime"] = pd.to_datetime(appt["date"].astype(str) + " " + appt["time"].astype(str))

pat = rt[rt["entity_type"] == "patient"].copy()
prov = rt[rt["entity_type"] == "provider"].copy()

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
df = df.sort_values(["date","examination_checkin"]).reset_index(drop=True)

# Base features
df["checkin_hour"]             = df["examination_checkin"].dt.hour
df["checkin_minute"]           = df["examination_checkin"].dt.minute
df["day_of_week"]              = df["examination_checkin"].dt.dayofweek
df["week"]                     = df["examination_checkin"].dt.isocalendar().week.astype(int)
df["bloodwork_duration"]       = (df["bloodwork_end"] - df["bloodwork_start"]).dt.total_seconds() / 60
df["total_pre_exam_time"]      = (df["examination_checkin"] - df["hospital_checkin"]).dt.total_seconds() / 60
df["early_late_min"]           = (df["examination_checkin"] - df["appt_datetime"]).dt.total_seconds() / 60
df["queue_position"]           = df.groupby("date").cumcount()
df["provider_patients_so_far"] = df.groupby(["date","provider_id"]).cumcount()

print(f"\n✓ Base dataset: {len(df):,} patients")

# ── Provider room lookup table (vectorised) ──────────────────────────────────
prov_room = prov[prov["event"].isin(["provider_entered_room","provider_left_room"])].copy()
prov_room = prov_room.sort_values("datetime")

# ── Build snapshot rows efficiently ─────────────────────────────────────────
SNAPSHOTS = [0, 5, 10, 15, 20]
BASE_FEATS = ["checkin_hour","checkin_minute","day_of_week","week",
              "bloodwork_duration","total_pre_exam_time","early_late_min",
              "queue_position","provider_patients_so_far"]

print("\nBuilding snapshot rows...")
all_rows = []

# Pre-index exam events by date for fast lookup
df_by_date = {d: grp for d, grp in df.groupby("date")}
prov_by_pid_date = {(p, d): grp for (p, d), grp in prov_room.groupby(["entity_id","date"])}

for _, pat_row in df.iterrows():
    ec  = pat_row["examination_checkin"]
    es  = pat_row["examination_start"]
    aw  = pat_row["wait_minutes"]
    pid = pat_row["provider_id"]
    d   = pat_row["date"]
    day_df = df_by_date.get(d, pd.DataFrame())

    for t in SNAPSHOTS:
        snap = ec + pd.Timedelta(minutes=t)
        if snap >= es:
            continue

        remaining = aw - t

        # Queue dynamics
        served_ahead = int(((day_df["examination_checkin"] < ec) &
                            (day_df["examination_start"]  <= snap)).sum())
        still_ahead  = int(((day_df["examination_checkin"] < ec) &
                            (day_df["examination_start"]  >  snap)).sum())

        # Provider status
        provider_busy = 0
        prov_in_room_dur = 0.0
        if not pd.isna(pid):
            key = (int(pid), d)
            pev = prov_by_pid_date.get(key, pd.DataFrame())
            if len(pev):
                pev_before = pev[pev["datetime"] <= snap]
                if len(pev_before):
                    last_ev = pev_before.iloc[-1]
                    if last_ev["event"] == "provider_entered_room":
                        provider_busy = 1
                        prov_in_room_dur = (snap - last_ev["datetime"]).total_seconds() / 60

        row = {feat: pat_row[feat] for feat in BASE_FEATS}
        row.update({
            "entity_id": pat_row["entity_id"],
            "actual_wait": aw,
            "t_elapsed": t,
            "remaining_wait": remaining,
            "patients_served_ahead": served_ahead,
            "patients_still_waiting_ahead": still_ahead,
            "provider_busy_at_t": provider_busy,
            "provider_in_room_duration": prov_in_room_dur,
        })
        all_rows.append(row)

dyn = pd.DataFrame(all_rows)
dyn = dyn[dyn["remaining_wait"] > 0].reset_index(drop=True)

print(f"✓ Dynamic dataset: {len(dyn):,} snapshot rows")
for t in SNAPSHOTS:
    n = (dyn["t_elapsed"] == t).sum()
    print(f"  t={t:>2} min: {n:,} samples")

# ── Train one model per snapshot ─────────────────────────────────────────────
print("\n" + "─"*60)
print("TRAINING (5-Fold CV per snapshot)")
print("─"*60)

DYN_FEATS = BASE_FEATS + ["t_elapsed","patients_served_ahead",
                           "patients_still_waiting_ahead",
                           "provider_busy_at_t","provider_in_room_duration"]
kf = KFold(n_splits=5, shuffle=True, random_state=42)
results = {}

for t in SNAPSHOTS:
    sub = dyn[dyn["t_elapsed"] == t]
    Xt  = sub[DYN_FEATS].fillna(0)
    yt  = sub["remaining_wait"]
    oof = np.zeros(len(yt))
    for tr, va in kf.split(Xt):
        m = GradientBoostingRegressor(n_estimators=150, max_depth=4,
                                      learning_rate=0.05, random_state=42)
        m.fit(Xt.iloc[tr], yt.iloc[tr])
        oof[va] = m.predict(Xt.iloc[va])
    rmse = np.sqrt(mean_squared_error(yt, oof))
    mae  = mean_absolute_error(yt, oof)
    r2   = r2_score(yt, oof)
    results[t] = {"RMSE": rmse, "MAE": mae, "R2": r2}
    print(f"  t={t:>2} min  RMSE={rmse:.2f}  MAE={mae:.2f}  R²={r2:.3f}")

# ── Plots ─────────────────────────────────────────────────────────────────────
ts    = list(results.keys())
rmses = [results[t]["RMSE"] for t in ts]
maes  = [results[t]["MAE"]  for t in ts]
r2s   = [results[t]["R2"]   for t in ts]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Task 4 — Dynamic Prediction: Performance Over Time",
             color=GREEN, fontsize=14, fontweight="bold")

for ax, vals, color, label in zip(
    axes,
    [rmses, maes, r2s],
    [RED, ORANGE, GREEN],
    ["RMSE (minutes)", "MAE (minutes)", "R²"]
):
    ax.plot(ts, vals, color=color, marker="o", linewidth=2, markersize=9)
    for t, v in zip(ts, vals):
        ax.text(t, v + (max(vals)-min(vals))*0.03, f"{v:.2f}",
                ha="center", fontsize=9, color=color)
    ax.set_xlabel("Minutes Elapsed Since Check-in")
    ax.set_ylabel(label)
    ax.set_title(label + " vs Elapsed Time")
    ax.set_xticks(ts)
    ax.grid(True)

plt.tight_layout()
plt.savefig("task4_dynamic_performance.png", dpi=150, bbox_inches="tight")
plt.close()
print("\n✓ Saved: task4_dynamic_performance.png")

print("\n" + "="*60)
print("TASK 4 SUMMARY")
print("="*60)
print(f"""
Dynamic features added at each snapshot:
  - t_elapsed                    : minutes already waited
  - patients_served_ahead        : patients ahead now done
  - patients_still_waiting_ahead : patients ahead still in queue
  - provider_busy_at_t           : is provider currently in a room?
  - provider_in_room_duration    : how long has provider been in room?

Performance (RMSE on remaining wait):""")
for t in SNAPSHOTS:
    print(f"  t={t:>2} min  RMSE={results[t]['RMSE']:.2f}  R²={results[t]['R2']:.3f}")

imp = (results[0]['RMSE'] - results[20]['RMSE']) / results[0]['RMSE'] * 100
print(f"\n  RMSE improvement t=0 → t=20: {imp:.1f}%")
print("\n✓ Task 4 complete.")
