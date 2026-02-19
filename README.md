# 🏥 Waiting Time Prediction in Healthcare Operations
### CIS432 — Project 2

Predicting patient examination waiting times at a cancer hospital using simulated RFID tracking data. The project develops and compares four complementary ML approaches: point prediction, interval prediction, quantile prediction, and dynamic (real-time updating) prediction.

---

## 📁 Project Structure

```
project2/
├── CIS432_Project2_WaitingTimePrediction.ipynb  # Main notebook (all tasks)
├── data.xlsx                                     # Source data (not included)
├── task1_point_prediction.py                     # Task 1 standalone script
├── task2_interval_prediction.py                  # Task 2 standalone script
├── task3_quantile_prediction.py                  # Task 3 standalone script
├── task4_dynamic_prediction.py                   # Task 4 standalone script
├── task5_comparison.py                           # Task 5 standalone script
└── README.md
```

---

## 📊 Dataset

Two sheets from a simulated hospital RFID system (Jan–Jun 2024):

| Sheet | Rows | Description |
|---|---|---|
| `realtime` | 116,946 | Patient & provider location events |
| `appointments` | 13,398 | Scheduled appointment slots |

**Target variable:** Minutes between `examination_checkin` and `examination_start`  
**Valid samples:** 12,762 patient visits  
**Mean wait:** 41.8 min | **Median:** 33.0 min | **Max:** 233 min

---

## 🔬 Tasks

### Task 1 — Point Prediction
Single predicted waiting time at the moment of examination check-in.

- Models compared: Baseline, Linear Regression, Ridge, Decision Tree, Random Forest, Gradient Boosting
- Evaluation: 5-Fold Cross-Validation
- Best model: **Gradient Boosting** — RMSE = 30.4 min, MAE = 22.8 min, R² = 0.311
- Top feature: `provider_patients_so_far` (59% importance)

### Task 2 — Interval Prediction
90% prediction intervals using two methods:

- **Residual Bootstrap** — Coverage: 90.0%, Avg Width: 92 min
- **Split Conformal Prediction** — Coverage: 89.9%, Avg Width: 86.5 min ✅ preferred

### Task 3 — Quantile Prediction
Conditional quantile estimates at q10, q25, q50, q75, q90.

- Model: Gradient Boosting Quantile Regression
- Metric: Pinball Loss (per quantile)
- All quantiles well-calibrated (actual coverage within ±1.5% of target)
- Clinical use: q50 → patient-facing ETA, q90 → staffing SLA target

### Task 4 — Dynamic Prediction
Predictions updated every 5 minutes as the patient continues to wait.

- Snapshots: t = 0, 5, 10, 15, 20 minutes after check-in
- Dynamic features: queue progress, provider room status, elapsed time
- RMSE improves from 29.3 min (t=0) → 27.7 min (t=20) — **5.5% gain**

### Task 5 — Comparison & Reflection
Side-by-side evaluation of all approaches with deployment recommendation.

**Recommended system:** Task 3 (Quantile) + Task 4 (Dynamic) combined
- q50 shown to patients as expected wait
- q90 used internally for SLA monitoring
- Dynamic updates every 5 min reduce patient anxiety

---

## ⚙️ Installation

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

Then open the notebook:

```bash
jupyter notebook CIS432_Project2_WaitingTimePrediction.ipynb
```

> Place `data.xlsx` in the same directory as the notebook before running.

---

## 🔑 Key Features Engineered

| Feature | Description |
|---|---|
| `provider_patients_so_far` | How many patients the provider has seen today |
| `early_late_min` | Minutes early/late vs scheduled appointment |
| `queue_position` | Patient's position in the day's queue |
| `checkin_hour` | Hour of examination check-in |
| `bloodwork_duration` | Time spent in bloodwork phase |
| `total_pre_exam_time` | Total time from hospital arrival to exam check-in |
