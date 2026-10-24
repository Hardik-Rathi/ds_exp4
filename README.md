EXP4 — Time Series Analysis and Failure Prediction
Dataset: Sub_Division_IMD_2017.csv — IMD subdivision-wise rainfall, 1901–2017 (same dataset as EXP1–EXP3).

Use case: KJS-AGR-01 — AI Irrigation Advisory System

Mapping used: timestamp = YEAR, sensor measurements = rainfall columns (JF, MAM, JJAS, OND, ANNUAL), failure = a year with ANNUAL rainfall more than 20% below the subdivision's average.

What's covered
Timestamp conversion and sorting, missing-value handling, time-series plots, 5/10-year moving averages, rolling standard deviation, anomaly detection, time-series features (lag, moving average, rolling std, diff, trend), time-based train/test split (train up to 1995, test 1996–2017), Random Forest failure prediction and evaluation.

Files
exp4_time_series.py — standalone script
EXP4_Time_Series_16014324020.ipynb — notebook with output
EXP4_notebook.pdf — notebook as PDF
Sub_Division_IMD_2017.csv — dataset
Run
pip install pandas numpy matplotlib scipy scikit-learn
python exp4_time_series.py
Key result
Random Forest (predicting next-year deficient rainfall): accuracy 0.688, precision 0.36, recall 0.48, F1 0.41, ROC-AUC 0.63 — modest skill above random guessing.
