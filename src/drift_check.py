import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset

from src.preprocessing import load_raw_data, clean_data

df = load_raw_data("data/telco_churn.csv")
df = clean_data(df)

split = len(df) // 2
reference = df.iloc[:split]
current = df.iloc[split:]

report = Report(metrics=[DataDriftPreset()])
result = report.run(reference_data=reference, current_data=current)
result.save_html("docs/drift_report.html")
print("Drift report saved to docs/drift_report.html")