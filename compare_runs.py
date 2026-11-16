"""results/ altındaki tüm çalıştırmaları tek tabloda karşılaştırır.

Kullanım: python compare_runs.py
"""
import json

import pandas as pd

import config

rows = []
for path in sorted(config.RESULTS_DIR.glob("*/summary.json")):
    s = json.loads(path.read_text(encoding="utf-8"))
    for model, mae in s["cv_mae"].items():
        rows.append({
            "çalıştırma": path.parent.name,
            "model": model,
            "CV MAE": mae,
            "test MAE": s["test"][model]["MAE"],
            "test r": s["test"][model]["r"],
            "n test": s["n_test"],
        })

if not rows:
    print("results/ altında summary.json bulunamadı. Önce run_pipeline.py çalıştır.")
else:
    df = pd.DataFrame(rows)
    df = df[df["model"] != "baseline"].sort_values(["model", "çalıştırma"])
    pd.set_option("display.width", 120)
    print(df.round(2).to_string(index=False))
