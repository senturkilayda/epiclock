"""Yazıda kullanılan bütün sayıları kaydedilmiş sonuçlardan yeniden hesaplar.

Amaç: Yazıdaki her sayının hangi dosyadan geldiği belli olsun ve elle
kopyalarken hata yapılmasın. Model yeniden eğitilmez.

Kullanım:
    python report_numbers.py --run gse40279
    python report_numbers.py --run gse40279_plate_sex
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from scipy import stats

import config
from src import data, evaluate


def line(label: str, value) -> None:
    print(f"  {label:<46} {value}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gse40279")
    args = p.parse_args()
    res = config.RESULTS_DIR / args.run
    s = json.loads((res / "summary.json").read_text(encoding="utf-8"))
    oof = pd.read_csv(res / "oof_predictions.csv")
    test = pd.read_csv(res / "test_predictions.csv")
    folds = pd.read_csv(res / "cv_fold_metrics.csv")
    best = s["best_model_by_cv"]

    if s["synthetic"]:
        X, cpg_ids, meta = data.make_synthetic_dataset(seed=config.SEED)
    else:
        X, cpg_ids, meta = data.load_cache(config.CACHE_DIR)
    meta.index = meta.index.astype(str)
    row_of = {sid: i for i, sid in enumerate(meta.index)}
    tr_rows = np.array([row_of[str(x)] for x in oof["sample_id"]])
    te_ids = test["sample_id"].astype(str)

    print(f"\n=== {args.run} ===")
    print("\n[Veri ve bölme]")
    line("eğitim / test kişi sayısı", f"{len(oof)} / {len(test)}")
    if "plate" in meta.columns:
        line("test plakaları", sorted(meta.loc[te_ids, "plate"].astype(str).unique()))

    print("\n[Hatalar] (yıl)")
    cv = folds.groupby("model")[["MAE", "RMSE"]].mean()
    cv_sd = folds.groupby("model")["MAE"].std()
    for m in [c for c in test.columns if c not in ("sample_id", "age")]:
        t = evaluate.regression_metrics(test["age"], test[m])
        line(f"{m}: CV MAE (± fold SD) | test MAE, RMSE, r",
             f"{cv.loc[m, 'MAE']:.2f} (±{cv_sd[m]:.2f}) | {t['MAE']:.2f}, {t['RMSE']:.2f}, {t['r']:.2f}")
    line("CV'ye göre seçilen model", best)

    y, pred = oof["age"].to_numpy(), oof[best].to_numpy()
    resid = pred - y
    print(f"\n[Fold dışı hatalar, {best}]")
    line("eğim (hata ~ yaş), yıl/yıl", f"{np.polyfit(y, resid, 1)[0]:+.3f}")
    for lo, hi in [(35, 45), (45, 55), (75, 85), (85, 110)]:
        sel = (y >= lo) & (y < hi)
        line(f"ortalama hata, {lo}-{hi} yaş (n={sel.sum()})", f"{resid[sel].mean():+.2f}")
    line("tahmini gerçek yaşından ≥8 yıl genç (%)", f"{np.mean(resid <= -8) * 100:.1f}")
    line("|hata| > 20 yıl olan kişi sayısı", int(np.sum(np.abs(resid) > 20)))

    coef_path = res / "enet_coefficients.csv"
    if coef_path.exists():
        coefs = pd.read_csv(coef_path, index_col=0)["coef"]
        cov = [c for c in coefs.index if c not in set(cpg_ids)]
        cpg = coefs.drop(labels=cov)
        print("\n[Elastic Net]")
        line("sıfır olmayan CpG katsayısı", len(cpg))
        if s.get("enet_alpha") is not None:
            line("seçilen alpha / l1_ratio", f"{s['enet_alpha']:.4g} / {s['enet_l1_ratio']}")
        for c in cov:
            line(f"klinik değişken: {c}", f"{coefs[c]:+.3f}")
        col_of = {c: i for i, c in enumerate(cpg_ids)}
        print("  en büyük 6 katsayı (katsayı işareti | yaşla Spearman ρ, eğitim verisi):")
        for name, c in cpg.head(6).items():
            rho = stats.spearmanr(y, X[tr_rows, col_of[name]], nan_policy="omit")[0]
            flag = "  <- işaretler ters" if np.sign(rho) != np.sign(c) else ""
            print(f"    {name}  {'+' if c > 0 else '−'} | ρ = {rho:+.2f}{flag}")

    if "plate" in meta.columns:
        plate = meta["plate"].iloc[tr_rows].astype(str).to_numpy()
        slope, icpt = np.polyfit(y, resid, 1)
        adj = resid - (slope * y + icpt)
        df = pd.DataFrame({"plate": plate, "raw": resid, "adj": adj, "age": y})
        g = df.groupby("plate")
        print("\n[Plakalar] (eğitim verisi, fold dışı hatalar)")
        print(pd.DataFrame({
            "n": g.size(), "medyan yaş": g["age"].median().round(0),
            "ham medyan hata": g["raw"].median().round(1),
            "yaşa göre düzeltilmiş": g["adj"].median().round(1),
        }).to_string())
        for col, name in [("raw", "ham"), ("adj", "yaşa göre düzeltilmiş")]:
            h, pv = stats.kruskal(*[grp[col].to_numpy() for _, grp in g])
            spread = g[col].median().max() - g[col].median().min()
            line(f"{name}: Kruskal-Wallis H, p | medyan aralığı", f"{h:.1f}, {pv:.2g} | {spread:.1f} yıl")
        if "ethnicity" in meta.columns:
            print("\n[Plaka x etnik köken] (plakalar örnek gruplarıyla örtüşüyor mu?)")
            print(pd.crosstab(meta["plate"].astype(str), meta["ethnicity"]).to_string())


if __name__ == "__main__":
    main()
