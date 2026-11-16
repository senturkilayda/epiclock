"""Kaydedilmiş sonuçlardan grafikleri yeniden üretir. Model yeniden EĞİTİLMEZ.

Grafik kodunu değiştirdiğinde 20 dakikalık eğitimi tekrarlamak yerine bunu çalıştır:
    python make_figures.py --run gse40279
    python make_figures.py --run gse40279_plate_sex

Gerekenler: results/<run>/ altında summary.json, oof_predictions.csv,
test_predictions.csv, cv_fold_metrics.csv, enet_coefficients.csv.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

import config
from src import data, evaluate, models, plots


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="gse40279", help="results/ altındaki klasör adı")
    args = p.parse_args()

    res_dir, fig_dir = config.RESULTS_DIR / args.run, config.FIG_DIR / args.run
    needed = ["summary.json", "oof_predictions.csv", "test_predictions.csv", "cv_fold_metrics.csv"]
    missing = [f for f in needed if not (res_dir / f).exists()]
    if missing:
        raise SystemExit(f"{res_dir} içinde eksik dosya: {missing}. "
                         "Bu çalıştırmayı güncel run_pipeline.py ile tekrar yap.")

    s = json.loads((res_dir / "summary.json").read_text(encoding="utf-8"))
    synthetic, split, best = s["synthetic"], s.get("split", "random"), s["best_model_by_cv"]
    label = models.MODEL_LABELS[best]
    plots.apply_style()

    oof = pd.read_csv(res_dir / "oof_predictions.csv")
    test = pd.read_csv(res_dir / "test_predictions.csv")
    folds = pd.read_csv(res_dir / "cv_fold_metrics.csv")
    model_names = [c for c in test.columns if c not in ("sample_id", "age")]
    test_metrics = pd.DataFrame([
        {"model": m, **evaluate.regression_metrics(test["age"], test[m])} for m in model_names])

    if synthetic:
        X, cpg_ids, meta = data.make_synthetic_dataset(seed=config.SEED)
    else:
        cached = data.load_cache(config.CACHE_DIR)
        if cached is None:
            raise SystemExit("data/cache bulunamadı. Önce run_pipeline.py çalıştır.")
        X, cpg_ids, meta = cached
    row_of = {sid: i for i, sid in enumerate(meta.index.astype(str))}
    tr_rows = np.array([row_of[str(sid)] for sid in oof["sample_id"]])
    y_tr, y_te = oof["age"].to_numpy(), test["age"].to_numpy()
    resid = oof[best].to_numpy() - y_tr

    plots.plot_age_distribution(meta["age"].dropna().to_numpy(), fig_dir / "01_yas_dagilimi.png", synthetic)
    plots.plot_pred_vs_true(y_te, test[best].to_numpy(), label,
                            fig_dir / "02_tahmin_vs_gercek.png", synthetic, split)
    plots.plot_model_comparison(folds, test_metrics, models.MODEL_LABELS,
                                fig_dir / "03_model_karsilastirma.png", synthetic, split)
    plots.plot_residual_vs_age(y_tr, oof[best].to_numpy(), label,
                               fig_dir / "04_yasa_bagli_sapma.png", synthetic)
    coef_path = res_dir / "enet_coefficients.csv"
    if coef_path.exists():
        coefs = pd.read_csv(coef_path, index_col=0)["coef"]
        coefs = coefs[coefs.index.isin(set(cpg_ids))]          # klinik değişkenleri çıkar
        plots.plot_top_cpgs(X[tr_rows], y_tr, cpg_ids, coefs,
                            fig_dir / "05_en_etkili_cpgler.png", synthetic)
    plots.plot_hypothetical_shift(resid, fig_dir / "06_varsayimsal_ornek.png", synthetic)
    if "plate" in meta.columns:
        plots.plot_plate_effect(resid, meta["plate"].iloc[tr_rows],
                                fig_dir / "07_plaka_etkisi.png", synthetic, age=y_tr)
    print(f"[bitti] {fig_dir}")


if __name__ == "__main__":
    main()
