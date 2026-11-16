"""Uçtan uca akış: veri -> bölme -> CV -> test -> grafikler.

Kullanım
--------
    python run_pipeline.py --synthetic             # kodu hızlıca denemek için
    python run_pipeline.py --k 5000                # gerçek veri, rastgele bölme
    python run_pipeline.py --k 5000 --split plate  # plakalar bütün olarak ayrılır
    python run_pipeline.py --k 5000 --sex          # cinsiyet modele eklenir
    python run_pipeline.py --k 5000 --split plate --sex
    python run_pipeline.py --k 5000 --inner-selection   # iç CV'de de CpG seçimi

Her ayar kombinasyonu kendi klasörüne yazar (ör. results/gse40279_plate_sex),
böylece çalıştırmalar birbirinin üzerine yazılmaz. Karşılaştırmak için:
    python compare_runs.py
"""
from __future__ import annotations

import argparse
import json

import numpy as np

import config
from src import data, evaluate, models, plots


def parse_args():
    p = argparse.ArgumentParser(description="GSE40279 epigenetik yaş tahmini")
    p.add_argument("--synthetic", action="store_true", help="Sentetik test verisi kullan")
    p.add_argument("--k", type=int, default=config.N_SELECTED_CPGS,
                   help="Fold içinde seçilecek CpG sayısı")
    p.add_argument("--split", choices=["random", "plate"], default="random",
                   help="random: yaşa göre tabakalı; plate: plakalar bütün olarak ayrılır")
    p.add_argument("--sex", action="store_true", help="Cinsiyeti klinik değişken olarak ekle")
    p.add_argument("--inner-selection", action="store_true",
                   help="CpG seçimini hiperparametre aramasındaki iç CV'nin her parçasında yeniden yap")
    return p.parse_args()


def run_name(args) -> str:
    name = "synthetic" if args.synthetic else "gse40279"
    if args.split == "plate":
        name += "_plate"
    if args.sex:
        name += "_sex"
    if args.inner_selection:
        name += "_inner"
    return name


def add_sex_column(X: np.ndarray, meta) -> np.ndarray:
    """Cinsiyeti 0/1 olarak X'in son sütununa ekler (kadın = 0, erkek = 1)."""
    sex = meta["sex"].astype(str).str.strip().str.upper().str[0].map({"F": 0.0, "M": 1.0})
    print(f"[veri] cinsiyet: {int((sex == 0).sum())} kadın, {int((sex == 1).sum())} erkek, "
          f"{int(sex.isna().sum())} bilinmiyor")
    # Not: bu adım X'in bir kopyasını oluşturur (yaklaşık 1,2 GB ek bellek).
    return np.hstack([X, sex.to_numpy(dtype=np.float32)[:, None]])


def main() -> None:
    args = parse_args()
    np.random.seed(config.SEED)
    plots.apply_style()
    name = run_name(args)
    fig_dir, res_dir = config.FIG_DIR / name, config.RESULTS_DIR / name
    res_dir.mkdir(parents=True, exist_ok=True)
    print(f"[ayar] çalıştırma: {name}")

    # 1) Veri ---------------------------------------------------------------
    if args.synthetic:
        X, cpg_ids, meta = data.make_synthetic_dataset(seed=config.SEED)
    else:
        X, cpg_ids, meta = data.load_gse40279(config.RAW_DIR, config.CACHE_DIR, config.GEO_URL)
    keep = meta["age"].notna().to_numpy()
    X, meta = X[keep], meta.loc[keep]
    y = meta["age"].to_numpy(dtype=float)
    print(f"[veri] {X.shape[0]} örnek, {X.shape[1]:,} CpG, eksik oranı {np.isnan(X).mean():.4%}")

    covariates: list[str] = []
    if args.sex:
        X = add_sex_column(X, meta)
        covariates = ["cinsiyet (erkek=1)"]
    plots.plot_age_distribution(y, fig_dir / "01_yas_dagilimi.png", args.synthetic)

    groups = meta["plate"].astype(str).to_numpy() if args.split == "plate" else None

    # 2) Bölme: test verisi burada ayrılır ve en sona kadar kullanılmaz ------
    idx_tr, idx_te = evaluate.split_train_test(
        y, config.TEST_SIZE, config.N_AGE_BINS, config.SEED, groups)
    X_tr, X_te, y_tr, y_te = X[idx_tr], X[idx_te], y[idx_tr], y[idx_te]
    g_tr = groups[idx_tr] if groups is not None else None
    print(f"[bölme] eğitim {len(idx_tr)}, test {len(idx_te)}")
    if groups is not None:
        print(f"[bölme] test plakaları: {sorted(set(groups[idx_te]))} "
              f"(eğitimde yok: {set(groups[idx_te]).isdisjoint(g_tr)})")

    # 3) Eğitim verisi içinde çapraz doğrulama -------------------------------
    model_set = models.build_models(X.shape[1], args.k, config.SEED,
                                    n_covariates=len(covariates),
                                    enet_max_iter=config.ENET_MAX_ITER,
                                    inner_selection=args.inner_selection)
    oof, fold_metrics = evaluate.cross_validate(
        model_set, X_tr, y_tr, config.N_OUTER_FOLDS, config.N_AGE_BINS, config.SEED, g_tr)
    fold_metrics.to_csv(res_dir / "cv_fold_metrics.csv", index=False)

    # 4) Tek seferlik test değerlendirmesi -----------------------------------
    fitted, test_pred, test_metrics = evaluate.fit_and_test(model_set, X_tr, y_tr, X_te, y_te)
    test_metrics.to_csv(res_dir / "test_metrics.csv", index=False)
    # Tahminleri sakla: yazıdaki her sayı bu dosyalardan yeniden hesaplanabilsin.
    oof.assign(sample_id=meta.index[idx_tr], age=y_tr).to_csv(res_dir / "oof_predictions.csv", index=False)
    test_pred.assign(sample_id=meta.index[idx_te], age=y_te).to_csv(res_dir / "test_predictions.csv", index=False)

    cv_mae = fold_metrics.groupby("model")["MAE"].mean()
    best = cv_mae.drop("baseline").idxmin()   # seçim CV'ye göre, test'e göre değil
    label = models.MODEL_LABELS[best]
    print("\nCV ortalaması:\n", fold_metrics.groupby("model")[["MAE", "RMSE", "R2"]].mean().round(2))
    print("\nTest:\n", test_metrics.round(2).to_string(index=False))
    print(f"\n[seçim] CV'ye göre en iyi model: {label}")

    # 5) Grafikler -------------------------------------------------------------
    plots.plot_pred_vs_true(y_te, test_pred[best].to_numpy(), label,
                            fig_dir / "02_tahmin_vs_gercek.png", args.synthetic, args.split)
    plots.plot_model_comparison(fold_metrics, test_metrics, models.MODEL_LABELS,
                                fig_dir / "03_model_karsilastirma.png", args.synthetic, args.split)
    resid_oof = oof[best].to_numpy() - y_tr
    plots.plot_residual_vs_age(y_tr, oof[best].to_numpy(), label,
                               fig_dir / "04_yasa_bagli_sapma.png", args.synthetic)
    if "enet" in fitted:
        coefs = models.enet_coefficients(fitted["enet"], cpg_ids, covariates)
        coefs.to_csv(res_dir / "enet_coefficients.csv", header=["coef"])
        enet = fitted["enet"].named_steps["model"]
        print(f"[enet] sıfır olmayan katsayı: {len(coefs)} · seçilen alpha = {enet.alpha_:.4g}, "
              f"l1_ratio = {enet.l1_ratio_} · denenen en küçük alpha = {np.min(enet.alphas_):.4g}")
        for cov in covariates:
            if cov in coefs.index:
                print(f"[enet] {cov}: katsayı {coefs[cov]:+.3f} (ölçeklenmiş)")
            else:
                print(f"[enet] {cov}: model katsayıyı sıfırladı")
        cpg_coefs = coefs.drop(labels=[c for c in covariates if c in coefs.index])
        plots.plot_top_cpgs(X_tr, y_tr, cpg_ids, cpg_coefs,
                            fig_dir / "05_en_etkili_cpgler.png", args.synthetic)
    plots.plot_hypothetical_shift(resid_oof, fig_dir / "06_varsayimsal_ornek.png", args.synthetic)
    if "plate" in meta.columns and meta["plate"].iloc[idx_tr].nunique() > 1:
        plots.plot_plate_effect(resid_oof, meta["plate"].iloc[idx_tr],
                                fig_dir / "07_plaka_etkisi.png", args.synthetic, age=y_tr)

    summary = {
        "run": name, "synthetic": args.synthetic, "split": args.split, "sex": args.sex,
        "inner_selection": args.inner_selection,
        "n_train": int(len(idx_tr)), "n_test": int(len(idx_te)),
        "k_selected_cpgs": args.k, "enet_max_iter": config.ENET_MAX_ITER,
        "best_model_by_cv": best,
        "enet_alpha": float(fitted["enet"].named_steps["model"].alpha_) if "enet" in fitted else None,
        "enet_l1_ratio": float(fitted["enet"].named_steps["model"].l1_ratio_) if "enet" in fitted else None,
        "cv_mae": cv_mae.round(3).to_dict(),
        "test": test_metrics.set_index("model").round(3).to_dict(orient="index"),
    }
    (res_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"\n[bitti] Grafikler: {fig_dir}  Sonuçlar: {res_dir}")


if __name__ == "__main__":
    main()
