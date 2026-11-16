"""Bölme, çapraz doğrulama ve metrikler.

İki bölme türü var:
- ``random``: Kişiler yaşa göre tabakalı, rastgele bölünür.
- ``plate``: Ölçüm plakaları bütün olarak ayrılır. Test verisindeki ve her CV
  fold'unun doğrulama kısmındaki plakaları model eğitimde hiç görmez.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedKFold, train_test_split


def age_strata(age: np.ndarray, n_bins: int) -> np.ndarray:
    """Yaşı yüzdelik dilimlere böler. Bölmelerde yaş dağılımı korunur."""
    return pd.qcut(age, q=n_bins, labels=False, duplicates="drop")


def split_train_test(age: np.ndarray, test_size: float, n_bins: int, seed: int,
                     groups: np.ndarray | None = None):
    idx = np.arange(len(age))
    if groups is None:
        return train_test_split(idx, test_size=test_size, random_state=seed,
                                stratify=age_strata(age, n_bins))
    # Plakalar bütün olarak test verisine gider. test_size burada plaka oranıdır,
    # bu yüzden kişi sayısı tam %20 tutmayabilir; çıktıda gerçek sayılar yazılır.
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    tr, te = next(gss.split(idx, age, groups))
    return tr, te


def regression_metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        "r": float(np.corrcoef(y_true, y_pred)[0, 1]) if np.std(y_pred) > 0 else np.nan,
    }


def _cv_splits(X, y, n_folds, n_bins, seed, groups):
    if groups is None:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        return list(cv.split(X, age_strata(y, n_bins)))
    n_folds = min(n_folds, len(np.unique(groups)))
    return list(GroupKFold(n_splits=n_folds).split(X, y, groups))


def cross_validate(models: dict, X, y, n_folds: int, n_bins: int, seed: int,
                   groups: np.ndarray | None = None):
    """Eğitim verisi içinde fold dışı (out-of-fold) tahminler üretir.

    Her fold'da model sıfırdan kurulur (``clone``). İmputasyon, CpG seçimi,
    ölçekleme ve ElasticNetCV'nin iç CV'si yalnızca o fold'un eğitim kısmını görür.
    """
    splits = _cv_splits(X, y, n_folds, n_bins, seed, groups)
    oof = pd.DataFrame(index=np.arange(len(y)), columns=list(models), dtype=float)
    rows = []
    for name, model in models.items():
        t0 = time.time()
        for fold, (tr, va) in enumerate(splits):
            m = clone(model).fit(X[tr], y[tr])
            pred = m.predict(X[va])
            oof.loc[va, name] = pred
            rows.append({"model": name, "fold": fold, **regression_metrics(y[va], pred)})
        print(f"[cv] {name:9s} tamam ({time.time() - t0:.0f} sn)")
    return oof, pd.DataFrame(rows)


def fit_and_test(models: dict, X_tr, y_tr, X_te, y_te):
    """Her modeli tüm eğitim verisiyle kurar, test verisinde bir kez değerlendirir."""
    fitted, preds, rows = {}, {}, []
    for name, model in models.items():
        m = clone(model).fit(X_tr, y_tr)
        fitted[name] = m
        preds[name] = m.predict(X_te)
        rows.append({"model": name, **regression_metrics(y_te, preds[name])})
    return fitted, pd.DataFrame(preds), pd.DataFrame(rows)
