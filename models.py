"""Karşılaştırılacak modeller.

Her model bir sklearn ``Pipeline``. Eksik değer doldurma, CpG seçimi ve
ölçekleme Pipeline'ın içinde olduğu için ``fit`` yalnızca eğitim verisini görür.
Çapraz doğrulamada da her fold kendi seçimini yeniden yapar. Veri sızıntısını
önleyen şey bu yapıdır.

Klinik değişkenler (ör. cinsiyet) kullanılıyorsa X'in SON sütunlarına eklenir.
``ColumnTransformer`` bu sütunları CpG seçiminden geçirmeden modele taşır;
böylece tek değişkenli seçim cinsiyeti "yaşla ilişkisiz" diye atamaz.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV, enet_path
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_LABELS = {
    "baseline": "Ortalama (taban çizgisi)",
    "ridge": "Ridge",
    "enet": "Elastic Net",
    "xgb": "XGBoost",
}


def _n_alphas_kwarg(n: int) -> dict:
    """sklearn 1.7+ 'alphas=int' kullanır; eski sürümler 'n_alphas' bekler."""
    params = inspect.signature(ElasticNetCV.__init__).parameters
    return {"n_alphas": n} if "n_alphas" in params else {"alphas": n}


def _features(n_cpgs: int, n_covariates: int, k: int, scale: bool):
    """Özellik hazırlama adımı: CpG'ler seçilir, klinik değişkenler olduğu gibi geçer."""
    cpg_steps = [
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectKBest(score_func=f_regression, k=min(k, n_cpgs))),
    ]
    if scale:
        cpg_steps.append(("scale", StandardScaler()))
    cpg_pipe = Pipeline(cpg_steps)
    if n_covariates == 0:
        return cpg_pipe

    cov_steps = [("impute", SimpleImputer(strategy="most_frequent"))]
    if scale:
        cov_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("cpg", cpg_pipe, slice(0, n_cpgs)),
        ("cov", Pipeline(cov_steps), slice(n_cpgs, n_cpgs + n_covariates)),
    ])


class InnerSelectionCV(RegressorMixin, BaseEstimator):
    """Hiperparametre seçimini, CpG seçimini iç CV'nin HER parçasında yeniden yaparak yapar.

    ElasticNetCV/RidgeCV, kendilerine verilen (önceden seçilmiş) CpG'lerle iç CV yapar.
    CpG seçimi tüm eğitim verisine bakılarak yapıldıysa, iç doğrulama parçaları seçimi
    dolaylı olarak etkilemiş olur ve ceza değeri seçimi iyimserleşir. Bu sınıf her iç
    parçada imputasyon + CpG seçimi + ölçeklemeyi yalnızca o parçanın eğitim kısmıyla
    yeniden öğrenir, sonra ceza değerlerini dener.

    kind="enet": l1_ratio x alpha ızgarası (alpha ızgarası ElasticNetCV ile aynı mantıkla).
    kind="ridge": alpha ızgarası.
    """

    def __init__(self, kind="enet", n_cpgs=0, n_covariates=0, k=5000,
                 l1_ratios=(0.1, 0.5, 0.9), n_alphas=30, ridge_alphas=None,
                 cv=5, max_iter=20_000, seed=0):
        self.kind = kind
        self.n_cpgs = n_cpgs
        self.n_covariates = n_covariates
        self.k = k
        self.l1_ratios = l1_ratios
        self.n_alphas = n_alphas
        self.ridge_alphas = ridge_alphas
        self.cv = cv
        self.max_iter = max_iter
        self.seed = seed

    def _features(self):
        return _features(self.n_cpgs, self.n_covariates, self.k, scale=True)

    def _enet_grid(self, Xs, y, l1):
        # ElasticNetCV ile aynı: en büyük alpha tüm katsayıları sıfırlayan değer.
        alpha_max = np.max(np.abs(Xs.T @ (y - y.mean()))) / (len(y) * l1)
        return np.logspace(np.log10(alpha_max), np.log10(alpha_max * 1e-3), self.n_alphas)

    def fit(self, X, y):
        y = np.asarray(y, dtype=float)
        feats = clone(self._features()).fit(X, y)
        Xf = np.asarray(feats.transform(X), dtype=np.float64)
        if self.kind == "enet":
            grids = {l1: self._enet_grid(Xf, y, l1) for l1 in self.l1_ratios}
            cands = [(l1, a) for l1 in self.l1_ratios for a in grids[l1]]
        else:
            alphas = self.ridge_alphas if self.ridge_alphas is not None else np.logspace(-1, 5, 25)
            cands = [(None, a) for a in alphas]

        err = np.zeros(len(cands))
        for tr, va in KFold(self.cv, shuffle=True, random_state=self.seed).split(X):
            f = clone(self._features()).fit(X[tr], y[tr])      # seçim yalnızca iç eğitim kısmıyla
            Xtr = np.asarray(f.transform(X[tr]), dtype=np.float64)
            Xva = np.asarray(f.transform(X[va]), dtype=np.float64)
            mu = y[tr].mean()
            j = 0
            if self.kind == "enet":
                for l1 in self.l1_ratios:
                    _, coefs, _ = enet_path(Xtr, y[tr] - mu, l1_ratio=l1, alphas=grids[l1],
                                            max_iter=self.max_iter)
                    pred = Xva @ coefs + mu                       # (n_va, n_alphas)
                    err[j:j + len(grids[l1])] += np.abs(pred - y[va][:, None]).mean(axis=0)
                    j += len(grids[l1])
            else:
                for _, a in cands:
                    pred = Ridge(alpha=a).fit(Xtr, y[tr]).predict(Xva)
                    err[j] += np.abs(pred - y[va]).mean()
                    j += 1

        best = int(np.argmin(err))
        self.l1_ratio_, self.alpha_ = cands[best]
        self.cv_mae_ = err / self.cv
        if self.kind == "enet":
            self.alphas_ = grids[self.l1_ratio_]
            self.model_ = ElasticNet(alpha=self.alpha_, l1_ratio=self.l1_ratio_,
                                     max_iter=self.max_iter).fit(Xf, y)
        else:
            self.alphas_ = np.array([c[1] for c in cands])
            self.model_ = Ridge(alpha=self.alpha_).fit(Xf, y)
        self.features_ = feats
        self.coef_ = self.model_.coef_
        return self

    def predict(self, X):
        return self.model_.predict(np.asarray(self.features_.transform(X), dtype=np.float64))


def build_models(n_features: int, k: int, seed: int, n_covariates: int = 0,
                 enet_max_iter: int = 20_000, inner_selection: bool = False) -> dict[str, Pipeline]:
    """n_features: X'in toplam sütun sayısı (CpG + klinik değişkenler).

    inner_selection=True: Ridge ve Elastic Net'in ceza değeri, CpG seçimini iç CV'nin
    her parçasında yeniden yapan InnerSelectionCV ile seçilir (daha yavaş, daha temiz).
    """
    n_cpgs = n_features - n_covariates
    models: dict[str, Pipeline] = {
        "baseline": Pipeline([("model", DummyRegressor(strategy="mean"))]),
        "ridge": Pipeline([
            ("features", _features(n_cpgs, n_covariates, k, scale=True)),
            ("model", RidgeCV(alphas=np.logspace(-1, 5, 25))),
        ]),
        "enet": Pipeline([
            ("features", _features(n_cpgs, n_covariates, k, scale=True)),
            ("model", ElasticNetCV(
                l1_ratio=[0.1, 0.5, 0.9], cv=5,
                # Varsayılan 1000-5000 iterasyon bazı ceza değerlerinde yetmedi
                # (ConvergenceWarning). Daha uzun yürümesine izin veriyoruz.
                max_iter=enet_max_iter, n_jobs=-1, random_state=seed,
                **_n_alphas_kwarg(30),
            )),
        ]),
    }
    if inner_selection:
        common = dict(n_cpgs=n_cpgs, n_covariates=n_covariates, k=k, seed=seed)
        models["ridge"] = Pipeline([("model", InnerSelectionCV(kind="ridge", **common))])
        models["enet"] = Pipeline([("model", InnerSelectionCV(
            kind="enet", max_iter=enet_max_iter, **common))])
    try:
        from xgboost import XGBRegressor
    except ImportError:
        print("[model] xgboost kurulu değil; XGBoost atlanıyor.")
    else:
        models["xgb"] = Pipeline([
            ("features", _features(n_cpgs, n_covariates, k, scale=False)),
            ("model", XGBRegressor(
                n_estimators=600, learning_rate=0.03, max_depth=3,
                subsample=0.8, colsample_bytree=0.3, min_child_weight=3,
                tree_method="hist", n_jobs=-1, random_state=seed,
            )),
        ])
    return models


def _cpg_selector(pipeline: Pipeline):
    if "features" in pipeline.named_steps:
        feats = pipeline.named_steps["features"]
    else:                                   # InnerSelectionCV kendi özellik adımını taşır
        feats = pipeline.named_steps["model"].features_
    if isinstance(feats, ColumnTransformer):
        feats = feats.named_transformers_["cpg"]
    return feats.named_steps["select"]


def selected_cpgs(pipeline: Pipeline, cpg_ids: list[str]) -> np.ndarray:
    """Pipeline'daki SelectKBest'in seçtiği CpG adları."""
    mask = _cpg_selector(pipeline).get_support()
    return np.asarray(cpg_ids)[mask]


def enet_coefficients(pipeline: Pipeline, cpg_ids: list[str],
                      covariate_names: list[str] | None = None) -> pd.Series:
    """Elastic Net'in sıfır olmayan katsayıları, mutlak değere göre büyükten küçüğe.

    Klinik değişkenlerin katsayıları da (varsa) listede yer alır.
    """
    names = list(selected_cpgs(pipeline, cpg_ids)) + list(covariate_names or [])
    coef = pipeline.named_steps["model"].coef_
    s = pd.Series(coef, index=names)
    s = s[s != 0]
    return s.reindex(s.abs().sort_values(ascending=False).index)
