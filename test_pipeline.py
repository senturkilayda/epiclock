"""Temel testler: ayrıştırma, metrikler ve veri sızıntısı koruması.

Çalıştırma: proje klasöründe ``python -m pytest -q``
"""
import gzip
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src import data, evaluate, models  # noqa: E402

FAKE_MATRIX = """!Series_title\t"Fake"
!Sample_title\t"s1"\t"s2"\t"s3"
!Sample_geo_accession\t"GSM1"\t"GSM2"\t"GSM3"
!Sample_characteristics_ch1\t"age (y): 30"\t"age (y): 55"\t"age (y): 80"
!Sample_characteristics_ch1\t"plate: 1"\t"plate: 1"\t"plate: 2"
!Sample_characteristics_ch1\t"gender: F"\t"gender: M"\t"gender: F"
!series_matrix_table_begin
"ID_REF"\t"GSM1"\t"GSM2"\t"GSM3"
"cg001"\t0.1\t0.2\t0.3
"cg002"\t0.9\t\t0.7
"cg003"\t0.5\t0.5\t0.5
!series_matrix_table_end
"""


@pytest.fixture
def fake_matrix(tmp_path):
    path = tmp_path / "fake_series_matrix.txt.gz"
    with gzip.open(path, "wt") as fh:
        fh.write(FAKE_MATRIX)
    return path


def test_parse_series_matrix(fake_matrix):
    X, cpg_ids, meta = data.parse_series_matrix(fake_matrix)
    assert X.shape == (3, 3)                 # örnek x CpG
    assert X.dtype == np.float32
    assert cpg_ids == ["cg001", "cg002", "cg003"]
    assert meta["age"].tolist() == [30, 55, 80]
    assert meta["sex"].tolist() == ["F", "M", "F"]
    assert meta["plate"].tolist() == ["1", "1", "2"]
    assert np.isnan(X[1, 1])                 # eksik değer korunuyor
    assert X[2, 0] == pytest.approx(0.3)     # transpoz doğru


def test_regression_metrics():
    m = evaluate.regression_metrics(np.array([10, 20, 30.0]), np.array([12, 18, 30.0]))
    assert m["MAE"] == pytest.approx(4 / 3)
    assert m["RMSE"] == pytest.approx(np.sqrt(8 / 3))


def test_split_is_disjoint_and_complete():
    age = np.random.default_rng(0).uniform(20, 90, 200)
    tr, te = evaluate.split_train_test(age, 0.2, 10, seed=1)
    assert len(set(tr) & set(te)) == 0
    assert len(tr) + len(te) == 200


def test_cpg_selection_uses_only_training_data():
    """Seçim skorları yalnızca eğitim satırlarından hesaplanmış olmalı.

    Testte yaşla mükemmel ilişkili ama eğitimde saf gürültü olan bir CpG var.
    Pipeline'ın skorları eğitim verisiyle elle hesaplanan skorlarla aynıysa ve
    bu CpG seçilmediyse test verisi seçime sızmamıştır.
    """
    from sklearn.feature_selection import f_regression

    rng = np.random.default_rng(0)
    n, p = 120, 50
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = rng.uniform(20, 80, n)
    X[:, 0] += 0.2 * y                    # gerçek sinyal, her yerde
    tr, te = np.arange(90), np.arange(90, 120)
    X[te, 1] = y[te]                      # yalnızca testte "mükemmel" CpG

    pipe = models.build_models(p, k=1, seed=0)["ridge"].fit(X[tr], y[tr])
    expected, _ = f_regression(X[tr], y[tr])
    np.testing.assert_allclose(models._cpg_selector(pipe).scores_, expected, rtol=1e-5)
    assert list(models.selected_cpgs(pipe, [f"c{i}" for i in range(p)])) == ["c0"]


def test_synthetic_shapes():
    X, cpg_ids, meta = data.make_synthetic_dataset(n_samples=50, n_cpgs=100, n_age_cpgs=10)
    assert X.shape == (50, 100) and len(cpg_ids) == 100 and len(meta) == 50


def test_plate_split_keeps_plates_apart():
    """Plakaya göre bölmede bir plaka hem eğitimde hem testte olamaz."""
    rng = np.random.default_rng(0)
    age = rng.uniform(20, 90, 300)
    plates = rng.integers(1, 12, 300).astype(str)
    tr, te = evaluate.split_train_test(age, 0.2, 10, seed=0, groups=plates)
    assert set(plates[tr]).isdisjoint(set(plates[te]))
    assert len(tr) + len(te) == 300

    splits = evaluate._cv_splits(age[tr, None], age[tr], 5, 10, 0, plates[tr])
    for f_tr, f_va in splits:
        assert set(plates[tr][f_tr]).isdisjoint(set(plates[tr][f_va]))


def test_sex_covariate_bypasses_cpg_selection():
    """Klinik değişken, CpG seçiminden geçmeden modele ulaşmalı."""
    rng = np.random.default_rng(1)
    n, p = 100, 30
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = rng.uniform(20, 80, n)
    X[:, 0] += 0.2 * y
    sex = rng.integers(0, 2, n).astype(np.float32)   # yaşla ilişkisiz
    X_full = np.hstack([X, sex[:, None]])

    pipe = models.build_models(p + 1, k=3, seed=0, n_covariates=1)["enet"].fit(X_full, y)
    n_inputs = pipe.named_steps["model"].coef_.shape[0]
    assert n_inputs == 3 + 1                          # 3 CpG + cinsiyet
    coefs = models.enet_coefficients(pipe, [f"c{i}" for i in range(p)], ["sex"])
    assert "c0" in coefs.index


def test_inner_selection_refits_selection_in_each_inner_fold(monkeypatch):
    """InnerSelectionCV, CpG seçimini her iç parçada ve sonda tüm veride yeniden yapmalı."""
    from sklearn.feature_selection import SelectKBest

    calls = []
    original_fit = SelectKBest.fit

    def counting_fit(self, X, y):
        calls.append(len(y))
        return original_fit(self, X, y)

    monkeypatch.setattr(SelectKBest, "fit", counting_fit)
    rng = np.random.default_rng(2)
    n, p = 100, 40
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = rng.uniform(20, 80, n)
    X[:, 0] += 0.2 * y
    est = models.InnerSelectionCV(kind="enet", n_cpgs=p, k=5, cv=5, n_alphas=5, seed=0).fit(X, y)

    assert len(calls) == 1 + 5                      # tüm eğitim + 5 iç parça
    assert sorted(calls)[:5] == [80] * 5            # her iç parça yalnızca kendi 80 örneğini gördü
    assert est.alpha_ in est.alphas_
    assert est.predict(X).shape == (n,)
