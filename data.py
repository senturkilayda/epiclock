"""GSE40279 verisini indirme, ayrıştırma ve önbelleğe alma.

GEO "series matrix" dosyası iki bölümden oluşur:
1. ``!Sample_...`` ile başlayan meta veri satırları (yaş, cinsiyet, plaka vb.)
2. ``!series_matrix_table_begin`` ile ``!series_matrix_table_end`` arasındaki
   tablo: satırlar CpG, sütunlar örnek (GSM kimliği), değerler beta (0-1).

Çıktı her zaman ``X`` (örnek x CpG, float32) ve ``meta`` (örnek x değişken)
olarak döner. Satır sırası iki tabloda da aynıdır.
"""
from __future__ import annotations

import gzip
import re
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

TABLE_BEGIN = "!series_matrix_table_begin"
TABLE_END = "!series_matrix_table_end"


# --------------------------------------------------------------------------
# İndirme
# --------------------------------------------------------------------------
def download_series_matrix(url: str, dest: Path) -> Path:
    """Dosya yoksa indirir. Dosya büyük (yüzlerce MB), bir kez indirilir."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"[veri] Zaten indirilmiş: {dest}")
        return dest
    print(f"[veri] İndiriliyor: {url}")
    tmp = dest.with_suffix(dest.suffix + ".part")
    urllib.request.urlretrieve(url, tmp)
    tmp.rename(dest)
    return dest


# --------------------------------------------------------------------------
# Ayrıştırma
# --------------------------------------------------------------------------
def _clean(token: str) -> str:
    return token.strip().strip('"')


def _normalize_key(key: str) -> str:
    """'age (y)' -> 'age_y'"""
    key = re.sub(r"[^0-9a-zA-Z]+", "_", key.strip().lower())
    return key.strip("_")


def _parse_header(path: Path) -> tuple[pd.DataFrame, int]:
    """Meta veriyi okur ve tablonun başladığı satır numarasını döndürür."""
    sample_ids: list[str] = []
    characteristics: list[list[str]] = []
    titles: list[str] = []

    with gzip.open(path, "rt", encoding="utf-8", errors="replace") as fh:
        for line_no, line in enumerate(fh):
            if line.startswith(TABLE_BEGIN):
                table_start = line_no + 1
                break
            fields = [_clean(t) for t in line.rstrip("\n").split("\t")]
            tag, values = fields[0], fields[1:]
            if tag == "!Sample_geo_accession":
                sample_ids = values
            elif tag == "!Sample_title":
                titles = values
            elif tag == "!Sample_characteristics_ch1":
                characteristics.append(values)
        else:
            raise ValueError(f"Tablo başlangıcı bulunamadı: {path}")

    if not sample_ids:
        raise ValueError("!Sample_geo_accession satırı bulunamadı.")

    meta = pd.DataFrame(index=pd.Index(sample_ids, name="sample_id"))
    if titles:
        meta["title"] = titles

    # Her characteristics satırı "anahtar: değer" biçimindedir.
    for row in characteristics:
        for sid, cell in zip(sample_ids, row):
            if ":" not in cell:
                continue
            key, value = cell.split(":", 1)
            meta.loc[sid, _normalize_key(key)] = value.strip()

    return meta, table_start


def _standardize_meta(meta: pd.DataFrame) -> pd.DataFrame:
    """Sütun adlarını ortak bir biçime getirir: age, sex, plate, ethnicity."""
    meta = meta.copy()
    rename = {}
    for col in meta.columns:
        if col.startswith("age"):
            rename[col] = "age"
        elif col in {"gender", "sex"}:
            rename[col] = "sex"
        elif col.startswith("plate"):
            rename[col] = "plate"
        elif col.startswith("ethnic"):
            rename[col] = "ethnicity"
    meta = meta.rename(columns=rename)
    if "age" not in meta.columns:
        raise ValueError(f"Yaş sütunu bulunamadı. Mevcut sütunlar: {list(meta.columns)}")
    meta["age"] = pd.to_numeric(meta["age"], errors="coerce")
    return meta


def parse_series_matrix(path: Path, chunksize: int = 20_000) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """Series matrix dosyasını okur.

    Bellek: 473 bin CpG x 656 örnek, float32 ile yaklaşık 1,2 GB tutar.
    Tablo parça parça okunur, böylece pandas'ın ara kopyaları küçük kalır.

    Returns
    -------
    X : (n_samples, n_cpgs) float32
    cpg_ids : CpG kimlikleri (X'in sütun sırası)
    meta : örnek meta verisi (X'in satır sırası)
    """
    meta, table_start = _parse_header(path)

    blocks, cpg_ids = [], []
    reader = pd.read_csv(
        path, sep="\t", skiprows=table_start, index_col=0,
        chunksize=chunksize, compression="gzip", low_memory=False,
    )
    columns = None
    for chunk in reader:
        chunk = chunk[~chunk.index.astype(str).str.startswith("!")]  # tablo sonu satırı
        if columns is None:
            columns = [_clean(c) for c in chunk.columns]
        blocks.append(chunk.to_numpy(dtype=np.float32))
        cpg_ids.extend(chunk.index.astype(str))
        print(f"[veri] {len(cpg_ids):,} CpG okundu", end="\r")
    print()

    # Tablodaki örnek sırasını meta veriyle hizala.
    meta = _standardize_meta(meta.loc[columns])
    X = np.ascontiguousarray(np.vstack(blocks).T)  # örnek x CpG
    return X, cpg_ids, meta


# --------------------------------------------------------------------------
# Önbellek
# --------------------------------------------------------------------------
def save_cache(X: np.ndarray, cpg_ids: list[str], meta: pd.DataFrame, cache_dir: Path) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "X.npy", X)
    (cache_dir / "cpg_ids.txt").write_text("\n".join(cpg_ids))
    meta.to_csv(cache_dir / "meta.csv")


def load_cache(cache_dir: Path) -> tuple[np.ndarray, list[str], pd.DataFrame] | None:
    files = [cache_dir / f for f in ("X.npy", "cpg_ids.txt", "meta.csv")]
    if not all(f.exists() for f in files):
        return None
    X = np.load(files[0], mmap_mode="r")  # belleğe tamamını kopyalamadan açar
    cpg_ids = files[1].read_text().splitlines()
    meta = pd.read_csv(files[2], index_col=0)
    return np.asarray(X), cpg_ids, meta


def load_gse40279(raw_dir: Path, cache_dir: Path, url: str) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    cached = load_cache(cache_dir)
    if cached is not None:
        print("[veri] Önbellekten yüklendi.")
        return cached
    path = download_series_matrix(url, raw_dir / Path(url).name)
    X, cpg_ids, meta = parse_series_matrix(path)
    save_cache(X, cpg_ids, meta, cache_dir)
    return X, cpg_ids, meta


# --------------------------------------------------------------------------
# Sentetik veri (yalnızca kodu test etmek için)
# --------------------------------------------------------------------------
def make_synthetic_dataset(
    n_samples: int = 656, n_cpgs: int = 4_000, n_age_cpgs: int = 250,
    n_plates: int = 9, seed: int = 0,
) -> tuple[np.ndarray, list[str], pd.DataFrame]:
    """GSE40279'a benzer yapıda yapay veri üretir.

    Bu veriden çıkan sayılar ve grafikler gerçek sonuç DEĞİLDİR. Yalnızca
    kodun uçtan uca çalıştığını doğrulamak için kullanılır.
    """
    rng = np.random.default_rng(seed)
    age = np.clip(rng.normal(64, 14, n_samples), 19, 101).round()
    sex = rng.choice(["F", "M"], n_samples)
    plate = rng.integers(1, n_plates + 1, n_samples)

    base = rng.normal(0, 1.5, n_cpgs)                  # CpG'lerin ortalama logit düzeyi
    slope = np.zeros(n_cpgs)
    slope[:n_age_cpgs] = rng.normal(0, 0.12, n_age_cpgs)  # yaşla ilişkili CpG'ler
    curve = np.zeros(n_cpgs)
    curve[:60] = rng.normal(0, 0.06, 60)                  # bir kısmı doğrusal değil
    plate_shift = rng.normal(0, 0.12, (n_plates + 1, 1))
    # Kişiye özgü "biyolojik yaş" sapması: gerçek verideki gibi tahmin hatası yaratır.
    bio_age = age + rng.normal(0, 4, n_samples)

    z = (bio_age - 60) / 20
    logit = (base[None, :] + np.outer(z, slope) + np.outer(z**2, curve)
             + plate_shift[plate] + rng.normal(0, 0.35, (n_samples, n_cpgs)))
    X = (1 / (1 + np.exp(-logit))).astype(np.float32)
    X[rng.random(X.shape) < 0.001] = np.nan               # az sayıda eksik değer

    ids = [f"GSM_SYN{i:04d}" for i in range(n_samples)]
    meta = pd.DataFrame(
        {"age": age, "sex": sex, "plate": plate.astype(str)},
        index=pd.Index(ids, name="sample_id"),
    )
    cpg_ids = [f"cg{i:08d}" for i in range(n_cpgs)]
    return X, cpg_ids, meta
