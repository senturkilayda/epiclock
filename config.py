"""Proje genelinde kullanılan sabitler.

Tüm yollar ve rastgelelik tohumu tek yerde durur; böylece sonuçlar yeniden üretilebilir.
"""
from pathlib import Path

SEED = 42

ROOT = Path(__file__).resolve().parent
RAW_DIR = ROOT / "data" / "raw"
CACHE_DIR = ROOT / "data" / "cache"
FIG_DIR = ROOT / "figures"
RESULTS_DIR = ROOT / "results"

GEO_ID = "GSE40279"
GEO_URL = (
    "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE40nnn/GSE40279/matrix/"
    "GSE40279_series_matrix.txt.gz"
)

# Değerlendirme ayarları
TEST_SIZE = 0.20          # Dokunulmayan test verisinin oranı
N_OUTER_FOLDS = 5         # Model karşılaştırması için eğitim verisi içindeki CV
N_AGE_BINS = 10           # Bölmeleri yaşa göre tabakalandırmak için

# Fold içinde, yalnızca eğitim verisiyle yapılan tek değişkenli CpG ön seçimi.
# 450 binden fazla değişkenin hepsiyle Elastic Net yavaş çalışır; bu adım
# Pipeline içinde olduğu için test verisini görmez (veri sızıntısı yok).
N_SELECTED_CPGS = 20_000

# Elastic Net'in en fazla kaç iterasyon yapabileceği. İlk çalıştırmada 5000
# bazı ceza değerlerinde yetmedi (ConvergenceWarning).
ENET_MAX_ITER = 20_000
