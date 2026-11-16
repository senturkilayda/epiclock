# epiclock — GSE40279 ile epigenetik yaş tahmini

Hannum ve ark. (2013) verisiyle (GEO **GSE40279**, 656 kan örneği, 19–101 yaş) DNA
metilasyonundan kronolojik yaş tahmini. Projenin amacı, birinci nesil bir epigenetik saatin
nasıl kurulduğunu ve nerelerde yanılabileceğini sızıntısız bir değerlendirmeyle göstermek.

İlgili Medium yazısının bağlantısı yayımlandığında buraya eklenecek.

**English summary.** Chronological age prediction from blood DNA methylation (GSE40279, n = 656)
with Ridge, Elastic Net and XGBoost. CpG selection, imputation and scaling are fitted inside
cross-validation folds to avoid leakage. With a random split, Ridge and Elastic Net reach a test
MAE of about 3.2–3.4 years. When whole measurement plates are held out, error increases
(Elastic Net CV MAE 3.30 → 3.99 years) and plate-level differences in age-adjusted error remain.

## Sonuçlar

Rastgele bölme (eğitim 524, test 132, yaşa göre tabakalı), `--k 5000`:

| Model | CV MAE | Test MAE | Test RMSE | Test r |
|---|---|---|---|---|
| Ortalama (taban çizgisi) | 12,00 | 11,56 | 13,93 | — |
| Ridge | 3,29 | 3,23 | 4,33 | 0,95 |
| Elastic Net | 3,30 | 3,36 | 4,51 | 0,95 |
| XGBoost (sabit ayarlar) | 3,95 | 4,15 | 5,43 | 0,92 |

Plakaya göre bölme (test plakaları 8 ve 10; eğitim 558, test 98), cinsiyet eklenmiş, `--k 5000`:

| Model | CV MAE | Test MAE | Test RMSE |
|---|---|---|---|
| Ridge | 4,22 | 3,95 | 5,56 |
| Elastic Net | 3,99 | 3,78 | 5,28 |
| XGBoost (sabit ayarlar) | 4,66 | 5,30 | 6,70 |

Karşılaştırma için: Hannum ve ark. RMSE olarak birincil grupta 3,88, bağımsız doğrulama
grubunda 4,89 yıl raporlamıştır. Kurulumlar birebir aynı değildir (aşağıdaki sınırlamalara bakın).

Yazıdaki sayılar bu kaynak kod arşivine dahil olmayan, yerel PyCharm çalıştırmalarından raporlanmıştır. Kodu çalıştırıp kaydedilmiş sonuç dosyalarını ürettikten sonra `report_numbers.py` ile yeniden hesaplanabilir.

## Kurulum

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Yazıdaki sonuçları yeniden üretme

```bash
python -m pytest -q                                      # 8 test
python run_pipeline.py --k 5000                          # rastgele bölme
python run_pipeline.py --k 5000 --split plate --sex      # plakaya göre bölme + cinsiyet
python run_pipeline.py --k 5000 --inner-selection        # iç CV'de de CpG seçimi
python report_numbers.py --run gse40279
python report_numbers.py --run gse40279_plate_sex
```

İlk çalıştırmada series matrix dosyası GEO'dan `data/raw/` klasörüne indirilir ve
`data/cache/` klasörüne `.npy` olarak kaydedilir (`data/` depoya dahil değildir).
Bellek: tüm matris float32 olarak yaklaşık 1,2 GB tutar; `--sex` bir kopya daha oluşturur.

Hızlı kod denemesi için gerçek veri gerekmez: `python run_pipeline.py --synthetic`.

## Veri sızıntısını nasıl önlüyoruz?

1. Test verisi en başta ayrılır ve yalnızca en sonda bir kez kullanılır.
2. Eksik değer doldurma, CpG seçimi (`SelectKBest`) ve ölçekleme bir `Pipeline` içindedir;
   her CV fold'unda yalnızca o fold'un eğitim kısmından öğrenilir.
3. "En iyi model" CV ortalamasına göre seçilir, test sonucuna göre seçilmez.
4. Varsayılan kurulumda ElasticNetCV/RidgeCV'nin **iç** CV'si, dış eğitim verisinin tamamıyla
   seçilmiş CpG'lerle çalışır. Bu, dış doğrulama ve test verisinden doğrudan bilgi sızdırmaz,
   ama hiperparametre seçimini iyimserleştirebilir. `--inner-selection` CpG seçimini iç CV'nin
   her parçasında yeniden yapar; gerçek veride sonuç değişmedi (Elastic Net CV MAE 3,30 → 3,29).
5. İlgili testler: `test_cpg_selection_uses_only_training_data`,
   `test_inner_selection_refits_selection_in_each_inner_fold`, `test_plate_split_keeps_plates_apart`.

## Seçenekler ve yardımcı betikler

| Komut | Ne yapar |
|---|---|
| `--split plate` | Plakaları bütün olarak ayırır (GroupShuffleSplit + GroupKFold) |
| `--sex` | Cinsiyeti, CpG seçiminden geçmeden modele ekler |
| `--inner-selection` | CpG seçimini iç CV'nin her parçasında yeniden yapar |
| `make_figures.py --run <klasör>` | Grafikleri modeli yeniden eğitmeden üretir |
| `report_numbers.py --run <klasör>` | Yazıdaki sayıları, AMAR ve plaka analizlerini yeniden hesaplar |
| `compare_runs.py` | Çalıştırmaları tek tabloda karşılaştırır |

## Grafikler

Aşağıdaki görseller ZIP paketinde bulunmaz; ilgili `run_pipeline.py` komutları çalıştırılınca yerel olarak üretilir.

| Dosya | Soru |
|---|---|
| `figures/gse40279/01_yas_dagilimi.png` | Veride hangi yaşlar var? |
| `figures/gse40279/02_tahmin_vs_gercek.png` | Model görmediği kişilerde ne kadar isabetli? |
| `figures/gse40279/03_model_karsilastirma.png` | Modeller taban çizgisinden ne kadar iyi? |
| `figures/gse40279/04_yasa_bagli_sapma.png` | Hata yaşa göre sistematik mi? |
| `figures/gse40279/05_en_etkili_cpgler.png` | Model hangi CpG'lere dayanıyor? |
| `figures/gse40279_plate_sex/06_varsayimsal_ornek.png` | Yazıdaki 60 → 58 → 52 örneği dağılımın neresinde? |
| `figures/gse40279_plate_sex/07_plaka_etkisi.png` | Yaşa göre düzeltilmiş hata plakalara göre değişiyor mu? |

## Bilinen sınırlamalar

- Hannum'un modeli cinsiyet, BMI, diyabet durumu, etnik köken ve ölçüm partisini cezasız
  değişken olarak içeriyordu. GEO dosyasında BMI ve diyabet bilgisi yok.
- Bu veride plaka ve etnik köken iç içe geçmiştir (Hispanik-Meksikalı örneklerin tamamı 5, 6 ve
  9 numaralı plakalarda). Plaka farklarının nedeni bu veriyle belirlenemiyor.
- Bağımsız bir dış veri setiyle doğrulama yapılmadı.
- `SelectKBest` hız için kullanılan tek değişkenli bir ön seçimdir, biyolojik bir seçim sayılmaz.
- Yazıdaki XGBoost sonuçları sabit ayarlarla elde edildi.
- Elastic Net'in seçtiği ceza değeri arama aralığının kenarına yakın ve bu bölgede yakınsama
  uyarıları görülüyor; sonuçlar iterasyon sayısından etkilenmedi.

## Veri ve kaynak

- Hannum, G. ve ark. (2013). Genome-wide Methylation Profiles Reveal Quantitative Views of Human
  Aging Rates. *Molecular Cell*, 49, 359–367. https://doi.org/10.1016/j.molcel.2012.10.016
- NCBI GEO, GSE40279: https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279
