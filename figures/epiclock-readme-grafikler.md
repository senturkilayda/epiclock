## Grafikler

Bu bölümdeki görseller, **GSE40279** verisinden Epiclock kapsamında elde edilen analizleri özetler. Rastgele eğitim/test bölmesi ve ölçüm plakalarının bütün olarak ayrıldığı değerlendirme farklı deneylerdir; aynı sonuçlar gibi yorumlanmamalıdır.

### 1. Veri setinde yaş dağılımı

656 kişinin kronolojik yaş dağılımı (19–101 yaş; medyan 65). Genç yaşlardaki örnek sayısı daha sınırlı olduğundan rastgele bölmede yaşa göre tabakalandırma kullanıldı.

![GSE40279 yaş dağılımı](figures/01_yas_dagilimi.png)

### 2. Görülmemiş ölçüm plakalarındaki yaş tahmini

Elastic Net'in **plaka bazlı ayrılmış test kümesindeki 98 kişi** için gerçek yaş ve tahmin edilen yaş karşılaştırması. Test MAE yaklaşık **3,78 yıl** (grafikte yuvarlanarak 3,8 yıl); Pearson korelasyonu yaklaşık **0,95**. Kesikli çizgi kusursuz tahmini gösterir.

![Elastic Net - görülmemiş plakalarda gerçek ve tahmin edilen yaş](figures/02_test_tahmin_vs_gercek.png)

### 3. Model karşılaştırması: plaka bazlı doğrulama

Ölçüm plakalarına göre gruplandırılmış çapraz doğrulamada ortalama MAE: **Elastic Net 3,99**, **Ridge 4,22**, **XGBoost 4,66 yıl**. Daireler çapraz doğrulama katlarını, elmaslar ayrılmış test kümesinin sonuçlarını gösterir. Basit ortalama yaş tahmini taban çizgisi olarak verilir.

![Plaka bazlı MAE model karşılaştırması](figures/03_model_karsilastirma.png)

### 4. Yaşa bağlı tahmin yanlılığı

Elastic Net'in kat dışı tahminlerinde (n = 558), hata yaşla değişiyor: gençler genel olarak daha büyük, ileri yaştakiler daha küçük tahmin edilebiliyor. Bu nedenle **tahmin edilen yaş − gerçek yaş** farkı tek başına biyolojik yaş ivmesi sayılmamalıdır.

![Kronolojik yaşa göre tahmin yanlılığı](figures/04_yasa_bagli_sapma.png)

### 5. Elastic Net'in en yüksek ağırlık verdiği altı CpG

En büyük mutlak model katsayısına göre seçilen altı CpG için metilasyonun yaşla ilişkisi. Katsayı büyüklüğü bir değişkenin biyolojik olarak nedensel olduğunu göstermez; ayrıca çok değişkenli model katsayısının işareti, tek başına korelasyonun işaretiyle farklı olabilir.

![En önemli altı CpG ve yaş ilişkileri](figures/05_en_onemli_6_cpg.png)

### 6. Sekiz yıl daha genç tahmin edilen kişiler

Plaka bazlı değerlendirmede bazı kişilerin tahmini yaşı gerçek yaşından sekiz yıl veya daha fazla düşük bulundu (yaklaşık **%6,5**). Grafik, **farklı kişilerin tahmin hatalarını** ve yazıdaki varsayımsal müdahale örneğini karşılaştırır; aynı kişiye yapılan tekrarlı ölçümü veya gerçek gençleşmeyi göstermez.

![Gerçek yaştan en az sekiz yıl düşük tahmin edilenlerin dağılımı](figures/06_en_az_8_yil_genc_tahmin_orani.png)

### 7. Plakalar arasında düzeltilmiş hata farklılıkları

Yaşa göre düzeltilmiş tahmin hatasının plakalara göre dağılımı. Plaka medyanları arasında yaklaşık **8,2 yıllık fark** görüldü (Kruskal–Wallis, p < 0,001). Ancak ölçüm plakaları ile örnek özellikleri birbiriyle ilişkili olduğundan fark **yalnızca teknik parti etkisine** bağlanamaz.

![Ölçüm plakalarına göre yaşa düzeltilmiş tahmin hataları](figures/07_plaka_farki.png)

**Yorum sınırı:** Grafikler kronolojik yaş tahminini ve model değerlendirmesini gösterir; herhangi bir tedavinin insanları gençleştirdiğini kanıtlamaz. Ana veriler: [NCBI GEO GSE40279](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE40279).
