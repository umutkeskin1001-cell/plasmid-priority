# Proje Özeti

## Temel Mesaj

Plasmid Priority, plasmid omurga sınıflarını eğitim dönemindeki genomik sinyallerle puanlayan ve bu puanların daha sonraki coğrafi görünürlük artışıyla ilişkili olup olmadığını test eden retrospektif bir analiz hattıdır.

Ana model `bio_clean_priority` için ROC AUC `0.747`, AP `0.660` ve Brier Skill Score `0.193` olarak raporlanır. Mevcut permütasyon denetimi sabit-skor label-permutation audit'i olduğu için model-seçim-düzeltilmiş anlamlılık iddiası olarak değil, keşifsel sinyal kontrolü olarak okunmalıdır. Sayım temelli karşılaştırma modeli `0.722` ROC AUC üretir; ana modelin bu taban modele karşı kazancı `0.024, 95% CI [-0.019, 0.065]` düzeyindedir.

## Model Seçimi

Model seçimi tek bir metriğe göre yapılmamıştır. Birlikte okunan ölçütler şunlardır:

1. Genel ROC AUC ve AP.
2. Düşük bilinirlik yarısındaki performans.
3. Eşleştirilmiş bilinirlik/kaynak katmanlarındaki performans.
4. Kaynak dışlama denetimi ve diğer sağlamlık analizleri.
5. Pratik kısa liste verimi.

Bu nedenle discovery hattında `bio_clean_priority` korunur; governance watch-only hattında ise `phylo_support_fusion_priority` daha temkinli yorum katmanı olarak ele alınır.

## Metodoloji

- Ham veri PLSDB, RefSeq ve Pathogen Detection metadata kaynaklarından gelir.
- Kayıtlar harmonize edilir, yinelenenler ayıklanır ve omurga sınıfı ataması yapılır.
- Zaman ayrımı `<=2015` eğitim ve `>2015` sonuç penceresi olacak şekilde kuruludur.
- T, H ve A eksenleri yalnızca eğitim döneminden hesaplanır.
- Yayılım etiketi, test döneminde en az `3` yeni ülke görülmesiyle tanımlanır.
- Değerlendirme OOF lojistik regresyon tahminleri üzerinden yapılır.

## Türkiye Bağlamı

WHO'nun 2025 GLASS özeti, antibiyotik direnci yükünün özellikle Güneydoğu Asya ve Doğu Akdeniz bölgelerinde yüksek olduğunu vurgular. Türkiye, AMR sürveyansı açısından bu bölgesel baskının doğrudan önemli olduğu bir ülkedir.

ECDC ve WHO Europe çerçevelerinde karbapenem dirençli *Klebsiella pneumoniae* ile GSBL/ESBL üreten *Escherichia coli*, Enterobacterales kaynaklı hastane yükünün temel başlıkları arasında yer almaktadır. Bu nedenle bu çalışmanın ürettiği omurga sınıfı önceliklendirmesi, Türkiye'de genomik AMR sürveyansına doğrudan uyarlanabilecek bir kanıt-konsept çerçevesi sunar.

Bu proje klinik karar desteği vermez; ancak Türkiye'de ulusal veya kurumsal genomik sürveyans akışlarında hangi omurga sınıflarının önce incelenmesi gerektiğini sistematikleştirebilir.

## Ana Bulgular

- Ana model: ROC AUC `0.747` | AP `0.660`.
- Koruyucu model: ROC AUC `0.752` | AP `0.664`.
- Baseline model: ROC AUC `0.722` | AP `0.647`.
- Yanlış negatif incelemesi: kısa liste dışında kalan `50` pozitif vardır; baskın nedenler `low_assignment_confidence, low_training_members, low_knownness`.
- Eşleştirilmiş bilinirlik/kaynak katmanları denetimi: ana model `0.701`, taban model `0.594`.
- Ağırlıklı yeni ülke yükü ile ilişki: Spearman ρ `0.620`.
- Ham yeni ülke sayısı ile ilişki: Spearman ρ `0.624` [0.580, 0.662].
- Mekânsal holdout denetimi: ağırlıklı ROC AUC `0.735`.

## Sınırlılıklar

- Bu çalışma retrospektiftir; prospektif erken uyarı sistemi iddiası taşımaz.
- Sonuç değişkeni doğrudan biyolojik yayılım değil, sonraki dönem ülke görünürlüğü artışıdır.
- En sıkı bilinirlik eşleştirmeli kaynak dışlama testi en zor alt kohorttur; bu katmanda tüm modeller temkinli yorumlanmalıdır.
- Fırsat yanlılığı tamamen giderilemez: daha erken görülen omurgaların daha uzun takip penceresi vardır.
- `risk_uncertainty` bir güven aralığı değildir; risk-bileşeni uyumsuzluğu, karar sınırına yakınlık ve düşük bilinirlik cezasından türetilen operasyonel bir belirsizlik skorudur.

## Örnek Adaylar

- `AA282`: baskın tür `Escherichia coli`, baskın replikon `IncI-gamma/K1`; bu aday `yerleşik yüksek risk kısa listesi` içinde değerlendirilir. Kaynak desteği `çok kaynaklı destek`, operasyonel karar katmanı `eylem` ve genel risk `0.81`, belirsizlik `0.28`. Öne çıkan AMR sınıfları: AMINOGLYCOSIDE,BETA-LACTAM,SULFONAMIDE,CEPHALOSPORIN,MONOBACTAM.
- `AC030`: baskın tür `Escherichia coli`, baskın replikon `IncFIA`; bu aday `erken-sinyal izleme hattı` içinde değerlendirilir. Kaynak desteği `çok kaynaklı destek`, operasyonel karar katmanı `eylem` ve genel risk `0.79`, belirsizlik `0.34`. Öne çıkan AMR sınıfları: belirgin AMR sınıfı sinyali yok.
- `AA324`: baskın tür `Escherichia coli`, baskın replikon `IncFIA`; bu aday `yerleşik yüksek risk kısa listesi` içinde değerlendirilir. Kaynak desteği `destek düzeyi belirtilmemiş`, operasyonel karar katmanı `eylem` ve genel risk `0.82`, belirsizlik `0.23`. Öne çıkan AMR sınıfları: MACROLIDE,PHENICOL,TETRACYCLINE,AMINOGLYCOSIDE,SULFONAMIDE.
- `AA859`: baskın tür `Salmonella enterica`, baskın replikon `IncC`; bu aday `erken-sinyal izleme hattı` içinde değerlendirilir. Kaynak desteği `destek düzeyi belirtilmemiş`, operasyonel karar katmanı `eylem` ve genel risk `0.80`, belirsizlik `0.31`. Öne çıkan AMR sınıfları: SULFONAMIDE,TETRACYCLINE.

## Sürüm Yüzeyi

- `blocked_holdout_summary.tsv`: iç kaynak/bölge stres testini raporlar; bloklanmış holdout denetimi `bio_clean model` için ağırlıklı ROC AUC `0.735` ve en zor grup `dominant_region_train:Europe` değerini içerir.
- `calibration_threshold_summary.png`: kalibrasyon ve eşik duyarlılığı için kompakt tanı grafiğidir.
- `jury_brief.md` ve `ozet_tr.md`: jüriye dönük anlatının dağıtım yüzeyleridir.
