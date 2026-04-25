# Proje Özeti

## Temel Mesaj

Plasmid Priority, plasmid omurga sınıflarını eğitim dönemindeki genomik sinyallerle puanlayan ve bu puanların daha sonraki coğrafi görünürlük artışıyla ilişkili olup olmadığını test eden retrospektif bir analiz hattıdır.

Ana model `discovery_boosted` için ROC AUC `0.811`, AP `0.721` ve Brier Skill Score `0.257` olarak raporlanır. Mevcut permütasyon denetimi sabit-skor label-permutation audit'i olduğu için model-seçim-düzeltilmiş anlamlılık iddiası olarak değil, keşifsel sinyal kontrolü olarak okunmalıdır. Sayım temelli karşılaştırma modeli `0.736` ROC AUC üretir; ana modelin bu taban modele karşı kazancı `+0.074 [+0.040, +0.107]` düzeyindedir.

## Model Seçimi

Model seçimi tek bir metriğe göre yapılmamıştır. Birlikte okunan ölçütler şunlardır:

1. Genel ROC AUC ve AP.
2. Düşük bilinirlik yarısındaki performans.
3. Eşleştirilmiş bilinirlik/kaynak katmanlarındaki performans.
4. Kaynak dışlama denetimi ve diğer sağlamlık analizleri.
5. Pratik kısa liste verimi.

Bu nedenle discovery hattında `discovery_boosted` korunur; governance watch-only hattında ise `governance_linear` daha temkinli yorum katmanı olarak ele alınır.

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

- Ana model: ROC AUC `0.811` | AP `0.721`.
- Koruyucu model: ROC AUC `0.747` | AP `0.657`.
- Baseline model: ROC AUC `0.736` | AP `0.648`.
- Yanlış negatif incelemesi: kısa liste dışında kalan `50` pozitif vardır; baskın nedenler `low_assignment_confidence, low_training_members, low_knownness`.
- Eşleştirilmiş bilinirlik/kaynak katmanları denetimi: ana model `0.752`, taban model `0.602`.
- Ağırlıklı yeni ülke yükü ile ilişki: Spearman ρ `0.620`.
- Ham yeni ülke sayısı ile ilişki: Spearman ρ `0.624` [0.582, 0.663].
- Mekânsal holdout denetimi: ağırlıklı ROC AUC `0.710`.

## Sıralama Kararlılığı

- `candidate_rank_stability.tsv` aday siralama kararliligini raporlar; en kararlı örnek `AA316` için ilk `10` içinde kalma sıklığı `1.00`.
- `candidate_variant_consistency.tsv` aday siralama kararliligini raporlar; en kararlı örnek `AA171` için ilk `10` içinde kalma sıklığı `1.00`.

## Sınırlılıklar

- Bu çalışma retrospektiftir; prospektif erken uyarı sistemi iddiası taşımaz.
- Sonuç değişkeni doğrudan biyolojik yayılım değil, sonraki dönem ülke görünürlüğü artışıdır.
- En sıkı bilinirlik eşleştirmeli kaynak dışlama testi en zor alt kohorttur; bu katmanda tüm modeller temkinli yorumlanmalıdır.
- Fırsat yanlılığı tamamen giderilemez: daha erken görülen omurgaların daha uzun takip penceresi vardır.
- `risk_uncertainty` bir güven aralığı değildir; risk-bileşeni uyumsuzluğu, karar sınırına yakınlık ve düşük bilinirlik cezasından türetilen operasyonel bir belirsizlik skorudur.

## Örnek Adaylar

- `AB039`: baskın tür `Citrobacter freundii`, baskın replikon `ColRNAI_rep_cluster_1987`; bu aday `yerleşik yüksek risk kısa listesi` içinde değerlendirilir. Kaynak desteği `çok kaynaklı destek`, operasyonel karar katmanı `eylem` ve genel risk `0.85`, belirsizlik `0.20`. Öne çıkan AMR sınıfları: belirgin AMR sınıfı sinyali yok.
- `AA434`: baskın tür `Klebsiella pneumoniae`, baskın replikon `IncFII`; bu aday `erken-sinyal izleme hattı` içinde değerlendirilir. Kaynak desteği `çok kaynaklı destek`, operasyonel karar katmanı `inceleme` ve genel risk `0.70`, belirsizlik `0.43`. Öne çıkan AMR sınıfları: AMINOGLYCOSIDE,CEPHALOSPORIN,MONOBACTAM,PENAM,PENEM.
- `AA346`: baskın tür `Escherichia coli`, baskın replikon `IncFIB`; bu aday `yerleşik yüksek risk kısa listesi` içinde değerlendirilir. Kaynak desteği `destek düzeyi belirtilmemiş`, operasyonel karar katmanı `eylem` ve genel risk `0.85`, belirsizlik `0.19`. Öne çıkan AMR sınıfları: BETA-LACTAM.
- `AF809`: baskın tür `Pseudomonas putida`, baskın replikon `rep_cluster_1115`; bu aday `erken-sinyal izleme hattı` içinde değerlendirilir. Kaynak desteği `destek düzeyi belirtilmemiş`, operasyonel karar katmanı `eylem` ve genel risk `0.81`, belirsizlik `0.29`. Öne çıkan AMR sınıfları: EFFLUX.

## Sürüm Yüzeyi

- `frozen_scientific_acceptance_audit.tsv`: doğrulayıcı kabul katmanını; eşleştirilmiş bilinirlik, kaynak dışlama, mekânsal holdout, kalibrasyon ve leakage incelemesini raporlar.
- `blocked_holdout_summary.tsv`: iç kaynak/bölge stres testi olarak raporlanır.
- `nonlinear_deconfounding_audit.tsv`: knownness residualization için kullanılan doğrusal olmayan karıştırma denetimini raporlar.
- `ordinal_outcome_audit.tsv`, `exposure_adjusted_event_outcomes.tsv` ve `macro_region_jump_outcome.tsv`: sırasal, maruziyet-düzeltilmiş ve makro-bölge sıçrama sonuçları için alternatif sonuç stres testlerini raporlar.
- `prospective_candidate_freeze.tsv` ve `annual_candidate_freeze_summary.tsv`: kısa listenin ileriye dönük bir holdout üzerinde ayakta kalıp kalmadığını kontrol eden quasi-prospective freeze yüzeyini raporlar.
- `future_sentinel_audit.tsv`, `mash_similarity_graph.tsv`, `counterfactual_shortlist_comparison.tsv`, `geographic_jump_distance_outcome.tsv` ve `amr_uncertainty_summary.tsv`: leakage canary, graph denetimi, counterfactual shortlist comparison, geographic-jump tanısı ve AMR belirsizlik özetini raporlar.
- `country_missingness_bounds.tsv` ve `country_missingness_sensitivity.tsv`: ülke eksikliği varsayımlarına göre etiket ve performans duyarlılığını raporlar.
- `candidate_rank_stability.tsv` ve `candidate_variant_consistency.tsv`: aday sıralama kararlılığını ve model-varyant tutarlılığını raporlar.
- `calibration_threshold_summary.png`: kalibrasyon ve eşik duyarlılığı için kompakt tanı grafiğidir.
- `jury_brief.md` ve `ozet_tr.md`: jüriye dönük anlatının dağıtım yüzeyleri.
