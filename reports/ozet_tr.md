# Proje Ozeti (TR)

## Temel Fikir

Bu proje plasmid omurgalarini egitim donemindeki biyolojik ve ekolojik sinyallerle puanlar; daha sonra bu omurgalarin 2015 sonrasinda yeni ulkelerde gorunur olup olmadigini retrospektif olarak test eder.

## Formal Hipotezler

- **H0 (sifir)**: <=2015 verisinden uretilen T/H/A temelli oncelik sinyali, 2015 sonrasi cok-ulkeli gorunurluk genislemesini ayirt etmez (ROC AUC = 0.50).
- **H1 (alternatif)**: Ayni oncelik sinyali, 2015 sonrasi cok-ulkeli gorunurluk genislemesi ile pozitif iliskilidir (ROC AUC > 0.50).
- **Anlamlilik olcutu**: Empirik permutasyon p-degeri < 0.01; ana null audit'i, yayinlanan ana modelin ozellik seti ve ayni L2/agirliklandirma ayarlari ile kurulur.

## Ana Model ve Denetim Baglami

- Mevcut ana benchmark: `support-synergy biological model` | ROC AUC `0.937` | AP `0.897`.
- En yuksek metrikli denetim modeli: `full_priority` | ROC AUC `0.948` | AP `0.910`.
- Koruyucu karsilastirma modeli: `bio-clean model` | ROC AUC `0.783` | AP `0.663`.
- Kurator odakli aday portfoyu: `10` yerlesik yuksek-risk + `10` erken-sinyal adayi.
- Cok amacli model secim scorecard'inda ana model `4/18` sirada; bu scorecard overall AUC, AP, lower-half/q1 knownness, matched-knownness, source holdout ve knownness-Spearman'i birlikte degerlendirir.
- Scorecard'da en ustte yer alan model `full_priority` olsa da headline benchmark `support_synergy_priority` olarak korunur; secim gerekcesi yukarida ayri satirda acik verilir.
- Guclendirilmis biyolojik denetim modeli: `augmented biological model` | ROC AUC `0.843` | AP `0.756`.
- Knownness-robust biyolojik denetim modeli: `knownness-robust biological model` | ROC AUC `0.863` | AP `0.791`.
- Support-calibrated biyolojik model: `support-calibrated biological model` | ROC AUC `0.880` | AP `0.826`.
- Support-synergy biyolojik model: `support-synergy biological model` | ROC AUC `0.937` | AP `0.897`.
- Hata-odakli host-transfer sinerji modeli: `host-transfer synergy biological model` | ROC AUC `0.863` | AP `0.791`.
- Tehdit-mimari denetim modeli: `threat-architecture biological model` | ROC AUC `0.864` | AP `0.791`.
- Taksonomi-duyarli H denetim modeli: `taxonomy-aware H model` | ROC AUC `0.840` | AP `0.760`.
- Yapisal sinyal denetim modeli: `structure-aware biological model` | ROC AUC `0.847` | AP `0.773`.
- `support-synergy biological model` icin top-10 kesinlik `1.000`; duyarlilik `0.028`.
- Top-10 kesinlik karsilastirmasi: koruyucu model `0.800` vs yalniz-sayim referans modeli `1.000`.
- Top-25 daha gercekci karar kesitidir: `support-synergy biological model` icin kesinlik `1.000` ve duyarlilik `0.069`.
- Yayinlanan ana model ile en yuksek metrikli modelin top-10 ortusmesi: `2/10` aday; top-25 ortusmesi `7/25`; top-50 ortusmesi `33/50`.
- Yayinlanan ana modelin top-25 listesinde dusuk-bilinirlik yarimindan gelen aday sayisi `0`'dir. Bu nedenle erken-sinyal adaylari ana kisa listeden ayri yorumlanmalidir.
- En zor alt grup olan en dusuk bilinirlik ceyregi icin ana model ROC AUC'si `0.979`; bu bolge ana modelin genel performansindan belirgin olarak daha zordur.
- Knownness-gated audit modeli (`adaptive_natural_priority`) ust yari icin `natural_auc_priority`, alt yari icin ise dusuk-bilinirlik uzman skorunu OOF temelli olarak kullanarak genel ROC AUC `0.867` ve AP `0.779` uretir.
- En guclu knownness-gated audit modeli (`adaptive_support_synergy_blend_priority`) `support-synergy biological model` tabanini kullanir; alt bilinirlik yariminda uzman novelty skoruna `0.50` agirlik vererek genel ROC AUC `0.934` ve AP `0.890` uretir.
- Gate consistency audit'i: `adaptive_support_synergy_blend_priority` icin aktif kapinin en yakin `99` omurgasinda rota degisimi altinda ortalama |Δskor| `0.071`, p90 |Δskor| `0.149` ve rota-spearman `0.948` bulundu; bu modelin gate tier'i `moderate` olarak raporlandi.
- Kaynak-dengeli tekrarlar ortalama ROC AUC `0.889` (sd `0.064`) verir; bu, veri kaynagi kompozisyonunun etkisinin tamamen yok olmadigini gosterir.
- Eslesik knownness/source strata audit'inde ana model agirlikli ROC AUC `0.698`, yalniz-sayim baseline ise `0.691` verir.
- Ikincil outcome olarak yeni makro-bolge sicrama audit'inde en guclu model ROC AUC `0.910` uretir; bu, sinyalin yalnizca ulke sayisina bagli olmadigini destekler.
- Weighted yeni-ulke burden audit'inde en iyi modelin Spearman korelasyonu `0.848`'tur.
- Binary esitli outcome'a ek olarak ham yeni-ulke sayisi icin en iyi count-alignment audit modeli Spearman `0.878` verir.
- Backbone metadata quality ortalamasi `0.574` olarak ayri raporlanir; veri kalitesi dusuk adaylar false-negative ve risk audit'lerinde ayrica isaretlenir.

## Nasil Yorumlanmali

- Outcome gercek biyolojik yayilimin birebir olcumu degil; daha cok sonraki donemde yeni ulke gorunurlugu artisidir.
- Ana outcome esigi: 2015 sonrasi en az `3` yeni ulke. `candidate_threshold_flip.tsv` bu etiketin esige ne kadar hassas oldugunu gosterir.
- Bu sistem tum pozitif backbone'lari yakalayan tam bir tarama araci degil; sinirli bir aday listesini onceliklendiren retrospektif bir kisa-liste aracidir.
- Ana model secimi yalniz tek bir metrikten degil; genel ayiricilik, low-knownness davranisi, matched audit ve kaynak-robustlugu birlikte okunarak yapilir.
- Model secim gerekcesi: mevcut ana benchmark, marjinal farklar olsa bile headline benchmark olarak acik tutulur ve alternatif denetim gorunumleri ayrica raporlanir
- Gozlenen host cesitliligi ekseni dogrudan biyolojik host range olarak okunmamalidir; bu eksen kismen ornekleme doygunlugu ve bilinirlik sinyali tasir.
- Guclendirilmis biyolojik denetim modelinde dis host-range sinyali, backbone safligi, atama guveni ve replikon mimarisi ek audit ozellikleri olarak ayri raporlanir; bunlar headline benchmark yerine biyolojik sinyali sikilastirma amaciyla kullanilir.
- `adaptive_*` aileleri etiket veya zaman ayrimini degistirmez; yalnizca pre-2015 knownness sinyali ile dusuk-bilinirlik omurgalarda uzman novelty skorunu switch veya blend seklinde kullanir. Bu nedenle routing audit'idir, yeni bir headline benchmark degildir.
- En yuksek dropout etkisinin `T_eff_norm` uzerinde olmasi, host cesitliligiyle iliskili terimlerin yonlu katsayi tasimasiyla celismez; ablation etkisi ile katsayi yorumu ayni sey degildir.
- Okuma sirasi: `candidate_portfolio.tsv` -> `candidate_evidence_matrix.tsv` -> `candidate_threshold_flip.tsv`.
- `novelty_watchlist` ana shortlist ile ayni sey degildir; yalniz-sayim referans modelini gecen dusuk-bilinirlik erken-sinyal adaylarini toplar.
- Ana rapor dili yalnizca mevcut primary benchmark ve koruyucu benchmark icin kullanilir; diger model aileleri kesifsel denetim olarak raporlanir.
- Backbone bootstrap araliklari outcome birimi olan backbone duzeyinde hesaplanir; ek group-bootstrap bu granulerlikte ayni birimi yeniden ornekler.
- Spatial genelleme artik `dominant_region_train` uzerinden strict holdout olarak ayri tabloda denetlenir; bu analiz, zaman ayirimina ek ikinci bir OOD kontroludur.
- Firsat yanliligi tam olarak sifirlanmis degildir: erken yillarda gozlenen backbone'larin daha uzun takip suresi vardir. Bu durum sinirlandirma olarak acikca kabul edilmelidir.
- Outcome-eligibility kasitli olarak yalnizca egitimde 1-3 ulke gorunurlugune sahip backbone'lari hedefler; sistem tum backbone evreni icin degil, erken-donem izleme kisa listesi icin optimize edilmis bir aractir.
- Ulke metadata kalitesi ayri `country_quality_summary.tsv` tablosunda raporlanir; eksik ulke kayitlari yayilim zayifligi gibi yorumlanmamalidir.
- Permutasyon audit'leri iki ayri soruya cevap verir: ana null audit'i headline sinyalin sans ustu olup olmadigini, model-karsilastirma permutasyonlari ise modeller arasi farkin ne kadar tesadufi olabilecegini test eder.
- Etik cerceve: yalnizca halka acik genom ve metadata kullanilir; bireysel hasta kimligi cikarimi veya klinik tani uretilmez.

## Zero-Floor Bilesen Davranisi

- T, H veya A eksenlerinden birinin ham degeri sifir ise ilgili normalize bilesen de 0.0 olarak kalir. Bu nedenle aritmetik `priority_index`, bazi backbone'larda fiilen iki aktif eksenin ortalamasi gibi davranabilir; bu bir bug degil, eksik biyolojik kanitin bilerek sifir puanlanmasidir.

## OLS Residual Yaklasimi

- `H_support_norm_residual`, gorunurluk ve bilinirlik proxy'lerine karsi OLS artiklasiyla hesaplanir. Amaç robust bir nedensellik modeli kurmak degil, destek sinyalinin sayim-proxy'lerinden arindirilmis deterministik bir audit ekseni elde etmektir.

## Ornek Adaylar

- AA627 omurgasi agirlikli olarak Salmonella enterica ile iliskilidir; egitim doneminde 119 kayit ve 3 ulke destegi vardir, 2015 sonrasinda ise 11 yeni ulkede gorulmustur. Juri yorumu icin bu aday yerlesik yuksek risk kisa listesi olarak ele alinmalidir. Kanit seviyesi A seviyesi; izlem duzeyi cekirdek izlem; coklu model uzlasi top-50 durumu: evet. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: stable internal signal, multi modal support, multiverse stable, oof supported. Baskin halk sagligi odakli AMR siniflari: AMINOGLYCOSIDE,SULFONAMIDE,TETRACYCLINE,PHENICOL,BETA-LACTAM.
- AE548 omurgasi agirlikli olarak Burkholderia sp. RPE67 ile iliskilidir; egitim doneminde 8 kayit ve 1 ulke destegi vardir, 2015 sonrasinda ise 4 yeni ulkede gorulmustur. Juri yorumu icin bu aday ayri erken sinyal izleme hatti olarak ele alinmalidir. Kanit seviyesi kesif amacli yenilik izleme listesi; izlem duzeyi dusuk guvenli inceleme havuzu; coklu model uzlasi top-50 durumu: hayir. Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: multi modal support, outperforms counts baseline, oof supported. Baskin halk sagligi odakli AMR siniflari: FLUOROQUINOLONE,TETRACYCLINE.
- AB193 omurgasi agirlikli olarak Klebsiella pneumoniae ile iliskilidir; egitim doneminde 144 kayit ve 1 ulke destegi vardir, 2015 sonrasinda ise 22 yeni ulkede gorulmustur. Juri yorumu icin bu aday yerlesik yuksek risk kisa listesi olarak ele alinmalidir. Kanit seviyesi A seviyesi; izlem duzeyi dusuk guvenli inceleme havuzu; coklu model uzlasi top-50 durumu: evet. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: stable internal signal, multi modal support, oof supported. Baskin halk sagligi odakli AMR siniflari: AMINOGLYCOSIDE,MACROLIDE,BETA-LACTAM,SULFONAMIDE,PHENICOL.
- AC030 omurgasi agirlikli olarak Escherichia coli ile iliskilidir; egitim doneminde 10 kayit ve 1 ulke destegi vardir, 2015 sonrasinda ise 4 yeni ulkede gorulmustur. Juri yorumu icin bu aday ayri erken sinyal izleme hatti olarak ele alinmalidir. Kanit seviyesi kesif amacli yenilik izleme listesi; izlem duzeyi dusuk guvenli inceleme havuzu; coklu model uzlasi top-50 durumu: evet. Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: outperforms counts baseline, oof supported. Baskin halk sagligi odakli AMR siniflari: tespit edilmedi.
- AA739 omurgasi agirlikli olarak Salmonella enterica ile iliskilidir; egitim doneminde 346 kayit ve 2 ulke destegi vardir, 2015 sonrasinda ise 36 yeni ulkede gorulmustur. Juri yorumu icin bu aday yerlesik yuksek risk kisa listesi olarak ele alinmalidir. Kanit seviyesi A seviyesi; izlem duzeyi cekirdek izlem; coklu model uzlasi top-50 durumu: evet. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: stable internal signal, multi modal support, multiverse stable, oof supported. Baskin halk sagligi odakli AMR siniflari: AMINOGLYCOSIDE,SULFONAMIDE,PEPTIDE,TETRACYCLINE,PHENICOL.
- AF563 omurgasi agirlikli olarak Alicycliphilus denitrificans ile iliskilidir; egitim doneminde 10 kayit ve 1 ulke destegi vardir, 2015 sonrasinda ise 6 yeni ulkede gorulmustur. Juri yorumu icin bu aday ayri erken sinyal izleme hatti olarak ele alinmalidir. Kanit seviyesi kesif amacli yenilik izleme listesi; izlem duzeyi dusuk guvenli inceleme havuzu; coklu model uzlasi top-50 durumu: hayir. Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: multi modal support, outperforms counts baseline, oof supported. Baskin halk sagligi odakli AMR siniflari: FLUOROQUINOLONE,TETRACYCLINE,AMINOGLYCOSIDE,BETA-LACTAM.
- AA304 omurgasi agirlikli olarak Escherichia coli ile iliskilidir; egitim doneminde 60 kayit ve 2 ulke destegi vardir, 2015 sonrasinda ise 14 yeni ulkede gorulmustur. Juri yorumu icin bu aday yerlesik yuksek risk kisa listesi olarak ele alinmalidir. Kanit seviyesi A seviyesi; izlem duzeyi cekirdek izlem; coklu model uzlasi top-50 durumu: evet. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: stable internal signal, multi modal support, multiverse stable, oof supported. Baskin halk sagligi odakli AMR siniflari: TETRACYCLINE,AMINOGLYCOSIDE,SULFONAMIDE,BETA-LACTAM,BLEOMYCIN.
- AB238 omurgasi agirlikli olarak Salmonella enterica ile iliskilidir; egitim doneminde 8 kayit ve 1 ulke destegi vardir, 2015 sonrasinda ise 4 yeni ulkede gorulmustur. Juri yorumu icin bu aday ayri erken sinyal izleme hatti olarak ele alinmalidir. Kanit seviyesi kesif amacli yenilik izleme listesi; izlem duzeyi dusuk guvenli inceleme havuzu; coklu model uzlasi top-50 durumu: hayir. Ana yayinlanan kisa listenin dogal parcasi degildir; ayri bir kesif ve erken-sinyal izleme hattidir. Mekanistik agirlik: coherence; profil etiketleri: coherence.  Izlem gerekcesi: outperforms counts baseline, oof supported. Baskin halk sagligi odakli AMR siniflari: tespit edilmedi.
