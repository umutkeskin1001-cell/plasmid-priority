# Projeyi En Ust Seviyeye Cikarma Onerileri

Bu dokumanin amaci projeyi sadece "iyi calisan bir repo" olmaktan cikarip, bilimsel olarak daha sert, muhendislik olarak daha temiz ve yari-final / final juri duzeyinde daha profesyonel bir arastirma sistemi haline getirmektir.

Buradaki oneriler iki olcekte dusunulmelidir:

- Kisa vadeli: mevcut projeyi bozmadan guclendiren, daha iyi metrik ve daha daha savunulabilir cikti veren degisiklikler
- Buyuk donusumler: repo mimarisini, model stratejisini ve bilimsel pozisyonlamayi ust seviyeye tasiyan koklu degisiklikler

## 1. Mevcut Durumun Durer Ozeti

Su an proje artik zayif degil. Ozellikle modelleme katmani ciddi ilerledi.

- En iyi tekil, sabit, dogal model: `knownness_robust_priority`
- En iyi leakage-free adaptive audit model: `adaptive_knownness_blend_priority`
- Knownness problemi tamamen bitmis degil ama onceki hallere gore daha temiz ele alinmis durumda
- Ana zayiflik artik "ham model yetersizligi" degil
- Ana zayifliklar artik daha cok:
  - model secimi / anlati ayrimi
  - rapor ve analysis yuzeyinin kalabalikligi
  - monolit Python dosyalari
  - deneysel arama ciktilarinin canonical analysis ile karismasi
  - reproducibility / release / CI eksigi

Bu da iyi haber: proje artik "calismiyor" asamasinda degil, "nasıl profesyonel arastirma urunune donusur" asamasinda.

## 2. Bence Projenin Yeni Resmi Kimligi

Bu projenin en dogru pozisyonlamasi su olmali:

- Bu bir "gercek biyolojik yayilimi kesin tahmin eden sistem" degil
- Bu bir "erken donem genomik ve ekolojik sinyallerle plasmid backbone izleme onceliklendirme sistemi"
- Ana amac:
  - daha sonra yeni ulke gorunurlugu kazanabilecek omurgalari erken donemde ayirt etmek
  - bunu sadece popularite sayarak degil, biyolojik ve yapisal sinyallerle yapmak

Bu kimlik etrafinda tum model secimi, rapor dili ve repo mimarisi hizalanmali.

## 3. En Yuksek Oncelikli Buyuk Kararlar

### 3.1 Primary modeli degistir

En kritik karar bu.

Su anda resmi headline model hala `parsimonious_priority`.
Ama en iyi tekil, dogal, savunulabilir model artik `knownness_robust_priority`.

Bence yapilmasi gereken:

- `PRIMARY_MODEL_NAME = "knownness_robust_priority"` yap
- `parsimonious_priority`'yi legacy published benchmark veya conservative published benchmark gibi ikinci plana al
- rapor dilini buna gore yeniden yaz

Neden?

- AUC daha yuksek
- AP daha yuksek
- lower-half knownness ve q1 performansi daha iyi
- matched-knownness audit'te daha iyi
- yine de tamamen proxy-agir, support-heavy bir model degil

Bu degisiklik projeyi bir anda daha guclu hale getirir.

### 3.2 Adaptive modeli audit olarak resmi sekilde sabitle

`adaptive_knownness_blend_priority` su anda en guclu leakage-free adaptive model.

Bunun resmi rolu su olmali:

- headline benchmark degil
- upper-bound audit model
- "knownness-aware routed evaluation ceiling"

Raporlarda acik dille su yazilmali:

- en iyi tekil model: `knownness_robust_priority`
- en iyi adaptive audit: `adaptive_knownness_blend_priority`

### 3.3 Tek bir model secmek yerine resmi model hiyerarsisi tanimla

Repo icinde model hiyerarsisi yazili hale getirilmeli:

1. `headline model`
2. `conservative comparator`
3. `counts-only baseline`
4. `adaptive upper-bound audit`
5. `exploratory research models`

Bu olmadan model hayvanat bahcesi hissi kaybolmaz.

## 4. Bilimsel Tasarimi Ust Seviyeye Tasiyacak Degisiklikler

### 4.1 Multi-objective model selection getir

Model secimi yalnizca ROC AUC ile yapilmamali.

Yeni resmi secim skoru su bileşenleri birlikte icermeli:

- overall ROC AUC
- average precision
- lower-half knownness ROC AUC
- q1 lowest-knownness ROC AUC
- matched-knownness weighted ROC AUC
- source holdout ROC AUC
- prediction-vs-knownness Spearman ceza terimi

Yani model secimi tek metrikli degil, cok amacli hale gelmeli.

Bu degisiklik bilimselligi ciddi seviyede artirir.

### 4.2 Frozen-year benchmark sistemini resmi hale getir

Su an rolling ve freeze audit mantigi var ama proje anlatisinin merkezi degil.

Yapilmasi gereken:

- 2008, 2010, 2012, 2014 freeze noktalarini resmi benchmark ailesi yap
- her bir freeze icin:
  - training window
  - later-country outcome
  - model metrics
  - shortlist yield
  - low-knownness performance
  raporla

Boylece tek bir 2015 split yerine "coklu tarihsel retrospektif kanit" gelmis olur.

### 4.3 Time-to-event / hazard audit'i resmi ikinci outcome yap

Ikili etiket iyi ama sinirli.

Eklenmesi gereken resmi ikinci outcome:

- "ilk yeni ulke kazanimi ne kadar hizli oluyor?"
- veya
- "takip penceresinde kac yeni ulke geliyor?"

Bu su sorunlari daha iyi ele alir:

- opportunity bias
- time-at-risk farki
- 3 ulke esigi keyfiligi

### 4.4 Positive-unlabeled dusuncesi ekle

En buyuk bilimsel sorunlardan biri su:

- görünmeyen = gercekten yayilmamis degil

Bu nedenle klasik binary yapi yaninda bir PU audit'i gerekli.

Oneri:

- pozitifler: sonra yeni ulke gorunurlugu kazananlar
- unlabeled: negatif diye davranilan ama gercekte sadece eksik gozlenmis olabilecekler
- PU-weighted audit ekle

Bu, database bias elestirisine cok guclu cevap olur.

### 4.5 Missingness bounds resmi hale getir

Ulke eksikligi ve metadata eksikligi icin:

- optimistic bound
- pessimistic bound
- midpoint estimate

uret ve bunlari reportta ana robustness kutusu olarak goster.

Bu, "zengin ulkeler biasi" elestirisini oldukca zayiflatir.

### 4.6 Backbone uncertainty'yi resmi model girdisi yap

Su an purity ve assignment confidence var ama bu hala yan audit gibi duruyor.

Yapilmasi gereken:

- ambiguous backbone membership ceza terimi
- high heterogeneity downweighting
- low purity candidate flag

Bunlar prediction'a doğrudan girmeli ve karar tablosunda gorunmeli.

### 4.7 Knownness problemini rapordan degil metrigin kendisinden yonet

Her ana model icin su kolonlar artik core tabloda olmali:

- overall_roc_auc
- lower_half_knownness_auc
- q1_knownness_auc
- matched_knownness_auc
- prediction_vs_knownness_spearman
- source_holdout_auc

Yani knownness problemi artik appendix'te degil ana karar tablosunda gorunmeli.

## 5. Modelleme Tarafinda Yeni Buyuk Oneriler

### 5.1 Monotonic GAM / EBM audit modeli ekle

Siyah kutu olmadan lineer logistikten daha iyi performans alinabilir.

Ozellikle su ozellikler icin:

- T_eff_norm
- H_specialization_norm
- A_eff_norm
- backbone_purity_norm
- assignment_confidence_norm
- mash_neighbor_distance_train_norm
- A_recurrence_norm

Monotonic kisitli GAM/EBM modeli denenmeli.

Bu dogru yapilirsa:

- AUC artabilir
- yorumlanabilirlik tamamen kaybolmaz

### 5.2 Pairwise ranking objective dene

Bugun esas dert shortlist kalitesi.
Bu nedenle klasik logistic loss yerine:

- pairwise ranking
- AUC-focused optimization
- ranking-based learning

audit modeli kurulabilir.

Bu özellikle top-k precision tarafinda kazanc getirebilir.

### 5.3 Calibration-aware model secimi yap

AUC guzel ama tek basina yeterli degil.

Yeni secim kriterine su da girmeli:

- Brier
- ECE
- risk-band stability

### 5.4 Group-aware hyperparameter tuning getir

L2 ve weight mode secimi su an hala biraz artisanal.

Yapilmasi gereken:

- source holdout
- region holdout
- lower-knownness performance
- matched-strata performance

uzerinden nested tuning.

### 5.5 Ensemble ama kontrollu

Her ensemble iyi degil.
Ama su mantikli:

- en iyi tekil model
- onun phylogeny-aware varyanti
- leakage-free novelty specialist

uzerinde kucuk, kontrollu, sparse stacked model.

Bu audit seviyesinde tutulur.

### 5.6 Candidate-level uncertainty interval

Her aday icin sadece skor degil:

- mean score
- bootstrap rank interval
- shortlist membership frequency

uretilmeli.

Bu juri icin cok guclu olur.

## 6. Feature Engineering Tarafinda En Buyuk Gelisim Alanlari

### 6.1 H 3.0 tasarla

H ekseni hala projedeki en hassas noktalardan biri.

Yeni H ailesi su bilesenlerden olusmali:

- observed host breadth
- host evenness
- host entropy
- genus/family/order dispersion
- taxonomy-weighted phylogenetic breadth
- rarefaction-corrected host diversity
- external host-range support
- literature-supported host-range confidence

Sonra H icinden:

- `H_biological_core`
- `H_sampling_sensitive`

diye iki alt eksen ayir.

Bu cok buyuk bir bilimsel iyilestirme olur.

### 6.2 A 2.0 tasarla

Sadece burden degil:

- burden
- recurrence
- persistence
- class diversity
- clinical threat weighting
- annotation uncertainty

ayrilmali.

Ozellikle:

- `AMR consensus uncertainty`
- `backbone-level AMR persistence`

ana A ailesine girmeli.

### 6.3 T 2.0 tasarla

Mobiliteyi sadece boole veya support sinyali olarak tutmak sinirli.

Yeni T tarafi:

- relaxase / MPF / oriT completeness
- transfer architecture coherence
- mobility consistency across members
- mobilizable vs conjugative confidence

uzerinden yeniden kurulabilir.

### 6.4 Context feature'larini daha sistematik hale getir

Su an klinik ve ecology context var ama hala tam formal degil.

Yeni context feature ailesi:

- clinical_context_fraction
- pathogenic_context_fraction
- ecology_context_diversity
- environmental_persistence_signal
- cross-ecosystem_transition_signal

Bu sayede "risk" ile "novelty" daha iyi ayrilir.

### 6.5 Mash graph'tan daha fazla sey cikar

Halihazirda novelty distance kullaniyorsun.
Ama graph tabanli ozellikler daha guclu olabilir:

- local density
- bridge centrality
- articulation-like novelty
- community rarity
- cluster volatility

Bunlar training-only olacak sekilde hesaplanmali.

## 7. Repo Mimarisinde Zorunlu Buyuk Donusumler

### 7.1 Reports ve analysis'i kesin ayir

Bugun en buyuk duzensizliklerden biri bu.

Onerilen son hali:

- `data/analysis/`
  - tum machine-readable canonical outputs
- `reports/core_tables/`
  - sadece sunum / juri / el kitabina girecek tablolar
- `reports/core_figures/`
  - sadece sunuma girecek figurlar
- `data/experiments/`
  - search, sweep, blend, greedy, what-if ciktilari

Bugun `data/analysis` icindeki su dosyalar `data/experiments` altina tasinmali:

- `knownness_greedy_search.tsv`
- `knownness_deep_search.tsv`
- `knownness_natural_model_search.tsv`
- `adaptive_blend_search.tsv`
- `adaptive_knownness_blend_search.tsv`
- `adaptive_knownness_oof_blend_search.tsv`

### 7.2 Report generator'i bol

[24_build_reports.py](/Users/umut/Projeler/plasmid-priority/scripts/24_build_reports.py) asiri buyuk.

Su yapida bolunmeli:

- `report_inputs.py`
- `report_tables.py`
- `report_narratives.py`
- `report_exports.py`

Script sadece orkestrasyon yapmali.

### 7.3 Reporting modullerini domain bazli ayir

[model_audit.py](/Users/umut/Projeler/plasmid-priority/src/plasmid_priority/reporting/model_audit.py) cok buyuk.

Boyle bolunmeli:

- `reporting/model_selection.py`
- `reporting/candidate_outputs.py`
- `reporting/holdouts.py`
- `reporting/calibration.py`
- `reporting/negative_controls.py`
- `reporting/robustness.py`

[advanced_audits.py](/Users/umut/Projeler/plasmid-priority/src/plasmid_priority/reporting/advanced_audits.py) de boyle bolunmeli:

- `advanced/geography.py`
- `advanced/missingness.py`
- `advanced/event_time.py`
- `advanced/counterfactuals.py`

### 7.4 Module A monolitini parcalara ayir

[module_a.py](/Users/umut/Projeler/plasmid-priority/src/plasmid_priority/modeling/module_a.py) su anda:

- registry
- weighting
- preprocessing
- fitting
- OOF logic
- coefficient audit
- convergence audit

hepsini ayni yerde tutuyor.

Bolunmesi gereken yapilar:

- `model_registry.py`
- `fit.py`
- `weights.py`
- `preprocess.py`
- `oof.py`
- `model_diagnostics.py`

### 7.5 Feature core'u da bol

[features/core.py](/Users/umut/Projeler/plasmid-priority/src/plasmid_priority/features/core.py) da fazla buyuk.

Bolunmesi gerekenler:

- `features_t.py`
- `features_h.py`
- `features_a.py`
- `features_backbone.py`
- `features_context.py`

## 8. Workflow ve Operasyon Tarafinda Buyuk Gelisimler

### 8.1 Workflow'lari katmanlara ayir

Su hedefler olsun:

- `make core`
- `make support`
- `make appendix`
- `make experiments`
- `make release`
- `make full`

Varsayilan `make pipeline` sadece core + rapor üretmeli.

### 8.2 Release bundle ekle

Tek komutla su paketi ureten bir release hedefi olmali:

- jury_brief
- ozet_tr
- core tables
- core figures
- tubitak metrics
- release manifest
- software version info
- data manifest

### 8.3 CI ekle

Su anda `.github/workflows` yok.

Mutlaka eklenmeli:

- unit tests
- smoke
- style / lint
- report generation dry-run

### 8.4 Experiment registry ekle

Her arama / sweep / deneme icin:

- name
- purpose
- date
- inputs
- metric summary
- winner

tutan `experiments/index.tsv` veya `experiments/manifest.json` olmali.

### 8.5 Locked benchmark mode ekle

Su mod eklenmeli:

- sadece pre-registered modeller
- sabit metric seti
- sabit outcome seti
- sabit export seti

Bu, "hareketli hedef" elestirisini ciddi azaltir.

## 9. Veri ve Reproducibility Tarafinda Buyuk Oneriler

### 9.1 Checksummed manifest yap

`data_contract.json` yeterli degil.

Yeni dosyalar lazim:

- `data_checksums.tsv`
- `release_manifest.json`
- `artifact_manifest.tsv`

Her biri:

- relative path
- bytes
- sha256
- producing script
- produced timestamp

icermeli.

### 9.2 Dev TSV'leri Parquet'e gecir

Ozellikle buyuk derived tablolar:

- bronze
- silver
- bazı analysis dump'lari

Parquet olursa:

- disk azalir
- okuma hizlanir
- rerun hizlanir

### 9.3 Ham veri repo disi politika

Repo icinde dev ham veri tutmak pratik degil.

Oneri:

- kod repo
- ham veri harici storage
- sadece manifest + bootstrap script

### 9.4 Optional reserve asset politikasini netlestir

Bugun reserve varliklarin bir kismi "belki kullaniriz" alaninda.

Uc sinifa ayir:

- core required
- support optional
- reserve archive

### 9.5 Derived output retention policy yaz

Her output icin acik karar lazim:

- canonical
- report-facing
- experiment-only
- ephemeral

Bu yazili olmadan repo tekrar kirlenir.

## 10. Rapor Kalitesi ve Juri Sunumu Icin Oneriler

### 10.1 Core report setini daha da daralt

Ideal final rapor yuzeyi:

- `jury_brief.md`
- `ozet_tr.md`
- `tubitak_final_metrics.txt`
- `model_metrics.tsv`
- `model_selection_summary.tsv`
- `candidate_portfolio.tsv`
- `candidate_evidence_matrix.tsv`
- `candidate_threshold_flip.tsv`
- `consensus_shortlist.tsv`
- `roc_curve.png`
- `pr_curve.png`
- `score_distribution.png`

Geri kalan her sey appendix veya analysis olmali.

### 10.2 Juri icin tek sayfalik model aciklamasi ekle

Yeni bir belge lazim:

- `model_card_primary.md`

Icerik:

- model neyi tahmin eder
- neyi tahmin etmez
- hangi feature ailelerini kullanir
- en buyuk riskleri neler
- low-knownness performansi ne durumda

### 10.3 Candidate one-pager sistemi

Her top aday icin tek sayfalik ozet:

- neden riskli
- neden dikkat cekici
- hangi eksenler tasiyor
- hangi audit'lerde dayaniyor
- hangi threshold'larda stabil
- hangi belirsizlikler var

Bu, sunum gucunu cok arttirir.

### 10.4 Figure dili daha sistematik olsun

Her core figür icin:

- tek mesaj
- tek cikarim
- tek caption

olmalı.

### 10.5 Turkish and English parity

TR ve EN anlatilar tamamen ayni mantikla senkron gitmeli.
Birinde daha guclu iddia olup digerinde yoksa juri bunu fark eder.

## 11. Test ve Kalite Guvencesi Onerileri

### 11.1 Report schema testleri

Su an genel test var ama rapor yuzeyi icin zorunlu schema testleri lazim:

- core_tables kolon testi
- row count sanity
- metric range sanity
- label consistency

### 11.2 Workflow dependency testleri

`run_workflow.py` icin test yaz:

- dependency order
- failure propagation
- deadlock prevention
- max_workers davranisi

### 11.3 Artifact consistency testleri

Test et:

- reporttaki ana metrikler tablolarla ayni mi
- primary model name her yerde ayni mi
- adaptive best model brief ile tsv'de ayni mi

### 11.4 Data leakage regression suite genislet

Halihazirda leakage testleri var.
Ama su eklenmeli:

- adaptive specialist leakage test
- future-country contamination test
- future-host contamination test
- future-AMR contamination test

### 11.5 Search artifact isolation test

Deneysel search dump'lari `data/analysis` yerine `data/experiments` altina gitmeli.
Bunun regression testi olsun.

## 12. En Cesur Buyuk Donusumler

Burasi "buyuk degisikliklerden korkmama" bolumu.

### 12.1 Projeyi iki urune ayir

Tek proje yerine iki katman:

- `plasmid-priority-core`
- `plasmid-priority-support`

Core:

- backbone
- features
- Module A
- validation
- reports

Support:

- pathogen detection
- card
- who mia
- amrfinder concordance
- enrichment

Bu ayrim projeyi inanilmaz temizler.

### 12.2 Model zoo'yu resmi olarak kapat

Tum modelleri default yuzeyde gostermek yerine:

- `headline models`
- `audit models`
- `ablation models`
- `retired models`

siniflarina ayir.

Retired / exploratory modelleri default export'tan cikar.

### 12.3 Paper-ready benchmark pack hazirla

Tek komutla:

- dataset summary
- benchmark metrics
- subgroup metrics
- matched-knownness metrics
- holdout metrics
- permutation summary
- candidate shortlist

ureten bir publication pack hazirla.

### 12.4 Dashboard veya local review app

Sunum disinda proje kullanimi icin lokal karar arayuzu dusunulebilir.

Her aday icin:

- score
- evidence
- threshold sensitivity
- stability
- audit profile

tek ekranda gorunsun.

### 12.5 Formal pre-registration mode

Repo icinde:

- `preregistered_models.json`
- `preregistered_outcomes.json`
- `preregistered_exports.json`

olabilir.

Bu, "sonucu guzel gosteren modeli secti" elestirisini guclu sekilde kirar.

## 13. En Yuksek ROI Sirasi

Bence bundan sonra en dogru uygulama sirasi su:

### Faz 1: Bilimsel netlik

1. `knownness_robust_priority`'yi resmi primary model yap
2. `adaptive_knownness_blend_priority`'yi resmi strongest adaptive audit yap
3. core metric tablosuna knownness ve holdout kolonlarini ekle
4. model secimini multi-objective hale getir

### Faz 2: Repo temizligi

5. `data/analysis` icindeki search dump'larini `data/experiments` altina tas
6. `reports` yuzeyini daha da daralt
7. default workflow'u `core / support / appendix / experiments` diye ayir

### Faz 3: Mimari refactor

8. `24_build_reports.py`yi bol
9. `model_audit.py`yi bol
10. `module_a.py`yi bol
11. `features/core.py`yi bol

### Faz 4: Reproducibility ve release

12. CI ekle
13. checksum manifest ekle
14. release bundle ekle
15. docs klasoru ve model cards ekle

### Faz 5: Yeni bilimsel kazanclar

16. multi-freeze historical benchmark
17. time-to-event secondary benchmark
18. PU / exposure-aware audit
19. H 3.0
20. A 2.0

## 14. Tek Cumlelik Sonuc

Bu proje artik "daha fazla ozellik ekleyelim" asamasinda degil.
En buyuk kazanc bundan sonra su uc seyden gelecek:

- dogru resmi model hiyerarsisini kurmak
- repo ve artifact yuzeyini profesyonelce temizlemek
- zaman / knownness / missingness sorunlarini benchmarkin merkezi parcasi yapmak

Eger bunlar yapilirse proje sadece daha yuksek AUC alan bir repo olmaz; gercekten ciddi, savunulabilir ve birincilik seviyesinde bir biyoinformatik arastirma sistemi haline gelir.
