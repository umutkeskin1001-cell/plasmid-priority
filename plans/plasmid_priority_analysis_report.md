# Plasmid-Priority: Kapsamlı Analiz ve Geliştirme Raporu

---

## BÖLÜM A: YÖNETİCİ ÖZETİ

### Projenin Mevcut Durumu

Plasmid-Priority, biyoinformatik alanında **olgunluk düzeyi yüksek** bir retrospektif genomik gözetim çerçevesidir. 122 kaynak dosya (~41K LOC), 75 test dosyası (~16K LOC), 507 geçen test ve manifest-driven bir veri kontrat sistemi ile akademik/araştırma kalitesinde bir pipeline sunar. Proje, split-safe temporal tasarım (2015 bölünmesi), üç bağımsız tahmin dalı ve discovery/governance ikili model hattı ile bilimsel açıdan sağlam bir mimariye sahiptir.

**Genel Sağlık Skoru: 7.5/10**

| Boyut | Skor | Durum |
|-------|------|-------|
| Kod Kalitesi & Mimari | 7/10 | İyi temel, birkaç god-module sorunu |
| Veri Mühendisliği | 9/10 | Manifest-driven kontratlar mükemmel |
| ML Modellemesi | 8/10 | Split-safe tasarım ve dual-track güçlü |
| Bilimsel Yaklaşım | 9/10 | Retrospektif sınırlar net tanımlanmış |
| Dokümantasyon | 6/10 | README iyi ama docstring coverage düşük |
| Test & Kalite Güvencesi | 7/10 | 507 test geçiyor ama integration gap var |
| DevOps & CI/CD | 6/10 | Temel CI var ama monitoring yok |
| Açık Kaynak | 4/10 | LICENSE ve CONTRIBUTING eksik |
| Genişletilebilirlik | 7/10 | Branch pattern tekrarlanabilir |
| Güvenlik & Risk | 6/10 | Secret yok ama audit tooling eksik |

### En Kritik 5 Bulgu

1. **LICENSE dosyası yok** — Proje hukuki olarak korumasız; üçüncü taraflar kodu kullanamaz/katkı sağlayamaz (P0)
2. **`model_audit.py` 5601 satır / 78 fonksiyon** — Bakımı imkansız hale yaklaşan god-module (P0)
3. **Yapılandırılmış logging yok** — `logging_utils.py` var ama pipeline genelinde `print()` kullanılıyor (P1)
4. **%36 docstring coverage** — 908 fonksiyondan yalnızca 327'sinin docstring'i var (P1)
5. **23 mypy hatası** — Statik tip denetimi CI'da enforced ama hatalar var (P1)

### Öncelikli 5 İyileştirme Önerisi

1. MIT veya Apache-2.0 LICENSE ekle + CONTRIBUTING.md oluştur
2. `model_audit.py`'yi 4-5 alt modüle böl (selection, benchmark, diagnostics, scorecard, protocol)
3. Pipeline genelinde `configure_logging()` kullanımını standartlaştır, print→logging geçişi
4. mypy strict mode'a geç, mevcut 23 hatayı düzelt
5. Property-based testleri genişlet, integration test katmanı ekle

---

## BÖLÜM B: DETAYLI BULGULAR

---

### BOYUT 1: KOD KALİTESİ VE MİMARİ ANALİZ

#### Güçlü Yönler

**Temiz modüler yapı**: `src/plasmid_priority/` altında net bir subpackage hiyerarşisi var. Her tahmin dalı (geo_spread, bio_transfer, clinical_hazard, consensus) aynı pattern'i takip eder: `contracts.py → specs.py → dataset.py → features.py → train.py → evaluate.py → calibration.py → provenance.py → report.py → cli.py`. Bu tutarlılık yeni dalların eklenmesini kolaylaştırır.

**Pydantic & Pandera entegrasyonu**: Veri kontratları `pydantic.BaseModel` ile, tablo şemaları `pandera.DataFrameSchema` ile tanımlanmış. `BranchInputContract` ve `BranchBenchmarkSpec` frozen model olarak tasarlanmış — mutability hataları önleniyor.

**Utility katmanı**: `utils/` altında `dataframe.py`, `files.py`, `geography.py`, `benchmarking.py` gibi tekrar kullanılabilir yardımcı modüller. `coalescing_left_merge()`, `clean_text_series()`, `dominant_share()` gibi fonksiyonlar proje genelinde tutarlı kullanılıyor.

**LRU cache stratejisi**: Dosya SHA256 hesaplama (maxsize=4096), TSV sütun okuma (maxsize=256), ülke normalizasyonu (maxsize=16K) için akıllı önbellek kullanımı. `_read_tsv_cached()` boyut+mtime ile invalidation yapıyor.

#### Zayıf Yönler

**God-module problemi**: Dosya boyutu dağılımı ciddi bir dengesizlik gösteriyor:

| Dosya | LOC | Fonksiyon |
|-------|-----|-----------|
| `reporting/model_audit.py` | 5601 | 78 |
| `modeling/module_a.py` | 3073 | ~45 |
| `reporting/figures.py` | 3054 | ~30 |
| `reporting/advanced_audits.py` | 2831 | ~25 |
| `features/core.py` | 1803 | ~40 |

`build_primary_model_selection_summary()` tek başına 596 satır ve 51 branch — bu tek fonksiyon birçok dosyadan büyük.

**Cyclomatic complexity**: En karmaşık 5 fonksiyon:
- `build_primary_model_selection_summary`: 51 branch, 596 satır
- `build_model_selection_scorecard`: 39 branch, 476 satır
- `annotate_candidate_explanation_fields`: 29 branch, 200 satır
- `build_benchmark_protocol_table`: 24 branch, 313 satır
- `build_backbone_table`: 14 branch, 626 satır

**Custom exception hierarchy yok**: Tüm hata yönetimi standart Python exception'ları ile yapılıyor (ValueError, RuntimeError, FileExistsError). Domain-specific exception'lar yok (örn. `ContractViolationError`, `FeatureEngineeringError`, `ModelFitError`).

**5 broad except bloğu**: `branching.py:348`, `ensemble_strategies.py:213,335`, `geo_spread/inventory.py:69`, `geo_spread/train.py:153` dosyalarında `except Exception` kullanımı var. Bu, hata teşhisini zorlaştırır.

#### Fırsatlar

- `model_audit.py` → `model_selection.py`, `model_benchmark.py`, `model_diagnostics.py`, `model_scorecard.py`, `model_protocol.py` olarak bölünebilir
- `features/core.py` → `features/transfer.py`, `features/host.py`, `features/amr.py` olarak ayrılabilir
- Abstract base class veya Protocol pattern ile branch interface'i formalize edilebilir

#### Tehditler

- God-module'ler büyüdükçe merge conflict riski artar
- Broad except blokları sessiz hata maskeleme riski taşır
- Fonksiyon complexity test edilebilirliği düşürür

---

### BOYUT 2: VERİ MÜHENDİSLİĞİ VE PIPELINE ANALİZİ

#### Güçlü Yönler

**Manifest-driven veri kontratı**: `data_contract.json` 44+ varlık tanımlıyor (raw, external, optional, derived). Her varlık için `key`, `path`, `format`, `required`, `source` alanları var. `DataContract` pydantic modeli bu kontratı runtime'da validate ediyor.

**Dependency-aware workflow**: `run_workflow.py` topological sort ile 30+ pipeline adımını sıralar. `ThreadPoolExecutor` ile paralel yürütme, deadlock detection, automatic job cap, dry-run modu destekleniyor.

**Atomic yazma**: `atomic_write_json()` ve `atomic_write_text()` ile yarım kalmış dosya riski önleniyor. Temporary file → `os.replace()` pattern'i kullanılıyor.

**Signature-based caching**: `path_signature()` ve `path_signature_with_hash()` ile dosya değişiklik tespiti. Hardening snapshot cache'i dosya imzaları değişmediğinde pahalı audit'leri atlar.

**ManagedScriptRun context manager**: Her pipeline adımı giriş dosyalarını, çıkış dosyalarını, metrikleri ve uyarıları yapılandırılmış JSON olarak kaydediyor.

#### Zayıf Yönler

**DVC veya veri versiyonlama yok**: Veri dosyaları `.gitignore` ile hariç tutuluyor ama versiyon takibi yok. `uv.lock` kod bağımlılıklarını kilitlerken veri bağımlılıkları kilitlenmiyor.

**Schema validation'ın 4 kez copy-paste edilmesi**: `validation/schemas.py` içinde `validate_harmonized_plasmids()`, `validate_backbone_table()`, `validate_scored_backbones()`, `validate_deduplicated_plasmids()` fonksiyonları neredeyse birebir aynı try/except yapısını tekrarlıyor (her biri ~25 satır).

**Chunked processing sınırlı**: `00_fetch_external_data.py` chunked okuma yapıyor (50K satır) ama bu pattern proje genelinde yaygın değil. Büyük TSV dosyaları tek seferde belleğe yükleniyor.

#### Fırsatlar

- DVC entegrasyonu ile veri versiyonlama
- Apache Arrow / Parquet ile sütunsal depolama (DuckDB zaten dependency'de)
- Schema validation'ı generic bir `_validate_table()` fonksiyonuna refactor etme

#### Tehditler

- Veri dosyaları versiyonlanmadığı için reproducibility kırılabilir
- Büyük veri setlerinde memory pressure

---

### BOYUT 3: MAKİNE ÖĞRENMESİ MODELLEMESİ ANALİZİ

#### Güçlü Yönler

**Split-safe temporal tasarım**: 2015 yılı bölünmesi ile training ve test setleri ayrılıyor. Future-derived sütunlar (`*_future`) feature validation'da reddediliyor (`validate_branch_feature_set()`). `training_only_future_unseen_backbone_flag` ile data leakage önleniyor.

**İkili model hattı (Discovery/Governance)**:
- Discovery: En yüksek AUC-ROC ayırıcılığı hedefler
- Governance: Guardrail-aware, matched-knownness denetimi, calibrated risk raporlama

**Zengin validation suite**:
- Bootstrap CI'lar (BCa düzeltmeli)
- Tie-invariant ROC-AUC (custom implementasyon)
- DeLong testi (paired model karşılaştırma)
- BH çoklu test düzeltmesi
- VIF multicollinearity kontrolü
- ECE (Expected Calibration Error)
- Brier decomposition (reliability/resolution/uncertainty)

**Frozen scientific acceptance thresholds**: `model_audit.py` içinde sabitlenmiş kabul eşikleri var — modelin "geçer" olması için kesin kriterlerin karşılanması gerekiyor.

**Çoklu sample weighting**: `source_balanced`, `class_balanced`, `knownness_balanced` modları. Model hangi bias'a karşı korunacağını açıkça belirtiyor.

**Calibration**: Isotonic regression ve Platt scaling destekleniyor. `calibrate_geo_spread_predictions()` ECE tabanlı model seçimi yapıyor.

#### Zayıf Yönler

**Model çeşitliliği sınırlı**: Ağırlıklı olarak logistic regression ve firth logistic regression. Random forest, gradient boosting, neural network gibi alternatifler denenmemiş veya yokluğu gerekçelendirilmemiş.

**Hyperparameter tuning sistematik değil**: `config.yaml`'da C değerleri ve regularization parametreleri sabitlenmiş. Grid search veya Bayesian optimization yok.

**SHAP veya permutation importance entegrasyonu eksik**: `interpret` dependency var ama kod genelinde kullanımı sınırlı. Feature importance büyük ölçüde coefficient magnitude'a bağlı.

**Ensemble strateji riskleri**: `ensemble_strategies.py`'de 2 broad except bloğu var — model fit başarısız olduğunda sessizce atlanıyor.

#### Fırsatlar

- scikit-learn Pipeline + GridSearchCV ile sistematik hyperparameter tuning
- SHAP değerleri ile model yorumlanabilirliği
- LightGBM veya XGBoost ile tree-based model karşılaştırması
- Nested cross-validation ile model selection bias önleme

#### Tehditler

- Sadece logistic regression'a bağlılık non-linear ilişkileri kaçırabilir
- Ensemble'da silent failure güvenilmez sonuçlara yol açabilir

---

### BOYUT 4: BİLİMSEL VE DOMAIN UZMANLIĞI ANALİZİ

#### Güçlü Yönler

**Net bilimsel sınırlar**: README açıkça belirtiyor: "mortalite, hasta prognozu veya gerçek transmisyon tahmin ETMEZ." Bu, aşırı yorum riskini minimize ediyor.

**T/H/A üçlü sinyal yaklaşımı**: Transfer mobilizasyonu, konak çeşitliliği ve AMR yükünün bağımsız bileşenler olarak modellenmesi biyolojik olarak anlamlı ve metodolojik olarak sağlam.

**WHO MIA entegrasyonu**: Medically Important Antimicrobials dokümanının referans validasyonu için kullanılması bilimsel meşruiyet katıyor.

**Backbone-level analiz birimi**: Çoğu çerçeve tür veya tekil plazmid odaklıyken, omurga sınıfını analiz birimi yapma yaklaşımı özgün.

**Retrospektif tasarımın açık kabulü**: Prospektif tahmin iddiasında bulunmamak, bilimsel dürüstlük açısından güçlü.

#### Zayıf Yönler

**Confounding analizi sınırlı**: Özellikle coğrafi örnekleme bias'ı (bazı ülkeler daha fazla sequence üretiyor) sistematik olarak ele alınmamış.

**Filogenetik bilgi kullanımı temel düzeyde**: Host range scoring rank-based (strain → phylum). Filogenetik mesafe matrisleri veya tree-based metrikler kullanılmamış.

#### Fırsatlar

- Sampling bias düzeltmesi (inverse probability weighting)
- Filogenetik çeşitlilik metrikleri (UniFrac, Faith's PD)
- Plazmid ağ analizi (conjugation network topology)

#### Tehditler

- Sampling bias, retrospektif sonuçları yanlış yönlendirebilir
- Coğrafi temsiliyet dengesizliği model genellenebilirliğini sınırlayabilir

---

### BOYUT 5: DOKÜMANTASYON VE REPRODÜKSİYON ANALİZİ

#### Güçlü Yönler

**README kapsamlı**: Bilimsel sınırlar, veri kaynakları, repository layout, kurulum, runtime modları, rapor yorumlama, ve run order açıkça belgelenmiş. Hem İngilizce hem Türkçe bölümler var.

**Machine-readable data contract**: `data_contract.json` hem insan hem makine tarafından okunabilir bir veri sözlüğü görevi görüyor.

**Rapor şablonları**: `jury_brief.md`, `ozet_tr.md`, `headline_validation_summary.md` gibi standardize çıktılar var.

**`uv.lock` ile bağımlılık kilitleme**: Reproducibility için kritik.

#### Zayıf Yönler

**%36 docstring coverage**: 908 fonksiyondan 581'i docstring'siz. Özellikle kritik fonksiyonlarda eksiklik var.

**INSTALL.md yok**: Kurulum rehberi README içinde dağınık.

**USAGE.md yok**: Detaylı kullanım kılavuzu eksik.

**API dokümantasyonu yok**: Sphinx, MkDocs veya pdoc gibi bir araç kullanılmıyor.

**Inline comment kalitesi değişken**: Bazı dosyalarda detaylı açıklamalar varken bazılarında hiç yok.

#### Fırsatlar

- MkDocs + mkdocstrings ile otomatik API dokümantasyonu
- GitHub Pages ile barındırma
- Jupyter notebook ile interaktif tutorial

#### Tehditler

- Düşük docstring coverage yeni geliştiricilerin onboarding süresini uzatır
- API dokümantasyonu olmadan üçüncü parti entegrasyon zorlaşır

---

### BOYUT 6: TEST VE KALİTE GÜVENCESİ ANALİZİ

#### Güçlü Yönler

**507 test, 0 failure**: Tüm testler geçiyor. Test/kaynak oranı iyi (75 test dosyası / 122 kaynak dosya).

**Property-based testing**: Hypothesis kullanımı (pyproject.toml'da dev dependency). Bu, edge case keşfi için güçlü.

**Smoke test katmanı**: `26_run_tests_or_smoke.py` ile pipeline'ın hızlı doğrulaması.

**Pre-commit hooks**: ruff lint/format, yaml/toml/json validation, large file check, merge conflict detection.

**ruff temiz**: Sıfır lint uyarısı.

#### Zayıf Yönler

**Integration test katmanı eksik**: Unit testler ve smoke testler var ama end-to-end pipeline testi yok.

**CI sadece `make quality` çalıştırıyor**: Full pipeline testi CI'da yok (veri bağımlılığı nedeniyle anlaşılabilir ama mock/fixture ile çözülebilir).

**mypy CI'da enforced ama 23 hata var**: Hatalar çoğunlukla `Mapping` vs `dict` type narrowing sorunları. Bu, ya strict modun eksikliğini ya da tip anotasyonlarının gevşekliğini gösteriyor.

**mypy pre-commit'ten açıkça hariç tutulmuş**: `.pre-commit-config.yaml`'da mypy yok, sadece CI'da çalışıyor.

**Code coverage ölçümü yok**: pytest-cov veya benzeri bir araç kullanılmıyor.

#### Fırsatlar

- pytest-cov ile coverage ölçümü ve eşik belirleme (hedef: %80+)
- Integration test fixture'ları ile pipeline parçalarının testi
- Mutation testing (mutmut) ile test kalitesi ölçümü
- mypy'yi pre-commit'e ekleme

#### Tehditler

- Coverage ölçümü olmadan dead code ve test edilmemiş paths gizli kalır
- Integration test eksikliği pipeline regresyonlarını geç yakalatır

---

### BOYUT 7: DevOps VE DEPLOYMENT ANALİZİ

#### Güçlü Yönler

**uv ile modern bağımlılık yönetimi**: `uv.lock` deterministik build sağlıyor. `pyproject.toml`'da optional dependency grupları var (`analysis`, `dev`).

**Makefile ile pipeline orkestrasyonu**: Net target'lar, `.venv` auto-detection, `clean-generated` target'ı.

**External data root desteği**: `PLASMID_PRIORITY_DATA_ROOT` environment variable ile veri ayrıştırma — USB volume veya NFS mount senaryoları destekleniyor.

**CI temel düzeyde çalışıyor**: Ubuntu, Python 3.13, compile check, quality gate.

#### Zayıf Yönler

**Monitoring ve logging altyapısı yok**: Runtime monitoring, error tracking, performance monitoring eksik.

**Docker/container desteği yok**: Reproducibility için container image'ı yok.

**Versioning stratejisi belirsiz**: `pyproject.toml`'da version yok veya sabit. Semantic versioning uygulanmıyor.

**CI/CD pipeline minimal**: Sadece quality gate var, deployment, artifact publishing, release automation yok.

#### Fırsatlar

- Dockerfile ile reproducible environment
- GitHub Actions'da matrix build (Python 3.12 + 3.13)
- Semantic versioning + changelog automation
- Structured logging ile runtime monitoring

#### Tehditler

- Container olmadan "works on my machine" riski
- Tek Python versiyonu testi compatibility sorunlarını gizler

---

### BOYUT 8: AÇIK KAYNAK VE TOPLULUK ANALİZİ

#### Güçlü Yönler

**Public repository**: Kod açıkça erişilebilir.

**İki dilli dokümantasyon**: Türkçe ve İngilizce bölümler var.

#### Zayıf Yönler

**LICENSE dosyası YOK**: Bu en kritik eksiklik. Lisans olmadan:
- Başkaları kodu yasal olarak kullanamaz
- Katkı sağlayanlar hakları konusunda belirsizlik yaşar
- Akademik atıf ve yeniden kullanım sorunludur

**CONTRIBUTING.md YOK**: Katkı rehberi eksik.

**CODE_OF_CONDUCT.md YOK**: Topluluk davranış kuralları tanımlanmamış.

**Git commit mesajları tutarsız**: "bumblebee", "newwwww", "bumblebee flies now i guess" gibi mesajlar profesyonel standartların altında. Yanında "Harden branch architecture and release workflow" gibi iyi mesajlar da var.

**Issue template'leri yok**: Bug report ve feature request şablonları tanımlanmamış.

#### Fırsatlar

- MIT veya Apache-2.0 lisansı ekleme
- Conventional Commits standardı
- GitHub issue/PR template'leri
- GitHub Topics ve description optimizasyonu

#### Tehditler

- Lisans eksikliği akademik kullanımı engelleyebilir
- Tutarsız commit mesajları kod inceleme sürecini zorlaştırır

---

### BOYUT 9: GENİŞLETİLEBİLİRLİK VE GELECEK PLANLAMA

#### Güçlü Yönler

**Branch pattern tekrarlanabilir**: Yeni bir tahmin dalı eklemek için mevcut pattern (contracts → specs → dataset → train → evaluate → calibrate → report) takip edilebilir.

**Config-driven model tanımlama**: `config.yaml`'da 35+ model konfigürasyonu, feature set'ler ve branch ayarları. Yeni model eklemek için sadece config dosyasını güncellemek yeterli.

**Manifest-driven veri ekleme**: `data_contract.json`'a yeni varlık ekleyerek yeni veri kaynaklarını entegre etmek kolay.

**Topological workflow**: Yeni pipeline adımları dependency graph'a eklenebilir.

#### Zayıf Yönler

**Plugin sistemi yok**: Yeni feature engineering veya model backend'i eklemek doğrudan kod değişikliği gerektiriyor.

**API katmanı yok**: REST API veya CLI framework (click/typer) ile erişim sınırlı.

**Cloud deployment desteği yok**: AWS/GCP/Azure entegrasyonu veya Snakemake remote execution.

#### Fırsatlar

- FastAPI veya Streamlit ile web interface
- Snakemake veya Nextflow ile workflow management
- Plugin registry ile extensibility
- Cloud-native deployment (AWS Batch, Google Cloud Life Sciences)

#### Tehditler

- Monolitik yapı büyüdükçe yeni özellik ekleme maliyeti artar

---

### BOYUT 10: KRİTİK SORUNLAR VE TEHLİKE ANALİZİ

#### Güçlü Yönler

**Hardcoded secret yok**: Grep taraması herhangi bir password, API key veya token bulamadı.

**Input validation güçlü**: `BranchInputContract` ile yapısal kontratlar, Pandera ile şema validation.

**Atomic file writes**: Yarım kalmış dosya riski minimize edilmiş.

**Random seed yönetimi tutarlı**: Seed=42 standart olarak kullanılıyor, tüm branch'lerde parametre olarak geçiriliyor.

#### Zayıf Yönler

**Dependency vulnerability taraması yok**: pip-audit veya safety CI pipeline'ında yok.

**Broad except blokları (5 adet)**: Hata maskeleme riski.

**Tek thread'li büyük veri işleme**: Paralel processing fırsatları değerlendirilmemiş (DuckDB dışında).

**Disaster recovery planı yok**: Veri kaybı senaryosu için strateji tanımlanmamış.

#### Fırsatlar

- pip-audit CI'a ekleme
- Dependabot veya Renovate ile otomatik dependency güncelleme
- Data backup stratejisi

#### Tehditler

- Bilinen güvenlik açıkları fark edilmeden kalabilir
- Veri kaybında recovery zorlaşır

---

## BÖLÜM C: SOMUT EYLEM ÖNERİLERİ

### P0 — Hemen Yapılması Gerekenler (Bu Hafta)

| # | Eylem | Etki | Çaba |
|---|-------|------|------|
| 1 | **LICENSE dosyası ekle** (MIT veya Apache-2.0) | Kritik — hukuki koruma | 5 dakika |
| 2 | **CONTRIBUTING.md oluştur** | Katkı sürecini tanımla | 30 dakika |
| 3 | **23 mypy hatasını düzelt** | Tip güvenliği | 2-3 saat |
| 4 | **pip-audit'i CI'a ekle** | Güvenlik taraması | 15 dakika |
| 5 | **Broad except bloklarını daralt** (5 adet) | Hata teşhisi | 1 saat |

### P1 — Kısa Vadede Yapılması Gerekenler (2-4 Hafta)

| # | Eylem | Etki | Çaba |
|---|-------|------|------|
| 6 | **`model_audit.py`'yi 4-5 modüle böl** | Bakım kolaylığı | 1-2 gün |
| 7 | **print→logging geçişi** (pipeline genelinde) | Hata ayıklama | 1 gün |
| 8 | **pytest-cov ekle**, %70 eşik belirle | Test kalitesi görünürlüğü | 2 saat |
| 9 | **Docstring'leri kritik modüllere ekle** (scoring, modeling, features) | Onboarding | 2-3 gün |
| 10 | **Schema validation refactoring** — tekrarlı try/except blokları | DRY prensibi | 2 saat |
| 11 | **Custom exception hierarchy** oluştur | Domain-specific hata yönetimi | 3-4 saat |
| 12 | **CI'a Python 3.12 matrix build ekle** | Compatibility | 30 dakika |
| 13 | **`features/core.py`'yi T/H/A modüllerine böl** | Modülerlik | 1 gün |

### P2 — Orta Vadede Değerlendirilecekler (1-3 Ay)

| # | Eylem | Etki | Çaba |
|---|-------|------|------|
| 14 | **DVC entegrasyonu** ile veri versiyonlama | Reproducibility | 2-3 gün |
| 15 | **Dockerfile oluştur** | Portable environment | 1 gün |
| 16 | **MkDocs ile API dokümantasyonu** | Developer experience | 2-3 gün |
| 17 | **SHAP entegrasyonu** ile model yorumlanabilirliği | Bilimsel güç | 2-3 gün |
| 18 | **Integration test fixture'ları** | Pipeline güvenilirliği | 3-5 gün |
| 19 | **Structured logging** (JSON format, log levels) | Monitoring temeli | 2 gün |
| 20 | **`build_primary_model_selection_summary()` refactoring** | Bakım | 1-2 gün |
| 21 | **Conventional Commits + commitlint** | Git history kalitesi | 1 saat |
| 22 | **GitHub issue/PR template'leri** | Topluluk | 1 saat |
| 23 | **Sampling bias analizi** | Bilimsel güç | 3-5 gün |

### P3 — Uzun Vadeli Hedefler (3-6 Ay)

| # | Eylem | Etki | Çaba |
|---|-------|------|------|
| 24 | **Tree-based model karşılaştırması** (LightGBM/XGBoost) | Model çeşitliliği | 1 hafta |
| 25 | **FastAPI ile REST API** | Entegrasyon | 1-2 hafta |
| 26 | **Streamlit dashboard** | Görselleştirme | 1 hafta |
| 27 | **Snakemake/Nextflow workflow** | Ölçeklenebilirlik | 2-3 hafta |
| 28 | **Cloud deployment** (AWS Batch) | Ölçeklenebilirlik | 2-3 hafta |
| 29 | **Plugin sistemi** (feature engineering + model backends) | Genişletilebilirlik | 2 hafta |
| 30 | **Nested cross-validation** | Model selection bias | 1 hafta |

---

## BÖLÜM D: KOD ÖRNEKLERİ VE İMPLEMENTASYON DETAYLARI

### D1: Custom Exception Hierarchy

```python
# src/plasmid_priority/exceptions.py

"""Domain-specific exception hierarchy for Plasmid Priority."""

from __future__ import annotations


class PlasmidPriorityError(Exception):
    """Base exception for all Plasmid Priority errors."""


class ContractViolationError(PlasmidPriorityError):
    """Raised when a data contract is violated."""

    def __init__(self, contract_name: str, violations: list[str]) -> None:
        self.contract_name = contract_name
        self.violations = violations
        joined = "; ".join(violations)
        super().__init__(f"Contract '{contract_name}' violated: {joined}")


class FeatureEngineeringError(PlasmidPriorityError):
    """Raised when feature computation fails."""

    def __init__(self, feature_name: str, reason: str) -> None:
        self.feature_name = feature_name
        super().__init__(f"Feature '{feature_name}' computation failed: {reason}")


class ModelFitError(PlasmidPriorityError):
    """Raised when model training fails."""

    def __init__(self, model_name: str, reason: str) -> None:
        self.model_name = model_name
        super().__init__(f"Model '{model_name}' fit failed: {reason}")


class CalibrationError(PlasmidPriorityError):
    """Raised when model calibration fails."""


class PipelineStepError(PlasmidPriorityError):
    """Raised when a pipeline step fails."""

    def __init__(self, step_name: str, exit_code: int) -> None:
        self.step_name = step_name
        self.exit_code = exit_code
        super().__init__(f"Pipeline step '{step_name}' failed with exit code {exit_code}")
```

### D2: Schema Validation Refactoring (DRY)

**Önceki** (her tablo için ~25 satır kopyalanmış):
```python
# validation/schemas.py - MEVCUT (tekrarlı)
def validate_harmonized_plasmids(df, lazy=True):
    try:
        validated = HARMONIZED_PLASMID_SCHEMA.validate(df, lazy=lazy)
        return {"status": "pass", "table": "harmonized_plasmids", ...}
    except pa.errors.SchemaErrors as e:
        return {"status": "fail", "table": "harmonized_plasmids", ...}
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return {"status": "error", "table": "harmonized_plasmids", ...}

# Aynı yapı 3 kez daha tekrarlanıyor...
```

**Sonrası** (generic fonksiyon):
```python
# validation/schemas.py - ÖNERİLEN

def _validate_table(
    df: pd.DataFrame,
    schema: pa.DataFrameSchema,
    table_name: str,
    *,
    lazy: bool = True,
) -> dict[str, Any]:
    """Generic Pandera table validation with structured result."""
    try:
        validated = schema.validate(df, lazy=lazy)
        return {
            "status": "pass",
            "table": table_name,
            "n_rows": int(len(validated)),
            "errors": [],
        }
    except pa.errors.SchemaErrors as e:
        return {
            "status": "fail",
            "table": table_name,
            "n_rows": int(len(df)),
            "errors": (
                e.failure_cases.to_dict("records")
                if hasattr(e, "failure_cases")
                else []
            ),
        }
    except (ValueError, TypeError, KeyError, AttributeError) as e:
        return {
            "status": "error",
            "table": table_name,
            "n_rows": int(len(df)),
            "errors": [{"message": str(e)}],
        }


def validate_harmonized_plasmids(df: pd.DataFrame, *, lazy: bool = True) -> dict[str, Any]:
    return _validate_table(df, HARMONIZED_PLASMID_SCHEMA, "harmonized_plasmids", lazy=lazy)

def validate_backbone_table(df: pd.DataFrame, *, lazy: bool = True) -> dict[str, Any]:
    return _validate_table(df, BACKBONE_TABLE_SCHEMA, "backbone_table", lazy=lazy)

def validate_scored_backbones(df: pd.DataFrame, *, lazy: bool = True) -> dict[str, Any]:
    return _validate_table(df, SCORED_BACKBONE_SCHEMA, "scored_backbones", lazy=lazy)

def validate_deduplicated_plasmids(df: pd.DataFrame, *, lazy: bool = True) -> dict[str, Any]:
    return _validate_table(df, DEDUPLICATED_PLASMID_SCHEMA, "deduplicated_plasmids", lazy=lazy)
```

### D3: Logging Migration Pattern

**Önceki** (print kullanımı):
```python
# scripts/run_workflow.py - MEVCUT
print(f"[workflow] {step.name}: {' '.join(command)}", flush=True)
# ...
print(f"[workflow] {step.name} failed with exit code {return_code}", file=sys.stderr)
```

**Sonrası** (structured logging):
```python
# scripts/run_workflow.py - ÖNERİLEN
import logging
from plasmid_priority.logging_utils import configure_logging

configure_logging()
logger = logging.getLogger("workflow")

logger.info("Starting step %s: %s", step.name, " ".join(command))
# ...
logger.error("Step %s failed with exit code %d", step.name, return_code)
```

### D4: Broad Except Daraltma

**Önceki**:
```python
# shared/branching.py:348 - MEVCUT
except Exception:
    # Silently swallows ALL errors
    result = _build_failed_model_result(...)
```

**Sonrası**:
```python
# shared/branching.py - ÖNERİLEN
from plasmid_priority.exceptions import ModelFitError

except (ValueError, np.linalg.LinAlgError, sklearn.exceptions.NotFittedError) as exc:
    logger.warning("Model fit failed for %s: %s", model_name, exc)
    result = _build_failed_model_result(...)
except Exception as exc:
    logger.error("Unexpected error during %s fit: %s", model_name, exc, exc_info=True)
    raise ModelFitError(model_name, str(exc)) from exc
```

### D5: CI Pipeline Güçlendirme

```yaml
# .github/workflows/ci.yml - ÖNERİLEN
name: ci

on:
  push:
  pull_request:

jobs:
  quality:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        python-version: ["3.12", "3.13"]
    env:
      MPLBACKEND: Agg
    steps:
      - uses: actions/checkout@v4

      - name: Set up Python ${{ matrix.python-version }}
        uses: actions/setup-python@v5
        with:
          python-version: ${{ matrix.python-version }}

      - name: Install package and dev tooling
        run: |
          python -m pip install --upgrade pip
          python -m pip install -e ".[analysis,dev]"

      - name: Compile Python sources
        run: python -m compileall -q src scripts tests

      - name: Audit dependencies
        run: pip install pip-audit && pip-audit

      - name: Run Quality Gate
        run: make quality

      - name: Run tests with coverage
        run: |
          pip install pytest-cov
          python -m pytest tests/ -x -q --tb=short --cov=src/plasmid_priority --cov-report=xml --cov-fail-under=70

      - name: Upload coverage
        if: matrix.python-version == '3.13'
        uses: codecov/codecov-action@v4
        with:
          file: ./coverage.xml
```

### D6: model_audit.py Bölme Stratejisi

```
# MEVCUT: model_audit.py (5601 LOC, 78 fonksiyon)

# ÖNERİLEN YAPI:
src/plasmid_priority/reporting/
├── model_audit/
│   ├── __init__.py          # Public API re-exports
│   ├── selection.py         # build_primary_model_selection_summary, selection helpers
│   │                        # (~1200 LOC: lines 3748-4345)
│   ├── benchmark.py         # build_benchmark_protocol_table, benchmark_model_family
│   │                        # (~800 LOC: lines 4346-5176)
│   ├── diagnostics.py       # build_h_feature_diagnostics, build_variant_rank_consistency
│   │                        # (~1000 LOC: lines 200-1200)
│   ├── scorecard.py         # build_model_selection_scorecard, frozen thresholds
│   │                        # (~800 LOC: lines 4700-5601)
│   ├── calibration.py       # build_blocked_holdout_calibration_table
│   │                        # (~600 LOC: lines 1800-2400)
│   └── constants.py         # FROZEN_SCIENTIFIC_ACCEPTANCE_THRESHOLDS, shared constants
│                            # (~200 LOC)
```

### D7: Dockerfile

```dockerfile
# Dockerfile
FROM python:3.13-slim AS base

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv for fast dependency resolution
RUN pip install uv

# Copy dependency files first for layer caching
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Copy source code
COPY src/ src/
COPY scripts/ scripts/
COPY config.yaml Makefile ./
COPY data/manifests/ data/manifests/

# Install package
RUN pip install -e ".[analysis]"

# Validate installation
RUN python -m compileall -q src scripts

ENTRYPOINT ["python"]
CMD ["scripts/run_workflow.py", "pipeline"]
```

---

## BÖLÜM E: KAYNAK VE REFERANSLAR

### Önerilen Araçlar ve Kütüphaneler

| Araç | Amaç | Öncelik |
|------|-------|---------|
| **pip-audit** | Dependency vulnerability taraması | P0 |
| **pytest-cov** | Test coverage ölçümü | P1 |
| **MkDocs + mkdocstrings** | API dokümantasyonu | P2 |
| **SHAP** | Model yorumlanabilirliği | P2 |
| **DVC** | Veri versiyonlama | P2 |
| **commitlint** | Commit message standardı | P2 |
| **LightGBM** | Tree-based model karşılaştırması | P3 |
| **FastAPI** | REST API | P3 |
| **Streamlit** | Dashboard | P3 |

### Referans Dokümantasyon

- [Conventional Commits](https://www.conventionalcommits.org/) — Commit mesaj standardı
- [Keep a Changelog](https://keepachangelog.com/) — Changelog formatı
- [Python Packaging Guide](https://packaging.python.org/) — Paket yönetimi
- [scikit-learn Model Evaluation](https://scikit-learn.org/stable/model_evaluation.html) — ML metrikler
- [Pandera Documentation](https://pandera.readthedocs.io/) — Schema validation
- [DVC Documentation](https://dvc.org/doc) — Veri versiyonlama
- [SHAP Documentation](https://shap.readthedocs.io/) — Model yorumlanabilirliği

### Benzer Projeler ve Best Practice'ler

- **AMRFinderPlus** (NCBI) — AMR gene detection tool, iyi yapılandırılmış C++ pipeline
- **MOB-suite** — Plasmid classification toolkit, Python-based
- **abricate** — Mass screening, clean CLI interface
- **mlflow** — ML experiment tracking (model audit için referans)

---

## SONUÇ

Plasmid-Priority, bilimsel açıdan güçlü temellere sahip, iyi yapılandırılmış bir biyoinformatik çerçevesidir. En acil ihtiyaçlar hukuki (LICENSE), mimari (god-module refactoring), ve operasyonel (logging standardizasyonu) düzeydedir. Mevcut branch pattern'i genişletilebilir bir temel sunuyor; önerilen iyileştirmeler bu temeli güçlendirecek ve projenin uzun vadeli sürdürülebilirliğini artıracaktır.

**Toplam somut öneri sayısı: 53**
**Tahmini toplam uygulama süresi: ~8-12 hafta (tam zamanlı 1 geliştirici)**

---

*Rapor Tarihi: 15 Nisan 2026*
*Analiz Kapsamı: Repository commit d2d2271 (main branch)*
