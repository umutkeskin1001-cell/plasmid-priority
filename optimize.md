# Plasmid Priority: Kapsamlı Mimari ve Kod Kalitesi Denetim Raporu

Bu belge, OmG ajanları (omg-architect, omg-reviewer) tarafından projenin `MAKSİMUM` derinlikte incelenmesi sonucunda tespit edilen tüm kusurları, anti-pattern'leri ve mimari çöküş noktalarını listeler. Proje, dışarıdan profesyonel görünmesine rağmen içeride ciddi yapısal borçlar barındırmaktadır.

---

## 🏗️ 1. Mimari Felaketler ve Yapısal Hatalar (Architecture)

### 1.1. Şizofrenik Çift Orkestrasyon (Schizophrenic Double-Orchestration)
* **Kusur:** Sistem eşzamanlı olarak hem DVC (`dvc.yaml`) hem de ısmarlama (bespoke), devasa bir Python DAG orkestratörü (`run_workflow.py`) tarafından yönetilmeye çalışılıyor. `run_workflow.py`, kendi topolojik sıralamasını, bağımlılık izlemesini ve `tmp/artifact_cache` dizinine yazan özel bir `ArtifactCache` yapısını uygularken; DVC de aynı çıktılar için `cache: true` talimatı alıyor.
* **Sonuç:** Bu durum, önbellek çakışmasını (cache thrashing), boşa harcanan disk G/Ç'sini ve bir sistemin önbelleğinin diğerini geçtiği durum tutarsızlıklarını garanti eder.

### 1.2. Eşzamanlılık Cehaleti ve Ölü Kod (Concurrency Illiteracy & Dead Code)
* **Kusur:** `run_workflow.py` içinde `_get_executor_for_step` adında, CPU'ya bağlı görevleri GIL'den kaçırmak için bir `ProcessPoolExecutor`'a yönlendiren karmaşık bir yapı tanımlanmıştır. **Ancak bu fonksiyon asla çağrılmıyor.** Tüm görevler körü körüne bir `ThreadPoolExecutor`'a gönderiliyor ve `subprocess.run()` üzerinden çalıştırılıyor.

### 1.3. Kırılgan, String'e Bağlı DAG (Brittle, String-Coupled DAG)
* **Kusur:** `scripts/00` ile `51` arasındaki bağımlılıklar `STEP_LIBRARY` içinde sihirli string listeleri olarak kodlanmıştır (örn: `deps=("15_normalize_and_score",)`). Boru hattını (pipeline) değiştirmek için `MODE_DEP_OVERRIDES`, `PIPELINE_STEP_NAMES` gibi birden fazla sözlük arasında string referanslarını avlamak gerekir.

### 1.4. Çökmüş Protokol Katmanı (Catastrophic Hash Cascade & Split-Brain Config)
* **Kusur:** `ScientificProtocol` veri sınıfı, `ece_max` gibi tamamen estetik/raporlama metriklerini bile içeren bir özet (snapshot) alır ve bunu `build_protocol_hash` ile şifreler. Bu hash, `run_workflow.py` tarafından tüm pipeline cache anahtarlarına eklenir. Yani raporlamadaki basit bir eşik değişikliği, tüm veri işleme hattının önbelleğini (cache) patlatır.
* **Kusur:** Çekirdek varsayılanlar `protocol.py` içinde kodlanmışken (`DEFAULT_MATCHED_KNOWNNESS_GAP_MIN = -0.005`), diğer ayarlar `config/*.yaml` içinde yaşıyor ("Split-Brain Configuration").

---

## 💩 2. Kod Kirliliği ve Tasarım Hataları (Code Smells)

### 2.1. Devasa "God-Files" ve Gölge Uygulamalar (Shadow Implementations)
* **Kusur:** Kod tabanı, korkunç bir "kopyala-yapıştır programlama" sendromu yaşıyor. Devasa monolitik modüller (3,000 ila 6,000 satır), alt sınıflara ayrılmak (subclassing) veya yeniden düzenlenmek (refactoring) yerine `_impl.py` varyantları olarak **tamamen** kopyalanmış.
    * `model_audit.py` (5,708 satır) vs. `model_audit.py` (5,792 satır)
    * `module_a.py` (3,392 satır) vs. `module_a.py` (3,607 satır)
    * `advanced_audits.py` (3,330 satır) vs. `advanced_audits.py` (3,264 satır)
    * `figures.py` (3,104 satır) vs. `figures.py` (3,261 satır)
* **Sonuç:** Bu durum, ~15.000 satırdan fazla kelimenin tam anlamıyla kopyalanmış teknik borç yaratıyor ve bakımı imkansız hale getiriyor. Ayrıca `module_a.py` Tek Sorumluluk Prensibini (SRP) hiçe sayarak ham DataFrame G/Ç'sini, lineer cebir işlemlerini ve ML modelleme süreçlerini aynı yere yığıyor.

### 2.2. Sessiz Matematiksel Başarısızlıklar (Silent Mathematical Failures)
* **Kusur:** Ön işleme mantığı (`_fit_knownness_residualizer`), ham matris ters çevirme işlemi (`np.linalg.solve`) yapıyor ve `np.linalg.LinAlgError` hatasını bilinçli olarak yakalayıp sessizce `np.linalg.pinv` (pseudo-inverse) kullanmaya geçiyor. Bu durum, özellik tasarım matrisindeki kritik çoklu doğrusal bağlantı (multicollinearity) sorunlarını aktif olarak araştırmacılardan gizler.

### 2.3. Üretime Sızmış Sahte Uç Noktalar (Fake API Endpoints)
* **Kusur:** `src/plasmid_priority/api/app.py` içindeki üretim API'si aktif olarak sahte (mock) veri dönüyor. `_batch_job(job_id)` hiçbir iş yapmıyor ve statik bir sözlük dönüyor. GraphQL ayrıştırıcısı, gerçek bir AST oluşturmak yerine `if "models" in query:` gibi basit string kontrolleriyle çalışıyor.

### 2.4. Hardcoded Yollar ve Geniş Exception Blokları
* **Kusur:** `data/scores/backbone_scored.parquet` gibi yollar scriptlerin ve çekirdek kütüphanenin içine ("magic strings" olarak) gömülmüş.
* **Kusur:** Kod tabanında 40'tan fazla yerde geniş kapsamlı `except Exception:` kullanımı var, bu da kritik çalışma zamanı çökmelerini (runtime crashes) ve eksik dosyaları gizliyor.

---

## 🧪 3. Güvenlik, Tip Güvenliği ve Test İhmalleri

### 3.1. Aldatıcı Test Kapsamı (The Coverage Illusion)
* **Kusur:** `pyproject.toml` içinde `fail_under = 70` metrik şartı koşulmasına rağmen, **30'dan fazla en kritik ve karmaşık modül** `tool.coverage.run.omit` bloğuna eklenerek kapsam raporlarından yapay olarak gizlenmiştir. Tüm o devasa 5,000 satırlık `_impl.py` dosyaları, `module_a.py`, API uç noktaları test edilmemektedir.

### 3.2. Fail-Open Kimlik Doğrulama Zafiyeti (Security Risk)
* **Kusur:** `src/plasmid_priority/api/app.py` içindeki yetkilendirme zorlaması mantığında: `if settings.api_key and _extract_api_key(request) != settings.api_key:` kullanılmış.
* **Sonuç:** Eğer `settings.api_key` boş (unset) bırakılırsa, API "fail-open" durumuna düşer ve kimlik doğrulamasını tamamen atlayarak herkese açık hale gelir. Ayrıca `!=` kullanımı zamanlama (timing) saldırılarına açıktır.

### 3.3. Tip Güvenliğinin İptal Edilmesi (Type Safety Bypass)
* **Kusur:** `pyproject.toml` içindeki `[[tool.mypy.overrides]]` blokları, sistemin en yapısal parçaları için `ignore_errors = true` ve `disallow_untyped_defs = false` ayarlarını içeriyor. Kod içine serpiştirilmiş 40'ın üzerinde `# type: ignore` komutuyla birleştiğinde `mypy`'nin tüm değeri sıfırlanıyor.
