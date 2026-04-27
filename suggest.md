# Plasmid Priority: Acil Eylem ve İyileştirme Planı (Suggest.md)

`optimize.md` dosyasında tespit edilen kusurların giderilmesi ve projenin modern bir veri bilimi / yazılım mühendisliği standardına yükseltilmesi için önerilen, önceliklendirilmiş adımlar.

---

## 🔴 Öncelik 0 (P0) - Sistemi Ayakta Tutacak Kritik Düzeltmeler

### 1. Güvenlik ve API İyileştirmesi
*   **Hemen Yapılmalı:** `src/plasmid_priority/api/app.py` içindeki `!=` bazlı güvensiz kimlik doğrulamasını kaldırın. `hmac.compare_digest` kullanın ve `api_key` ayarlanmamışsa API'nin çalışmayı reddetmesini (fail-closed) sağlayın.
*   **Aksiyon:** Sahte (mock) çalışan `_batch_job` ve GraphQL uç noktalarını gerçek implementasyonlarla değiştirin ya da tamamen silin.

### 2. "Gölge Modüllerin" (Shadow Implementations) İmhası
*   **Hemen Yapılmalı:** `_impl.py` son ekiyle biten tüm o devasa kopya dosyaları (`model_audit.py`, `module_a.py` vb.) silin.
*   **Aksiyon:** Mantığı asıl dosyalarda birleştirin ve versiyon kontrolünü Git'e bırakın. Kodu kopyalamak teknik borcun en tehlikeli halidir.

### 3. Sessiz Hataların (Silent Failures) Önlenmesi
*   **Hemen Yapılmalı:** Kod tabanındaki tüm `except Exception:` bloklarını kaldırın.
*   **Aksiyon:** Sadece beklenilen hataları (örn: `FileNotFoundError`) yakalayın. `module_a.py` içindeki matris ters çevirme hatasını gizleyen `try/except LinAlgError` bloğunu kaldırın veya araştırmacılara uyarı fırlatacak (Warning) şekilde yeniden yapılandırın.

---

## 🟠 Öncelik 1 (P1) - Mimari ve Orkestrasyon Değişiklikleri

### 4. DVC ve Orkestratör Senkronizasyonu
*   **Hemen Yapılmalı:** 900 satırlık `run_workflow.py` dosyasını tamamen silin. Kendi caching mekanizmanızı yazmak yerine bu işi DVC'ye bırakın.
*   **Aksiyon:** Tüm boru hattını (pipeline) sadece `dvc.yaml` üzerinden yönetecek şekilde sadeleştirin veya halihazırda bağımlılıklarda bulunan `snakemake` veya `prefect`'e taşıyın.

### 5. "God-File"ların Parçalanması (Shatter God-Files)
*   **Hemen Yapılmalı:** `module_a.py` (3600 satır) dosyasını parçalara ayırın.
*   **Aksiyon:** 
    *   `src/plasmid_priority/modeling/data_prep.py`
    *   `src/plasmid_priority/modeling/engines.py` (Model wrapper'ları)
    *   `src/plasmid_priority/modeling/evaluation.py` (Metrikler)
    şeklinde alt modüller oluşturarak SRP (Tek Sorumluluk Prensibi) kuralını uygulayın.

### 6. Protokol Hash'inin Ayrıştırılması
*   **Hemen Yapılmalı:** `ScientificProtocol` sınıfındaki raporlama estetiklerini (örn: `ece_max`) ve ham veri işleme parametrelerini birbirinden ayırın.
*   **Aksiyon:** `ExecutionProtocol` (veri önbelleğini etkileyenler) ve `EvaluationProtocol` (sadece son raporları etkileyenler) olmak üzere iki farklı sınıfa bölün. Bu, gereksiz cache invalidation'ı önleyecektir.

---

## 🟡 Öncelik 2 (P2) - Kalite ve Sürdürülebilirlik (Quality & Maintainability)

### 7. Test Kapsamının (Coverage) Gerçekçi Hale Getirilmesi
*   **Hemen Yapılmalı:** `pyproject.toml` içindeki `tool.coverage.run.omit` listesini silin.
*   **Aksiyon:** Çekirdek algoritmalara (özellikle `module_a` ve raporlama mantığına) sahte API çağrıları yerine matematiksel doğrulama yapan birim testleri (Unit Tests) yazın. Aşırı Mocking kullanımını terk edin.

### 8. Hardcoded Yolların Merkezi Yönetimi
*   **Hemen Yapılmalı:** Scriptlerde dağılmış olan `"data/raw/"` gibi dizin yollarını temizleyin.
*   **Aksiyon:** `src/plasmid_priority/config.py` içinde global `pathlib.Path` nesneleri (veya `pydantic-settings`) tanımlayın ve yolları sadece bu merkezi yerden okuyun.

### 9. Tip Güvenliğinin (Type Safety) Zorunlu Kılınması
*   **Hemen Yapılmalı:** `pyproject.toml`'daki `ignore_errors = true` ayarlarını ve koddaki `# type: ignore` yorumlarını temizleyin.
*   **Aksiyon:** Tüm fonksiyonlara doğru Type Hint'leri vererek `mypy`'nin kod kalitesini artırmasına izin verin.
