# Pitch Notes

## Olası Jüri Soruları ve Yanıtları

**S: "Model sadece zaten iyi bilinen büyük backbone'ları mı buluyor?"**
C: `baseline_both` ROC AUC `0.722`, `discovery_12f_source` ROC AUC `0.804`. Delta `0.081, 95% CI [0.046, 0.116]`; bu, biyolojik sinyalin yalnızca popülerlik sayacı olmadığını gösterir.

**S: "Tüm modeller strict testi kaybediyorsa metodoloji geçerli mi?"**
C: Evet. Strict matched-knownness/source-holdout testi en zor alt kohortu izole eder. Burada başarısız olmak metodolojinin çöktüğünü değil, mevcut veri yoğunluğunun bu alt dilimde sınırlı olduğunu gösterir. Bu kısıt raporda proaktif olarak açıkça belirtilir.

**S: "29 özellik, 989 örnek; overfit değil mi?"**
C: Ana model L2 düzenlemeli lojistik regresyondur ve OOF tahminlerle değerlendirilir. Katsayı kararlılığı için 5-fold CV özeti ayrı verilir; en kararlı örnek sinyaller: `orit_support, T_raw_norm, H_obs_specialization_norm`.

**S: "Bu gerçek bir tahmin sistemi mi, yoksa retrospektif analiz mi?"**
C: Bu çalışma kasıtlı olarak retrospektiftir. Soru şudur: eğitim dönemindeki genomik sinyaller, sonraki dönemdeki coğrafi görünürlük artışıyla ilişkili mi? Prospektif klinik erken uyarı iddiası yapılmaz.

**S: "Governance modeli neden discovery modelinden ayrı?"**
C: Discovery modeli `discovery_12f_source` ayırma gücünü optimize eder. Governance watch-only modeli `phylo-support fusion model` ise kalibrasyon, belirsizlik ve abstention davranışını öne çıkarır; en yüksek AUC'u kovalamaz.
