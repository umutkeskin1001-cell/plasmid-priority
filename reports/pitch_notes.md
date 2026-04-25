# Pitch Notes

## Olası Jüri Soruları ve Yanıtları

**S: "Model sadece zaten iyi bilinen büyük backbone'ları mı buluyor?"**
C: `baseline_both` ROC AUC `0.736`, `discovery_boosted` ROC AUC `0.811`. Delta `+0.074 [+0.040, +0.107]`; bu, biyolojik sinyalin yalnızca popülerlik sayacı olmadığını gösterir.

**S: "Tüm modeller strict testi kaybediyorsa metodoloji geçerli mi?"**
C: Evet. Strict matched-knownness/source-holdout testi en zor alt kohortu izole eder. Burada başarısız olmak metodolojinin çöktüğünü değil, mevcut veri yoğunluğunun bu alt dilimde sınırlı olduğunu gösterir. Bu kısıt raporda proaktif olarak açıkça belirtilir.

**S: "29 özellik, 989 örnek; overfit değil mi?"**
C: Ana model L2 düzenlemeli lojistik regresyondur ve OOF tahminlerle değerlendirilir. Katsayı kararlılığı için 5-fold CV özeti ayrı verilir; en kararlı örnek sinyaller: `NA`.

**S: "Bu gerçek bir tahmin sistemi mi, yoksa retrospektif analiz mi?"**
C: Bu çalışma kasıtlı olarak retrospektiftir. Soru şudur: eğitim dönemindeki genomik sinyaller, sonraki dönemdeki coğrafi görünürlük artışıyla ilişkili mi? Prospektif klinik erken uyarı iddiası yapılmaz.

**S: "Governance modeli neden discovery modelinden ayrı?"**
C: Discovery modeli `discovery-boosted primary model` ayırma gücünü optimize eder. Governance watch-only modeli `governance linear model` ise kalibrasyon, belirsizlik ve abstention davranışını öne çıkarır; en yüksek AUC'u kovalamaz.
