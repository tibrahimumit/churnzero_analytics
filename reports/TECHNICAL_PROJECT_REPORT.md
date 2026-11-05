# E-commerce Customer Churn Prediction
## Güncel Teknik Proje Raporu

**Tarih:** 2026-09-07  
**Dataset:** `data/raw/ecommerce_customer_churn_dataset.csv`  
**Target:** `Churned`  
**Önerilen model:** CatBoost

## 1. Güncel Proje Yapısı

Eski Telecom denemesi projeden kaldırıldı. Güncel proje yalnızca e-commerce churn dataset’i ile çalışır.

```text
proje/
├── data/
│   ├── raw/ecommerce_customer_churn_dataset.csv
│   └── processed/cleaned_ecommerce_customer.csv
├── src/ecommerce_churn/
│   ├── config.py
│   ├── data.py
│   ├── preprocessing.py
│   ├── models.py
│   ├── pipeline.py
│   └── visualization.py
├── scripts/clean_ecommerce_dataset.py
├── artifacts/
│   ├── models/
│   ├── results/
│   └── plots/
├── reports/
├── run_ecommerce_training.py
├── requirements.txt
└── README.md
```

Tek çalıştırma komutu:

```bash
.ml-venv/bin/python run_ecommerce_training.py
```

## 2. Dataset Profili

Raw dataset ilk profillemede 50.000 satır ve 25 sütundan oluşuyordu.

- Target: `Churned`
- Numeric sütun: 21, target dahil
- Categorical sütun: 4
- Identifier sütunu: bulunmadı
- Exact duplicate: 0
- Satır uzunlukları: tutarlı

Categorical sütunlar:

- `Gender`
- `Country`
- `City`
- `Signup_Quarter`

Target dağılımı:

- `Churned = 0`: 35.550, `%71.10`
- `Churned = 1`: 14.450, `%28.90`

Majority-class baseline yaklaşık `%71.10` olduğu için accuracy tek başına değerlendirme metriği değildir.

## 3. Veri Temizleme

Raw dataset değiştirilmedi. Temiz çıktı:

- `data/processed/cleaned_ecommerce_customer.csv`
- Temizlik scripti: `scripts/clean_ecommerce_dataset.py`
- Temizlik raporu: `reports/ecommerce_cleaning_report.md`

Uygulanan filtreler:

- Age boş olan 2.545 satır kaldırıldı.
- `Age > 65` olan 458 satır kaldırıldı.
- `Session_Duration_Avg` boş olan 3.182 satır kaldırıldı.
- Final cleaned dataset: 43.815 satır, 25 sütun.
- Final yaş aralığı: 18-65.
- Final duplicate: 0.

Geçersiz değerler missing’e çevrildi:

- `Age`: 50
- `Cart_Abandonment_Rate`: 30
- `Discount_Usage_Rate`: 207
- `Total_Purchases`: 40
- Toplam: 327

Sayısal hassasiyetler sütun tipine göre standardize edildi. Örneğin `Membership_Years` tek, para ve değer alanları uygun şekilde iki ondalık basamakla kaydedildi.

Görüntüde boş hücre kalmaması için temizleme çıktısındaki kalan missing değerler numeric median ve categorical mode ile dolduruldu. Model eğitiminde ise preprocessing tekrar pipeline içinde ve yalnızca train verisinde öğrenildi.

## 4. Modelleme Pipeline’ı

Güncel kodun ana parçaları:

- `src/ecommerce_churn/data.py`: raw veriyi yükler ve model öncesi filtreleri uygular.
- `src/ecommerce_churn/preprocessing.py`: train-only imputation, encoding ve scaling işlemlerini tanımlar.
- `src/ecommerce_churn/models.py`: beş modelin ayarlarını içerir.
- `src/ecommerce_churn/pipeline.py`: split, eğitim, metrik, model kaydı ve grafik üretimini yönetir.
- `src/ecommerce_churn/visualization.py`: tüm görselleştirmeleri üretir.
- `run_ecommerce_training.py`: tek giriş noktasıdır.

Train/test split:

- Train: `%80`
- Test: `%20`
- Stratified split: aktif
- `random_state=42`
- Test seti preprocessing fit sürecine dahil edilmedi.

Preprocessing:

- Logistic Regression: median imputation, mode imputation, one-hot encoding, StandardScaler.
- Random Forest/XGBoost/LightGBM: median/mode imputation ve one-hot encoding.
- CatBoost: numeric median imputation ve native categorical feature kullanımı.

## 5. Güncel Model Sonuçları

Kaynak: `artifacts/results/structured_model_results.csv`

| Model | Test Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | **92.09%** | **87.89%** | **92.58%** | 78.24% | **84.81%** | 0.9184 | 0.8977 |
| LightGBM | 92.05% | 87.88% | 92.32% | **78.32%** | 84.75% | **0.9186** | **0.9019** |
| XGBoost | 91.99% | 87.68% | **92.63%** | 77.79% | 84.56% | 0.9180 | 0.8960 |
| Random Forest | 91.02% | 85.96% | 92.32% | 74.35% | 82.37% | 0.9123 | 0.8729 |
| Logistic Regression | 77.34% | 65.66% | 66.92% | 38.88% | 49.18% | 0.7794 | 0.6168 |

### CatBoost confusion matrix

- True Negative: 6.136
- False Positive: 155
- False Negative: 538
- True Positive: 1.934

CatBoost gerçek churn müşterilerinin yaklaşık `%78.24`ünü yakalıyor. False Negative oranı yaklaşık `%21.76`.

## 6. Overfitting Kontrolü

| Model | Train Accuracy | Test Accuracy | Accuracy Gap | ROC-AUC Gap |
|---|---:|---:|---:|---:|
| CatBoost | 93.41% | 92.09% | **1.32 puan** | 2.18 puan |
| LightGBM | 93.57% | 92.05% | 1.52 puan | 3.09 puan |
| XGBoost | 93.72% | 91.99% | 1.73 puan | 2.97 puan |
| Random Forest | 94.52% | 91.02% | 3.50 puan | 5.21 puan |
| Logistic Regression | 78.08% | 77.34% | 0.74 puan | 0.94 puan |

CatBoost ve LightGBM iyi genelliyor. Random Forest daha yüksek overfitting gösteriyor. Logistic Regression’da gap düşük olsa da model nonlinear ilişkileri yakalayamadığı için underfitting eğilimi var.

## 7. Görselleştirmeler

Kod: `src/ecommerce_churn/visualization.py`  
Çıktı klasörü: `artifacts/plots/`

- `structured_metric_comparison.png`: model metriklerinin karşılaştırması
- `structured_target_distribution.png`: target sınıf dağılımı
- `structured_roc_curves.png`: ROC eğrileri
- `structured_confusion_matrices.png`: confusion matrix’ler
- `structured_feature_importance.png`: ağaç modellerinin top-20 feature importance grafikleri

## 8. Model Dosyaları ve Sonuçlar

Model dosyaları `artifacts/models/` içindedir:

- `structured_catboost.pkl`
- `structured_lightgbm.pkl`
- `structured_xgboost.pkl`
- `structured_random_forest.pkl`
- `structured_logistic_regression.pkl`

Sonuç dosyaları `artifacts/results/` içindedir:

- `structured_model_results.csv`
- `structured_metadata.json`
- `structured_catboost_importance.csv`
- `structured_lightgbm_importance.csv`
- `structured_xgboost_importance.csv`
- `structured_random_forest_importance.csv`

## 9. Final Karar

Genel denge açısından **CatBoost** önerilmektedir:

1. En yüksek test accuracy: `%92.09`
2. En yüksek F1-score: `0.8481`
3. En yüksek precision: `%92.58`
4. Düşük train-test accuracy farkı: `1.32 puan`
5. Numeric ve categorical feature’ları uygun şekilde işleyebilmesi

LightGBM güçlü alternatiftir; ROC-AUC ve PR-AUC değerleri CatBoost’tan biraz daha yüksektir. Churn kaçırma maliyeti daha önemliyse threshold, validation seti ve iş maliyetleri kullanılarak ayrıca belirlenmelidir.

Streamlit ve deployment henüz yapılmadı.
