# E-commerce Customer Churn

Modüler ve leakage kontrollü e-commerce müşteri churn projesi. Proje; veri temizleme, train-only preprocessing, beş model eğitimi, metrik karşılaştırması ve otomatik görselleştirme adımlarını içerir.

## Hızlı Başlangıç

Aktif sanal ortamı kullanarak bağımlılıkları kur:

```bash
.ml-venv/bin/pip install -r requirements.txt
```

Model eğitimi, değerlendirme ve grafik üretimi:

```bash
.ml-venv/bin/python run_ecommerce_training.py
```

Bu tek giriş noktası:

1. Raw dataset’i yükler.
2. Model öncesi deterministik filtreleri uygular.
3. Stratified `%80/%20` train/test split yapar.
4. Imputer ve encoder’ları yalnızca train verisinde fit eder.
5. Beş modeli eğitir.
6. Test metriklerini ve confusion matrix değerlerini kaydeder.
7. Model dosyalarını ve grafiklerini üretir.

Veri temizleme script’i:

```bash
.ml-venv/bin/python scripts/clean_ecommerce_dataset.py
```

Bu script görsel inceleme için processed CSV üretir. Model eğitim pipeline’ı ise leakage riskini önlemek için raw veriyi alıp imputation işlemlerini model pipeline’ı içinde train-only yapar.

## Proje Yapısı

```text
proje/
├── data/
│   ├── raw/
│   │   └── ecommerce_customer_churn_dataset.csv
│   └── processed/
│       └── cleaned_ecommerce_customer.csv
├── src/ecommerce_churn/
│   ├── config.py          # yollar, target ve split ayarları
│   ├── data.py            # veri yükleme ve model filtreleri
│   ├── preprocessing.py   # train-only imputation/encoding/scaling
│   ├── models.py          # beş model tanımı
│   ├── pipeline.py        # eğitim, metrik, kayıt akışı
│   └── visualization.py   # tüm grafik fonksiyonları
├── scripts/
│   └── clean_ecommerce_dataset.py
├── artifacts/
│   ├── models/             # kaydedilmiş model pipeline’ları
│   ├── results/            # CSV metrikleri, metadata, feature importance
│   └── plots/              # PNG grafikler
├── reports/
│   ├── ecommerce_cleaning_report.md
│   └── TECHNICAL_PROJECT_REPORT.md
├── run_ecommerce_training.py
├── requirements.txt
└── README.md
```

## Dataset

Raw dosya: `data/raw/ecommerce_customer_churn_dataset.csv`

- İlk boyut: 50.000 satır, 25 sütun
- Target: `Churned`
- Categorical sütunlar: `Gender`, `Country`, `City`, `Signup_Quarter`
- Identifier sütunu: bulunamadı
- Initial target dağılımı: `0 = 35.550`, `1 = 14.450`

Temizlik sonrası processed dosya:

- `data/processed/cleaned_ecommerce_customer.csv`
- 43.815 satır, 25 sütun
- `Age` boş kayıtları ve `Age > 65` kayıtları kaldırıldı.
- `Session_Duration_Avg` boş kayıtları kaldırıldı.
- Açık invalid değerler missing’e çevrildi.
- Numeric formatlar sütun tipine göre standardize edildi.
- Görsel CSV için kalan missing değerler median/mode ile dolduruldu.

Raw dataset hiçbir zaman değiştirilmez.

## Modeller

Pipeline şu beş modeli karşılaştırır:

- Logistic Regression
- Random Forest
- XGBoost
- LightGBM
- CatBoost

Preprocessing model tipine göre ayrıdır:

- Logistic Regression: median/mode imputation, One-Hot Encoding, StandardScaler
- Random Forest, XGBoost, LightGBM: median/mode imputation ve One-Hot Encoding
- CatBoost: numeric median imputation ve native categorical feature kullanımı

Split ayarları:

- `test_size=0.20`
- `random_state=42`
- `stratify=y`

## Güncel Sonuçlar

Kaynak: [structured_model_results.csv](artifacts/results/structured_model_results.csv)

| Model | Test Accuracy | Balanced Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|---:|
| CatBoost | **92.09%** | **87.89%** | **92.58%** | 78.24% | **84.81%** | 0.9184 | 0.8977 |
| LightGBM | 92.05% | 87.88% | 92.32% | **78.32%** | 84.75% | **0.9186** | **0.9019** |
| XGBoost | 91.99% | 87.68% | **92.63%** | 77.79% | 84.56% | 0.9180 | 0.8960 |
| Random Forest | 91.02% | 85.96% | 92.32% | 74.35% | 82.37% | 0.9123 | 0.8729 |
| Logistic Regression | 77.34% | 65.66% | 66.92% | 38.88% | 49.18% | 0.7794 | 0.6168 |

Çoğunluk sınıfı baseline accuracy yaklaşık `%71.10` olduğu için Balanced Accuracy, Recall, F1 ve ROC-AUC birlikte değerlendirilmelidir.

### Overfitting özeti

| Model | Train Accuracy | Test Accuracy | Accuracy Gap |
|---|---:|---:|---:|
| CatBoost | 93.41% | 92.09% | 1.32 puan |
| LightGBM | 93.57% | 92.05% | 1.52 puan |
| XGBoost | 93.72% | 91.99% | 1.73 puan |
| Random Forest | 94.52% | 91.02% | 3.50 puan |
| Logistic Regression | 78.08% | 77.34% | 0.74 puan |

CatBoost final öneridir: en yüksek test accuracy ve F1-score’a, düşük train-test farkına sahiptir. LightGBM ROC-AUC ve PR-AUC açısından güçlü alternatiftir.

## Görselleştirmeler

Tüm grafikler `src/ecommerce_churn/visualization.py` içindeki fonksiyonlarla oluşturulur:

- [Model metric comparison](artifacts/plots/structured_metric_comparison.png)
- [Target distribution](artifacts/plots/structured_target_distribution.png)
- [ROC curves](artifacts/plots/structured_roc_curves.png)
- [Confusion matrices](artifacts/plots/structured_confusion_matrices.png)
- [Feature importance](artifacts/plots/structured_feature_importance.png)

## Çıktılar

Model dosyaları:

- [CatBoost](artifacts/models/structured_catboost.pkl)
- [LightGBM](artifacts/models/structured_lightgbm.pkl)
- [XGBoost](artifacts/models/structured_xgboost.pkl)
- [Random Forest](artifacts/models/structured_random_forest.pkl)
- [Logistic Regression](artifacts/models/structured_logistic_regression.pkl)

Sonuç ve metadata:

- [Model results](artifacts/results/structured_model_results.csv)
- [Experiment metadata](artifacts/results/structured_metadata.json)
- [CatBoost importance](artifacts/results/structured_catboost_importance.csv)
- [LightGBM importance](artifacts/results/structured_lightgbm_importance.csv)
- [XGBoost importance](artifacts/results/structured_xgboost_importance.csv)
- [Random Forest importance](artifacts/results/structured_random_forest_importance.csv)

## Raporlar

- [Cleaning report](reports/ecommerce_cleaning_report.md)
- [Technical project report](reports/TECHNICAL_PROJECT_REPORT.md)

Streamlit ve deployment henüz eklenmedi.
