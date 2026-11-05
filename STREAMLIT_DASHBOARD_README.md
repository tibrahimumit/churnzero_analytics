# 📊 E-Commerce Churn Model Dashboard - Streamlit Arayüzü

Streamlit tabanlı interaktif model performans gösterge panosu. Confusion matrices, hata metrikleri ve model karşılaştırması görebilirsiniz.

## 🚀 Hızlı Başlangıç

### 1. Model Eğitimi (Henüz Yapmadıysanız)

Önce modelleri eğitin ve sonuçları kaydedin:

```bash
.ml-venv/bin/python run_ecommerce_training.py
```

Bu komut:
- 5 model eğitir (Logistic Regression, Random Forest, XGBoost, LightGBM, CatBoost)
- Confusion matrices hesaplar
- Metrikleri (F1, Accuracy, Precision, Recall, ROC-AUC, PR-AUC) kaydeder
- Sonuçları `artifacts/results/` klasörüne kaydeder

### 2. Streamlit Arayüzünü Çalıştırın

```bash
.ml-venv/bin/streamlit run app.py
```

Tarayıcınızda otomatik olarak açılacaktır (genellikle `http://localhost:8501`)

## 📋 Arayüz Özellikleri

### 📈 Tab 1: Özet (Summary)

**En İyi Model Özeti:**
- 🏆 En iyi performans gösteren model
- F1 Score, Accuracy, Precision, Recall metrikleri
- Model eğitim süresi

**Model Sıralaması:**
- Tüm modellerin ROC-AUC puanlarının yatay bar grafiği
- Modellerin sıralanmış tablosu

### 🎯 Tab 2: Confusion Matrices (Karmaşıklık Matrisleri)

**Interaktif Seçim:**
- Modeli seçin (dropdown)
- Confusion matrix ısı haritası görüntüleyin

**Confusion Matrix Açıklaması:**
```
            Tahmin
         Neg    Pos
       ┌─────────────┐
     N │ TN  | FP    │   TN = True Negative (Doğru negatif)
Real   │─────┼─────  │   FP = False Positive (Yanlış pozitif)
     P │ FN  | TP    │   FN = False Negative (Yanlış negatif)
       │     |       │   TP = True Positive (Doğru pozitif)
       └─────────────┘
```

**Metrikleri Görüntüleyin:**
- F1 Score, Precision, Recall
- Accuracy, Balanced Accuracy, PR-AUC
- TN, FP, FN, TP değerleri

### 📊 Tab 3: Metrikleri Karşılaştır

**Multi-Select Metrik Seçimi:**
- Karşılaştırmak istediğiniz metrikleri seçin
- F1-Score, Precision, Recall, Test Accuracy, Balanced Accuracy, Test ROC-AUC, PR-AUC

**Dinamik Görselleştirme:**
- Seçilen metriklerin tüm modeller arasındaki karşılaştırması
- Gruplandırılmış bar grafikleri
- Tooltip'lerde detaylı bilgiler

**Metrik Açıklamaları:**
- F1 Score: Precision ve Recall'ın harmonik ortalaması
- Precision: TP / (TP + FP) - Pozitif tahminlerin doğruluğu
- Recall: TP / (TP + FN) - Gerçek pozitiflerin bulunması oranı
- Accuracy: (TP + TN) / Tüm - Genel doğruluk
- ROC-AUC: Sınıf dengesizliğine dayanıklı performans ölçüsü

### 📋 Tab 4: Detaylı Sonuçlar

**Tam Tablo:**
- Tüm modellerin tüm metrikleri
- Scroll yapılabilir, sıralanabilir

**CSV İndirme:**
- 📥 Sonuçları CSV dosyası olarak indirin
- Excel veya diğer araçlarda analiz edin

**Dataset Metadata:**
- Dataset bilgileri (satır sayısı, sütun sayısı)
- Hedef değişken dağılımı
- Özellik türleri (sayısal, kategorik)
- Veri işleme ayarları

## 🎨 Tasarım Özellikleri

- **Dark Theme:** Siyah tema, mata öğrenmeye uygun
- **Responsive Design:** Tüm cihazlarda uygun
- **Interactive Plotly Charts:** Hover, zoom, pan özellikleri
- **Streamlit Styling:** Modern, profesyonel arayüz

## 📊 Metriklerin Tanımı

### Hata Matrisi Metrikleri

| Metrik | Formül | Anlamı |
|--------|--------|--------|
| **Accuracy** | (TP+TN)/(TP+TN+FP+FN) | Genel doğruluk oranı |
| **Precision** | TP/(TP+FP) | Pozitif tahminlerin doğruluğu |
| **Recall (Sensitivity/TPR)** | TP/(TP+FN) | Gerçek pozitiflerin bulunması |
| **F1-Score** | 2·(Precision·Recall)/(Precision+Recall) | Precision ve Recall dengesi |
| **Specificity (TNR)** | TN/(TN+FP) | Gerçek negatiflerin bulunması |

### Sınıflandırma Metrikleri

| Metrik | Anlamı |
|--------|--------|
| **ROC-AUC** | Sınıf dengesizliğine dayanıklı, eşik bağımsız ölçü |
| **PR-AUC** | Precision-Recall eğrisi altındaki alan |
| **Balanced Accuracy** | (Sensitivity + Specificity) / 2 |

## 💾 Proje Yapısı

```
proje/
├── app.py                          # Streamlit dashboard
├── run_ecommerce_training.py       # Model eğitim scripti
├── requirements.txt                # Bağımlılıklar
├── artifacts/
│   ├── models/                     # Kaydedilen model dosyaları
│   ├── plots/                      # Eğitim grafikleri
│   └── results/                    # Model sonuçları CSV
│       ├── structured_model_results.csv
│       ├── structured_metadata.json
│       └── feature_importance csvleri
└── src/ecommerce_churn/
    ├── pipeline.py                 # Eğitim pipeline'ı
    ├── models.py                   # Model tanımları
    ├── preprocessing.py            # Veri işleme
    └── ...
```

## 🔧 Teknik Detaylar

### Kullanılan Teknolojiler

- **Streamlit:** Web arayüzü framework'ü
- **Plotly:** İnteraktif görselleştirme
- **Pandas:** Veri işleme
- **Scikit-learn:** ML modelleri ve metrikleri

### Veriler Nerede Saklanıyor?

Model sonuçları `artifacts/results/` klasöründe saklanıyor:
- `structured_model_results.csv` → Tüm metriklerin ana tablosu
- `structured_metadata.json` → Dataset bilgileri
- `structured_*_importance.csv` → Feature importance dosyaları

## ⚙️ Özelleştirme

### Metrikleri Eklemek/Değiştirmek

`app.py` dosyasında `metric_options` listesini düzenleyin:

```python
metric_options = ["F1-Score", "Precision", "Recall", "Test Accuracy", ...]
```

### Renkler Değiştirmek

CSS section'unda `:root` değişkenlerini düzenleyin:

```css
:root {
  --blue: #3b82f6; 
  --green: #2ecc71; 
  /* ... */
}
```

### Yeni Grafik Eklemek

`create_*_chart()` fonksiyonlarını örnek alarak yeni fonksiyonlar yazın ve tab'lara ekleyin.

## 🐛 Sorun Giderme

### "Model sonuçları bulunamadı" hatası

Çözüm:
```bash
.ml-venv/bin/python run_ecommerce_training.py
```

Modelleri eğiterek sonuçları oluşturun.

### Streamlit kütüphanesi yok hatası

Çözüm:
```bash
.ml-venv/bin/pip install streamlit plotly
```

### Yavaş yükleme

- Cache'i temizleyin: `.streamlit/cache` klasörünü silin
- Dosya boyutunu azaltın (Plotly grafikleri büyük olabilir)

## 📞 İletişim & Destek

Sorular veya önerileriniz için proje sahibine başvurun.

---

**Son Güncelleme:** 2026-09-11  
**Versiyon:** 1.0
