# 🎯 HIZLI BAŞLANGIÇ REHBERI

## 📊 Streamlit Dashboard'ı Hemen Çalıştır

### Adım 1: Terminal'i Aç ve Proje Klasörüne Git

```bash
cd /Users/ibrahim/Desktop/proje
```

### Adım 2: Dashboard'ı Başlat

```bash
./run_dashboard.sh
```

VEYA doğrudan:

```bash
.ml-venv/bin/streamlit run app.py
```

### Adım 3: Tarayıcıda Görüntüle

Otomatik açılacaktır ya da ziyaret edin: **http://localhost:8501**

---

## 📈 Dashboard Sekmelerini Keşfet

### 🏆 TAB 1: ÖZET
- En iyi performans gösteren model
- Tüm modellerin ROC-AUC karşılaştırması
- Hızlı metrik özeti

### 🎯 TAB 2: CONFUSION MATRICES
- Model seçin (dropdown)
- Karmaşıklık matrisini görüntüleyin
- F1 Score, Accuracy, Precision, Recall, vb.

### 📊 TAB 3: METRİKLERİ KARŞILAŞTIR
- Metrikleri seçin (multi-select)
- Tüm modeller arasında karşılaştır
- İnteraktif bar grafiği

### 📋 TAB 4: DETAYLΙ SONUÇLAR
- Tüm metriklerin tam tablosu
- CSV olarak indirme
- Dataset bilgileri (metadata)

---

## 📊 Model Performansı Hızlı Bakış

| Sıra | Model | ROC-AUC | F1-Score | Accuracy |
|------|-------|---------|----------|----------|
| 1️⃣ | LightGBM | 0.9186 | 0.8475 | 0.9205 |
| 2️⃣ | CatBoost | 0.9184 | 0.8481 | 0.9209 |
| 3️⃣ | XGBoost | 0.9180 | 0.8456 | 0.9199 |
| 4️⃣ | Random Forest | 0.9123 | 0.8237 | 0.9102 |
| 5️⃣ | Logistic Regression | 0.7794 | 0.4918 | 0.7734 |

---

## 🎨 Arayüz Özellikleri

✅ **Dark Theme** - Gözlere kolay  
✅ **İnteraktif Grafikler** - Zoom, Pan, Hover  
✅ **Responsive Tasarım** - Tüm cihazlarda  
✅ **Türkçe Etiketler** - Anlaşılması kolay  
✅ **CSV İndirme** - Sonuçları dışarı aktarın  

---

## 💡 İpuçları

| Soru | Cevap |
|------|-------|
| 🤔 Hangi model seçmeliyim? | LightGBM veya CatBoost en iyi sonuç veriyor |
| 🔍 Hangi metrik önemli? | F1-Score ve Recall churn tespiti için kritik |
| 💾 Sonuçları nasıl kaydederim? | Tab 4'te CSV İndir butonunu kullanın |
| ⚡ Dashboard yavaş mı? | Tarayıcıyı yenileyin ya da cache'i temizleyin |

---

## 🚀 Gelişmiş Kullanım

### Modelleri Yeniden Eğit

```bash
.ml-venv/bin/python run_ecommerce_training.py
```

### Streamlit Ayarlarını Değiştir

`.streamlit/config.toml` dosyasını edit edin:

```toml
[theme]
primaryColor="#3b82f6"
backgroundColor="#050505"
secondaryBackgroundColor="#121212"
```

### Debug Modu

```bash
.ml-venv/bin/streamlit run app.py --logger.level=debug
```

---

## 📞 Dosyalar & Linkler

| Dosya | Açıklama |
|-------|----------|
| `app.py` | Streamlit arayüzü (Ana dosya) |
| `STREAMLIT_DASHBOARD_README.md` | Detaylı döküman |
| `PROJECT_SUMMARY.md` | Proje özeti |
| `run_dashboard.sh` | Hızlı başlatma scripti |
| `artifacts/results/` | Model sonuçları CSV |

---

## ✨ Başarılı!

Artık model performansınızı profesyonel bir arayüzde görüntüleyebilirsiniz! 🎉

**Soru mu var?** Dökümentasyona bakın:
- 📖 STREAMLIT_DASHBOARD_README.md
- 📖 PROJECT_SUMMARY.md

---

**Happy Analyzing! 📊✨**
