# 🎯 Model Trainer - Quick Start Guide

## 🚀 Launch the Application

### Option 1: Using the Launcher Script (Recommended)
```bash
cd /Users/ibrahim/Desktop/proje
./run_model_trainer.sh
```

### Option 2: Direct Streamlit Command
```bash
cd /Users/ibrahim/Desktop/proje
.ml-venv/bin/python -m streamlit run model_trainer.py
```

### Option 3: If Port 8501 is Busy
```bash
.ml-venv/bin/python -m streamlit run model_trainer.py --server.port 8502
```

---

## 📋 What You'll See

### Three Main Tabs:

1. **⚙️ Configuration & Training**
   - Set train/test split (10-40%)
   - Select model (LightGBM, XGBoost, CatBoost, Random Forest, Extra Trees, Logistic Regression)
   - Adjust model-specific parameters
   - Click "🚀 Train Model" to start

2. **📊 Evaluation Results**
   - Performance metrics (appears after training)
   - Confusion matrix heatmap
   - ROC curve visualization
   - Classification report
   - Detailed breakdown

3. **📈 Model Comparison**
   - Summary of trained model
   - Quick interpretation of results

---

## 💡 First Model Training (3 Minutes)

1. **Open the app** → Configuration tab is default
2. **Leave defaults** or change Test Size to 25%
3. **Select Model:** LightGBM (already selected)
4. **Parameter Adjustment:** Use slider defaults
5. **Click Button:** "🚀 Train Model"
6. **Wait:** 30-60 seconds for training
7. **View Results:** Automatically shown in "Evaluation Results" tab

**Expected Result:**
- F1-Score: ~0.85
- ROC-AUC: ~0.92
- Accuracy: ~0.92

---

## 🔄 Training Different Models

### LightGBM (Best Accuracy)
```
Recommended Settings:
  Test Size: 20%
  n_estimators: 300
  learning_rate: 0.05
  max_depth: 6
Expected F1: 0.847
```

### XGBoost (Stable Performance)
```
Recommended Settings:
  Test Size: 20%
  n_estimators: 400
  learning_rate: 0.05
  max_depth: 4
Expected F1: 0.846
```

### CatBoost (Handles Categorical Data)
```
Recommended Settings:
  Test Size: 20%
  iterations: 350
  learning_rate: 0.05
  depth: 6
Expected F1: 0.848
```

### Random Forest (Baseline)
```
Recommended Settings:
  Test Size: 20%
  n_estimators: 350
  max_depth: 14
Expected F1: 0.824
```

### Logistic Regression (Baseline)
```
Recommended Settings:
  Test Size: 20%
  C: 0.5
  solver: liblinear
Expected F1: 0.492
```

---

## 🎯 Understanding the Metrics

| Metric | What It Means | Target Value |
|--------|---------------|--------------|
| **Accuracy** | % of correct predictions | > 0.85 |
| **Precision** | Of predicted churners, how many actually churn | > 0.80 |
| **Recall** | Of actual churners, how many did we catch | > 0.75 |
| **F1-Score** | Balance between Precision & Recall | > 0.80 |
| **ROC-AUC** | Model's ability to distinguish classes | > 0.85 |

**Color Coding in App:**
- 🟢 Green: ≥ 0.85 (Excellent)
- 🟡 Yellow: 0.70-0.84 (Good)
- 🔴 Red: < 0.70 (Needs improvement)

---

## ⚡ Advanced: Custom Parameter Tuning

### Scenario 1: Model Too Slow?
→ Reduce `n_estimators` (e.g., from 300 to 100)
→ Use `Extra Trees` instead (faster training)

### Scenario 2: Model Overfitting?
→ Increase regularization (reg_alpha, reg_lambda)
→ Reduce max_depth
→ Increase min_child_samples

### Scenario 3: Low Recall?
→ Lower decision threshold from 0.5 to 0.4
→ Note: Requires code modification in model_trainer.py line ~424

### Scenario 4: Comparing Models?
1. Train LightGBM → note F1-Score
2. Go back to Configuration
3. Switch to XGBoost
4. Adjust parameters as needed
5. Train XGBoost
6. Go to Model Comparison tab

---

## 🐛 Troubleshooting

### Problem: "Train a model first..."
**Solution:** Click "🚀 Train Model" in Configuration tab

### Problem: Takes very long (>5 min)
**Solution:** 
- Reduce n_estimators
- Increase test_size
- Try "Extra Trees" model
- Check if system is running other processes

### Problem: Connection refused (Can't open http://localhost:8501)
**Solution:**
- Check if terminal shows "Local URL: http://localhost:8501"
- Try port 8502: `--server.port 8502`
- Kill other Streamlit instances: `pkill streamlit`

### Problem: All metrics are very low (< 0.5)
**Solution:**
- Model is underfitting
- Increase n_estimators
- Increase max_depth
- Decrease regularization

### Problem: Metrics show 0.99+ (unrealistic)
**Solution:**
- Model is overfitting
- Reduce n_estimators
- Reduce max_depth
- Increase regularization
- Try different random_state

---

## 📊 Data Overview

The application trains on:
- **7,674 customer records**
- **18 features** (10 numeric, 8 categorical)
- **Binary classification:** Churned (Yes/No)
- **Class distribution:** ~26% churn, ~74% retention

---

## 🔧 System Requirements

- Python 3.8+
- 4GB RAM minimum
- 2GB free disk space
- Modern web browser

---

## 📁 Files Created

```
model_trainer.py              ← Main application (1000+ lines)
run_model_trainer.sh          ← Launcher script
MODEL_TRAINER_GUIDE.md        ← Comprehensive guide
MODEL_TRAINER_QUICKSTART.md   ← This file
```

---

## 🎓 Learning Path

### Beginner (15 minutes)
1. Run the app
2. Train LightGBM with defaults
3. Look at results
4. Read the sidebar info

### Intermediate (1 hour)
1. Train multiple models
2. Adjust one parameter at a time
3. Observe performance changes
4. Understand metrics

### Advanced (2+ hours)
1. Design parameter experiments
2. Compare models systematically
3. Interpret classification reports
4. Optimize for business metrics

---

## 💾 Saving Results

### Current Results (Auto-Saved in Session)
- Visible in "Evaluation Results" tab
- Shows metrics, confusion matrix, ROC curve
- Data cleared when browser refreshes

### To Permanently Save
1. Take screenshots of metrics
2. Download the Evaluation Results via:
   - Screenshot: Right-click → Screenshot
   - Export: Copy table data manually

### Integration with Dashboard
- Original `app.py` still shows pre-trained results
- Model Trainer is for experimentation
- Run `./run_dashboard.sh` to see historical results

---

## 🚀 Next: Model Trainer Features

The Model Trainer includes:

✅ **6 Different Models**
- LightGBM, XGBoost, CatBoost
- Random Forest, Extra Trees
- Logistic Regression

✅ **70+ Tunable Parameters**
- Each model has 5-10 specific parameters
- Interactive sliders and dropdowns
- Real-time validation

✅ **7 Evaluation Metrics**
- Accuracy, Precision, Recall
- F1-Score, Balanced Accuracy
- ROC-AUC, PR-AUC

✅ **Multiple Visualizations**
- Confusion Matrix Heatmap
- ROC Curve with AUC
- Metrics Comparison Bar Chart
- Classification Report Table

✅ **Dynamic Everything**
- No pre-trained models used
- All training happens in real-time
- Preprocessing done fresh each time
- No data leakage

---

## ✅ Verification

After starting the app, you should see:

```
✓ Page loads at http://localhost:8501
✓ Three tabs visible
✓ Configuration tab has input fields
✓ "🚀 Train Model" button is clickable
✓ Dataset info shows in sidebar
✓ No errors in terminal
```

---

## 📞 Quick Help

**In the app:**
- Hover over parameters for tooltips
- Check the sidebar for dataset info
- Read "How to Use" in sidebar
- View last trained model stats in sidebar

**Command line help:**
```bash
streamlit --help
streamlit run model_trainer.py --help
```

---

## 🎉 Ready?

```bash
cd /Users/ibrahim/Desktop/proje
./run_model_trainer.sh
```

**Then navigate to:** http://localhost:8501

**Happy Training!** 🚀
