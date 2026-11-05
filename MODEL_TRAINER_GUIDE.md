# 🚀 Model Trainer - Interactive ML Experimentation Platform

## Overview

The **Model Trainer** is a dynamic, interactive machine learning interface that allows you to:

- ✅ **Select Models Dynamically** - Choose from 6 different machine learning models
- ✅ **Configure Parameters Interactively** - Adjust model-specific parameters in real-time
- ✅ **Train Models On-Demand** - Train models with your exact specifications
- ✅ **Evaluate Dynamically** - Get live evaluation results based on actual trained models
- ✅ **Compare Results** - View detailed metrics, confusion matrices, and ROC curves
- ✅ **Experiment Multiple Times** - Train different models without re-running scripts

## Key Features

### 🎯 1. Dynamic Model Selection

The application supports **6 machine learning models**:

1. **LightGBM** - Gradient boosting framework optimized for speed
2. **XGBoost** - Scalable gradient boosting with regularization
3. **CatBoost** - Gradient boosting handling categorical features natively
4. **Random Forest** - Ensemble of random decision trees
5. **Extra Trees** - Extremely randomized trees with faster training
6. **Logistic Regression** - Classical linear classification model

### ⚙️ 2. Model-Specific Parameters

Each model has its own parameter panel with relevant controls:

#### LightGBM Parameters
- `n_estimators` (50-500) - Number of boosting rounds
- `num_leaves` (15-50) - Maximum leaves per tree
- `max_depth` (1-20) - Maximum tree depth
- `learning_rate` (0.001-0.3) - Step size for boosting
- `min_child_samples` (5-100) - Minimum samples in leaf
- `subsample` (0.5-1.0) - Fraction of samples for training
- `colsample_bytree` (0.5-1.0) - Fraction of features per tree
- `reg_alpha` (0-10) - L1 regularization
- `reg_lambda` (0-10) - L2 regularization

#### XGBoost Parameters
- `n_estimators` (50-500) - Boosting rounds
- `max_depth` (1-15) - Tree depth
- `learning_rate` (0.001-0.3) - Learning rate
- `subsample` (0.5-1.0) - Row sampling ratio
- `colsample_bytree` (0.5-1.0) - Column sampling ratio
- `min_child_weight` (1-10) - Minimum sum of weights
- `reg_alpha` (0-10) - L1 regularization
- `reg_lambda` (0-10) - L2 regularization
- `gamma` (0-5) - Minimum loss reduction

#### CatBoost Parameters
- `iterations` (50-500) - Boosting rounds
- `depth` (1-15) - Tree depth
- `learning_rate` (0.001-0.3) - Learning rate
- `l2_leaf_reg` (0.1-20) - L2 regularization for leaves
- `subsample` (0.5-1.0) - Subsample ratio
- `min_child_samples` (1-20) - Minimum samples per leaf

#### Random Forest Parameters
- `n_estimators` (10-500) - Number of trees
- `max_depth` (5-30) - Maximum tree depth
- `min_samples_leaf` (1-20) - Minimum samples in leaf
- `min_samples_split` (2-20) - Minimum samples to split
- `max_features` (sqrt, log2) - Features to consider
- `bootstrap` (True/False) - Use bootstrap samples

#### Extra Trees Parameters
- Same as Random Forest (more randomness in splits)

#### Logistic Regression Parameters
- `C` (0.001-10) - Inverse regularization strength
- `max_iter` (100-1000) - Maximum iterations
- `penalty` (l2) - Regularization type
- `solver` (lbfgs, liblinear) - Optimization algorithm

### 📊 3. Dynamic Evaluation

All results are generated **in real-time** based on the trained model:

**Metrics Calculated:**
- **Accuracy** - Proportion of correct predictions
- **Precision** - True positives / (True positives + False positives)
- **Recall** - True positives / (True positives + False negatives)
- **F1-Score** - Harmonic mean of Precision and Recall
- **Balanced Accuracy** - Average of per-class recall
- **ROC-AUC** - Area under the ROC curve
- **PR-AUC** - Area under the Precision-Recall curve

**Visualizations Generated:**
- 🔥 **Confusion Matrix Heatmap** - Shows TN, FP, FN, TP
- 📈 **ROC Curve** - True Positive vs False Positive rate
- 📊 **Metrics Comparison** - Bar chart of all metrics
- 📋 **Classification Report** - Detailed per-class statistics

## Workflow

### Step 1: Dataset Configuration
```
┌─────────────────────────────────────┐
│  Dataset: 7,674 samples × 18 features
│  ✓ Numerical: 10 features
│  ✓ Categorical: 8 features
│  ✓ Target: Churned (binary)
└─────────────────────────────────────┘
```

### Step 2: Train/Test Configuration
```
Select:
  • Test Size: 10% - 40% (default: 20%)
  • Random State: 0 - 1000 (for reproducibility)
  
Result: Automatic stratified split preserving class distribution
```

### Step 3: Model Selection
```
Choose from:
  ☐ LightGBM
  ☐ XGBoost
  ☐ CatBoost
  ☐ Random Forest
  ☐ Extra Trees
  ☐ Logistic Regression
```

### Step 4: Parameter Configuration
```
Automatically displays parameters relevant to the selected model

Example (LightGBM):
┌──────────────────────┬──────────────────────┬──────────────────────┐
│ n_estimators: 300    │ learning_rate: 0.05  │ num_leaves: 31       │
│ max_depth: 6         │ min_child_samples: 20│ subsample: 0.8       │
│ colsample_bytree: 0.8│ reg_alpha: 0.1       │ reg_lambda: 1.0      │
└──────────────────────┴──────────────────────┴──────────────────────┘
```

### Step 5: Model Training
```
1. Click "🚀 Train Model"
2. System:
   ✓ Splits data (train/test with stratification)
   ✓ Builds pipeline with preprocessing
   ✓ Instantiates model with parameters
   ✓ Trains on training set
   ✓ Generates predictions on test set
   ✓ Calculates all metrics
   ✓ Creates visualizations
3. Results displayed in "Evaluation Results" tab
```

### Step 6: View Results
```
Automatic display of:
  ✓ Performance metrics with color coding
  ✓ Confusion matrix
  ✓ ROC curve (if applicable)
  ✓ Metrics comparison chart
  ✓ Detailed classification report
  ✓ Confusion matrix breakdown
```

## Usage Guide

### Quick Start

```bash
# Navigate to project directory
cd /Users/ibrahim/Desktop/proje

# Start the model trainer
./run_model_trainer.sh

# Open in browser
http://localhost:8501
```

### Example: Training LightGBM

1. **Go to Configuration Tab**
2. **Set Parameters:**
   - Test Size: 25%
   - Random State: 42
   - Model: LightGBM
   - n_estimators: 200
   - learning_rate: 0.1
   - max_depth: 8

3. **Click "🚀 Train Model"**
4. **Wait for completion**
5. **Go to Evaluation Results tab**
6. **Review metrics and visualizations**

### Example: Comparing Two Models

**Sequence:**
```
1. Train LightGBM with params A
   → View results
   
2. Go back to Configuration
   
3. Select XGBoost
   
4. Adjust parameters for XGBoost
   
5. Train XGBoost
   
6. Compare results in Model Comparison tab
```

## Important Notes

### ✅ Genuine Dynamic Training

- **NOT** displaying pre-computed results
- **NOT** loading saved models
- **ACTUALLY** training new models with your exact parameters
- **GENUINELY** evaluating on test set

Each training run:
```python
✓ Preprocesses fresh data
✓ Splits into new train/test
✓ Creates new model instance
✓ Trains with specified parameters
✓ Generates predictions
✓ Calculates metrics
✓ Creates visualizations
```

### ⚠️ Data Leakage Prevention

- Preprocessing steps fit on training set only
- Transformations applied consistently to test set
- Stratified split maintains class distribution
- No information flows from test to training

### 🎯 Evaluation Metrics Interpretation

| Metric | Range | Interpretation | Good Value |
|--------|-------|-----------------|-----------|
| **Accuracy** | 0-1 | % correct predictions | > 0.85 |
| **Precision** | 0-1 | % predicted positives correct | > 0.80 |
| **Recall** | 0-1 | % actual positives found | > 0.75 |
| **F1-Score** | 0-1 | Balance of precision/recall | > 0.80 |
| **Balanced Acc.** | 0-1 | Per-class average | > 0.80 |
| **ROC-AUC** | 0-1 | Classification threshold ability | > 0.85 |

### 🔍 Confusion Matrix Interpretation

```
                Predicted Negative    Predicted Positive
Actual Negative │       TN             │       FP         │
                ├──────────────────────┼──────────────────┤
Actual Positive │       FN             │       TP         │
                └──────────────────────┴──────────────────┘

TN (True Negative):   Correctly identified non-churners
FP (False Positive):  Non-churners incorrectly flagged as churn (Type I error)
FN (False Negative):  Churners missed by model (Type II error)
TP (True Positive):   Correctly identified churners
```

## Advanced Tips

### Parameter Tuning Strategy

1. **Start with defaults** - Run with preset values to establish baseline
2. **Adjust one at a time** - Change one parameter, observe impact
3. **Focus on key params** - n_estimators, learning_rate, max_depth
4. **Iterate quickly** - Re-train to see changes instantly
5. **Monitor trade-offs** - Precision vs Recall, Accuracy vs Balanced Accuracy

### Finding Optimal Configuration

```
Technique: Grid Search (Manual)

1. Select promising range for key parameter
2. Test 3-5 values
3. Pick best performer
4. Narrow down around winner
5. Fine-tune secondary parameters
```

### When to Use Each Model

| Model | Best For | Pros | Cons |
|-------|----------|------|------|
| **LightGBM** | Fast training | Speed, accuracy | Complex tuning |
| **XGBoost** | General purpose | Stable, proven | Higher memory |
| **CatBoost** | Categorical data | Handles categories | Slower inference |
| **Random Forest** | Baseline | Robust, simple | Lower accuracy |
| **Extra Trees** | Parallel training | Very fast | Less stable |
| **Log. Regression** | Interpretability | Simple, linear | Limited capacity |

## Troubleshooting

### ❌ "Train a model first in Configuration tab"
**Solution:** Go to the Configuration & Training tab and click "🚀 Train Model"

### ❌ Model training is slow
**Reason:** Large n_estimators, complex model
**Solution:** Reduce n_estimators or use Extra Trees (faster)

### ❌ Metrics are all 0 or 1
**Reason:** Model overfitting or underfitting
**Solution:** Adjust regularization parameters (reg_alpha, reg_lambda, C)

### ❌ Error during preprocessing
**Reason:** Categorical/numerical column mismatch
**Solution:** Check that CATEGORICAL_COLUMNS in config.py is correct

## File Structure

```
proje/
├── model_trainer.py                    ← NEW: Interactive trainer app
├── run_model_trainer.sh                ← NEW: Launcher script
├── app.py                              ← Dashboard (original)
├── run_dashboard.sh                    ← Dashboard launcher
├── run_ecommerce_training.py           ← Batch training script
├── src/ecommerce_churn/
│   ├── models.py                       ← Model definitions
│   ├── pipeline.py                     ← Training pipeline
│   ├── preprocessing.py                ← Data preprocessing
│   ├── data.py                         ← Data loading
│   └── config.py                       ← Configuration
└── artifacts/
    └── results/                        ← Pre-trained model results
```

## Two Interfaces

### 🎨 Dashboard (app.py)
- **Purpose:** Visualize pre-trained model results
- **Use When:** Comparing already-trained models
- **Run:** `./run_dashboard.sh`

### 🚀 Model Trainer (model_trainer.py)
- **Purpose:** Interactively train and experiment
- **Use When:** Developing, tuning, testing models
- **Run:** `./run_model_trainer.sh`

## Next Steps

1. ✅ **Start the trainer:** `./run_model_trainer.sh`
2. ✅ **Train your first model:** Select LightGBM and click Train
3. ✅ **Experiment:** Try different models and parameters
4. ✅ **Compare:** Use Model Comparison tab to review results
5. ✅ **Optimize:** Iterate until satisfied with performance

## Technical Details

### Data Preprocessing Pipeline

For **tree-based models:**
```python
Pipeline([
    ("preprocessor", make_preprocessor(numerical, categorical)),
    ("model", TreeModel())
])
```

For **Logistic Regression:**
```python
Pipeline([
    ("preprocessor", make_preprocessor(numerical, categorical, scale=True)),
    ("model", LogisticRegression())
])
```

For **CatBoost:**
```python
Pipeline([
    ("preprocessor", CatBoostPreprocessor(numerical, categorical)),
    ("model", CatBoostClassifier())
])
```

### Model Instantiation Example

When you select LightGBM with parameters:
```python
model = Pipeline([
    ("preprocessor", make_preprocessor(...)),
    ("model", LGBMClassifier(
        n_estimators=300,           # Your selected value
        num_leaves=31,              # Your selected value
        max_depth=6,                # Your selected value
        learning_rate=0.05,         # Your selected value
        # ... all other parameters from your selection
        random_state=RANDOM_STATE   # For reproducibility
    ))
])

# Fit on training data
model.fit(X_train, y_train)

# Predict on test data
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = (y_pred_proba >= 0.5).astype(int)

# Calculate metrics from predictions
metrics = {
    "Accuracy": accuracy_score(y_test, y_pred),
    "Precision": precision_score(y_test, y_pred),
    # ... all metrics
}
```

## Version History

- **v1.0** (2024-09-14)
  - ✅ Dynamic model selection
  - ✅ Interactive parameter tuning
  - ✅ Real-time training
  - ✅ Dynamic evaluation
  - ✅ Multiple visualizations
  - ✅ 6 ML models supported

---

**Ready to experiment?** 🚀

```bash
./run_model_trainer.sh
```

**Questions?** Check the sidebar in the app for more information!

---

*Built with Streamlit, Scikit-learn, and LightGBM*
