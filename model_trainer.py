# ══════════════════════════════════════════════════════════════════════════════
#  E-COMMERCE CHURN - DYNAMIC MODEL TRAINER
#  Interactive model selection, parameter configuration, and dynamic evaluation
# ══════════════════════════════════════════════════════════════════════════════

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import json
import warnings
import time
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    balanced_accuracy_score, average_precision_score, roc_curve
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import joblib

warnings.filterwarnings("ignore")

# ─────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Model Trainer - Churn Prediction",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
#  CSS - Dark Theme
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #050505; 
  --surface: #0a0a0a; 
  --surface2: #121212;
  --border: #27272a; 
  --border2: #3f3f46;
  --accent: #52525b;
  --blue: #3b82f6; 
  --green: #2ecc71; 
  --amber: #f59e0b; 
  --purple: #8b5cf6; 
  --teal: #06b6d4;
  --txt1: #ffffff; 
  --txt2: #d4d4d8; 
  --txt3: #a1a1aa;
}

html, body, .stApp { 
  background: #050505 !important; 
  font-family: 'Inter', sans-serif !important; 
}

.main .block-container { 
  padding: 2rem 2.5rem 3rem !important; 
  max-width: 1800px !important; 
}

[data-testid="stSidebar"] > div:first-child { 
  background: #0a0a0a !important; 
  border-right: 1px solid #1a1a1a !important; 
}

.stTabs [data-baseweb="tab-list"] { 
  background: transparent !important; 
  border-bottom: 1px solid #27272a !important; 
  gap: 2px !important; 
}

.stTabs [data-baseweb="tab"] { 
  background: transparent !important; 
  color: #71717a !important; 
  border-radius: 4px 4px 0 0 !important; 
  padding: 12px 20px !important; 
  font-size: 0.9rem !important; 
  font-weight: 500 !important; 
  border-bottom: 2px solid transparent !important; 
}

.stTabs [aria-selected="true"] { 
  background: transparent !important; 
  color: #ffffff !important; 
  border-bottom: 2px solid #3b82f6 !important; 
}

.stButton>button { 
  background: #3b82f6 !important; 
  color: white !important; 
  border: none !important; 
  border-radius: 6px !important; 
  padding: 8px 16px !important; 
  font-weight: 600 !important;
}

.stButton>button:hover { 
  background: #2563eb !important; 
}

[data-testid="stDataFrame"] { 
  border: 1px solid #27272a !important; 
  border-radius: 4px !important; 
}

.stDataFrame th { 
  background: #121212 !important; 
  color: #d4d4d8 !important; 
  font-weight: 600 !important;
}

.stDataFrame td { 
  color: #a1a1aa !important; 
}

.metric-card {
  background: #121212;
  border: 1px solid #27272a;
  border-radius: 8px;
  padding: 20px;
  text-align: center;
}

h1, h2, h3 { 
  color: #ffffff !important; 
}

p, label { 
  color: #d4d4d8 !important; 
}

.stSelectbox, .stSlider, .stNumberInput, .stMultiSelect { 
  color: #ffffff !important; 
}

[data-testid="stMarkdownContainer"] p { 
  color: #a1a1aa !important; 
}

</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────
@st.cache_data
def load_data():
    """Load preprocessed data from CSV"""
    from src.ecommerce_churn.data import load_modeling_data, split_features_target
    data = load_modeling_data()
    X, y = split_features_target(data)
    return X, y, data

@st.cache_data
def get_categorical_numerical_columns():
    """Get column types from config"""
    from src.ecommerce_churn.config import CATEGORICAL_COLUMNS
    X, y, _ = load_data()
    categorical = [col for col in X.columns if col in CATEGORICAL_COLUMNS]
    numerical = [col for col in X.columns if col not in CATEGORICAL_COLUMNS]
    return numerical, categorical

# ─────────────────────────────────────
#  MODEL PARAMETER DEFINITIONS
# ─────────────────────────────────────
MODEL_PARAMS = {
    "LightGBM": {
        "n_estimators": (50, 500, 300),
        "num_leaves": (15, 50, 31),
        "max_depth": (1, 20, 6),
        "learning_rate": (0.001, 0.3, 0.05),
        "min_child_samples": (5, 100, 20),
        "subsample": (0.5, 1.0, 0.8),
        "colsample_bytree": (0.5, 1.0, 0.8),
        "reg_alpha": (0.0, 10.0, 0.1),
        "reg_lambda": (0.0, 10.0, 1.0),
    },
    "XGBoost": {
        "n_estimators": (50, 500, 400),
        "max_depth": (1, 15, 4),
        "learning_rate": (0.001, 0.3, 0.05),
        "subsample": (0.5, 1.0, 0.8),
        "colsample_bytree": (0.5, 1.0, 0.8),
        "min_child_weight": (1, 10, 5),
        "reg_alpha": (0.0, 10.0, 0.1),
        "reg_lambda": (0.0, 10.0, 1.0),
        "gamma": (0.0, 5.0, 0.0),
    },
    "CatBoost": {
        "iterations": (50, 500, 350),
        "depth": (1, 15, 6),
        "learning_rate": (0.001, 0.3, 0.05),
        "l2_leaf_reg": (0.1, 20.0, 8.0),
        "subsample": (0.5, 1.0, 0.8),
        "min_child_samples": (1, 20, 5),
    },
    "Random Forest": {
        "n_estimators": (10, 500, 350),
        "max_depth": (5, 30, 14),
        "min_samples_leaf": (1, 20, 5),
        "min_samples_split": (2, 20, 2),
        "max_features": ("sqrt", "log2"),
        "bootstrap": (True, False),
    },
    "Extra Trees": {
        "n_estimators": (10, 500, 350),
        "max_depth": (5, 30, 14),
        "min_samples_leaf": (1, 20, 5),
        "min_samples_split": (2, 20, 2),
        "max_features": ("sqrt", "log2"),
        "bootstrap": (True, False),
    },
    "Logistic Regression": {
        "C": (0.001, 10.0, 0.5),
        "max_iter": (100, 1000, 500),
        "penalty": ("l2",),
        "solver": ("lbfgs", "liblinear"),
    }
}

# ─────────────────────────────────────
#  MODEL BUILDING FUNCTIONS
# ─────────────────────────────────────
def build_model(model_name, params, numerical_cols, categorical_cols):
    """Build model with user-specified parameters"""
    from lightgbm import LGBMClassifier
    from xgboost import XGBClassifier
    from catboost import CatBoostClassifier
    from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
    from sklearn.linear_model import LogisticRegression
    from src.ecommerce_churn.config import RANDOM_STATE
    from src.ecommerce_churn.preprocessing import (
        make_preprocessor, CatBoostPreprocessor
    )
    
    # Select preprocessor based on model
    if model_name == "CatBoost":
        preprocessor = CatBoostPreprocessor(numerical_cols, categorical_cols)
    else:
        preprocessor = make_preprocessor(
            numerical_cols, 
            categorical_cols, 
            scale=(model_name == "Logistic Regression")
        )
    
    # Build model with parameters
    if model_name == "LightGBM":
        model = LGBMClassifier(
            n_estimators=int(params["n_estimators"]),
            num_leaves=int(params["num_leaves"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            min_child_samples=int(params["min_child_samples"]),
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            objective="binary",
            verbosity=-1,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "XGBoost":
        model = XGBClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]),
            learning_rate=params["learning_rate"],
            subsample=params["subsample"],
            colsample_bytree=params["colsample_bytree"],
            min_child_weight=int(params["min_child_weight"]),
            reg_alpha=params["reg_alpha"],
            reg_lambda=params["reg_lambda"],
            gamma=params["gamma"],
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "CatBoost":
        model = CatBoostClassifier(
            iterations=int(params["iterations"]),
            depth=int(params["depth"]),
            learning_rate=params["learning_rate"],
            l2_leaf_reg=params["l2_leaf_reg"],
            subsample=params["subsample"],
            min_child_samples=int(params["min_child_samples"]),
            loss_function="Logloss",
            eval_metric="AUC",
            cat_features=list(range(len(numerical_cols), len(numerical_cols) + len(categorical_cols))),
            random_seed=RANDOM_STATE,
            thread_count=-1,
            verbose=False,
            allow_writing_files=False,
        )
    elif model_name == "Random Forest":
        model = RandomForestClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Extra Trees":
        model = ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]),
            max_depth=int(params["max_depth"]) if params["max_depth"] else None,
            min_samples_leaf=int(params["min_samples_leaf"]),
            min_samples_split=int(params["min_samples_split"]),
            max_features=params["max_features"],
            bootstrap=params["bootstrap"],
            n_jobs=-1,
            random_state=RANDOM_STATE,
        )
    elif model_name == "Logistic Regression":
        model = LogisticRegression(
            C=params["C"],
            max_iter=int(params["max_iter"]),
            penalty=params["penalty"],
            solver=params["solver"],
            random_state=RANDOM_STATE,
        )
    
    return Pipeline([("preprocessor", preprocessor), ("model", model)])

# ─────────────────────────────────────
#  EVALUATION FUNCTIONS
# ─────────────────────────────────────
def evaluate_model(model, X_test, y_test, model_name):
    """Dynamically evaluate model on test set"""
    # Generate predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    # Calculate metrics
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred, zero_division=0),
        "Recall": recall_score(y_test, y_pred, zero_division=0),
        "F1-Score": f1_score(y_test, y_pred, zero_division=0),
        "Balanced Accuracy": balanced_accuracy_score(y_test, y_pred),
        "PR-AUC": average_precision_score(y_test, y_pred_proba),
    }
    
    # Add ROC-AUC for applicable models
    try:
        metrics["ROC-AUC"] = roc_auc_score(y_test, y_pred_proba)
    except:
        metrics["ROC-AUC"] = 0.0
    
    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred, labels=[0, 1]).ravel()
    cm = {"tn": tn, "fp": fp, "fn": fn, "tp": tp}
    
    # Classification report
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    
    return metrics, cm, y_pred, y_pred_proba, report

# ─────────────────────────────────────
#  VISUALIZATION FUNCTIONS
# ─────────────────────────────────────
def create_confusion_matrix_plot(tn, fp, fn, tp, model_name):
    """Create confusion matrix heatmap"""
    data = [[tn, fp], [fn, tp]]
    labels = [["TN", "FP"], ["FN", "TP"]]
    
    fig = go.Figure(data=go.Heatmap(
        z=data,
        text=labels,
        texttemplate="%{text}<br>%{z}",
        textfont={"size": 14},
        colorscale=[[0, "#1e293b"], [1, "#06b6d4"]],
        showscale=False,
        hovertemplate="<b>%{text}</b><br>Değer: %{z}<extra></extra>",
    ))
    
    fig.update_layout(
        title=f"<b>{model_name} - Confusion Matrix</b>",
        xaxis_title="Predicted",
        yaxis_title="Actual",
        height=400,
        width=450,
        template="plotly_dark",
        paper_bgcolor="rgba(18,18,18,0.5)",
        plot_bgcolor="rgba(18,18,18,0.5)",
        font=dict(color="#a1a1aa", family="Inter"),
        xaxis=dict(tickvals=[0, 1], ticktext=["Negative", "Positive"]),
        yaxis=dict(tickvals=[0, 1], ticktext=["Negative", "Positive"]),
    )
    return fig

def create_roc_curve_plot(y_test, y_pred_proba, roc_auc):
    """Create ROC curve"""
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fpr, y=tpr,
        mode='lines',
        name=f'ROC Curve (AUC = {roc_auc:.3f})',
        line=dict(color='#06b6d4', width=2),
        fill='tozeroy',
        fillcolor='rgba(6, 182, 212, 0.2)',
    ))
    fig.add_trace(go.Scatter(
        x=[0, 1], y=[0, 1],
        mode='lines',
        name='Random Classifier',
        line=dict(color='#71717a', width=1, dash='dash'),
    ))
    
    fig.update_layout(
        title="<b>ROC Curve</b>",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        height=400,
        template="plotly_dark",
        paper_bgcolor="rgba(18,18,18,0.5)",
        plot_bgcolor="rgba(18,18,18,0.5)",
        font=dict(color="#a1a1aa", family="Inter"),
    )
    return fig

def create_metrics_comparison(metrics):
    """Create metrics comparison visualization"""
    metric_names = list(metrics.keys())
    metric_values = list(metrics.values())
    
    colors = ["#3b82f6" if v >= 0.8 else "#f59e0b" if v >= 0.7 else "#ef4444" 
              for v in metric_values]
    
    fig = go.Figure(data=go.Bar(
        x=metric_names,
        y=metric_values,
        marker=dict(color=colors),
        text=[f"{v:.4f}" for v in metric_values],
        textposition="outside",
        hovertemplate="<b>%{x}</b><br>Score: %{y:.4f}<extra></extra>",
    ))
    
    fig.update_layout(
        title="<b>Model Performance Metrics</b>",
        yaxis_title="Score",
        height=400,
        yaxis=dict(range=[0, 1.1]),
        template="plotly_dark",
        paper_bgcolor="rgba(18,18,18,0.5)",
        plot_bgcolor="rgba(18,18,18,0.5)",
        font=dict(color="#a1a1aa", family="Inter"),
        showlegend=False,
    )
    return fig

# ─────────────────────────────────────
#  MAIN APPLICATION
# ─────────────────────────────────────

st.title("🚀 Interactive Model Trainer")
st.markdown("**Dynamic model selection, parameter tuning, and live evaluation**")
st.markdown("---")

# Initialize session state
if "trained_model" not in st.session_state:
    st.session_state.trained_model = None
if "training_results" not in st.session_state:
    st.session_state.training_results = None
if "selected_model" not in st.session_state:
    st.session_state.selected_model = None

# Load data
X, y, _ = load_data()
numerical_cols, categorical_cols = get_categorical_numerical_columns()

# Create tabs
tab1, tab2, tab3 = st.tabs([
    "⚙️ Configuration & Training",
    "📊 Evaluation Results",
    "📈 Model Comparison"
])

# ─────────────────────────────────────
#  TAB 1: CONFIGURATION & TRAINING
# ─────────────────────────────────────
with tab1:
    st.header("Step-by-Step Model Configuration")
    
    # Step 1: Dataset Configuration
    with st.container():
        st.subheader("📂 Step 1: Dataset Configuration")
        col1, col2 = st.columns(2)
        with col1:
            st.info(f"📊 Dataset Size: {X.shape[0]:,} samples × {X.shape[1]} features")
            st.info(f"🔢 Numerical Features: {len(numerical_cols)}")
        with col2:
            st.info(f"📝 Categorical Features: {len(categorical_cols)}")
            st.info(f"🎯 Target Distribution: {dict(y.value_counts())}")
    
    st.markdown("---")
    
    # Step 2: Train/Test Split
    with st.container():
        st.subheader("🔀 Step 2: Train/Test Configuration")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            test_size = st.slider(
                "Test Set Size (%)",
                min_value=10,
                max_value=40,
                value=20,
                step=5,
                help="Percentage of data to use for testing"
            )
        
        with col2:
            random_state = st.number_input(
                "Random State",
                min_value=0,
                max_value=1000,
                value=42,
                help="Random seed for reproducibility"
            )
        
        with col3:
            st.metric(
                "Test Samples",
                f"{int(X.shape[0] * test_size / 100):,}"
            )
    
    st.markdown("---")
    
    # Step 3: Model Selection
    with st.container():
        st.subheader("🤖 Step 3: Model Selection")
        
        available_models = list(MODEL_PARAMS.keys())
        selected_model = st.selectbox(
            "Select a Machine Learning Model",
            available_models,
            index=0,
            help="Choose the model to train"
        )
        
        st.session_state.selected_model = selected_model
    
    st.markdown("---")
    
    # Step 4: Model Parameters (Dynamic)
    with st.container():
        st.subheader(f"⚙️ Step 4: {selected_model} Parameters")
        
        params = {}
        params_cols = st.columns(min(3, len(MODEL_PARAMS[selected_model])))
        
        for idx, (param_name, param_config) in enumerate(MODEL_PARAMS[selected_model].items()):
            col = params_cols[idx % len(params_cols)]
            
            with col:
                if isinstance(param_config, tuple) and len(param_config) == 3:
                    # Numeric slider parameter
                    if isinstance(param_config[0], float):
                        params[param_name] = st.slider(
                            param_name,
                            min_value=param_config[0],
                            max_value=param_config[1],
                            value=param_config[2],
                            step=(param_config[1] - param_config[0]) / 20,
                        )
                    else:
                        params[param_name] = st.slider(
                            param_name,
                            min_value=param_config[0],
                            max_value=param_config[1],
                            value=param_config[2],
                        )
                elif isinstance(param_config, tuple) and len(param_config) == 2:
                    # Multi-select parameter (for categorical)
                    if isinstance(param_config[0], str):
                        params[param_name] = st.selectbox(
                            param_name,
                            param_config,
                            index=0,
                        )
                    else:
                        params[param_name] = st.selectbox(
                            param_name,
                            param_config,
                            index=0,
                        )
                elif param_name == "bootstrap":
                    params[param_name] = st.checkbox(param_name, value=True)
    
    st.markdown("---")
    
    # Step 5: Train Model
    with st.container():
        st.subheader("▶️ Step 5: Train Model")
        
        col1, col2 = st.columns([1, 3])
        
        with col1:
            train_button = st.button(
                "🚀 Train Model",
                use_container_width=True,
                key="train_btn"
            )
        
        with col2:
            st.info("Click the button to start training with the selected configuration")
        
        if train_button:
            with st.spinner("🔄 Training in progress..."):
                try:
                    # Split data
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y,
                        test_size=test_size/100,
                        stratify=y,
                        random_state=random_state
                    )
                    
                    # Build and train model
                    start_time = time.time()
                    model = build_model(selected_model, params, numerical_cols, categorical_cols)
                    model.fit(X_train, y_train)
                    training_time = time.time() - start_time
                    
                    # Evaluate
                    metrics, cm, y_pred, y_pred_proba, report = evaluate_model(
                        model, X_test, y_test, selected_model
                    )
                    
                    # Store results
                    st.session_state.trained_model = model
                    st.session_state.training_results = {
                        "model_name": selected_model,
                        "params": params,
                        "X_train": X_train,
                        "X_test": X_test,
                        "y_train": y_train,
                        "y_test": y_test,
                        "metrics": metrics,
                        "cm": cm,
                        "y_pred": y_pred,
                        "y_pred_proba": y_pred_proba,
                        "report": report,
                        "training_time": training_time,
                        "test_size": test_size,
                        "random_state": random_state,
                    }
                    
                    st.success(f"✅ Model trained successfully in {training_time:.2f}s!")
                    st.balloons()
                
                except Exception as e:
                    st.error(f"❌ Training failed: {str(e)}")

# ─────────────────────────────────────
#  TAB 2: EVALUATION RESULTS
# ─────────────────────────────────────
with tab2:
    if st.session_state.training_results is None:
        st.warning("⚠️ Train a model first in the Configuration tab!")
    else:
        results = st.session_state.training_results
        model_name = results["model_name"]
        metrics = results["metrics"]
        cm = results["cm"]
        y_test = results["y_test"]
        y_pred_proba = results["y_pred_proba"]
        report = results["report"]
        
        st.header(f"📊 {model_name} - Evaluation Results")
        
        # Model info
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Model", model_name)
        with col2:
            st.metric("Training Time", f"{results['training_time']:.2f}s")
        with col3:
            st.metric("Test Set Size", f"{results['test_size']}%")
        with col4:
            st.metric("Samples Evaluated", len(y_test))
        
        st.markdown("---")
        
        # Metrics Cards
        st.subheader("📈 Performance Metrics")
        cols = st.columns(len(metrics))
        for col, (metric_name, metric_value) in zip(cols, metrics.items()):
            with col:
                # Color code based on performance
                if metric_value >= 0.85:
                    color_indicator = "🟢"
                elif metric_value >= 0.70:
                    color_indicator = "🟡"
                else:
                    color_indicator = "🔴"
                
                st.metric(
                    f"{metric_name} {color_indicator}",
                    f"{metric_value:.4f}",
                    delta=f"{(metric_value - 0.5)*100:.1f}% above baseline"
                )
        
        st.markdown("---")
        
        # Visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            cm_fig = create_confusion_matrix_plot(
                cm["tn"], cm["fp"], cm["fn"], cm["tp"], model_name
            )
            st.plotly_chart(cm_fig, use_container_width=True)
        
        with col2:
            if "ROC-AUC" in metrics:
                roc_fig = create_roc_curve_plot(y_test, y_pred_proba, metrics["ROC-AUC"])
                st.plotly_chart(roc_fig, use_container_width=True)
            else:
                metrics_fig = create_metrics_comparison(metrics)
                st.plotly_chart(metrics_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Metrics Comparison
        st.subheader("📊 All Metrics Comparison")
        metrics_fig = create_metrics_comparison(metrics)
        st.plotly_chart(metrics_fig, use_container_width=True)
        
        st.markdown("---")
        
        # Classification Report
        st.subheader("📋 Detailed Classification Report")
        
        report_df = pd.DataFrame(report).transpose()
        report_df = report_df.round(4)
        st.dataframe(report_df, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrix Details
        st.subheader("🎯 Confusion Matrix Breakdown")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("True Negatives (TN)", cm["tn"])
        with col2:
            st.metric("False Positives (FP)", cm["fp"])
        with col3:
            st.metric("False Negatives (FN)", cm["fn"])
        with col4:
            st.metric("True Positives (TP)", cm["tp"])

# ─────────────────────────────────────
#  TAB 3: MODEL COMPARISON
# ─────────────────────────────────────
with tab3:
    st.header("📈 Model Comparison")
    
    if st.session_state.training_results is None:
        st.warning("⚠️ Train at least one model to see comparison!")
    else:
        results = st.session_state.training_results
        metrics = results["metrics"]
        
        # Show current model metrics
        st.subheader(f"Current Model: {results['model_name']}")
        
        comparison_df = pd.DataFrame({
            "Metric": list(metrics.keys()),
            "Score": list(metrics.values()),
        })
        
        # Add interpretation
        def interpret_metric(score):
            if score >= 0.85:
                return "Excellent ✅"
            elif score >= 0.75:
                return "Good ⚠️"
            else:
                return "Needs Improvement ❌"
        
        comparison_df["Interpretation"] = comparison_df["Score"].apply(interpret_metric)
        
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        
        st.info(
            "💡 **Tip:** Go back to the Configuration tab to train different models "
            "and compare their performance!"
        )

# ─────────────────────────────────────
#  SIDEBAR INFO
# ─────────────────────────────────────
with st.sidebar:
    st.header("ℹ️ Information")
    
    st.subheader("📚 How to Use")
    st.markdown("""
    1. **Configure Dataset** - Set train/test split
    2. **Select Model** - Choose from 6 models
    3. **Tune Parameters** - Adjust model-specific settings
    4. **Train** - Click to train the model
    5. **Evaluate** - View results in the next tab
    """)
    
    st.markdown("---")
    
    st.subheader("🎯 Available Models")
    for model in available_models:
        st.caption(f"• {model}")
    
    st.markdown("---")
    
    st.subheader("📊 Dataset Info")
    st.caption(f"Samples: {X.shape[0]:,}")
    st.caption(f"Features: {X.shape[1]}")
    st.caption(f"Positive Class: {(y == 1).sum():,} ({(y == 1).sum()/len(y)*100:.1f}%)")
    st.caption(f"Negative Class: {(y == 0).sum():,} ({(y == 0).sum()/len(y)*100:.1f}%)")
    
    st.markdown("---")
    
    if st.session_state.training_results:
        st.subheader("✅ Last Trained Model")
        st.caption(f"Model: {st.session_state.training_results['model_name']}")
        st.caption(f"F1-Score: {st.session_state.training_results['metrics']['F1-Score']:.4f}")
        st.caption(f"Time: {st.session_state.training_results['training_time']:.2f}s")

st.markdown("---")
st.caption("🔬 Dynamic Model Trainer v1.0 | Built with Streamlit + Scikit-learn")
