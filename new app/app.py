# ══════════════════════════════════════════════════════════════════════════════
#  END-TO-END MLOPS & DATA SCIENCE PLATFORM
#  Dataset Agnostic | Colorful SaaS Theme | Advanced Hyperparameters
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib, io, time, math, warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import (train_test_split, RandomizedSearchCV, GridSearchCV,
                                      StratifiedKFold, cross_val_score)
from sklearn.preprocessing import (StandardScaler, MinMaxScaler, RobustScaler,
                                    OneHotEncoder, OrdinalEncoder, LabelEncoder)
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import (accuracy_score, f1_score, roc_auc_score,
                              precision_score, recall_score, confusion_matrix,
                              classification_report, balanced_accuracy_score,
                              average_precision_score, precision_recall_curve, roc_curve)
from scipy.stats import randint, uniform, loguniform

try:
    from xgboost import XGBClassifier; XGB_OK = True
except ImportError: XGB_OK = False
try:
    from lightgbm import LGBMClassifier; LGBM_OK = True
except ImportError: LGBM_OK = False
try:
    from catboost import CatBoostClassifier; CB_OK = True
except ImportError: CB_OK = False
try:
    import shap; SHAP_OK = True
except ImportError: SHAP_OK = False
try:
    from mrmr import mrmr_classif; MRMR_OK = True
except ImportError: MRMR_OK = False

try:
    import optuna; OPTUNA_OK = True
except ImportError: OPTUNA_OK = False

# ─────────────────────────────────────
#  PAGE CONFIG
# ─────────────────────────────────────
st.set_page_config(
    page_title="Data Science & MLOps Platform",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────
#  CSS INJECTION - Pure Black & Colorful Accents (Sidebar Toggle Fixed)
# ─────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

:root {
  --bg: #050505; --surface: #0a0a0a; --surface2: #121212;
  --border: #27272a; --border2: #3f3f46;
  --accent: #3b82f6; /* Blue Accent */
  --blue: #3b82f6; --green: #2ecc71; --amber: #f59e0b; --purple: #8b5cf6; --teal: #06b6d4; --red: #e74c3c;
  --txt1: #ffffff; --txt2: #d4d4d8; --txt3: #a1a1aa;
}

html, body, .stApp, .stApp > div { background: var(--bg) !important; font-family: 'Inter', sans-serif !important; }
.main > div { background: var(--bg) !important; }
.main .block-container { padding: 1.2rem 2.2rem 3rem !important; max-width: 1500px !important; background: var(--bg) !important; }

/* ═════ ÖZEL SOL PANEL (native st.sidebar KULLANILMIYOR) ═════ */
[data-testid="stSidebar"] { display: none !important; }
[data-testid="stSidebarCollapsedControl"],
[data-testid="collapsedControl"],
[data-testid="stSidebarCollapseButton"] {
  display: none !important;
}

#MainMenu, footer {
    display: none !important;
    visibility: hidden !important;
}

[data-testid="stDecoration"], [data-testid="stStatusWidget"] {
    display: none !important;
}

header[data-testid="stHeader"] {
    background: transparent !important;
    pointer-events: none !important;
    z-index: 999998 !important;
}
header[data-testid="stHeader"] * { pointer-events: none !important; }

[data-testid="stToolbar"] { display: none !important; }

/* Kendi panelimiz */
.st-key-custom_sidebar {
    position: fixed !important;
    top: 0 !important;
    left: 0 !important;
    width: 280px !important;
    height: 100vh !important;
    overflow-y: auto !important;
    background: #0a0a0a !important;
    border-right: 1px solid #1a1a1a !important;
    z-index: 999990 !important;
    padding: 16px 16px !important;
    box-sizing: border-box !important;
}
.st-key-custom_sidebar * { color: #a1a1aa !important; }

/* AÇMA butonu */
.st-key-sb_open_btn {
    position: fixed !important;
    top: 14px !important;
    left: 14px !important;
    z-index: 1000005 !important;
    width: 42px !important;
}
.st-key-sb_open_btn .stButton > button {
    width: 42px !important;
    height: 42px !important;
    padding: 0 !important;
    border-radius: 8px !important;
    background: #18181b !important;
    border: 1px solid #3b82f6 !important;
    color: #3b82f6 !important;
    font-size: 1.2rem !important;
    box-shadow: 0 4px 14px rgba(59, 130, 246, 0.35) !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    transition: all 0.2s ease !important;
}
.st-key-sb_open_btn .stButton > button:hover {
    background: #27272a !important;
    border-color: #60a5fa !important;
    color: #60a5fa !important;
    transform: scale(1.05) !important;
}

/* KAPATMA butonu */
.st-key-sb_close_btn_wrap {
    display: flex !important;
    justify-content: flex-end !important;
    margin-bottom: 6px !important;
}
.st-key-sb_close_btn .stButton > button {
    width: 34px !important;
    height: 34px !important;
    padding: 0 !important;
    min-height: 34px !important;
    border-radius: 6px !important;
    background: #18181b !important;
    border: 1px solid #27272a !important;
    color: #a1a1aa !important;
    font-size: 1rem !important;
    display: flex !important;
    align-items: center !important;
    justify-content: center !important;
    margin-left: auto !important;
}
.st-key-sb_close_btn .stButton > button:hover {
    background: #27272a !important;
    border-color: #3b82f6 !important;
    color: #3b82f6 !important;
}

/* MENÜ - DİKDÖRTGEN KUTU BUTONLAR */
.st-key-custom_sidebar .stButton {
  width: 100% !important;
  margin-bottom: 6px !important;
}
.st-key-custom_sidebar .stButton > button {
  width: 100% !important;
  text-align: left !important;
  justify-content: flex-start !important;
  border-radius: 8px !important;
  padding: 12px 16px !important;
  margin: 0 !important;
  color: #9ca3af !important;
  font-size: 0.95rem !important;
  font-weight: 500 !important;
  background: transparent !important;
  border: 1px solid transparent !important;
  transition: all 0.2s ease !important;
  cursor: pointer !important;
  display: flex !important;
  align-items: center !important;
  min-height: 44px !important;
  box-shadow: none !important;
}
.st-key-custom_sidebar .stButton > button:hover {
  background: rgba(255,255,255,0.05) !important;
  color: #f3f4f6 !important;
  border: 1px solid #27272a !important;
  transform: none !important;
}
.st-key-custom_sidebar .stButton > button[kind="primary"] {
  background: #0f172a !important;
  color: #bae6fd !important;
  border: 1px solid #3b82f6 !important;
  box-shadow: 0 4px 6px rgba(0,0,0,0.2) !important;
  font-weight: 600 !important;
}
.st-key-custom_sidebar .stButton > button[kind="primary"]:hover {
  background: #14213d !important;
  border-color: #60a5fa !important;
  color: #e0f2fe !important;
}

/* BUTTONS */
.stButton > button {
  background: #18181b !important;
  color: #ffffff !important; border: 1px solid #3f3f46 !important; border-radius: 4px !important;
  padding: 0.45rem 1.35rem !important; font-weight: 600 !important; font-size: 0.85rem !important;
  letter-spacing: 0.02em !important; box-shadow: 0 4px 6px rgba(0,0,0,0.5) !important;
  transition: all 0.2s ease !important; min-height: 38px !important; font-family: 'Inter', sans-serif !important;
}
.stButton > button:hover { transform: translateY(-1px) !important; box-shadow: 0 6px 12px rgba(59,130,246,0.3) !important; background: #27272a !important; border-color: var(--accent) !important; color: #ffffff !important;}
.stButton > button:active { transform: translateY(0) !important; }
[data-testid="stDownloadButton"] > button { background: #0a0a0a !important; color: #d4d4d8 !important; border: 1px solid #3f3f46 !important; box-shadow: none !important; }
[data-testid="stDownloadButton"] > button:hover { background: #18181b !important; border-color: var(--accent) !important; box-shadow: none !important; color: #fff !important; transform: none !important; }

/* INPUTS */
.stSelectbox > div > div, [data-baseweb="select"] > div, .stMultiSelect > div > div {
  background: #09090b !important; border: 1px solid #3f3f46 !important; border-radius: 4px !important; color: #e4e4e7 !important;
}
.stSelectbox > div > div:hover { border-color: var(--accent) !important; }
.stTextInput > div > div > input, .stNumberInput > div > div > input {
  background: #09090b !important; border: 1px solid #3f3f46 !important; border-radius: 4px !important;
  color: #e4e4e7 !important; padding: 8px 12px !important; font-size: 0.85rem !important;
}
.stTextInput > div > div > input:focus, .stNumberInput > div > div > input:focus {
  border-color: var(--accent) !important; box-shadow: 0 0 0 1px var(--accent) !important; outline: none !important;
}

/* SLIDER */
[data-testid="stSlider"] > div > div > div { background: #18181b !important; }
[data-testid="stSlider"] > div > div > div > div { background: var(--accent) !important; }
[data-testid="stSlider"] div[role="slider"] {
    background-color: #a1a1aa !important; border: 2px solid #3f3f46 !important; box-shadow: none !important;
}
[data-testid="stSlider"] div[role="slider"] > div {
    background: transparent !important; 
    color: #ffffff !important; 
    border: none !important;
    box-shadow: none !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
}
[data-testid="stThumbValue"], [data-testid="stSliderTickBarMin"], [data-testid="stSliderTickBarMax"] {
    color: #a1a1aa !important;
    background: transparent !important;
    border: none !important;
    box-shadow: none !important;
}

/* TABS */
.stTabs [data-baseweb="tab-list"] { background: #050505 !important; border-radius: 6px !important; padding: 4px !important; gap: 4px !important; border: 1px solid #1a1a1a !important; flex-wrap: wrap !important; }
.stTabs [data-baseweb="tab"] { background: transparent !important; color: #71717a !important; border-radius: 4px !important; padding: 8px 16px !important; font-size: 0.85rem !important; font-weight: 500 !important; border: none !important; transition: all 0.2s !important; font-family: 'Inter', sans-serif !important; }
.stTabs [aria-selected="true"] { background: #1e293b !important; color: #bae6fd !important; border: 1px solid #3b82f6 !important;}
.stTabs [data-baseweb="tab"]:hover:not([aria-selected="true"]) { color: #d4d4d8 !important; background: #18181b !important; }
.stTabs [data-baseweb="tab-panel"] { background: transparent !important; padding-top: 20px !important; }

/* FILE UPLOADER */
[data-testid="stFileUploadDropzone"] { background: #09090b !important; border: 1px dashed #3f3f46 !important; border-radius: 6px !important; }
[data-testid="stFileUploadDropzone"]:hover { border-color: var(--accent) !important; background: rgba(59,130,246,0.05) !important; }
[data-testid="stFileUploaderDeleteBtn"] { color: #a1a1aa !important; }

/* DATAFRAME */
[data-testid="stDataFrame"], .stDataFrame { width: 100% !important; }
[data-testid="stDataFrame"] { border: 1px solid #27272a !important; border-radius: 4px !important; overflow: hidden !important; }
.stDataFrame th { background: #121212 !important; color: #d4d4d8 !important; font-size: 0.8rem !important; font-weight: 600 !important;}
.stDataFrame td { color: #a1a1aa !important; font-size: 0.8rem !important; }

/* EXPANDER */
[data-testid="stExpander"] details { background: #09090b !important; border: 1px solid #27272a !important; border-radius: 4px !important; }
[data-testid="stExpander"] summary { color: #d4d4d8 !important; font-size: 0.85rem !important; padding: 12px 16px !important; font-weight: 600 !important; }
[data-testid="stExpander"] summary:hover { color: #ffffff !important; background: #121212 !important;}

/* TEXT */
p, .stMarkdown p { color: var(--txt2) !important; font-size: 0.85rem !important; line-height: 1.6 !important; }
h1 { color: var(--txt1) !important; font-size: 1.55rem !important; font-weight: 600 !important; letter-spacing: -0.03em !important; margin-bottom: 0 !important; }
h2 { color: var(--txt1) !important; font-size: 1.15rem !important; font-weight: 500 !important; letter-spacing: -0.01em !important; }
h3 { color: #e4e4e7 !important; font-size: 0.95rem !important; font-weight: 500 !important; }
label, [data-testid="stWidgetLabel"] { color: #a1a1aa !important; font-size: 0.8rem !important; font-weight: 500 !important; }
small { color: #71717a !important; }

/* ALERTS */
[data-testid="stAlert"] { border-radius: 4px !important; }
.stSuccess { background: rgba(46,204,113,0.1) !important; border: 1px solid rgba(46,204,113,0.3) !important; color: #e4e4e7 !important;}
.stWarning { background: rgba(245,158,11,0.1) !important; border: 1px solid rgba(245,158,11,0.3) !important; color: #e4e4e7 !important;}
.stError { background: rgba(239,68,68,0.1) !important; border: 1px solid rgba(239,68,68,0.3) !important; color: #e4e4e7 !important;}
.stInfo { background: rgba(59,130,246,0.1) !important; border: 1px solid rgba(59,130,246,0.3) !important; color: #e4e4e7 !important;}

/* CHECKBOX */
.stCheckbox label { color: #a1a1aa !important; font-size: 0.85rem !important; }
.stCheckbox label span { border-color: var(--accent) !important; background: #0e0e0e !important; }

/* METRIC */
[data-testid="stMetric"] { background: #121212 !important; border: 1px solid #27272a !important; border-radius: 4px !important; padding: 14px 16px !important; }
[data-testid="stMetricValue"] { color: #ffffff !important; font-weight: 600 !important; font-size: 1.55rem !important; }
[data-testid="stMetricLabel"] { color: #a1a1aa !important; font-size: 0.75rem !important; }

.stSpinner > div > div { border-top-color: var(--accent) !important; }
hr { border: none !important; border-top: 1px solid #27272a !important; margin: 1.2rem 0 !important; }

::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #000000; }
::-webkit-scrollbar-thumb { background: #3f3f46; border-radius: 4px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent); }

[data-testid="stDecoration"], [data-testid="stStatusWidget"] { display: none !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────
#  CONSTANTS & MATH HELPERS
# ─────────────────────────────────────
ACCENT = "#3b82f6" # Blue
PALETTE = ["#3b82f6", "#e74c3c", "#2ecc71", "#f59e0b", "#8b5cf6", "#06b6d4", "#ec4899"]

# İndirilen grafiğin boyutunu ve kalitesini zorunlu kılan ayar matrisi
PLOTLY_CONFIG = {
    'toImageButtonOptions': {
        'format': 'png',
        'filename': 'ds_mlops_chart',
        'height': 1080,    # Ekran boyutundan bağımsız sabit 16:9 yükseklik
        'width': 1920,    # Ekran boyutundan bağımsız sabit 16:9 genişlik
        'scale': 2        # Çözünürlüğü 2 katına çıkarır (2560x1440 kalitesi)
    },
    'displayModeBar': True,
    'displaylogo': False
}

# Tüm grafiklerin standart olarak render edilmesini sağlayan merkezi fonksiyon
def render_chart(fig):
    st.plotly_chart(dark_fig(fig), config=PLOTLY_CONFIG, width='stretch')

def _hex_rgb(h):
    h = h.lstrip("#")
    return ",".join(str(int(h[i:i+2], 16)) for i in (0, 2, 4))

def safe_randint(a, b):
    if a >= b: b = a + 1
    return randint(a, b)

def safe_uniform(a, b):
    if a >= b: b = a + 1e-5
    return uniform(a, b - a)

def safe_loguniform(a, b):
    if a >= b: b = a * 1.001
    return loguniform(a, b)

def dark_fig(fig, title=""):
    if title:
        fig.update_layout(title=dict(text=title, font=dict(color="#ffffff", size=15, family="Inter"), x=0, pad=dict(l=0)))
    fig.update_layout(
        paper_bgcolor="#050505",  # Şeffaf yerine solid siyah arka plan
        plot_bgcolor="#121212",   # Grafik içi için solid koyu gri
        font=dict(color="#d4d4d8", family="Inter, system-ui, sans-serif", size=12),
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(bgcolor="rgba(0,0,0,0.9)", bordercolor="#27272a", borderwidth=1, font=dict(color="#d4d4d8", size=11)),
        hoverlabel=dict(bgcolor="#18181b", bordercolor="#3f3f46", font=dict(color="#ffffff", size=12)),
        coloraxis_colorbar=dict(tickfont=dict(color="#d4d4d8"), title_font=dict(color="#ffffff")),
    )
    axis_cfg = dict(gridcolor="#18181b", zerolinecolor="#27272a", tickfont=dict(color="#a1a1aa", size=11), linecolor="#27272a", showline=True)
    for ax in list(fig.layout):
        if ax.startswith("xaxis") or ax.startswith("yaxis"):
            fig.layout[ax].update(**axis_cfg)
    return fig

def section_head(title, sub=""):
    sub_html = f'<p style="color:#a1a1aa;font-size:.85rem;margin:4px 0 0">{sub}</p>' if sub else ""
    return f"""
    <div style="margin-bottom:20px;padding-bottom:14px;border-bottom:1px solid #1a1a1a">
      <h2 style="color:#ffffff;font-size:1.25rem;font-weight:600;margin:0;letter-spacing:-.01em">{title}</h2>{sub_html}
    </div>"""

def empty_state(msg, hint=""):
    hint_html = f'<div style="color:#a1a1aa;font-size:.85rem;margin-top:6px">{hint}</div>' if hint else ""
    return f"""
    <div style="background:#09090b;border:1px dashed #3f3f46;border-radius:6px;padding:50px 32px;text-align:center;margin:24px 0">
      <div style="color:#ffffff;font-size:1.05rem;font-weight:500">{msg}</div>{hint_html}
    </div>"""

def kpi(title, value, sub="", color=ACCENT):
    rgb = _hex_rgb(color)
    return f"""
    <div style="background:linear-gradient(145deg,#09090b,#121212);border:1px solid #27272a;
                border-radius:8px;padding:18px 20px;position:relative;overflow:hidden;min-height:92px">
      <div style="position:absolute;top:-12px;right:-12px;width:68px;height:68px;background:rgba({rgb},.15);border-radius:50%"></div>
      <div style="font-size:0.75rem;color:#a1a1aa;text-transform:uppercase;letter-spacing:.08em;font-weight:600;margin-bottom:6px">{title}</div>
      <div style="font-size:1.8rem;font-weight:700;color:#ffffff;line-height:1.05;font-variant-numeric:tabular-nums;letter-spacing:-.02em">{value}</div>
      <div style="font-size:0.8rem;color:#d4d4d8;margin-top:5px">{sub}</div>
    </div>"""

# ─────────────────────────────────────
#  DYNAMIC MODEL REGISTRY (exact params)
# ─────────────────────────────────────
MODEL_DISPLAY = {
    "LR": "Logistic Regression",
    "RF": "Random Forest",
    "ET": "Extra Trees",
    "XGB": "XGBoost",
    "LGBM": "LightGBM",
    "CB": "CatBoost",
    "SVM": "SVM",
}

def available_model_options():
    opts = [("LR", "Logistic Regression"), ("RF", "Random Forest"), ("ET", "Extra Trees")]
    if XGB_OK: opts.append(("XGB", "XGBoost"))
    if LGBM_OK: opts.append(("LGBM", "LightGBM"))
    if CB_OK: opts.append(("CB", "CatBoost"))
    opts.append(("SVM", "SVM"))
    return opts

def render_exact_params(model_key):
    """Render ONLY the selected model's parameters and return exact values."""
    params = {}
    st.markdown(f"<p style='color:#a1a1aa;font-size:.85rem'>Parameters below apply only to <b style='color:#ffffff'>{MODEL_DISPLAY.get(model_key, model_key)}</b>.</p>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)

    if model_key == "LR":
        with c1:
            params["C"] = st.number_input("C (inverse regularization)", 0.001, 100.0, 1.0, 0.01, key="ex_lr_c")
            params["max_iter"] = st.number_input("max_iter", 100, 5000, 1000, 50, key="ex_lr_mi")
        with c2:
            params["penalty"] = st.selectbox("penalty", ["l2", "l1", "elasticnet", None], index=0, key="ex_lr_pen", format_func=lambda x: "None" if x is None else x)
            params["solver"] = st.selectbox("solver", ["lbfgs", "liblinear", "saga"], index=0, key="ex_lr_sol")
        with c3:
            params["l1_ratio"] = st.slider("l1_ratio (elasticnet)", 0.0, 1.0, 0.5, 0.05, key="ex_lr_l1r")

    elif model_key == "RF":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 1000, 200, 10, key="ex_rf_n")
            params["max_depth"] = st.slider("max_depth", 2, 50, 12, 1, key="ex_rf_d")
        with c2:
            params["min_samples_split"] = st.slider("min_samples_split", 2, 50, 2, 1, key="ex_rf_mss")
            params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 50, 1, 1, key="ex_rf_msl")
        with c3:
            params["max_features"] = st.selectbox("max_features", ["sqrt", "log2", None], index=0, key="ex_rf_mf", format_func=lambda x: "None" if x is None else x)
            params["criterion"] = st.selectbox("criterion", ["gini", "entropy"], index=0, key="ex_rf_crit")
            params["bootstrap"] = st.checkbox("bootstrap", value=True, key="ex_rf_boot")

    elif model_key == "ET":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 10, 1000, 200, 10, key="ex_et_n")
            params["max_depth"] = st.slider("max_depth", 2, 50, 12, 1, key="ex_et_d")
        with c2:
            params["min_samples_split"] = st.slider("min_samples_split", 2, 50, 2, 1, key="ex_et_mss")
            params["min_samples_leaf"] = st.slider("min_samples_leaf", 1, 50, 1, 1, key="ex_et_msl")
        with c3:
            params["max_features"] = st.selectbox("max_features", ["sqrt", "log2", None], index=0, key="ex_et_mf", format_func=lambda x: "None" if x is None else x)
            params["criterion"] = st.selectbox("criterion", ["gini", "entropy"], index=0, key="ex_et_crit")
            params["bootstrap"] = st.checkbox("bootstrap", value=False, key="ex_et_boot")

    elif model_key == "XGB":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 50, 1000, 300, 10, key="ex_xgb_n")
            params["max_depth"] = st.slider("max_depth", 1, 20, 6, 1, key="ex_xgb_d")
            params["learning_rate"] = st.slider("learning_rate", 0.001, 0.5, 0.05, 0.001, key="ex_xgb_lr")
        with c2:
            params["subsample"] = st.slider("subsample", 0.1, 1.0, 0.8, 0.05, key="ex_xgb_sub")
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.1, 1.0, 0.8, 0.05, key="ex_xgb_col")
            params["min_child_weight"] = st.slider("min_child_weight", 1, 20, 1, 1, key="ex_xgb_mcw")
        with c3:
            params["gamma"] = st.slider("gamma", 0.0, 5.0, 0.0, 0.1, key="ex_xgb_gam")
            params["reg_alpha"] = st.slider("reg_alpha", 0.0, 10.0, 0.0, 0.1, key="ex_xgb_a")
            params["reg_lambda"] = st.slider("reg_lambda", 0.0, 10.0, 1.0, 0.1, key="ex_xgb_l")

    elif model_key == "LGBM":
        with c1:
            params["n_estimators"] = st.slider("n_estimators", 50, 1000, 300, 10, key="ex_lgbm_n")
            params["learning_rate"] = st.slider("learning_rate", 0.001, 0.5, 0.05, 0.001, key="ex_lgbm_lr")
            params["max_depth"] = st.slider("max_depth", -1, 30, 6, 1, key="ex_lgbm_d")
        with c2:
            params["num_leaves"] = st.slider("num_leaves", 8, 256, 31, 1, key="ex_lgbm_nl")
            params["subsample"] = st.slider("subsample", 0.1, 1.0, 0.8, 0.05, key="ex_lgbm_sub")
            params["colsample_bytree"] = st.slider("colsample_bytree", 0.1, 1.0, 0.8, 0.05, key="ex_lgbm_col")
        with c3:
            params["min_child_samples"] = st.slider("min_child_samples", 1, 100, 20, 1, key="ex_lgbm_mcs")
            params["reg_alpha"] = st.slider("reg_alpha", 0.0, 10.0, 0.0, 0.1, key="ex_lgbm_a")
            params["reg_lambda"] = st.slider("reg_lambda", 0.0, 10.0, 1.0, 0.1, key="ex_lgbm_l")

    elif model_key == "CB":
        with c1:
            params["iterations"] = st.slider("iterations", 50, 1000, 300, 10, key="ex_cb_i")
            params["depth"] = st.slider("depth", 2, 16, 6, 1, key="ex_cb_d")
            params["learning_rate"] = st.slider("learning_rate", 0.001, 0.5, 0.05, 0.001, key="ex_cb_lr")
        with c2:
            params["l2_leaf_reg"] = st.slider("l2_leaf_reg", 0.1, 30.0, 3.0, 0.1, key="ex_cb_l2")
            params["subsample"] = st.slider("subsample", 0.1, 1.0, 0.8, 0.05, key="ex_cb_sub")
            params["min_data_in_leaf"] = st.slider("min_data_in_leaf", 1, 50, 1, 1, key="ex_cb_mdl")
        with c3:
            params["border_count"] = st.selectbox("border_count", [32, 64, 128, 254], index=2, key="ex_cb_bc")
            params["bagging_temperature"] = st.slider("bagging_temperature", 0.0, 5.0, 1.0, 0.1, key="ex_cb_bt")

    elif model_key == "SVM":
        with c1:
            params["C"] = st.number_input("C", 0.001, 100.0, 1.0, 0.01, key="ex_svm_c")
            params["kernel"] = st.selectbox("kernel", ["rbf", "linear", "poly", "sigmoid"], index=0, key="ex_svm_k")
        with c2:
            params["gamma"] = st.selectbox("gamma", ["scale", "auto"], index=0, key="ex_svm_g")
            params["degree"] = st.slider("degree (poly)", 2, 8, 3, 1, key="ex_svm_deg")
        with c3:
            st.caption("probability=True is always enabled for ROC/threshold analysis.")

    return params

def build_classifier(model_key, params):
    """Instantiate the selected model with the exact user-provided parameters."""
    if model_key == "LR":
        kw = dict(C=float(params["C"]), max_iter=int(params["max_iter"]),
                  penalty=params["penalty"], solver=params["solver"],
                  random_state=42, n_jobs=1)
        if params["penalty"] == "elasticnet":
            kw["l1_ratio"] = float(params.get("l1_ratio", 0.5))
        if params["penalty"] is None:
            kw["penalty"] = None
        return LogisticRegression(**kw)
    if model_key == "RF":
        return RandomForestClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]), min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"], criterion=params["criterion"],
            bootstrap=bool(params["bootstrap"]), random_state=42, n_jobs=1)
    if model_key == "ET":
        return ExtraTreesClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            min_samples_split=int(params["min_samples_split"]), min_samples_leaf=int(params["min_samples_leaf"]),
            max_features=params["max_features"], criterion=params["criterion"],
            bootstrap=bool(params["bootstrap"]), random_state=42, n_jobs=1)
    if model_key == "XGB":
        return XGBClassifier(
            n_estimators=int(params["n_estimators"]), max_depth=int(params["max_depth"]),
            learning_rate=float(params["learning_rate"]), subsample=float(params["subsample"]),
            colsample_bytree=float(params["colsample_bytree"]), min_child_weight=int(params["min_child_weight"]),
            gamma=float(params["gamma"]), reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]), eval_metric="logloss",
            verbosity=0, random_state=42, n_jobs=1)
    if model_key == "LGBM":
        return LGBMClassifier(
            n_estimators=int(params["n_estimators"]), learning_rate=float(params["learning_rate"]),
            max_depth=int(params["max_depth"]), num_leaves=int(params["num_leaves"]),
            subsample=float(params["subsample"]), colsample_bytree=float(params["colsample_bytree"]),
            min_child_samples=int(params["min_child_samples"]), reg_alpha=float(params["reg_alpha"]),
            reg_lambda=float(params["reg_lambda"]), verbosity=-1, random_state=42, n_jobs=1)
    if model_key == "CB":
        return CatBoostClassifier(
            iterations=int(params["iterations"]), depth=int(params["depth"]),
            learning_rate=float(params["learning_rate"]), l2_leaf_reg=float(params["l2_leaf_reg"]),
            subsample=float(params["subsample"]), min_data_in_leaf=int(params["min_data_in_leaf"]),
            border_count=int(params["border_count"]), bagging_temperature=float(params["bagging_temperature"]),
            verbose=0, random_state=42, thread_count=1, allow_writing_files=False)
    if model_key == "SVM":
        return SVC(
            C=float(params["C"]), kernel=params["kernel"], gamma=params["gamma"],
            degree=int(params["degree"]), probability=True, random_state=42)
    raise ValueError(f"Unknown model: {model_key}")

def evaluate_trained_model(estimator, X_test, y_test):
    """Dynamically compute metrics/predictions for the currently trained model."""
    y_pred = estimator.predict(X_test)
    is_binary = len(np.unique(y_test)) <= 2
    avg = "binary" if is_binary else "weighted"

    y_proba = None
    if hasattr(estimator, "predict_proba"):
        proba_full = estimator.predict_proba(X_test)
        y_proba = proba_full[:, 1] if is_binary else proba_full

    metrics = {
        "Accuracy": float(accuracy_score(y_test, y_pred)),
        "Precision": float(precision_score(y_test, y_pred, average=avg, zero_division=0)),
        "Recall": float(recall_score(y_test, y_pred, average=avg, zero_division=0)),
        "F1 Score": float(f1_score(y_test, y_pred, average=avg, zero_division=0)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_test, y_pred)),
    }
    if y_proba is not None:
        try:
            if is_binary:
                metrics["ROC-AUC"] = float(roc_auc_score(y_test, y_proba))
                metrics["PR-AUC"] = float(average_precision_score(y_test, y_proba))
            else:
                metrics["ROC-AUC"] = float(roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted"))
        except Exception:
            pass

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    return metrics, cm, y_pred, y_proba, report

# ─────────────────────────────────────
#  SESSION STATE INITIALIZATION
# ─────────────────────────────────────
if "initialized" not in st.session_state:
    st.session_state.df = None
    st.session_state.df_eng = None
    st.session_state.target_col = None
    st.session_state.X_train = None
    st.session_state.X_test = None
    st.session_state.y_train = None
    st.session_state.y_test = None
    st.session_state.preprocessor = None
    st.session_state.X_train_p = None
    st.session_state.X_test_p = None
    st.session_state.model_results = {}
    st.session_state.best_model = None
    st.session_state.best_model_name = ""
    st.session_state.last_eval = None
    st.session_state.selected_features = []
    st.session_state.feature_names = []
    st.session_state.split_done = False
    st.session_state.preprocess_done = False
    st.session_state.training_done = False
    st.session_state.eng_cols_added = []
    st.session_state.baseline_results = None
    st.session_state.numeric_cols_pp = []
    st.session_state.nominal_cols_pp = []
    st.session_state.ordinal_cols_pp = []
    st.session_state.ordinal_orders = {}
    st.session_state.uploaded_filename = ""
    st.session_state.initialized = True

if "sidebar_open" not in st.session_state:
    st.session_state.sidebar_open = True

if "menu" not in st.session_state:
    st.session_state.menu = "Data Upload & Analysis"

MENU_ITEMS = [
    "Data Upload & Analysis",
    "Visualization",
    "Feature Engineering",
    "Preprocessing",
    "Feature Selection",
    "Model Training",
    "Evaluation & SHAP",
    "Prediction & Export"
]

# ─────────────────────────────────────
#  ÖZEL SOL PANEL
# ─────────────────────────────────────
if st.session_state.sidebar_open:
    st.markdown("""
    <style>
    section[data-testid="stMain"] {
        padding-left: 280px !important;
        transition: padding-left 0.15s ease;
    }
    </style>
    """, unsafe_allow_html=True)

    with st.container(key="custom_sidebar"):
        with st.container(key="sb_close_btn_wrap"):
            with st.container(key="sb_close_btn"):
                if st.button("\u2715", key="sidebar_close_button"):
                    st.session_state.sidebar_open = False
                    st.rerun()
        st.markdown("""
        <div style="padding:0 0 20px 0;border-bottom:1px solid #181818;margin-bottom:16px;text-align:center">
          <div style="font-size:0.8rem;font-weight:800;color:#bae6fd;letter-spacing:0.05em;text-transform:uppercase;">Data Science & MLOps</div>
          <div style="font-size:1.2rem;color:#ffffff;margin-top:4px;font-weight:700">Analytical Platform</div>
        </div>""", unsafe_allow_html=True)

        for item in MENU_ITEMS:
            is_active = st.session_state.menu == item
            if st.button(item, key=f"nav_{item}", type="primary" if is_active else "secondary", width="stretch"):
                st.session_state.menu = item
                st.rerun()

        st.markdown("<div style='margin-top:30px;padding:0 4px'>", unsafe_allow_html=True)
        if st.session_state.df is not None:
            df_ = st.session_state.df
            n, c = df_.shape
            st.markdown(f"""
            <div style="background:#09090b;border:1px solid #27272a;border-radius:6px;padding:14px 16px">
              <div style="font-size:.7rem;color:#a1a1aa;text-transform:uppercase;letter-spacing:.08em;margin-bottom:10px;font-weight:600">PIPELINE STATUS</div>
              <div style="display:flex;flex-direction:column;gap:6px">
                <div style="display:flex;justify-content:space-between"><span style="color:#d4d4d8;font-size:.85rem">Rows</span><span style="color:#3b82f6;font-size:.85rem;font-weight:600">{n:,}</span></div>
                <div style="display:flex;justify-content:space-between"><span style="color:#d4d4d8;font-size:.85rem">Columns</span><span style="color:#3b82f6;font-size:.85rem;font-weight:600">{c}</span></div>
                <div style="display:flex;justify-content:space-between"><span style="color:#d4d4d8;font-size:.85rem">Split</span><span style="color:{'#2ecc71' if st.session_state.split_done else '#71717a'};font-size:.85rem;font-weight:600">{'Completed' if st.session_state.split_done else 'Pending'}</span></div>
                <div style="display:flex;justify-content:space-between"><span style="color:#d4d4d8;font-size:.85rem">Preprocess</span><span style="color:{'#2ecc71' if st.session_state.preprocess_done else '#71717a'};font-size:.85rem;font-weight:600">{'Completed' if st.session_state.preprocess_done else 'Pending'}</span></div>
                <div style="display:flex;justify-content:space-between"><span style="color:#d4d4d8;font-size:.85rem">Training</span><span style="color:{'#2ecc71' if st.session_state.training_done else '#71717a'};font-size:.85rem;font-weight:600">{'Completed' if st.session_state.training_done else 'Pending'}</span></div>
              </div>
            </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""<div style="background:#09090b;border:1px dashed #27272a;border-radius:4px;padding:12px 14px;text-align:center">
              <div style="color:#a1a1aa;font-size:.85rem">No Data Uploaded</div></div>""", unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)
else:
    with st.container(key="sb_open_btn"):
        if st.button("\u2630", key="sidebar_open_button"):
            st.session_state.sidebar_open = True
            st.rerun()

menu = st.session_state.menu

# ══════════════════════════════════════════════════════
#  MODULE 1 — VERI YUKLEME & ANALIZ
# ══════════════════════════════════════════════════════
if menu == "Data Upload & Analysis":
    st.markdown(section_head("Data Upload & Exploratory Analysis",
                              "Upload your CSV or Excel (xls, xlsx) file, set the index/target variable, and generate a profile"), unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload Data File (CSV, XLS, XLSX)", type=["csv", "xls", "xlsx"], label_visibility="collapsed")
    if uploaded:
        if st.session_state.uploaded_filename != uploaded.name:
            try:
                if uploaded.name.endswith('.csv'):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)
                    
                st.session_state.uploaded_filename = uploaded.name
                st.session_state.df = df_raw
                st.session_state.df_eng = df_raw.copy()
                st.session_state.split_done = False
                st.session_state.preprocess_done = False
                st.session_state.training_done = False
                st.session_state.model_results = {}
                st.session_state.best_model = None
                st.session_state.last_eval = None
                st.session_state.selected_features = []
                st.session_state.feature_names = []
                st.success(f"{uploaded.name} uploaded successfully: {df_raw.shape[0]:,} rows, {df_raw.shape[1]} columns")
            except Exception as e:
                st.error(f"File read error: {e}. For Excel files, make sure 'openpyxl' or 'xlrd' is installed.")

    if st.session_state.df is None:
        st.markdown(empty_state("No data uploaded yet", "Upload a CSV or Excel file from the area above"), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df

    with st.expander("Index & Target Variable Settings", expanded=False):
        c1, c2 = st.columns(2)
        with c1:
            id_col = st.selectbox("ID Column (to set as index)", ["Select..."] + df.columns.tolist(), key="id_col_sel")
            if st.button("Set as Index", key="set_index_btn") and id_col != "Select...":
                df2 = st.session_state.df.copy()
                df2 = df2.set_index(id_col)
                st.session_state.df = df2
                st.session_state.df_eng = df2.copy()
                st.success(f"'{id_col}' set as index.")
                st.rerun()
        with c2:
            st.markdown("<small style='color:#a1a1aa;'>Target Variable Distribution</small>", unsafe_allow_html=True)
            chk_col = st.selectbox("Column to show distribution", df.columns.tolist(),
                                    index=len(df.columns)-1, key="imbalance_col")
            vc = df[chk_col].value_counts(normalize=True).mul(100).round(2)
            st.dataframe(vc.to_frame("Ratio (%)"), width="stretch")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()
    miss_tot = df.isna().sum().sum()
    miss_cols = (df.isna().sum() > 0).sum()

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1: st.markdown(kpi("Total Rows", f"{df.shape[0]:,}", color="#3b82f6"), unsafe_allow_html=True)
    with col2: st.markdown(kpi("Total Columns", f"{df.shape[1]}", color="#8b5cf6"), unsafe_allow_html=True)
    with col3: st.markdown(kpi("Numeric Features", f"{len(num_cols)}", color="#06b6d4"), unsafe_allow_html=True)
    with col4: st.markdown(kpi("Categorical Features", f"{len(cat_cols)}", color="#f59e0b"), unsafe_allow_html=True)
    with col5: st.markdown(kpi("Missing Cells", f"{miss_tot:,}", sub=f"{miss_cols} columns", color="#e74c3c" if miss_tot > 0 else "#2ecc71"), unsafe_allow_html=True)

    st.markdown("<div style='margin-top:20px'/>", unsafe_allow_html=True)
    tab_ov, tab_stat, tab_miss, tab_dtypes = st.tabs(["Preview", "Statistics", "Missing Data", "Data Types"])

    with tab_ov:
        n_rows = st.slider("Number of rows to display", 5, 50, 10, key="prev_n")
        st.dataframe(df.head(n_rows), width="stretch")

    with tab_stat:
        c1, c2 = st.columns([3, 2])
        with c1:
            st.markdown("<h3>Numeric Statistics</h3>", unsafe_allow_html=True)
            desc = df[num_cols].describe().T[["mean", "50%", "std", "min", "max"]].round(3)
            desc["skewness"] = df[num_cols].skew().round(3)
            desc["kurtosis"] = df[num_cols].kurt().round(3)
            st.dataframe(desc, width="stretch")
        with c2:
            st.markdown("<h3>Mode (Most Frequent Value)</h3>", unsafe_allow_html=True)
            st.dataframe(df.mode().iloc[0].to_frame("Mode Value").astype(str), width="stretch")

    with tab_miss:
        miss_s = df.isna().sum()
        miss_df = miss_s[miss_s > 0].reset_index()
        miss_df.columns = ["Columns", "Missing Count"]
        if len(miss_df) == 0:
            st.markdown(empty_state("No missing data", "Your dataset has no missing values."), unsafe_allow_html=True)
        else:
            miss_df["Ratio (%)"] = (miss_df["Missing Count"] / len(df) * 100).round(2)
            fig = px.bar(miss_df, x="Columns", y="Ratio (%)", color="Ratio (%)",
                         color_continuous_scale="Reds", title="Missing Data Ratio (%)")
            render_chart(fig)
            st.dataframe(miss_df, width="stretch")

    with tab_dtypes:
        dtype_df = pd.DataFrame({"Columns": df.columns, "Type": df.dtypes.values.astype(str),
                                  "Unique": df.nunique().values, "Null": df.isna().sum().values})
        st.dataframe(dtype_df, width="stretch")

# ══════════════════════════════════════════════════════
#  MODULE 2 — GORSELLESTIRME
# ══════════════════════════════════════════════════════
elif menu == "Visualization":
    if st.session_state.df is None:
        st.markdown(empty_state("No data uploaded", "Upload data from Module 1 first"), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df_eng.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown(section_head("Dynamic Data Visualization",
                              "Interactive data exploration with chart types"), unsafe_allow_html=True)

    tabs = st.tabs(["Histogram/KDE", "Boxplot", "Violin", "Scatter",
                    "Correlation", "Categorical", "Pie", "Radar",
                    "Pivot Heatmap", "Dual Axis", "Joint Density", "Small Multiples",
                    "PCA Projection", "Scatter Matrix", "Parallel Coordinates", "3D Scatter"])

    with tabs[0]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: feat = st.selectbox("Feature", num_cols, key="hist_feat")
        with c2: target = st.selectbox("Target (Hue)", ["None"] + df.columns.tolist(), index=0, key="hist_hue")
        with c3: bins = st.slider("Number of bins", 10, 80, 30, key="hist_bins")
        color_arg = target if target != "None" else None
        fig = px.histogram(df, x=feat, color=color_arg, nbins=bins, barmode="overlay",
                            color_discrete_sequence=PALETTE, opacity=0.8, marginal="rug", title=f"{feat} Distribution")
        fig.update_traces(marker_line_width=0)
        render_chart(fig)

    with tabs[1]:
        c1, c2, c3 = st.columns([2, 2, 1])
        with c1: bx_y = st.selectbox("Value (Y) ", num_cols, key="bx_y")
        with c2: bx_x = st.selectbox("Group (X)", df.columns.tolist(), index=len(df.columns)-1, key="bx_x")
        with c3: bx_pts = st.selectbox("Points", ["outliers", "all", "False"], key="bx_pts")
        fig = px.box(df, x=bx_x, y=bx_y, color=bx_x, color_discrete_sequence=PALETTE,
                     points=bx_pts if bx_pts != "False" else False, title=f"{bx_y} - Boxplot by {bx_x}")
        render_chart(fig)

    with tabs[2]:
        c1, c2 = st.columns(2)
        with c1: vl_y = st.selectbox("Value (Y)  ", num_cols, key="vl_y")
        with c2: vl_x = st.selectbox("Group (X) ", df.columns.tolist(), index=len(df.columns)-1, key="vl_x")
        fig = px.violin(df, x=vl_x, y=vl_y, color=vl_x, box=True, points="outliers",
                         color_discrete_sequence=PALETTE, title=f"{vl_y} - Violin by {vl_x}")
        render_chart(fig)

    with tabs[3]:
        c1, c2, c3, c4 = st.columns(4)
        with c1: sc_x = st.selectbox("X Axis", num_cols, key="sc_x")
        with c2: sc_y = st.selectbox("Y Axis", num_cols, index=min(1, len(num_cols)-1), key="sc_y")
        with c3: sc_c = st.selectbox("Color (Hue) ", ["None"] + df.columns.tolist(), index=0, key="sc_c")
        with c4: sc_sz = st.selectbox("Size", ["None"] + num_cols, key="sc_sz")
        sz_col = None if sc_sz == "None" else sc_sz
        color_col = None if sc_c == "None" else sc_c
        fig = px.scatter(df, x=sc_x, y=sc_y, color=color_col, size=sz_col, color_discrete_sequence=PALETTE,
                          opacity=0.7, trendline="ols", trendline_color_override="#d4d4d8", title=f"{sc_x} vs {sc_y}")
        render_chart(fig)

    with tabs[4]:
        c1, c2 = st.columns([3, 1])
        with c2:
            corr_method = st.selectbox("Method", ["pearson", "spearman", "kendall"], key="corr_m")
            corr_cols = st.multiselect("Columns", num_cols, default=num_cols[:min(12, len(num_cols))], key="corr_cols")
        with c1:
            if len(corr_cols) >= 2:
                corr = df[corr_cols].corr(method=corr_method).round(3)
                fig = px.imshow(corr, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, text_auto=".2f",
                                 aspect="auto", title=f"{corr_method.capitalize()} Correlation Matrix")
                fig.update_traces(textfont_size=10)
                render_chart(fig)
            else:
                st.info("Select at least 2 columns.")

    with tabs[5]:
        if not cat_cols:
            st.info("No categorical columns found.")
        else:
            c1, c2, c3 = st.columns([2, 2, 1])
            with c1: cat_feat = st.selectbox("Categorical Column", cat_cols, key="cat_feat")
            with c2: cat_tgt = st.selectbox("Target (For Numeric Rate)", ["None"] + num_cols, index=0, key="cat_tgt")
            with c3: cat_type = st.selectbox("Chart", ["Countplot", "Bar Plot (Average)"], key="cat_type")
            
            if cat_type == "Countplot" or cat_tgt == "None":
                fig = px.histogram(df, x=cat_feat, color=cat_feat, barmode="group",
                                    color_discrete_sequence=PALETTE, title=f"{cat_feat} - Distribution (Count)")
                fig.update_traces(marker_line_width=0)
            else:
                try:
                    rate = df.groupby(cat_feat)[cat_tgt].mean().reset_index()
                    rate = rate.sort_values(cat_tgt, ascending=False)
                    fig = px.bar(rate, x=cat_feat, y=cat_tgt, color=cat_tgt,
                                 color_continuous_scale="Plasma", text_auto=".2f", title=f"{cat_tgt} Average - {cat_feat}")
                    fig.update_traces(textfont_size=11)
                except Exception as e:
                    st.warning(f"Chart could not be created: {e}")
                    fig = go.Figure()
            render_chart(fig)

    with tabs[6]:
        c1, c2 = st.columns([2, 2])
        with c1: pie_col = st.selectbox("Columns", cat_cols if cat_cols else df.columns.tolist(), key="pie_col")
        with c2: pie_top = st.slider("Max categories", 3, 20, 8, key="pie_top")
        counts = df[pie_col].value_counts().head(pie_top).reset_index()
        counts.columns = [pie_col, "count"]
        fig = px.pie(counts, names=pie_col, values="count", color_discrete_sequence=PALETTE, hole=0.45, title=f"{pie_col} - Distribution")
        fig.update_traces(textfont_size=12, marker=dict(line=dict(color="#050505", width=2)))
        render_chart(fig)

    with tabs[7]:
        if not cat_cols:
            st.info("A categorical column is required.")
        else:
            c1, c2 = st.columns(2)
            with c1: rad_grp = st.selectbox("Group (Categorical)", cat_cols, key="rad_grp")
            with c2: rad_val = st.selectbox("Value (Numeric)", num_cols, key="rad_val")
            try:
                radar = df.groupby(rad_grp)[rad_val].mean().reset_index()
                categories = radar[rad_grp].astype(str).tolist()
                values = radar[rad_val].tolist()
                fig = go.Figure(go.Scatterpolar(r=values + [values[0]], theta=categories + [categories[0]],
                                                 fill="toself", fillcolor=f"rgba(59,130,246,.3)", 
                                                 line=dict(color="#3b82f6", width=2), marker=dict(size=7, color="#e74c3c"))) 
                fig.update_layout(polar=dict(bgcolor="rgba(24,24,27,.3)",
                                              radialaxis=dict(visible=True, gridcolor="#3f3f46", tickfont=dict(color="#d4d4d8")),
                                              angularaxis=dict(gridcolor="#3f3f46", tickfont=dict(color="#e4e4e7"))),
                                  title=f"Radar: {rad_val} by {rad_grp}")
                render_chart(fig)
            except Exception as e:
                st.warning(str(e))

    with tabs[8]:
        c1, c2, c3 = st.columns(3)
        with c1: piv_idx = st.selectbox("Row (Index)", cat_cols if cat_cols else df.columns.tolist(), key="piv_idx")
        with c2: piv_col = st.selectbox("Columns ", [c for c in cat_cols if c != piv_idx] if len(cat_cols)>1 else df.columns.tolist(), key="piv_col")
        with c3: piv_val = st.selectbox("Value (Numeric) ", num_cols, key="piv_val")
        try:
            pivot = df.pivot_table(index=piv_idx, columns=piv_col, values=piv_val, aggfunc="mean", observed=False).round(2)
            fig = px.imshow(pivot, color_continuous_scale="Viridis", text_auto=".2f", aspect="auto",
                             title=f"Pivot: {piv_val} ({piv_idx} x {piv_col})")
            fig.update_traces(textfont_size=10)
            render_chart(fig)
        except Exception as e:
            st.warning(f"Pivot could not be created: {e}")

    with tabs[9]:
        c1, c2, c3 = st.columns(3)
        with c1: da_x = st.selectbox("X Axis ", df.columns.tolist(), key="da_x")
        with c2: da_y1 = st.selectbox("Y1 (Bar)", num_cols, key="da_y1")
        with c3: da_y2 = st.selectbox("Y2 (Line)", num_cols, index=min(1, len(num_cols)-1), key="da_y2")
        try:
            grp = df.groupby(da_x)[[da_y1, da_y2]].mean().reset_index()
            fig = make_subplots(specs=[[{"secondary_y": True}]])
            fig.add_trace(go.Bar(x=grp[da_x], y=grp[da_y1], name=da_y1, marker_color="#3b82f6", opacity=0.85), secondary_y=False)
            fig.add_trace(go.Scatter(x=grp[da_x], y=grp[da_y2], name=da_y2, line=dict(color="#e74c3c", width=3),
                                      mode="lines+markers", marker=dict(size=8)), secondary_y=True)
            fig.update_layout(title=f"Dual Axis: {da_y1} & {da_y2} by {da_x}")
            render_chart(fig)
        except Exception as e:
            st.warning(str(e))

    with tabs[10]:
        c1, c2, c3 = st.columns(3)
        with c1: jd_x = st.selectbox("X ", num_cols, key="jd_x")
        with c2: jd_y = st.selectbox("Y ", num_cols, index=min(1, len(num_cols)-1), key="jd_y")
        with c3: jd_filter_col = st.selectbox("Filter Column (Optional)", ["None"] + cat_cols, key="jd_filter_col")
        df_jd = df
        if jd_filter_col != "None":
            jd_val = st.selectbox("Filter Value", sorted(df[jd_filter_col].dropna().unique().tolist(), key=str), key="jd_filter_val")
            df_jd = df[df[jd_filter_col] == jd_val]
        fig = px.density_heatmap(df_jd, x=jd_x, y=jd_y, nbinsx=25, nbinsy=25,
                                  color_continuous_scale="Inferno", title=f"Density Map: {jd_x} vs {jd_y}")
        render_chart(fig)

    with tabs[11]:
        if len(cat_cols) < 2:
            st.info("At least 2 categorical columns are required for small multiples.")
        else:
            c1, c2 = st.columns(2)
            with c1: sm_group = st.selectbox("Group Column ", cat_cols, key="sm_group")
            with c2: sm_target = st.selectbox("Column to Display ", cat_cols, index=min(1, len(cat_cols)-1), key="sm_target")
            groups = df[sm_group].dropna().unique().tolist()
            n = len(groups)
            ncols = 4
            nrows = math.ceil(n / ncols)
            specs = [[{"type": "domain"} for _ in range(ncols)] for _ in range(nrows)]
            fig = make_subplots(rows=nrows, cols=ncols, specs=specs, subplot_titles=[str(g) for g in groups])
            for i, g in enumerate(groups):
                sub = df[df[sm_group] == g][sm_target].value_counts()
                fig.add_trace(go.Pie(labels=[str(l) for l in sub.index], values=sub.values, hole=0.35,
                                      marker=dict(colors=PALETTE, line=dict(color="#050505", width=1.5)),
                                      showlegend=(i == 0), textfont_size=10),
                              row=(i // ncols) + 1, col=(i % ncols) + 1)
            fig.update_layout(height=280 * nrows, title_text=f"{sm_target} Distribution - By {sm_group}")
            render_chart(fig)

    with tabs[12]:
        st.markdown("<p style='color:#a1a1aa;'>Principal Component Analysis (PCA): Çok boyutlu veriyi 2 boyuta indirgeyerek sınıfların ayrılabilirliğini gösterir.</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        with c1: 
            pca_hue = st.selectbox("Hedef Değişken (Renk)", ["None"] + cat_cols, key="pca_hue")
            pca_feats = st.multiselect("PCA için Özellikler", num_cols, default=num_cols[:min(5, len(num_cols))], key="pca_feats")
        with c2:
            if len(pca_feats) >= 2:
                try:
                    df_pca = df.dropna(subset=pca_feats)
                    X_pca = StandardScaler().fit_transform(df_pca[pca_feats])
                    pca = PCA(n_components=2)
                    components = pca.fit_transform(X_pca)
                    
                    df_pca["PC1"] = components[:, 0]
                    df_pca["PC2"] = components[:, 1]
                    var_ratio = pca.explained_variance_ratio_
                    
                    fig = px.scatter(df_pca, x="PC1", y="PC2", color=pca_hue if pca_hue != "None" else None,
                                     color_discrete_sequence=PALETTE, opacity=0.7,
                                     title=f"PCA Projeksiyonu (Varyans: %{var_ratio[0]*100:.1f} ve %{var_ratio[1]*100:.1f})")
                    render_chart(fig)
                except Exception as e:
                    st.error(f"PCA Hatası: {e}")
            else:
                st.info("PCA analizi için en az 2 sayısal özellik seçmelisiniz.")

    with tabs[13]:
        st.markdown("<p style='color:#a1a1aa;'>Scatter Matrix: Seçili sayısal özelliklerin birbirleri ile olan tüm ikili ilişkilerini ve dağılımlarını eş zamanlı sunar.</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 4])
        with c1:
            sm_cols = st.multiselect("Matris Özellikleri", num_cols, default=num_cols[:min(4, len(num_cols))], key="sm_cols")
            sm_color = st.selectbox("Renk Değişkeni", ["None"] + cat_cols, key="sm_color")
        with c2:
            if len(sm_cols) >= 2:
                fig = px.scatter_matrix(df, dimensions=sm_cols, color=sm_color if sm_color != "None" else None,
                                        color_discrete_sequence=PALETTE, title="Özellik İlişki Matrisi")
                fig.update_traces(diagonal_visible=False) 
                fig.update_layout(height=700)
                render_chart(fig)
            else:
                st.info("Scatter Matrix için en az 2 özellik seçmelisiniz.")

    with tabs[14]:
        st.markdown("<p style='color:#a1a1aa;'>Parallel Coordinates: Farklı ölçeklerdeki özelliklerin profillerini izlemek ve aykırı değerleri bulmak için kullanılır.</p>", unsafe_allow_html=True)
        pc_cols = st.multiselect("Paralel Eksenler", num_cols, default=num_cols[:min(6, len(num_cols))], key="pc_cols")
        pc_color = st.selectbox("Sayısal Renk Skalası", num_cols, key="pc_color")
        if len(pc_cols) >= 2:
            fig = px.parallel_coordinates(df, dimensions=pc_cols, color=pc_color,
                                          color_continuous_scale="Viridis", title="Paralel Koordinat Analizi")
            render_chart(fig)
        else:
            st.info("En az 2 eksen seçmelisiniz.")

    with tabs[15]:
        c1, c2, c3, c4 = st.columns(4)
        with c1: s3_x = st.selectbox("X Ekseni", num_cols, key="s3_x")
        with c2: s3_y = st.selectbox("Y Ekseni", num_cols, index=min(1, len(num_cols)-1), key="s3_y")
        with c3: s3_z = st.selectbox("Z Ekseni", num_cols, index=min(2, len(num_cols)-1), key="s3_z")
        with c4: s3_c = st.selectbox("Renk", ["None"] + df.columns.tolist(), index=0, key="s3_c")
        
        color_col = None if s3_c == "None" else s3_c
        fig = px.scatter_3d(df, x=s3_x, y=s3_y, z=s3_z, color=color_col, color_discrete_sequence=PALETTE,
                            opacity=0.7, title=f"3D Uzay: {s3_x}, {s3_y}, {s3_z}")
        fig.update_layout(height=650, scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#27272a"),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#27272a"),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor="#27272a")
        ))
        render_chart(fig)

# ══════════════════════════════════════════════════════
#  MODULE 3 — FEATURE ENGINEERING
# ══════════════════════════════════════════════════════
elif menu == "Feature Engineering":
    if st.session_state.df is None:
        st.markdown(empty_state("No data uploaded"), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df.copy() 
    num_cols = st.session_state.df_eng.select_dtypes(include=np.number).columns.tolist()
    cat_cols = st.session_state.df_eng.select_dtypes(exclude=np.number).columns.tolist()

    st.markdown(section_head("Feature Engineering", "Derive new features with mathematical operations, binning, and custom formulas"), unsafe_allow_html=True)

    tab_math, tab_bin, tab_custom = st.tabs(["Mathematical Operations", "Column Binning", "Custom Formulaa"])

    with tab_math:
        c1, c2, c3, c4 = st.columns(4)
        with c1: col_a = st.selectbox("Column A", num_cols, key="math_cola")
        with c2: op = st.selectbox("Operation", ["Add (+)", "Subtract (-)", "Multiply (*)", "Divide (/)"], key="math_op")
        with c3: col_b = st.selectbox("Column B", num_cols, index=min(1, len(num_cols)-1), key="math_colb")
        with c4: feat_name = st.text_input("New Column Name", value="Yeni_Feature", key="math_name")
        
        if st.button("Apply Operation", key="math_apply"):
            try:
                df_cur = st.session_state.df_eng.copy()
                if op == "Add (+)": df_cur[feat_name] = df_cur[col_a] + df_cur[col_b]
                elif op == "Subtract (-)": df_cur[feat_name] = df_cur[col_a] - df_cur[col_b]
                elif op == "Multiply (*)": df_cur[feat_name] = df_cur[col_a] * df_cur[col_b]
                elif op == "Divide (/)": df_cur[feat_name] = df_cur[col_a] / df_cur[col_b].replace(0, np.nan)
                
                st.session_state.df_eng = df_cur
                st.success(f"Column '{feat_name}' added successfully.")
            except Exception as e:
                st.error(f"Error during operation: {e}")

    with tab_bin:
        c1, c2, c3, c4 = st.columns(4)
        with c1: bin_col = st.selectbox("Column  ", num_cols, key="bin_col")
        with c2: bin_type = st.selectbox("Method", ["qcut (Equal Frequency)", "cut (Equal Interval)"], key="bin_type")
        with c3: bin_q = st.slider("Number of bins", 2, 10, 4, key="bin_q")
        with c4: bin_name = st.text_input("New column name ", value=f"{bin_col}_Bin", key="bin_name")
        if st.button("Apply Binning", key="bin_apply"):
            try:
                df_cur = st.session_state.df_eng.copy()
                if "qcut" in bin_type:
                    df_cur[bin_name] = pd.qcut(df_cur[bin_col], q=bin_q, labels=False, duplicates="drop")
                else:
                    df_cur[bin_name] = pd.cut(df_cur[bin_col], bins=bin_q, labels=False)
                st.session_state.df_eng = df_cur
                st.success(f"Column '{bin_name}' created.")
            except Exception as e:
                st.error(str(e))

    with tab_custom:
        st.markdown("<p style='color:#a1a1aa;'>The current data is accessed via the <code>df</code> variable. Enter a valid Pandas expression.</p>", unsafe_allow_html=True)
        c1, c2 = st.columns([1, 3])
        with c1: cust_name = st.text_input("Feature name", value="Custom_Feature", key="cust_name")
        with c2: cust_expr = st.text_input("Formula", value="df.iloc[:,0] / df.iloc[:,1]", key="cust_expr")
        if st.button("Apply Formulaa", key="cust_apply"):
            try:
                df_cur = st.session_state.df_eng.copy()
                result = eval(cust_expr, {"df": df_cur, "pd": pd, "np": np})
                df_cur[cust_name] = result
                st.session_state.df_eng = df_cur
                st.success(f"Column '{cust_name}' added successfully.")
            except Exception as e:
                st.error(f"Formulaa error: {e}")

    st.markdown("---")
    st.markdown(f"<div style='color:#a1a1aa;font-size:.85rem'>Current Dataset Size: <b style='color:#ffffff'>{st.session_state.df_eng.shape[0]:,}</b> rows, <b style='color:#ffffff'>{st.session_state.df_eng.shape[1]}</b> sutun</div>", unsafe_allow_html=True)
    with st.expander("Current Dataset Preview"):
        st.dataframe(st.session_state.df_eng.head(10), width="stretch")

# ══════════════════════════════════════════════════════
#  MODULE 4 — PREPROCESSING
# ══════════════════════════════════════════════════════
elif menu == "Preprocessing":
    if st.session_state.df is None:
        st.markdown(empty_state("No data uploaded"), unsafe_allow_html=True)
        st.stop()

    df = st.session_state.df_eng.copy()

    st.markdown(section_head("Data Preprocessing Pipeline", "Train/Test Split, Scaling, and Categorical Encoding steps"), unsafe_allow_html=True)

    with st.expander("My Dataset Is Already Ready (Skip Preprocessing Steps)", expanded=False):
        st.markdown("<p style='color:#a1a1aa;'>Use this button if your data is fully ready for training. The system will still filter hidden text and NaN values for safety.</p>", unsafe_allow_html=True)
        fast_target = st.selectbox("Target Variable  ", df.columns.tolist(), index=len(df.columns)-1, key="fast_tgt")
        
        if st.button("Go Directly to Training"):
            X = df.drop(columns=[fast_target])
            y = df[fast_target]
            
            y = LabelEncoder().fit_transform(y)
            
            Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
            
            st.session_state.X_train, st.session_state.X_test = Xtr, Xte
            st.session_state.y_train, st.session_state.y_test = ytr, yte
            st.session_state.target_col = fast_target
            
            num_cols_auto = Xtr.select_dtypes(include=np.number).columns.tolist()
            cat_cols_auto = Xtr.select_dtypes(exclude=np.number).columns.tolist()
            
            num_pipe = Pipeline([("imputer", SimpleImputer(strategy="median"))])
            cat_pipe = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
            ])
            
            dummy_preprocessor = ColumnTransformer(
                transformers=[
                    ("num", num_pipe, num_cols_auto),
                    ("cat", cat_pipe, cat_cols_auto)
                ],
                remainder="drop"
            )
            
            st.session_state.X_train_p = dummy_preprocessor.fit_transform(Xtr)
            st.session_state.X_test_p = dummy_preprocessor.transform(Xte)
            st.session_state.preprocessor = dummy_preprocessor
            
            try:
                f_names = dummy_preprocessor.get_feature_names_out()
                st.session_state.feature_names = [f.split("__")[-1] for f in f_names]
            except:
                st.session_state.feature_names = num_cols_auto + cat_cols_auto
            
            st.session_state.split_done = True
            st.session_state.preprocess_done = True
            st.success("Locks opened! Model set to safe mode. You can go to 'Model Training' from the left menu.")
            st.rerun()

    with st.expander("Remove Unwanted Columns", expanded=False):
        to_drop = st.multiselect("Columns to remove", df.columns.tolist(), key="drop_eda_cols")
        if st.button("Remove Selected Columns", key="drop_eda_btn") and to_drop:
            df2 = st.session_state.df_eng.drop(columns=to_drop)
            st.session_state.df_eng = df2
            st.success(f"{len(to_drop)} column(s) removed successfully.")
            st.rerun()

    df = st.session_state.df_eng.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

    tab_split, tab_scale, tab_encode = st.tabs(["Train/Test Split", "Scaling", "Encoding"])

    with tab_split:
        c1, c2, c3, c4 = st.columns(4)
        with c1: target_col = st.selectbox("Target Variable (Target)", df.columns.tolist(), index=len(df.columns)-1, key="pp_target")
        with c2: test_size = st.slider("Test Ratio", 0.1, 0.4, 0.2, 0.05, key="pp_test")
        with c3: rand_state = st.number_input("Random State", 0, 999, 42, key="pp_rs")
        with c4: stratify = st.checkbox("Stratify (Preserve Target Distribution)", value=True, key="pp_strat")

        if st.button("Split the Data", key="pp_split"):
            try:
                st.session_state.target_col = target_col
                X = df.drop(columns=[target_col])
                y = df[target_col]
                y = LabelEncoder().fit_transform(y)
                
                strat_y = y if stratify else None
                Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=test_size, random_state=int(rand_state), stratify=strat_y)
                st.session_state.X_train, st.session_state.X_test = Xtr, Xte
                st.session_state.y_train, st.session_state.y_test = ytr, yte
                st.session_state.split_done = True
                st.session_state.preprocess_done = False
                st.success(f"Train: {Xtr.shape[0]:,} rows | Test: {Xte.shape[0]:,} rows | Feature: {Xtr.shape[1]}")
                
                fig = go.Figure(go.Pie(labels=["Train Set", "Test Set"], values=[len(Xtr), len(Xte)], hole=0.55,
                                        marker_colors=["#3b82f6", "#e74c3c"], marker_line=dict(color="#050505", width=2)))
                fig.update_layout(title="Train/Test Distribution", height=280)
                render_chart(fig)
            except Exception as e:
                st.error(str(e))

    with tab_scale:
        if not st.session_state.split_done:
            st.info("Perform Train/Test Split first to proceed.")
        else:
            Xtr = st.session_state.X_train
            num_tr = Xtr.select_dtypes(include=np.number).columns.tolist()
            c1, c2 = st.columns([1, 2])
            with c1:
                scaler_choice = st.selectbox("Scaling Algorithm", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="scaler_ch")
                scale_cols = st.multiselect("Columns to Scale", num_tr, default=num_tr, key="scale_cols")
            with c2:
                if scale_cols:
                    scalers = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
                    sc = scalers[scaler_choice]()
                    scaled_sample = sc.fit_transform(Xtr[scale_cols].copy())
                    sc_df = pd.DataFrame(scaled_sample, columns=scale_cols).describe().T[["mean", "std", "min", "max"]].round(3)
                    st.markdown("<p style='color:#a1a1aa;'>Statistics After Scaling</p>", unsafe_allow_html=True)
                    st.dataframe(sc_df, width="stretch")

    with tab_encode:
        if not st.session_state.split_done:
            st.info("Perform Train/Test Split first to proceed.")
        else:
            Xtr = st.session_state.X_train.copy()
            cat_tr = Xtr.select_dtypes(exclude=np.number).columns.tolist()
            num_tr = Xtr.select_dtypes(include=np.number).columns.tolist()

            st.markdown("<p style='color:#a1a1aa;'>Assign categorical columns as Ordinal (ranked) or Nominal (unranked).</p>", unsafe_allow_html=True)

            c1, c2 = st.columns(2)
            with c1:
                scaler_ch = st.selectbox("Numeric Variable Scaler", ["StandardScaler", "MinMaxScaler", "RobustScaler"], key="enc_scaler")
                num_cols_sel = st.multiselect("Numeric Columns to Process", num_tr, default=num_tr, key="enc_num_cols")
                nom_cols = st.multiselect("Nominal Columns (One-Hot / Label Encoder)", cat_tr,
                                           default=[c for c in cat_tr], key="enc_nom_cols")
            with c2:
                ord_cols = st.multiselect("Ordinal Columns (Ranked Encoding)", [c for c in cat_tr if c not in nom_cols],
                                           key="enc_ord_cols")
                nom_cols = [c for c in nom_cols if c not in ord_cols]
                nominal_ch = st.selectbox("Nominal Encoding Type", ["OneHotEncoder", "OrdinalEncoder (Label)"], key="enc_nom_type")

            ordinal_orders = {}
            if ord_cols:
                st.markdown("<p style='color:#a1a1aa;'>Ordinal Ranking Levels (separate from smallest to largest with commas)</p>", unsafe_allow_html=True)
                for oc in ord_cols:
                    uniq_vals = [str(v) for v in Xtr[oc].dropna().unique().tolist()]
                    default_order = ", ".join(sorted(uniq_vals))
                    order_str = st.text_input(f"'{oc}' Levels", value=default_order, key=f"order_{oc}")
                    ordinal_orders[oc] = [v.strip() for v in order_str.split(",") if v.strip()]

            if st.button("Build and Apply Pipeline", key="enc_apply"):
                try:
                    scalers = {"StandardScaler": StandardScaler, "MinMaxScaler": MinMaxScaler, "RobustScaler": RobustScaler}
                    sc = scalers[scaler_ch]()
                    transformers = []
                    if num_cols_sel:
                        transformers.append(("num", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", sc)]), num_cols_sel))
                    if nom_cols:
                        if nominal_ch == "OneHotEncoder":
                            transformers.append(("nom", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ohe", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False))]), nom_cols))
                        else:
                            transformers.append(("nom_le", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("le", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), nom_cols))
                    if ord_cols:
                        cats_list = [ordinal_orders[c] for c in ord_cols]
                        transformers.append(("ord", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("oe", OrdinalEncoder(categories=cats_list, handle_unknown="use_encoded_value", unknown_value=-1))]), ord_cols))

                    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
                    Xte = st.session_state.X_test
                    Xtr_p = preprocessor.fit_transform(st.session_state.X_train)
                    Xte_p = preprocessor.transform(Xte)

                    try:
                        feat_names = preprocessor.get_feature_names_out()
                        feat_names = [f.split("__")[-1] for f in feat_names]
                    except Exception:
                        feat_names = [f"f{i}" for i in range(Xtr_p.shape[1])]

                    st.session_state.preprocessor = preprocessor
                    st.session_state.X_train_p = Xtr_p
                    st.session_state.X_test_p = Xte_p
                    st.session_state.feature_names = feat_names
                    st.session_state.preprocess_done = True
                    st.session_state.training_done = False
                    st.session_state.model_results = {}
                    st.session_state.best_model = None
                    st.session_state.last_eval = None
                    st.session_state.numeric_cols_pp, st.session_state.nominal_cols_pp, st.session_state.ordinal_cols_pp = num_cols_sel, nom_cols, ord_cols
                    st.session_state.ordinal_orders = ordinal_orders

                    st.success(f"Preprocessing Successful. Processed Train Matrix Size: {Xtr_p.shape}")
                    st.markdown(f"<div style='color:#71717a;font-size:.8rem'>Resulting Features (First 20): {feat_names[:20]}</div>", unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Pipeline Error: {e}")

    if st.session_state.preprocess_done and st.session_state.X_train_p is not None:
        st.markdown("---")
        with st.expander("Final Processed Dataset Preview", expanded=False):
            feat_names_pv = st.session_state.feature_names
            Xtr_p_pv = st.session_state.X_train_p
            ytr_pv = st.session_state.y_train
            preview_df = pd.DataFrame(Xtr_p_pv, columns=feat_names_pv)
            tgt_name = st.session_state.target_col if st.session_state.target_col else "Target"
            preview_df[tgt_name] = np.asarray(ytr_pv)
            c1, c2 = st.columns(2)
            with c1: st.markdown(kpi("Processed Rows", f"{preview_df.shape[0]:,}", color="#3b82f6"), unsafe_allow_html=True)
            with c2: st.markdown(kpi("Processed Columns", f"{preview_df.shape[1]}", color="#8b5cf6"), unsafe_allow_html=True)
            n_rows_pv = st.slider("Number of rows to display", 5, 50, 10, key="pp_preview_n")
            st.dataframe(preview_df.head(n_rows_pv), width="stretch")
            buf_pv = io.BytesIO()
            preview_df.to_csv(buf_pv, index=False)
            st.download_button("Download Processed Dataset (.csv)", buf_pv.getvalue(), file_name="processed_train_dataset.csv", mime="text/csv", key="pp_preview_download")

# ══════════════════════════════════════════════════════
#  MODULE 5 — FEATURE SELECTION
# ══════════════════════════════════════════════════════
elif menu == "Feature Selection":
    if not st.session_state.preprocess_done:
        st.markdown(empty_state("Missing processing step", "Apply the Preprocessing pipeline in Module 4 first"), unsafe_allow_html=True)
        st.stop()

    st.markdown(section_head("Feature Selection", "MRMR and Mutual Information Statistical Filtering Methods"), unsafe_allow_html=True)

    Xtr_p = st.session_state.X_train_p
    ytr = st.session_state.y_train
    if isinstance(ytr, pd.Series): ytr = np.asarray(ytr)

    feat_names = st.session_state.feature_names
    df_train_p = pd.DataFrame(Xtr_p, columns=feat_names)

    tab_mrmr, tab_mi = st.tabs(["MRMR (Min-Redundancy Max-Relevance)", "Mutual Information"])

    with tab_mrmr:
        if not MRMR_OK:
            st.warning("mrmr-selection package not found. Install it via `pip install mrmr-selection` in the terminal.")
        else:
            c1, c2 = st.columns([1, 2])
            with c1:
                k_mrmr = st.slider("Optimal Number of Features to Select (K)", 3, len(feat_names), min(15, len(feat_names)), key="mrmr_k")
                if st.button("Run MRMR Algorithm", key="mrmr_btn"):
                    with st.spinner("Calculating MRMR matrix, please wait..."):
                        try:
                            selected_features = mrmr_classif(X=df_train_p, y=ytr, K=k_mrmr)
                            st.session_state.selected_features = selected_features
                            st.success(f"MRMR Completed. {k_mrmr} feature(s) selected.")
                        except Exception as e:
                            st.error(f"Error during MRMR: {e}")
            with c2:
                if st.session_state.selected_features:
                    st.markdown("<p style='color:#a1a1aa;'>Best Features Selected by MRMR</p>", unsafe_allow_html=True)
                    sel_df = pd.DataFrame({"Feature": st.session_state.selected_features, "Rank": range(1, len(st.session_state.selected_features)+1)})
                    st.dataframe(sel_df, width="stretch")

    with tab_mi:
        c1, c2 = st.columns([1, 2])
        with c1:
            k_mi = st.slider("Top-K for Mutual Information", 3, len(feat_names), min(15, len(feat_names)), key="mi_k")
            if st.button("Calculate MI Values", key="mi_btn"):
                with st.spinner("Extracting information gain values..."):
                    try:
                        scores = mutual_info_classif(df_train_p, ytr, random_state=42)
                        mi_df = pd.DataFrame({"Feature": feat_names, "MI Score": scores}).sort_values("MI Score", ascending=False)
                        top_k = mi_df.head(k_mi)
                        st.session_state.selected_features = top_k["Feature"].tolist()
                        
                        fig = px.bar(top_k.sort_values("MI Score"), x="MI Score", y="Feature", orientation="h",
                                     color="MI Score", color_continuous_scale="Viridis", title=f"Top {k_mi} Features (Mutual Information)")
                        render_chart(fig)
                        st.success("Analysis completed and features saved to the system.")
                    except Exception as e:
                        st.error(str(e))

    if st.session_state.selected_features:
        st.markdown("---")
        st.markdown(f"<div style='color:#a1a1aa;font-size:.85rem'>Input Features for Current Model: <b style='color:#ffffff'>{len(st.session_state.selected_features)}</b> feature(s) determined.</div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════
#  MODULE 6 — MODEL EGITIMI (Dynamic Model Selection)
# ══════════════════════════════════════════════════════
elif menu == "Model Training":
    st.markdown(section_head(
        "Dynamic Model Training",
        "Select one model → configure its parameters → train → evaluate dynamically"
    ), unsafe_allow_html=True)

    if not st.session_state.preprocess_done:
        st.markdown(empty_state("Preprocessing not completed", "Process the dataset in Module 4 first"), unsafe_allow_html=True)
        st.stop()

    if "last_eval" not in st.session_state:
        st.session_state.last_eval = None

    Xtr_p = st.session_state.X_train_p
    Xte_p = st.session_state.X_test_p

    if st.session_state.selected_features:
        sel_indices = [st.session_state.feature_names.index(f) for f in st.session_state.selected_features if f in st.session_state.feature_names]
        if sel_indices:
            Xtr_p = Xtr_p[:, sel_indices]
            Xte_p = Xte_p[:, sel_indices]
            current_features = [st.session_state.feature_names[i] for i in sel_indices]
        else:
            current_features = st.session_state.feature_names
    else:
        current_features = st.session_state.feature_names

    ytr = st.session_state.y_train
    yte = st.session_state.y_test
    if isinstance(ytr, pd.Series): ytr = np.asarray(ytr)
    if isinstance(yte, pd.Series): yte = np.asarray(yte)

    n_classes = len(np.unique(ytr))
    is_binary = n_classes <= 2

    st.markdown("""
    <div style="background:#121212;border:1px solid #27272a;border-radius:8px;padding:14px 18px;margin-bottom:18px">
      <div style="color:#a1a1aa;font-size:.8rem;margin-bottom:6px">Workflow</div>
      <div style="color:#d4d4d8;font-size:.9rem">
        <b style="color:#3b82f6">1</b> Dataset ready &nbsp;→&nbsp;
        <b style="color:#3b82f6">2</b> Train/Test already split in Preprocessing &nbsp;→&nbsp;
        <b style="color:#3b82f6">3</b> Select model &nbsp;→&nbsp;
        <b style="color:#3b82f6">4</b> Set parameters &nbsp;→&nbsp;
        <b style="color:#3b82f6">5</b> Train &nbsp;→&nbsp;
        <b style="color:#3b82f6">6</b> Dynamic evaluation
      </div>
    </div>
    """, unsafe_allow_html=True)

    c_info1, c_info2, c_info3 = st.columns(3)
    with c_info1:
        st.markdown(kpi("Train Samples", f"{Xtr_p.shape[0]:,}", color="#3b82f6"), unsafe_allow_html=True)
    with c_info2:
        st.markdown(kpi("Test Samples", f"{Xte_p.shape[0]:,}", color="#e74c3c"), unsafe_allow_html=True)
    with c_info3:
        st.markdown(kpi("Features", f"{Xtr_p.shape[1]}", color="#8b5cf6"), unsafe_allow_html=True)

    tab_cfg, tab_train = st.tabs(["Model Selection & Parameters", "Train & Live Evaluation"])

    # ─── TAB: CONFIG ───
    with tab_cfg:
        st.subheader("Step 3 — Model Selection")
        model_opts = available_model_options()
        labels = [label for _, label in model_opts]
        keys = [key for key, _ in model_opts]
        selected_label = st.selectbox(
            "Choose the machine learning model to create and evaluate",
            labels,
            index=0,
            key="dyn_model_select",
            help="Only the selected model will be instantiated and trained."
        )
        model_key = keys[labels.index(selected_label)]
        st.session_state.active_model_key = model_key

        st.markdown("---")
        st.subheader(f"Step 4 — {selected_label} Parameters")
        exact_params = render_exact_params(model_key)

        st.markdown("---")
        with st.expander("Advanced: Optional Hyperparameter Optimization (search ranges)", expanded=False):
            st.caption("Leave disabled to train with the exact parameters above. Enable only if you want search-based optimization.")
            use_hpo = st.checkbox("Enable hyperparameter optimization", value=False, key="dyn_use_hpo")
            up = {}
            if use_hpo:
                c1, c2, c3 = st.columns(3)
                with c1:
                    hpo_method = st.selectbox(
                        "Optimization Algorithm",
                        ["RandomizedSearchCV", "GridSearchCV", "Optuna"],
                        key="dyn_hpo_method"
                    )
                    n_iter = st.slider("Number of Trials (Random/Optuna)", 1, 100, 15, key="dyn_n_iter")
                with c2:
                    cv_folds = st.selectbox("Cross-Validation Folds", [3, 5, 7, 10], index=0, key="dyn_cv_folds")
                with c3:
                    refit_m = st.selectbox("Optimization Target Metric", ["F1", "AUC", "Accuracy"], index=0, key="dyn_refit_m")

                st.markdown("##### Search Bounds (used only when HPO is enabled)")
                if model_key == "LR":
                    c1_lr, c2_lr = st.columns(2)
                    up["LR_C"] = c1_lr.slider("C (Log10 range)", -5.0, 5.0, (-3.0, 3.0), step=0.5, key="hpo_lr_c")
                    up["LR_pen"] = c2_lr.multiselect("Penalty", ["l2", "l1", "elasticnet", "None"], default=["l2"], key="hpo_lr_pen")
                    up["LR_sol"] = c1_lr.multiselect("Solver", ["lbfgs", "saga", "liblinear"], default=["lbfgs", "saga"], key="hpo_lr_sol")
                elif model_key == "RF":
                    c1_rf, c2_rf = st.columns(2)
                    up["RF_N"] = c1_rf.slider("n_estimators range", 50, 1000, (100, 400), step=50, key="hpo_rf_n")
                    up["RF_D"] = c2_rf.slider("max_depth range", 2, 50, (5, 25), key="hpo_rf_d")
                    up["RF_MSS"] = c1_rf.slider("min_samples_split range", 2, 30, (2, 10), key="hpo_rf_mss")
                    up["RF_MF"] = c2_rf.multiselect("max_features", ["sqrt", "log2", "None"], default=["sqrt"], key="hpo_rf_mf")
                elif model_key == "ET":
                    c1_et, c2_et = st.columns(2)
                    up["ET_N"] = c1_et.slider("n_estimators range", 50, 1000, (100, 400), step=50, key="hpo_et_n")
                    up["ET_D"] = c2_et.slider("max_depth range", 2, 50, (5, 25), key="hpo_et_d")
                    up["ET_MSS"] = c1_et.slider("min_samples_split range", 2, 30, (2, 10), key="hpo_et_mss")
                    up["ET_MF"] = c2_et.multiselect("max_features", ["sqrt", "log2", "None"], default=["sqrt"], key="hpo_et_mf")
                elif model_key == "XGB":
                    c1_xb, c2_xb = st.columns(2)
                    up["XGB_N"] = c1_xb.slider("n_estimators range", 50, 1000, (100, 500), step=50, key="hpo_xgb_n")
                    up["XGB_D"] = c2_xb.slider("max_depth range", 2, 20, (3, 10), key="hpo_xgb_d")
                    up["XGB_LR"] = c1_xb.slider("learning_rate range", 0.001, 0.5, (0.01, 0.2), step=0.001, key="hpo_xgb_lr")
                    up["XGB_SUB"] = c2_xb.slider("subsample range", 0.3, 1.0, (0.6, 1.0), step=0.05, key="hpo_xgb_sub")
                elif model_key == "LGBM":
                    c1_lg, c2_lg = st.columns(2)
                    up["LGBM_N"] = c1_lg.slider("n_estimators range", 50, 1000, (100, 500), step=50, key="hpo_lgbm_n")
                    up["LGBM_L"] = c2_lg.slider("num_leaves range", 10, 256, (20, 96), step=2, key="hpo_lgbm_l")
                    up["LGBM_LR"] = c1_lg.slider("learning_rate range", 0.001, 0.5, (0.01, 0.2), step=0.001, key="hpo_lgbm_lr")
                    up["LGBM_D"] = c2_lg.slider("max_depth range", -1, 30, (-1, 12), key="hpo_lgbm_d")
                elif model_key == "CB":
                    c1_cb, c2_cb = st.columns(2)
                    up["CB_I"] = c1_cb.slider("iterations range", 50, 1000, (100, 500), step=50, key="hpo_cb_i")
                    up["CB_D"] = c2_cb.slider("depth range", 2, 12, (4, 8), key="hpo_cb_d")
                    up["CB_LR"] = c1_cb.slider("learning_rate range", 0.001, 0.5, (0.01, 0.2), step=0.001, key="hpo_cb_lr")
                    up["CB_L2"] = c2_cb.slider("l2_leaf_reg range", 1.0, 30.0, (1.0, 10.0), step=0.5, key="hpo_cb_l2")
                elif model_key == "SVM":
                    c1_sv, c2_sv = st.columns(2)
                    up["SVM_C"] = c1_sv.slider("C (Log10 range)", -4.0, 4.0, (-2.0, 2.0), step=0.5, key="hpo_svm_c")
                    up["SVM_K"] = c2_sv.multiselect("kernel", ["rbf", "poly", "linear", "sigmoid"], default=["rbf", "linear"], key="hpo_svm_k")
            else:
                hpo_method = "Exact Parameters"
                n_iter = 1
                cv_folds = 3
                refit_m = "F1"

        st.session_state.dyn_exact_params = exact_params
        st.session_state.dyn_hpo_cfg = {
            "use_hpo": bool(use_hpo),
            "method": hpo_method,
            "n_iter": int(n_iter),
            "cv_folds": int(cv_folds),
            "refit_m": refit_m,
            "up": up,
        }

    # ─── TAB: TRAIN & EVALUATE ───
    with tab_train:
        model_key = st.session_state.get("active_model_key", "LR")
        selected_label = MODEL_DISPLAY.get(model_key, model_key)
        exact_params = st.session_state.get("dyn_exact_params", {})
        hpo_cfg = st.session_state.get("dyn_hpo_cfg", {"use_hpo": False, "method": "Exact Parameters", "n_iter": 1, "cv_folds": 3, "refit_m": "F1", "up": {}})
        use_hpo = bool(hpo_cfg.get("use_hpo", False))
        hpo_method = hpo_cfg.get("method", "Exact Parameters")
        n_iter = int(hpo_cfg.get("n_iter", 15))
        cv_folds = int(hpo_cfg.get("cv_folds", 3))
        refit_m = hpo_cfg.get("refit_m", "F1")
        up = hpo_cfg.get("up", {})

        st.subheader("Step 5 — Train Model")
        st.info(
            f"Selected model: **{selected_label}**  |  Mode: **{'HPO (' + hpo_method + ')' if use_hpo else 'Exact user parameters'}**  |  "
            f"Train={Xtr_p.shape[0]:,} / Test={Xte_p.shape[0]:,} (no leakage — test used only after fit)"
        )

        with st.expander("Parameters that will be applied", expanded=True):
            st.json(exact_params if not use_hpo else {"mode": "HPO", "algorithm": hpo_method, "bounds": {k: str(v) for k, v in up.items()}})

        if st.button(f"Train {selected_label}", key="train_selected_model", type="primary", width="stretch"):
            if use_hpo and hpo_method == "Optuna" and not OPTUNA_OK:
                st.error("Optuna is not installed. Run: pip install optuna")
                st.stop()

            scoring_map = {"AUC": "roc_auc", "F1": "f1", "Accuracy": "accuracy"} if is_binary else \
                          {"AUC": "roc_auc_ovr_weighted", "F1": "f1_weighted", "Accuracy": "accuracy"}
            opt_metric = scoring_map[refit_m]
            mode_tag = hpo_method if use_hpo else "ExactParams"
            combo_key = f"{model_key}_{mode_tag}_{refit_m}_{cv_folds}F"
            prog = st.progress(0, text=f"Creating {selected_label}...")
            live_status = st.empty()
            t0 = time.time()

            try:
                best_est = None
                best_params_found = exact_params
                best_score = np.nan

                if not use_hpo:
                    prog.progress(0.2, text=f"Instantiating {selected_label} with user parameters...")
                    best_est = build_classifier(model_key, exact_params)
                    prog.progress(0.5, text=f"Fitting {selected_label} on the training set...")
                    best_est.fit(Xtr_p, ytr)
                    best_params_found = exact_params
                    # CV score for reference only (still fitted on full train for final model)
                    try:
                        n_splits = min(int(cv_folds), max(2, int(np.min(np.bincount(ytr.astype(int))))))
                        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
                        best_score = float(cross_val_score(build_classifier(model_key, exact_params), Xtr_p, ytr, cv=skf, scoring=opt_metric, n_jobs=1).mean())
                    except Exception:
                        best_score = np.nan
                    prog.progress(0.75, text="Evaluating on held-out test set...")

                else:
                    skf = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
                    base_model = build_classifier(model_key, exact_params)

                    if hpo_method == "Optuna":
                        import optuna
                        optuna.logging.set_verbosity(optuna.logging.WARNING)

                        def objective(trial):
                            params = {}
                            if model_key == "LR":
                                params["C"] = trial.suggest_float("C", 10**up["LR_C"][0], 10**up["LR_C"][1], log=True)
                                sol = trial.suggest_categorical("solver", up["LR_sol"])
                                params["solver"] = sol
                                valid_pen = []
                                for p in up["LR_pen"]:
                                    mapped = None if p == "None" else p
                                    if sol == "lbfgs" and mapped not in ["l2", None]: continue
                                    if sol == "liblinear" and mapped not in ["l1", "l2"]: continue
                                    if sol == "saga" and mapped not in ["l1", "l2", "elasticnet", None]: continue
                                    valid_pen.append(p)
                                if not valid_pen: valid_pen = ["l2"]
                                chosen = trial.suggest_categorical("penalty", valid_pen)
                                params["penalty"] = None if chosen == "None" else chosen
                            elif model_key == "RF":
                                params["n_estimators"] = trial.suggest_int("n_estimators", up["RF_N"][0], up["RF_N"][1])
                                params["max_depth"] = trial.suggest_int("max_depth", up["RF_D"][0], up["RF_D"][1])
                                params["min_samples_split"] = trial.suggest_int("min_samples_split", up["RF_MSS"][0], up["RF_MSS"][1])
                                mf = trial.suggest_categorical("max_features", up["RF_MF"])
                                params["max_features"] = None if mf == "None" else mf
                            elif model_key == "ET":
                                params["n_estimators"] = trial.suggest_int("n_estimators", up["ET_N"][0], up["ET_N"][1])
                                params["max_depth"] = trial.suggest_int("max_depth", up["ET_D"][0], up["ET_D"][1])
                                params["min_samples_split"] = trial.suggest_int("min_samples_split", up["ET_MSS"][0], up["ET_MSS"][1])
                                mf = trial.suggest_categorical("max_features", up["ET_MF"])
                                params["max_features"] = None if mf == "None" else mf
                            elif model_key == "XGB":
                                params = {
                                    "n_estimators": trial.suggest_int("n_estimators", up["XGB_N"][0], up["XGB_N"][1]),
                                    "max_depth": trial.suggest_int("max_depth", up["XGB_D"][0], up["XGB_D"][1]),
                                    "learning_rate": trial.suggest_float("learning_rate", up["XGB_LR"][0], up["XGB_LR"][1], log=True),
                                    "subsample": trial.suggest_float("subsample", up["XGB_SUB"][0], up["XGB_SUB"][1]),
                                }
                            elif model_key == "LGBM":
                                params = {
                                    "n_estimators": trial.suggest_int("n_estimators", up["LGBM_N"][0], up["LGBM_N"][1]),
                                    "num_leaves": trial.suggest_int("num_leaves", up["LGBM_L"][0], up["LGBM_L"][1]),
                                    "learning_rate": trial.suggest_float("learning_rate", up["LGBM_LR"][0], up["LGBM_LR"][1], log=True),
                                    "max_depth": trial.suggest_int("max_depth", up["LGBM_D"][0], up["LGBM_D"][1]),
                                }
                            elif model_key == "CB":
                                params = {
                                    "iterations": trial.suggest_int("iterations", up["CB_I"][0], up["CB_I"][1]),
                                    "depth": trial.suggest_int("depth", up["CB_D"][0], up["CB_D"][1]),
                                    "learning_rate": trial.suggest_float("learning_rate", up["CB_LR"][0], up["CB_LR"][1], log=True),
                                    "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", up["CB_L2"][0], up["CB_L2"][1]),
                                }
                            elif model_key == "SVM":
                                params = {
                                    "C": trial.suggest_float("C", 10**up["SVM_C"][0], 10**up["SVM_C"][1], log=True),
                                    "kernel": trial.suggest_categorical("kernel", up["SVM_K"]),
                                }

                            clone_model = build_classifier(model_key, {**exact_params, **params})
                            return float(cross_val_score(clone_model, Xtr_p, ytr, cv=skf, scoring=opt_metric, n_jobs=1).mean())

                        def live_callback(study, trial):
                            live_status.markdown(
                                f"<p style='color:#3b82f6;font-size:0.9rem;'>Optuna Trial {trial.number + 1}/{n_iter} — best so far: <b>{study.best_value:.4f}</b></p>",
                                unsafe_allow_html=True
                            )

                        study = optuna.create_study(direction="maximize")
                        study.optimize(objective, n_trials=int(n_iter), callbacks=[live_callback], n_jobs=1)
                        best_params_found = {**exact_params, **study.best_params}
                        best_est = build_classifier(model_key, best_params_found)
                        best_est.fit(Xtr_p, ytr)
                        best_score = float(study.best_value)

                    else:
                        pg = {}
                        if model_key == "LR":
                            lr_pg = []
                            for sol in up.get("LR_sol", ["lbfgs"]):
                                valid_pen = []
                                for p in up.get("LR_pen", ["l2"]):
                                    mapped = None if p == "None" else p
                                    if sol == "lbfgs" and mapped not in ["l2", None]: continue
                                    if sol == "liblinear" and mapped not in ["l1", "l2"]: continue
                                    if sol == "saga" and mapped not in ["l1", "l2", "elasticnet", None]: continue
                                    valid_pen.append(mapped)
                                if valid_pen:
                                    c_space = np.logspace(up["LR_C"][0], up["LR_C"][1], 3) if hpo_method == "GridSearchCV" else safe_loguniform(10**up["LR_C"][0], 10**up["LR_C"][1])
                                    lr_pg.append({"solver": [sol], "penalty": valid_pen, "C": c_space})
                            pg = lr_pg or [{"solver": ["lbfgs"], "penalty": ["l2"], "C": [1.0]}]
                        elif model_key == "RF":
                            mapped_mf = [None if x == "None" else x for x in up.get("RF_MF", ["sqrt"])]
                            pg = {
                                "n_estimators": np.linspace(up["RF_N"][0], up["RF_N"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["RF_N"][0], up["RF_N"][1]),
                                "max_depth": np.linspace(up["RF_D"][0], up["RF_D"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["RF_D"][0], up["RF_D"][1]),
                                "min_samples_split": np.linspace(up["RF_MSS"][0], up["RF_MSS"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["RF_MSS"][0], up["RF_MSS"][1]),
                                "max_features": mapped_mf,
                            }
                        elif model_key == "ET":
                            mapped_mf = [None if x == "None" else x for x in up.get("ET_MF", ["sqrt"])]
                            pg = {
                                "n_estimators": np.linspace(up["ET_N"][0], up["ET_N"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["ET_N"][0], up["ET_N"][1]),
                                "max_depth": np.linspace(up["ET_D"][0], up["ET_D"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["ET_D"][0], up["ET_D"][1]),
                                "min_samples_split": np.linspace(up["ET_MSS"][0], up["ET_MSS"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["ET_MSS"][0], up["ET_MSS"][1]),
                                "max_features": mapped_mf,
                            }
                        elif model_key == "XGB":
                            pg = {
                                "n_estimators": np.linspace(up["XGB_N"][0], up["XGB_N"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["XGB_N"][0], up["XGB_N"][1]),
                                "max_depth": np.linspace(up["XGB_D"][0], up["XGB_D"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["XGB_D"][0], up["XGB_D"][1]),
                                "learning_rate": np.logspace(np.log10(up["XGB_LR"][0]), np.log10(up["XGB_LR"][1]), 3) if hpo_method == "GridSearchCV" else safe_loguniform(up["XGB_LR"][0], up["XGB_LR"][1]),
                            }
                        elif model_key == "LGBM":
                            pg = {
                                "n_estimators": np.linspace(up["LGBM_N"][0], up["LGBM_N"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["LGBM_N"][0], up["LGBM_N"][1]),
                                "num_leaves": np.linspace(up["LGBM_L"][0], up["LGBM_L"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["LGBM_L"][0], up["LGBM_L"][1]),
                                "learning_rate": np.logspace(np.log10(up["LGBM_LR"][0]), np.log10(up["LGBM_LR"][1]), 3) if hpo_method == "GridSearchCV" else safe_loguniform(up["LGBM_LR"][0], up["LGBM_LR"][1]),
                            }
                        elif model_key == "CB":
                            pg = {
                                "iterations": np.linspace(up["CB_I"][0], up["CB_I"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["CB_I"][0], up["CB_I"][1]),
                                "depth": np.linspace(up["CB_D"][0], up["CB_D"][1], 3, dtype=int) if hpo_method == "GridSearchCV" else safe_randint(up["CB_D"][0], up["CB_D"][1]),
                                "learning_rate": np.logspace(np.log10(up["CB_LR"][0]), np.log10(up["CB_LR"][1]), 3) if hpo_method == "GridSearchCV" else safe_loguniform(up["CB_LR"][0], up["CB_LR"][1]),
                            }
                        elif model_key == "SVM":
                            pg = {
                                "C": np.logspace(up["SVM_C"][0], up["SVM_C"][1], 3) if hpo_method == "GridSearchCV" else safe_loguniform(10**up["SVM_C"][0], 10**up["SVM_C"][1]),
                                "kernel": up.get("SVM_K", ["rbf"]),
                            }

                        if hpo_method == "GridSearchCV":
                            search = GridSearchCV(base_model, pg, cv=skf, scoring=scoring_map, refit=refit_m, n_jobs=1, verbose=0)
                        else:
                            search = RandomizedSearchCV(base_model, pg, n_iter=int(n_iter), cv=skf, scoring=scoring_map, refit=refit_m, random_state=42, n_jobs=1, verbose=0)
                        search.fit(Xtr_p, ytr)
                        best_est = search.best_estimator_
                        best_params_found = search.best_params_
                        best_score = float(search.best_score_)

                # Dynamic evaluation on the held-out test set
                metrics, cm, y_pred, y_proba, report = evaluate_trained_model(best_est, Xte_p, yte)
                elapsed = time.time() - t0

                results = st.session_state.model_results if isinstance(st.session_state.model_results, dict) else {}
                results.setdefault(model_key, {})
                results[model_key][combo_key] = {
                    "combo_key": combo_key,
                    "refit": mode_tag,
                    "kfold": cv_folds,
                    "best_params": best_params_found,
                    "cv_best_score": best_score,
                    "test_auc": metrics.get("ROC-AUC", np.nan),
                    "test_f1": metrics["F1 Score"],
                    "test_acc": metrics["Accuracy"],
                    "test_prec": metrics["Precision"],
                    "test_rec": metrics["Recall"],
                    "estimator": best_est,
                    "elapsed": elapsed,
                    "metrics": metrics,
                    "cm": cm,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "report": report,
                    "model_label": selected_label,
                }
                st.session_state.model_results = results
                st.session_state.training_done = True
                st.session_state.best_model = best_est
                st.session_state.best_model_name = f"{selected_label} ({combo_key})"
                st.session_state.last_eval = {
                    "model_key": model_key,
                    "model_label": selected_label,
                    "combo_key": combo_key,
                    "params": best_params_found,
                    "metrics": metrics,
                    "cm": cm,
                    "y_pred": y_pred,
                    "y_proba": y_proba,
                    "report": report,
                    "elapsed": elapsed,
                    "mode": mode_tag,
                }

                prog.progress(1.0, text="Training and evaluation complete.")
                live_status.empty()
                st.success(
                    f"Trained **{selected_label}** in {elapsed:.2f}s — "
                    f"Accuracy={metrics['Accuracy']:.4f}, F1={metrics['F1 Score']:.4f}"
                    + (f", ROC-AUC={metrics['ROC-AUC']:.4f}" if "ROC-AUC" in metrics else "")
                )
            except Exception as ex:
                live_status.empty()
                err_str = str(ex) or repr(ex)
                st.error(f"Training failed for {selected_label}: {err_str}")

        # ─── Step 6: Dynamic evaluation display ───
        st.markdown("---")
        st.subheader("Step 6 — Dynamic Evaluation")

        last = st.session_state.get("last_eval")
        if not last:
            st.markdown(empty_state(
                "No model trained yet in this session",
                "Configure a model in the first tab, then click Train above."
            ), unsafe_allow_html=True)
        else:
            st.markdown(
                f"""
                <div style="background:#18181b;border:1px solid #3b82f6;border-radius:8px;padding:14px 18px;margin-bottom:14px">
                  <span style="color:#a1a1aa">Results generated from:</span>
                  <b style="color:#ffffff;margin-left:8px">{last['model_label']}</b>
                  <span style="color:#71717a;margin-left:10px">({last['mode']})</span>
                  <span style="color:#71717a;margin-left:10px">{last['elapsed']:.2f}s</span>
                </div>
                """,
                unsafe_allow_html=True,
            )

            metrics = last["metrics"]
            mcols = st.columns(min(6, len(metrics)))
            for col, (name, val) in zip(mcols, metrics.items()):
                with col:
                    st.markdown(kpi(name, f"{val:.4f}", color="#3b82f6"), unsafe_allow_html=True)

            c_left, c_right = st.columns(2)
            with c_left:
                cm = np.asarray(last["cm"])
                cls_names = [f"Class {c}" for c in range(cm.shape[0])]
                fig_cm = px.imshow(
                    cm, text_auto=True, color_continuous_scale="Viridis",
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=cls_names, y=cls_names,
                    title=f"Confusion Matrix — {last['model_label']}"
                )
                fig_cm.update_layout(title_font_color="#ffffff")
                render_chart(fig_cm)

            with c_right:
                y_proba = last.get("y_proba")
                if is_binary and y_proba is not None and "ROC-AUC" in metrics:
                    fpr, tpr, _ = roc_curve(yte, y_proba)
                    fig_roc = go.Figure()
                    fig_roc.add_trace(go.Scatter(
                        x=fpr, y=tpr, mode="lines", name=f"AUC={metrics['ROC-AUC']:.3f}",
                        line=dict(color="#3b82f6", width=2.5),
                        fill="tozeroy", fillcolor="rgba(59,130,246,0.15)"
                    ))
                    fig_roc.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#71717a"))
                    fig_roc.update_layout(
                        title=dict(text=f"ROC Curve — {last['model_label']}", font=dict(color="#ffffff")),
                        xaxis_title="False Positive Rate", yaxis_title="True Positive Rate"
                    )
                    render_chart(fig_roc)
                else:
                    fig_bar = go.Figure(go.Bar(
                        x=list(metrics.keys()), y=list(metrics.values()),
                        marker_color="#3b82f6",
                        text=[f"{v:.3f}" for v in metrics.values()], textposition="outside"
                    ))
                    fig_bar.update_layout(
                        title=dict(text=f"Metrics — {last['model_label']}", font=dict(color="#ffffff")),
                        yaxis=dict(range=[0, 1.15])
                    )
                    render_chart(fig_bar)

            st.markdown("##### Classification Report")
            report_df = pd.DataFrame(last["report"]).transpose().round(4)
            st.dataframe(report_df, width="stretch")

            with st.expander("Applied model parameters"):
                st.json(last["params"])

        # Comparison table across independently trained models
        if st.session_state.training_done and st.session_state.model_results:
            st.markdown("---")
            st.subheader("Trained Models Comparison")
            rows = []
            for model_k, refit_dict in st.session_state.model_results.items():
                for combo_k, info in refit_dict.items():
                    if "error" in info:
                        rows.append({
                            "Model": MODEL_DISPLAY.get(model_k, model_k), "ComboKey": info["combo_key"],
                            "Mode": info.get("refit", "?"), "K-Fold": f"{info.get('kfold', '?')}F",
                            "Test F1": np.nan, "Test AUC": np.nan, "Test Acc": np.nan,
                            "Precision": np.nan, "Recall": np.nan, "Duration (s)": np.nan,
                            "Error Info": info.get("error", "Error"),
                        })
                    else:
                        rows.append({
                            "Model": info.get("model_label", MODEL_DISPLAY.get(model_k, model_k)),
                            "ComboKey": info["combo_key"], "Mode": info["refit"], "K-Fold": f"{info['kfold']}F",
                            "Test F1": info["test_f1"], "Test AUC": info["test_auc"], "Test Acc": info["test_acc"],
                            "Precision": info["test_prec"], "Recall": info["test_rec"], "Duration (s)": info["elapsed"],
                            "Error Info": "",
                        })
            res_df = pd.DataFrame(rows)
            display_df = res_df.drop(columns=["ComboKey"])
            for col in ["Test F1", "Test AUC", "Test Acc", "Precision", "Recall", "Duration (s)"]:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
            st.dataframe(display_df.style.format(precision=4, na_rep="ERROR"), width="stretch")

            best_rows = [r for r in rows if not pd.isna(r.get("Test F1", np.nan))]
            if best_rows:
                best_row = max(best_rows, key=lambda x: float(x["Test F1"]))
                st.markdown(f"""
                <div style="background:#18181b;border:1px solid #3f3f46;border-radius:6px;padding:14px 18px;margin-top:12px">
                  <span style="color:#a1a1aa;font-weight:600">Best among trained models: </span>
                  <span style="color:#ffffff">{best_row['Model']}</span>
                  <span style="color:#71717a"> | {best_row['Mode']} | {best_row['K-Fold']}</span>
                  <span style="color:#ffffff;margin-left:16px;font-weight:600">F1 {float(best_row['Test F1']):.4f}</span>
                  <span style="color:#d4d4d8;margin-left:10px;font-weight:600">AUC {float(best_row['Test AUC']):.4f}</span>
                </div>""", unsafe_allow_html=True)

                bm_key = None
                for mk, rd in st.session_state.model_results.items():
                    if best_row["ComboKey"] in rd and "estimator" in rd[best_row["ComboKey"]]:
                        bm_key = mk
                        break
                if bm_key is not None:
                    st.session_state.best_model_name = best_row["ComboKey"]
                    st.session_state.best_model = st.session_state.model_results[bm_key][best_row["ComboKey"]]["estimator"]

# ══════════════════════════════════════════════════════
#  MODULE 7 — DEGERLENDIRME & SHAP
# ══════════════════════════════════════════════════════
elif menu == "Evaluation & SHAP":
    if not st.session_state.training_done:
        st.markdown(empty_state("No results to display", "You can train models via Module 6."), unsafe_allow_html=True)
        st.stop()

    st.markdown(section_head("Model Interpretability (Explainability)", "ROC, Confusion Matrix, Threshold Optimization, Feature Importance and SHAP Analyses"), unsafe_allow_html=True)

    results = st.session_state.model_results
    
    # ÇÖZÜM: Eğitim verisi (Xtr_p) çalışma uzayına dahil ediliyor
    Xtr_p = st.session_state.X_train_p 
    Xte_p = st.session_state.X_test_p
    yte = st.session_state.y_test
    
    if isinstance(yte, pd.Series): yte = yte.values
    n_classes = len(np.unique(yte))
    
    # ÇÖZÜM: Eğer Feature Selection yapıldıysa, hem eğitim hem test setindeki boyutlar eşitleniyor
    if st.session_state.selected_features:
        sel_indices = [st.session_state.feature_names.index(f) for f in st.session_state.selected_features if f in st.session_state.feature_names]
        if sel_indices:
            Xtr_p = Xtr_p[:, sel_indices] # SHAP background verisi boyutlandırıldı
            Xte_p = Xte_p[:, sel_indices]
            current_features = [st.session_state.feature_names[i] for i in sel_indices]
        else:
            current_features = st.session_state.feature_names
    else:
        current_features = st.session_state.feature_names

    estimator_map = {}
    for mk, rd in results.items():
        for rk, info in rd.items():
            if "estimator" in info:
                label = info.get("model_label", MODEL_DISPLAY.get(mk, mk))
                display_name = f"{label} ({info.get('refit', 'Unknown')}, {info.get('kfold', '?')}F)"
                estimator_map[display_name] = info["estimator"]

    if not estimator_map:
        st.warning("No successful estimator found.")
        st.stop()

    tab_roc, tab_pr, tab_cm, tab_thr, tab_shap, tab_fi = st.tabs(
        ["ROC Curve", "PR Curve", "Confusion Matrix", "Threshold Analysis", "SHAP Values", "Feature Importance"])

    with tab_roc:
        if n_classes > 2:
            st.info("The ROC curve is currently only supported for Binary classification problems.")
        else:
            fig = go.Figure()
            fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(dash="dash", color="#71717a", width=2))
            for i, (nm, est) in enumerate(estimator_map.items()):
                try:
                    if hasattr(est, "predict_proba"):
                        yp = est.predict_proba(Xte_p)[:, 1]
                        fpr, tpr, _ = roc_curve(yte, yp)
                        auc_v = roc_auc_score(yte, yp)
                        fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=f"{nm} (AUC={auc_v:.3f})",
                                                  line=dict(color=PALETTE[i % len(PALETTE)], width=2.5),
                                                  hovertemplate="FPR: %{x:.3f}<br>TPR: %{y:.3f}<extra></extra>"))
                except Exception:
                    pass
            fig.update_layout(title=dict(text="ROC Curve", font=dict(color="#ffffff", size=16)), 
                              xaxis_title="False Positive Rate", yaxis_title="True Positive Rate")
            render_chart(fig)

    with tab_pr:
        if n_classes > 2:
            st.info("The Precision-Recall curve is currently only supported for Binary classification problems.")
        else:
            fig = go.Figure()
            for i, (nm, est) in enumerate(estimator_map.items()):
                try:
                    if hasattr(est, "predict_proba"):
                        yp = est.predict_proba(Xte_p)[:, 1]
                        prec, rec, _ = precision_recall_curve(yte, yp)
                        fig.add_trace(go.Scatter(x=rec, y=prec, mode="lines", name=f"{nm}", line=dict(color=PALETTE[i % len(PALETTE)], width=2.5)))
                except:
                    pass
            fig.update_layout(title=dict(text="Precision-Recall Curve", font=dict(color="#ffffff", size=16)), xaxis_title="Recall", yaxis_title="Precision")
            render_chart(fig)

    with tab_cm:
        sel_model = st.selectbox("Model to Inspect (Confusion Matrix)", list(estimator_map.keys()), key="cm_sel")
        est = estimator_map[sel_model]
        try:
            yp = est.predict(Xte_p)
            cm = confusion_matrix(yte, yp)
            cls_names = [str(c) for c in np.unique(yte)]
            if len(cls_names) == 2: cls_names = ["Class 0", "Class 1"]
            else: cls_names = [f"Class {c}" for c in cls_names]
            
            fig = px.imshow(cm, text_auto=True, color_continuous_scale="Viridis", 
                             labels=dict(x="Model Prediction (Predicted)", y="Actual Value (Actual)", color="Count"),
                             x=cls_names, y=cls_names)
            
            fig.update_layout(title=dict(text=f"Confusion Matrix - {sel_model}", font=dict(color="#ffffff", size=16)),
                              paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
            fig.update_traces(textfont_size=22, textfont_color="#ffffff", hovertemplate="Actual: %{y}<br>Predicted: %{x}<br>Count: %{z}<extra></extra>")
            
            st.plotly_chart(fig, width="stretch")
        except Exception as e:
            st.error(str(e))

    with tab_thr:
        if n_classes > 2:
            st.info("Threshold Optimization is currently only supported for Binary classification problems.")
        else:
            sel_thr = st.selectbox("Model for Threshold Analysis", list(estimator_map.keys()), key="thr_sel")
            est = estimator_map[sel_thr]
            try:
                if hasattr(est, "predict_proba"):
                    yproba = est.predict_proba(Xte_p)[:, 1]
                    prec, rec, thr = precision_recall_curve(yte, yproba)
                    f1_arr = (2 * prec[:-1] * rec[:-1]) / (prec[:-1] + rec[:-1] + 1e-9)
                    opt_idx = np.argmax(f1_arr)
                    opt_thr = thr[opt_idx]
                    
                    fig = go.Figure()
                    fig.add_trace(go.Scatter(x=thr, y=f1_arr, mode="lines", fill='tozeroy', fillcolor='rgba(59,130,246,0.15)', 
                                             line=dict(color="#3b82f6", width=3), name="F1 Score"))
                    
                    fig.add_vline(x=0.5, line_dash="dash", line_color="#71717a", annotation_text="0.50 (Default)", annotation_position="top left", annotation_font_color="#71717a")
                    fig.add_vline(x=opt_thr, line_dash="solid", line_color="#2ecc71", line_width=2, annotation_text=f"Optimal Threshold: {opt_thr:.3f}", annotation_font_color="#2ecc71", annotation_position="top right")
                    
                    fig.update_layout(title=dict(text="Probability Threshold Optimization (F1 Maximization)", font=dict(color="#ffffff", size=16)), 
                                      xaxis_title="Decision Threshold (Threshold)", yaxis_title="F1 Score")
                    render_chart(fig)
                    
                    yp_opt = (yproba >= opt_thr).astype(int)
                    c1, c2, c3 = st.columns(3)
                    with c1: st.markdown(kpi("Optimal Threshold", f"{opt_thr:.3f}", color="#2ecc71"), unsafe_allow_html=True)
                    with c2: st.markdown(kpi("Max F1 Score", f"{f1_score(yte,yp_opt):.3f}", color="#3b82f6"), unsafe_allow_html=True)
                    with c3: st.markdown(kpi("Accuracy", f"{accuracy_score(yte,yp_opt):.3f}", color="#f59e0b"), unsafe_allow_html=True)
                else:
                    st.info("This model cannot produce probabilities (no predict_proba support).")
            except Exception as e:
                st.error(str(e))

    with tab_shap:
        if not SHAP_OK:
            st.warning("SHAP package not installed. Please run `pip install shap`.")
        else:
            sel_shap = st.selectbox("Model for SHAP Explanation", list(estimator_map.keys()), key="shap_sel")
            max_disp = st.slider("Number of Features to Display", 5, 30, 15, key="shap_nd")
            if st.button("Interpret Algorithm (SHAP)", key="shap_run"):
                with st.spinner("Building the contribution values matrix..."):
                    try:
                        est = estimator_map[sel_shap]
                        Xte_df = pd.DataFrame(Xte_p, columns=current_features)
                        
                        # Artık Xtr_p tanımlı olduğu için Linear ve Kernel Explainer'lar sorunsuz çalışır
                        if isinstance(est, LogisticRegression):
                            explainer = shap.LinearExplainer(est, Xtr_p)
                        elif isinstance(est, SVC):
                            explainer = shap.KernelExplainer(est.predict_proba, shap.kmeans(Xtr_p, 10))
                        else:
                            explainer = shap.TreeExplainer(est)
                            
                        shap_vals = explainer(Xte_df)
                        
                        if isinstance(shap_vals, list):
                            sv = shap_vals
                        elif len(shap_vals.shape) > 2:
                            sv = shap_vals[:, :, 1]
                        else:
                            sv = shap_vals
                            
                        with plt.style.context('dark_background'):
                            fig_shap = plt.figure(figsize=(9, 6)) 
                            fig_shap.patch.set_facecolor('#050505') 
                            ax = fig_shap.gca()
                            ax.set_facecolor('#050505')
                            
                            ax.spines['top'].set_visible(False)
                            ax.spines['right'].set_visible(False)
                            ax.spines['bottom'].set_color('#3f3f46')
                            ax.spines['left'].set_color('#3f3f46')
                            
                            ax.tick_params(colors='#d4d4d8', labelsize=11)
                            ax.xaxis.label.set_color('#a1a1aa')
                            ax.xaxis.label.set_fontsize(12)
                            
                            shap.summary_plot(sv, Xte_df, max_display=max_disp, show=False, plot_type="dot", plot_size=(9, 6))
                            
                            cb = plt.gcf().axes[-1] if len(plt.gcf().axes) > 1 else None
                            if cb:
                                cb.tick_params(colors='#a1a1aa')
                                cb.set_ylabel('Feature Value', color='#d4d4d8', fontsize=12)

                            plt.title(f"Overall Feature Impact (SHAP) - {sel_shap}", color="#ffffff", fontsize=15, pad=20)
                            plt.tight_layout()
                            st.pyplot(fig_shap, width="stretch")
                            plt.close()
                    except Exception as e:
                        st.error(f"Chart generation error: {e}")

    with tab_fi:
        sel_fi = st.selectbox("Model to Inspect (Feature Importance)", list(estimator_map.keys()), key="fi_sel")
        est = estimator_map[sel_fi]
        try:
            if hasattr(est, "feature_importances_"):
                importances = est.feature_importances_
            elif hasattr(est, "coef_"):
                importances = np.abs(est.coef_)
                if len(importances.shape) > 1 and importances.shape[0] > 1:
                    importances = np.mean(importances, axis=0)
                else:
                    importances = importances[0]
            else:
                importances = None
                
            if importances is not None:
                fi_df = pd.DataFrame({"Feature": current_features, "Importance": importances[:len(current_features)]})
                fi_df = fi_df.sort_values("Importance", ascending=False).head(20)
                fig = px.bar(fi_df, x="Importance", y="Feature", orientation="h", color="Importance", color_continuous_scale="Viridis", title=f"Feature Importance - {sel_fi}")
                render_chart(fig)
            else:
                st.info("Feature Importance is not supported for this model.")
        except Exception as e:
            st.error(str(e))

# ══════════════════════════════════════════════════════
#  MODULE 8 — TAHMIN & EXPORT
# ══════════════════════════════════════════════════════
elif menu == "Prediction & Export":
    st.markdown(section_head("System Integration & Prediction", "Export the trained model or test it on new data"), unsafe_allow_html=True)

    tab_pred, tab_exp = st.tabs(["Production Prediction", "Serialize Model (.pkl)"])

    with tab_pred:
        if st.session_state.best_model is None:
            st.markdown(empty_state("No model available for prediction", "You can train a model via Module 6."), unsafe_allow_html=True)
        else:
            st.info(f"Active Model: **{st.session_state.best_model_name}**")
            pred_file = st.file_uploader("Upload Raw CSV or Excel for Testing", type=["csv", "xls", "xlsx"], key="pred_upload")
            if pred_file:
                try:
                    if pred_file.name.endswith('.csv'):
                        df_new = pd.read_csv(pred_file)
                    else:
                        df_new = pd.read_excel(pred_file)
                        
                    preprocessor = st.session_state.preprocessor
                    tc = st.session_state.target_col
                    
                    if tc and tc in df_new.columns:
                        df_new = df_new.drop(columns=[tc])

                    X_new = preprocessor.transform(df_new)
                    
                    if st.session_state.selected_features:
                        sel_indices = [st.session_state.feature_names.index(f) for f in st.session_state.selected_features if f in st.session_state.feature_names]
                        X_new = X_new[:, sel_indices]
                        
                    model = st.session_state.best_model
                    preds = model.predict(X_new)
                    
                    df_out = df_new.copy()
                    df_out["System Prediction"] = preds
                    
                    if hasattr(model, "predict_proba"):
                        probas = model.predict_proba(X_new)
                        if probas.shape[1] == 2:
                            df_out["Probability_Score (Class 1)"] = probas[:, 1].round(4)
                        else:
                            df_out["Prediction_Confidence (Max Prob)"] = np.max(probas, axis=1).round(4)

                    c1, c2, c3 = st.columns(3)
                    with c1: st.markdown(kpi("Processed Records", f"{len(df_out):,}", color="#3b82f6"), unsafe_allow_html=True)
                    with c2: st.markdown(kpi("Prediction Distribution", f"{len(np.unique(preds))} Classes", color="#e74c3c"), unsafe_allow_html=True)
                    with c3: 
                        if hasattr(model, "predict_proba"):
                            st.markdown(kpi("Average Probability", f"{np.mean(np.max(probas, axis=1)):.3f}", color="#8b5cf6"), unsafe_allow_html=True)

                    st.dataframe(df_out.head(50), width="stretch")
                    buf = io.BytesIO()
                    df_out.to_csv(buf, index=False)
                    st.download_button("Download Predictions (.csv)", buf.getvalue(), file_name="ml_predictions.csv", mime="text/csv")
                except Exception as e:
                    st.error(f"System Error: {e}")

    with tab_exp:
        if st.session_state.best_model is None and st.session_state.preprocessor is None:
            st.markdown(empty_state("Not ready for export"), unsafe_allow_html=True)
        else:
            c1, c2 = st.columns(2)
            with c1:
                st.markdown("<h3>Scikit-Learn Pipeline Export</h3>", unsafe_allow_html=True)
                st.markdown("<p style='color:#a1a1aa;'>Preprocessing steps and the algorithm are combined into a single serialized object.</p>", unsafe_allow_html=True)
                if st.session_state.best_model and st.session_state.preprocessor:
                    try:
                        export_obj = {
                            "preprocessor": st.session_state.preprocessor,
                            "model": st.session_state.best_model,
                            "selected_features": st.session_state.selected_features,
                            "feature_names_out": st.session_state.feature_names
                        }
                        buf = io.BytesIO()
                        joblib.dump(export_obj, buf)
                        buf.seek(0)
                        st.download_button("Export System (.pkl)", buf.read(), file_name="ml_production_pipeline.pkl", mime="application/octet-stream")
                    except Exception as e:
                        st.error(str(e))
            with c2:
                st.markdown("<h3>Analysis Report (.csv)</h3>", unsafe_allow_html=True)
                if st.session_state.model_results:
                    rows = []
                    for mk, rd in st.session_state.model_results.items():
                        for rk, info in rd.items():
                            if "estimator" in info:
                                rows.append({
                                    "Model": info.get("model_label", MODEL_DISPLAY.get(mk, mk)),
                                    "Optimization": info.get("refit", ""),
                                    "K-Fold": info.get("kfold", ""),
                                    "AUC": info.get("test_auc", np.nan),
                                    "F1_Score": info.get("test_f1", np.nan),
                                    "Accuracy": info.get("test_acc", np.nan),
                                    "Precision": info.get("test_prec", np.nan),
                                    "Recall": info.get("test_rec", np.nan),
                                })
                    if rows:
                        res_df = pd.DataFrame(rows).round(4)
                        buf2 = io.BytesIO()
                        res_df.to_csv(buf2, index=False)
                        buf2.seek(0)
                        st.download_button("Download Report", buf2.read(), file_name="ml_model_metrics.csv", mime="text/csv")
                        st.dataframe(res_df.style.background_gradient(cmap="Blues", subset=["F1_Score", "AUC"]), width="stretch")
