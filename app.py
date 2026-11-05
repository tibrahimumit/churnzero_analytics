# ══════════════════════════════════════════════════════════════════════════════
#  CHURNZERO — ENTERPRISE CUSTOMER RETENTION & RISK INTELLIGENCE PLATFORM
#  100% Automated | English UI | Zero ML Knowledge Required | Dynamic SaaS Theme
# ══════════════════════════════════════════════════════════════════════════════
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from pathlib import Path
import io

from churn_client_lib import (
    auto_train_pipeline,
    suggest_target_column,
    detect_id_column,
    detect_revenue_column,
    explain_customer_360,
    simulate_what_if_intervention,
    get_global_feature_importance,
    friendly_feature_name,
)

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIGURATION
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="ChurnZero | Customer Churn Intelligence Platform",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CUSTOM SAAS CSS THEME (Obsidian Dark, Glassmorphism, Modern Typography)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #090a0f;
    --surface: #0f1219;
    --card: #151924;
    --card-border: #232938;
    --card-hover: #1c2232;
    --accent: #3b82f6;
    --accent-glow: rgba(59, 130, 246, 0.25);
    --txt-primary: #f8fafc;
    --txt-secondary: #94a3b8;
    --txt-muted: #64748b;
    --danger: #ef4444;
    --warning: #f59e0b;
    --success: #10b981;
}

html, body, .stApp {
    background-color: var(--bg) !important;
    font-family: 'Plus Jakarta Sans', 'Inter', -apple-system, sans-serif !important;
    color: var(--txt-primary) !important;
}

.main .block-container {
    padding: 1.5rem 2.2rem 3rem !important;
    max-width: 1550px !important;
}

/* Sidebar styling */
[data-testid="stSidebar"] {
    background-color: var(--surface) !important;
    border-right: 1px solid var(--card-border) !important;
}

/* Tabs styling */
.stTabs [data-baseweb="tab-list"] {
    background: var(--surface) !important;
    padding: 6px !important;
    border-radius: 12px !important;
    border: 1px solid var(--card-border) !important;
    gap: 8px !important;
    margin-bottom: 1.5rem !important;
}

.stTabs [data-baseweb="tab"] {
    background: transparent !important;
    color: var(--txt-secondary) !important;
    border-radius: 8px !important;
    padding: 10px 22px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    border: none !important;
    transition: all 0.2s ease !important;
}

.stTabs [aria-selected="true"] {
    background: #1e293b !important;
    color: #60a5fa !important;
    box-shadow: 0 4px 12px var(--accent-glow) !important;
    border: 1px solid rgba(59, 130, 246, 0.4) !important;
}

/* Metric KPI card */
.kpi-card {
    background: linear-gradient(145deg, #131722, #0d1017);
    border: 1px solid var(--card-border);
    border-radius: 14px;
    padding: 1.25rem 1.4rem;
    position: relative;
    overflow: hidden;
    box-shadow: 0 4px 20px rgba(0, 0, 0, 0.35);
    transition: transform 0.2s ease, border-color 0.2s ease;
}
.kpi-card:hover {
    transform: translateY(-2px);
    border-color: #3b82f6;
}
.kpi-label {
    font-size: 0.78rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--txt-muted);
}
.kpi-value {
    font-size: 1.9rem;
    font-weight: 800;
    color: var(--txt-primary);
    margin-top: 0.35rem;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.82rem;
    color: var(--txt-secondary);
    margin-top: 0.25rem;
}

/* Callout & Playbook Boxes */
.playbook-box {
    background: linear-gradient(135deg, rgba(59, 130, 246, 0.12), rgba(15, 23, 42, 0.6));
    border: 1px solid rgba(59, 130, 246, 0.4);
    border-radius: 12px;
    padding: 1.2rem 1.4rem;
    margin-top: 1rem;
}
.reason-item {
    background: var(--card);
    border: 1px solid var(--card-border);
    border-radius: 8px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
    color: #e2e8f0;
    font-size: 0.9rem;
}

/* Buttons */
.stButton>button {
    background: #2563eb !important;
    color: #ffffff !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
}
.stButton>button:hover {
    background: #1d4ed8 !important;
    box-shadow: 0 4px 16px rgba(37, 99, 235, 0.45) !important;
    transform: translateY(-1px) !important;
}

/* Tables */
[data-testid="stDataFrame"] {
    background: var(--surface) !important;
    border: 1px solid var(--card-border) !important;
    border-radius: 12px !important;
}

#MainMenu, footer, header {
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# SESSION STATE INITIALIZATION
# ─────────────────────────────────────────────────────────────────────────────
for key, default in {
    "df_raw": None,
    "file_hash": None,
    "file_name": None,
    "target_col": None,
    "rev_col": None,
    "bundle": None,
    "selected_customer_idx": 0,
    "sim_result": None,
    "active_tab": "Overview",
}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def render_kpi(label: str, value: str, sub: str = "", color: str = "#f8fafc"):
    return f"""
    <div class="kpi-card">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value" style="color: {color}">{value}</div>
      <div class="kpi-sub">{sub}</div>
    </div>
    """


def render_gauge(pct: float, color: str, title: str = "Churn Probability"):
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=float(pct),
            number={"suffix": "%", "font": {"size": 40, "color": "#f8fafc", "family": "Plus Jakarta Sans"}},
            title={"text": f"<b>{title}</b>", "font": {"size": 13, "color": "#94a3b8"}},
            gauge={
                "axis": {"range": [0, 100], "tickcolor": "#475569", "tickwidth": 1},
                "bar": {"color": color, "thickness": 0.25},
                "bgcolor": "#131722",
                "borderwidth": 0,
                "steps": [
                    {"range": [0, 35], "color": "rgba(16, 185, 129, 0.15)"},
                    {"range": [35, 65], "color": "rgba(245, 158, 11, 0.15)"},
                    {"range": [65, 100], "color": "rgba(239, 68, 68, 0.15)"},
                ],
            },
        )
    )
    fig.update_layout(
        height=260,
        margin=dict(l=20, r=20, t=35, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Plus Jakarta Sans"),
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR — CONNECT DATA SOURCE
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="padding: 0.5rem 0 1rem;">
      <div style="display:flex;align-items:center;gap:8px;">
        <span style="font-size:1.6rem">🛡️</span>
        <div>
          <div style="font-weight:800;font-size:1.15rem;color:#f8fafc;letter-spacing:-0.01em">ChurnZero</div>
          <div style="font-size:0.75rem;color:#60a5fa;font-weight:600;text-transform:uppercase;letter-spacing:0.06em">Customer Success SaaS</div>
        </div>
      </div>
      <div style="font-size:0.8rem;color:#94a3b8;margin-top:8px">
        Enterprise Churn Prevention & Health Intelligence
      </div>
    </div>
    <hr style="border-color: #232938; margin: 0.5rem 0 1rem;">
    """, unsafe_allow_html=True)

    st.markdown("#### 📂 Customer Data Ingestion")
    uploaded_file = st.file_uploader(
        "Upload Customer CSV or Excel file",
        type=["csv", "xls", "xlsx"],
        help="Upload customer dataset. Target column and IDs are auto-detected.",
    )

    if uploaded_file is not None:
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.file_hash != file_id:
            try:
                if uploaded_file.name.lower().endswith(".csv"):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                df.columns = [str(c).strip() for c in df.columns]
                st.session_state.df_raw = df
                st.session_state.file_hash = file_id
                st.session_state.file_name = uploaded_file.name
                st.session_state.bundle = None
                st.session_state.sim_result = None
                st.session_state.target_col = suggest_target_column(df)
                st.session_state.rev_col = detect_revenue_column(df)
            except Exception as e:
                st.error(f"Error loading file: {e}")

    df_raw = st.session_state.df_raw

    if df_raw is not None:
        st.markdown(f"""
        <div style="background:rgba(16,185,129,0.12);border:1px solid rgba(16,185,129,0.35);padding:10px 14px;border-radius:10px;margin-bottom:14px">
          <div style="color:#6ee7b7;font-size:0.85rem;font-weight:700">✓ Connected: {len(df_raw):,} Accounts</div>
          <div style="color:#94a3b8;font-size:0.75rem">{st.session_state.file_name} ({len(df_raw.columns)} attributes)</div>
        </div>
        """, unsafe_allow_html=True)

        cols = list(df_raw.columns)
        # Select Target Column
        auto_tgt = st.session_state.target_col or suggest_target_column(df_raw)
        tgt_idx = cols.index(auto_tgt) if auto_tgt in cols else (len(cols) - 1)
        selected_target = st.selectbox(
            "Churn Target Column (Label)",
            cols,
            index=tgt_idx,
            help="Select the column indicating if a customer churned (e.g. Churned, Attrition, Exited, Status).",
        )
        st.session_state.target_col = selected_target

        # Select Revenue Column
        auto_rev = st.session_state.rev_col or detect_revenue_column(df_raw)
        rev_options = ["None (Default $5,000 / account)"] + cols
        rev_idx = (cols.index(auto_rev) + 1) if auto_rev in cols else 0
        selected_rev = st.selectbox(
            "Account Value / Revenue Column",
            rev_options,
            index=rev_idx,
            help="Used to compute Revenue at Risk (e.g. Lifetime_Value, MRR, MonthlyCharges).",
        )
        st.session_state.rev_col = None if selected_rev.startswith("None") else selected_rev

        st.markdown("<div style='height: 8px'></div>", unsafe_allow_html=True)
        if st.button("🚀 Run ChurnZero Risk Analysis", type="primary", use_container_width=True):
            with st.spinner("Executing automated intelligence pipeline…"):
                try:
                    bundle = auto_train_pipeline(
                        df_raw,
                        target_col=selected_target,
                        max_train_rows=25000,
                    )
                    st.session_state.bundle = bundle
                    st.session_state.sim_result = None
                    st.session_state.selected_customer_idx = int(bundle["risk_table"].iloc[0]["_row_idx"])
                    st.success("Analysis Complete!")
                    st.rerun()
                except Exception as ex:
                    st.error(f"Pipeline error: {ex}")
    else:
        st.info("Upload customer CSV/Excel above to begin analysis.")

    st.markdown("""
    <hr style="border-color: #232938; margin: 1.5rem 0 1rem;">
    <div style="font-size:0.75rem;color:#64748b;line-height:1.4">
      <b>ChurnZero Intelligence Engine</b><br>
      • Automated Data Preprocessing<br>
      • Self-Tuning Gradient Boosted Risk Scoring<br>
      • Explainable Local Drivers (SHAP)<br>
      • Actionable CSM Retention Playbooks
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN DASHBOARD AREA
# ─────────────────────────────────────────────────────────────────────────────

# Landing State if no data analyzed yet
if st.session_state.bundle is None:
    st.markdown("""
    <div style="padding: 1.5rem 0 2rem;">
      <div style="display:inline-block;background:rgba(59,130,246,0.15);border:1px solid rgba(59,130,246,0.4);border-radius:20px;padding:4px 14px;color:#93c5fd;font-size:0.8rem;font-weight:700;margin-bottom:12px">
        ENTERPRISE CHURN SOFTWARE SOLUTION
      </div>
      <h1 style="font-size:2.4rem;font-weight:800;color:#ffffff;margin:0 0 0.5rem;letter-spacing:-0.02em">
        ChurnZero Customer Retention Intelligence
      </h1>
      <p style="font-size:1.05rem;color:#94a3b8;max-width:820px;line-height:1.5;margin-bottom:2rem">
        Proactively identify at-risk customers, uncover exact churn drivers, simulate intervention playbooks,
        and protect portfolio revenue — with zero technical or machine learning background required.
      </p>
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div class="kpi-card" style="height:100%">
          <div style="font-size:1.8rem;margin-bottom:8px">⚡</div>
          <div style="font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:6px">100% Zero-Touch Automation</div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.5">
            No machine learning setup, no data normalization settings, and no hyperparameter tuning needed.
            Just upload your CSV and get instant enterprise health scores.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div class="kpi-card" style="height:100%">
          <div style="font-size:1.8rem;margin-bottom:8px">🎯</div>
          <div style="font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:6px">Explainable Churn Drivers</div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.5">
            Understand exactly <i>why</i> an individual customer is at risk using transparent attribution and
            peer benchmark comparison charts.
          </div>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div class="kpi-card" style="height:100%">
          <div style="font-size:1.8rem;margin-bottom:8px">🧪</div>
          <div style="font-size:1.1rem;font-weight:700;color:#ffffff;margin-bottom:6px">Intervention Simulator</div>
          <div style="font-size:0.88rem;color:#94a3b8;line-height:1.5">
            Simulate retention plays (e.g. resolve support tickets, offer contract discounts) and observe
            immediate churn probability reduction in real time.
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div style="margin-top:2.5rem;background:#121620;border:1px dashed #283248;border-radius:14px;padding:2.5rem;text-align:center">
      <div style="font-size:2.2rem;margin-bottom:0.75rem">📁</div>
      <div style="font-size:1.2rem;font-weight:700;color:#ffffff;margin-bottom:0.4rem">Ready to analyze your customer data?</div>
      <div style="font-size:0.9rem;color:#94a3b8;margin-bottom:1.2rem">
        Use the sidebar on the left to upload your customer CSV or Excel file.
      </div>
      <div style="font-size:0.8rem;color:#64748b">
        Supports any B2B SaaS, E-Commerce, or Subscription customer dataset.
      </div>
    </div>
    """, unsafe_allow_html=True)

    st.stop()


# ─────────────────────────────────────────────────────────────────────────────
# ACTIVE CHURNZERO DASHBOARD
# ─────────────────────────────────────────────────────────────────────────────
bundle = st.session_state.bundle
risk_table = bundle["risk_table"]
total_customers = len(risk_table)
n_high = bundle["n_high"]
n_med = bundle["n_med"]
n_low = bundle["n_low"]
high_rate = (n_high / max(total_customers, 1)) * 100
rev_at_risk = bundle["total_revenue_at_risk"]
metrics = bundle["metrics"]

# Dashboard Header
st.markdown(f"""
<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:1.2rem;padding-bottom:1rem;border-bottom:1px solid #1e2433">
  <div>
    <h1 style="font-size:1.95rem;font-weight:800;color:#ffffff;margin:0 0 4px;letter-spacing:-0.02em">
      Customer Health & Churn Risk Hub
    </h1>
    <div style="font-size:0.88rem;color:#94a3b8">
      Portfolio overview for <b>{total_customers:,} monitored accounts</b> • Target: <code style="color:#60a5fa">{bundle['target_col']}</code>
    </div>
  </div>
  <div style="display:flex;gap:10px;align-items:center">
    <div style="background:#131824;border:1px solid #273144;border-radius:8px;padding:6px 14px;font-size:0.8rem;color:#94a3b8">
      Model: <b style="color:#38bdf8">{bundle['model_name']}</b> (ROC-AUC: <b style="color:#4ade80">{metrics['ROC-AUC']:.3f}</b>)
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# Main Navigation Tabs
tab_overview, tab_cust360, tab_simulator, tab_playbook = st.tabs([
    "📊 Portfolio Health Overview",
    "👤 Customer 360 & Risk Diagnosis",
    "🧪 What-If Playbook Simulator",
    "📋 Retention Action Queue & Export",
])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1: PORTFOLIO HEALTH OVERVIEW
# ═════════════════════════════════════════════════════════════════════════════
with tab_overview:
    # 4 Executive KPI Cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        st.markdown(render_kpi("Monitored Accounts", f"{total_customers:,}", "Active Customer Base"), unsafe_allow_html=True)
    with k2:
        st.markdown(render_kpi("High Churn Danger Rate", f"{high_rate:.1f}%", f"{n_high:,} Critical Accounts", "#ef4444"), unsafe_allow_html=True)
    with k3:
        st.markdown(render_kpi("Accounts at Risk", f"{n_high + n_med:,}", f"Critical: {n_high:,} | Moderate: {n_med:,}", "#f59e0b"), unsafe_allow_html=True)
    with k4:
        st.markdown(render_kpi("Est. Revenue at Risk", f"${rev_at_risk:,.0f}", "High-risk account value", "#3b82f6"), unsafe_allow_html=True)

    st.markdown("<div style='height: 1.25rem'></div>", unsafe_allow_html=True)

    # Charts Row
    chart_c1, chart_c2 = st.columns([1, 1.25])

    with chart_c1:
        st.markdown("### Portfolio Risk Distribution")
        donut_fig = go.Figure(data=[go.Pie(
            labels=["Critical Risk (>65%)", "Medium Risk (35-65%)", "Low Risk / Healthy (<35%)"],
            values=[n_high, n_med, n_low],
            hole=0.55,
            marker_colors=["#ef4444", "#f59e0b", "#10b981"],
            textinfo="percent+label",
            textposition="outside",
        )])
        donut_fig.update_layout(
            height=340,
            margin=dict(l=15, r=15, t=10, b=15),
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(color="#cbd5e1", family="Plus Jakarta Sans"),
            showlegend=False,
        )
        st.plotly_chart(donut_fig, use_container_width=True)

    with chart_c2:
        st.markdown("### Top Global Churn Drivers (Enterprise Impact)")
        fi_df = get_global_feature_importance(bundle, top_n=8)
        if len(fi_df) > 0:
            bar_fig = go.Figure(go.Bar(
                x=fi_df["Importance"],
                y=fi_df["Feature"],
                orientation="h",
                marker=dict(
                    color=fi_df["Importance"],
                    colorscale=[[0, "#1e3a8a"], [1, "#3b82f6"]],
                ),
                text=[f"{v:.1f}%" for v in fi_df["Importance"]],
                textposition="outside",
            ))
            bar_fig.update_layout(
                height=340,
                margin=dict(l=10, r=30, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="Relative Influence on Churn (%)", gridcolor="#1e293b"),
            )
            st.plotly_chart(bar_fig, use_container_width=True)
        else:
            st.info("Feature importance not available.")

    st.markdown("<hr style='border-color:#1e2433; margin:1.5rem 0'>", unsafe_allow_html=True)

    # Priority Action Queue (Top 10 High-Risk Customers)
    st.markdown("### 🚨 Critical Retention Queue (Top 10 Accounts)")
    st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1rem'>Accounts with the highest probability of leaving within the next billing cycle.</div>", unsafe_allow_html=True)

    top_queue = risk_table.head(10)[["Customer ID", "Churn Risk %", "Risk Tier", "Revenue at Risk ($)", "Recommended Playbook Action", "_row_idx"]]
    display_queue = top_queue.drop(columns=["_row_idx"])
    st.dataframe(display_queue, use_container_width=True, hide_index=True)

    col_sel, col_btn = st.columns([3, 1])
    with col_sel:
        pick_customer = st.selectbox(
            "Select customer from critical queue for immediate deep-dive:",
            top_queue["Customer ID"].tolist(),
            key="quick_dive_select",
        )
    with col_btn:
        st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
        if st.button("Open Customer 360", use_container_width=True):
            chosen_row = top_queue.loc[top_queue["Customer ID"] == pick_customer].iloc[0]
            st.session_state.selected_customer_idx = int(chosen_row["_row_idx"])
            st.info(f"Loaded {pick_customer}. Click the 'Customer 360' tab above.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 2: CUSTOMER 360 & RISK DIAGNOSIS
# ═════════════════════════════════════════════════════════════════════════════
with tab_cust360:
    st.markdown("### Individual Account Health & Churn Diagnostics")

    # Dropdown to pick customer
    customer_options = [
        f"{r['Customer ID']}  •  {r['Churn Risk %']:.1f}% Risk ({r['Risk Tier']})"
        for _, r in risk_table.head(300).iterrows()
    ]
    idx_lookup = {customer_options[i]: int(risk_table.iloc[i]["_row_idx"]) for i in range(len(customer_options))}

    default_pick_idx = 0
    if st.session_state.selected_customer_idx is not None:
        for i, opt in enumerate(customer_options):
            if idx_lookup[opt] == st.session_state.selected_customer_idx:
                default_pick_idx = i
                break

    selected_option = st.selectbox("Search / Select Customer Account:", customer_options, index=default_pick_idx)
    active_row_idx = idx_lookup[selected_option]
    st.session_state.selected_customer_idx = active_row_idx

    # Explain Customer 360
    cust_info = explain_customer_360(bundle, active_row_idx)

    # Top Row: Gauge + Diagnosis & Playbook
    d_col1, d_col2 = st.columns([1, 1.3])

    with d_col1:
        st.plotly_chart(render_gauge(cust_info["churn_pct"], cust_info["risk_color"]), use_container_width=True)
        st.markdown(f"""
        <div style="text-align:center;margin-top:-0.5rem">
          <div style="font-size:1.35rem;font-weight:800;color:{cust_info['risk_color']}">
            {cust_info['risk_tier']} ({cust_info['churn_pct']:.1f}%)
          </div>
          <div style="color:#94a3b8;font-size:0.85rem;margin-top:4px">{cust_info['summary_message']}</div>
        </div>
        """, unsafe_allow_html=True)

    with d_col2:
        st.markdown("#### 🔍 Why is this customer at risk?")
        for reason in cust_info["narrative_reasons"]:
            st.markdown(f"<div class='reason-item'>{reason}</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div class="playbook-box">
          <div style="font-size:0.75rem;font-weight:700;color:#93c5fd;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">
            RECOMMENDED RETENTION PLAYBOOK
          </div>
          <div style="font-size:0.95rem;font-weight:600;color:#ffffff">
            {cust_info['playbook_action']}
          </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<hr style='border-color:#1e2433; margin:1.5rem 0'>", unsafe_allow_html=True)

    # Bottom Row: Local SHAP Feature Drivers & Peer Radar Chart
    r_col1, r_col2 = st.columns([1, 1])

    with r_col1:
        st.markdown("#### Local Risk Drivers (SHAP Attribution)")
        st.caption("Attributes pushing the customer towards churn (Red) vs protecting the account (Green).")
        drivers = cust_info["drivers_df"]
        if len(drivers) > 0:
            driver_fig = go.Figure(go.Bar(
                x=drivers["Impact"],
                y=drivers["Feature"],
                orientation="h",
                marker_color=["#ef4444" if v > 0 else "#10b981" for v in drivers["Impact"]],
                text=[f"{v:+.2f}" for v in drivers["Impact"]],
                textposition="outside",
            ))
            driver_fig.update_layout(
                height=350,
                margin=dict(l=10, r=25, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                yaxis=dict(autorange="reversed"),
                xaxis=dict(title="Attribution Impact", gridcolor="#1e293b"),
            )
            st.plotly_chart(driver_fig, use_container_width=True)

    with r_col2:
        st.markdown("#### Peer Cohort Benchmark")
        st.caption("How this account compares against portfolio median metrics.")
        peer_data = cust_info["peer_data"]
        if peer_data:
            categories = [p["Metric"] for p in peer_data]
            radar = go.Figure()
            radar.add_trace(go.Scatterpolar(
                r=[p["Account Ratio"] for p in peer_data],
                theta=categories,
                fill="toself",
                name="This Account",
                line_color="#ef4444",
            ))
            radar.add_trace(go.Scatterpolar(
                r=[p["Benchmark Ratio"] for p in peer_data],
                theta=categories,
                fill="toself",
                name="Portfolio Benchmark",
                line_color="#3b82f6",
            ))
            radar.update_layout(
                height=350,
                polar=dict(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(visible=True, gridcolor="#1e293b"),
                    angularaxis=dict(gridcolor="#1e293b"),
                ),
                paper_bgcolor="rgba(0,0,0,0)",
                font=dict(color="#94a3b8", family="Plus Jakarta Sans"),
                legend=dict(orientation="h", y=-0.15),
                margin=dict(l=30, r=30, t=10, b=30),
            )
            st.plotly_chart(radar, use_container_width=True)
        else:
            st.info("No comparative metrics available for radar chart.")


# ═════════════════════════════════════════════════════════════════════════════
# TAB 3: WHAT-IF PLAYBOOK SIMULATOR
# ═════════════════════════════════════════════════════════════════════════════
with tab_simulator:
    st.markdown("### Interactive Retention Plays Simulator")
    st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1.2rem'>Adjust customer behavioral parameters to calculate the exact impact of proactive retention strategies before execution.</div>", unsafe_allow_html=True)

    sim_options = [
        f"{r['Customer ID']}  •  {r['Churn Risk %']:.1f}% Risk"
        for _, r in risk_table.head(100).iterrows()
    ]
    sim_idx_map = {sim_options[i]: int(risk_table.iloc[i]["_row_idx"]) for i in range(len(sim_options))}

    sim_pick = st.selectbox("Select Account to Simulate:", sim_options, key="sim_account_select")
    sim_row_idx = sim_idx_map[sim_pick]

    customer_row = bundle["X_all"].iloc[sim_row_idx]
    numeric_levers = bundle["numeric_cols"][:6]

    st.markdown("#### 🎛️ Intervention Levers")
    st.caption("Change account values to simulate customer success interventions.")

    slider_cols = st.columns(3)
    overrides = {}
    for i, col in enumerate(numeric_levers):
        with slider_cols[i % 3]:
            raw_val = pd.to_numeric(customer_row[col], errors="coerce")
            current_val = float(raw_val) if pd.notna(raw_val) else 0.0

            series = pd.to_numeric(bundle["X_all"][col], errors="coerce").dropna()
            min_v = float(series.quantile(0.01))
            max_v = float(series.quantile(0.99))
            if min_v >= max_v or not np.isfinite(min_v) or not np.isfinite(max_v):
                min_v, max_v = current_val * 0.5, current_val * 1.5 + 1.0

            step = 1.0 if (max_v - min_v) > 20 else 0.1
            overrides[col] = st.slider(
                friendly_feature_name(col),
                min_value=float(min_v),
                max_value=float(max_v),
                value=float(np.clip(current_val, min_v, max_v)),
                step=float(step),
                key=f"sim_slider_{col}_{sim_row_idx}",
            )

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    if st.button("⚡ Simulate Intervention Impact", type="primary", use_container_width=True):
        st.session_state.sim_result = simulate_what_if_intervention(bundle, sim_row_idx, overrides)

    sim_res = st.session_state.sim_result
    if sim_res is not None:
        st.markdown("<hr style='border-color:#1e2433; margin:1.5rem 0'>", unsafe_allow_html=True)
        st.markdown("#### 📊 Simulation Results")

        sc1, sc2, sc3 = st.columns(3)
        with sc1:
            st.markdown(render_kpi("Initial Risk", f"{sim_res['old_risk_pct']:.1f}%", "Prior to Intervention"), unsafe_allow_html=True)
        with sc2:
            st.markdown(render_kpi("Simulated Risk", f"{sim_res['new_risk_pct']:.1f}%", sim_res['new_level'], sim_res['new_color']), unsafe_allow_html=True)
        with sc3:
            delta = sim_res["delta_pct"]
            d_col = "#10b981" if delta < 0 else ("#ef4444" if delta > 0 else "#94a3b8")
            st.markdown(render_kpi("Net Churn Delta", f"{delta:+.1f}%", "Negative = Risk Drop", d_col), unsafe_allow_html=True)

        st.markdown(f"""
        <div class="playbook-box">
          <div style="font-size:0.75rem;font-weight:700;color:#93c5fd;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:4px">
            UPDATED PLAYBOOK ACTION POST-INTERVENTION
          </div>
          <div style="font-size:0.95rem;font-weight:600;color:#ffffff">
            {sim_res['new_action']}
          </div>
        </div>
        """, unsafe_allow_html=True)


# ═════════════════════════════════════════════════════════════════════════════
# TAB 4: RETENTION ACTION QUEUE & CRM EXPORT
# ═════════════════════════════════════════════════════════════════════════════
with tab_playbook:
    st.markdown("### Enterprise Retention Action Queue & CRM Export")
    st.markdown("<div style='font-size:0.85rem;color:#94a3b8;margin-bottom:1.2rem'>Export enriched risk priorities and tailored CSM actions directly into your CRM or Customer Success workflow.</div>", unsafe_allow_html=True)

    f_col1, f_col2 = st.columns([1, 2])
    with f_col1:
        tier_filter = st.selectbox("Filter by Risk Tier:", ["All Accounts", "Critical Risk (>65%)", "Medium Risk (35-65%)", "Low Risk (<35%)"])

    filtered_table = risk_table.copy()
    if tier_filter.startswith("Critical"):
        filtered_table = filtered_table[filtered_table["Risk Tier"] == "Critical Risk"]
    elif tier_filter.startswith("Medium"):
        filtered_table = filtered_table[filtered_table["Risk Tier"] == "Medium Risk"]
    elif tier_filter.startswith("Low"):
        filtered_table = filtered_table[filtered_table["Risk Tier"] == "Low Risk"]

    export_df = filtered_table[["Customer ID", "Churn Risk %", "Risk Tier", "Revenue at Risk ($)", "Recommended Playbook Action"]].reset_index(drop=True)

    st.markdown(f"**Showing {len(export_df):,} accounts matching criteria:**")
    st.dataframe(export_df, use_container_width=True, hide_index=True)

    # CSV Download Button
    csv_buffer = io.StringIO()
    export_df.to_csv(csv_buffer, index=False)
    csv_bytes = csv_buffer.getvalue().encode("utf-8")

    st.download_button(
        label="📥 Download Retention Action Plan (CSV)",
        data=csv_bytes,
        file_name="churnzero_retention_action_plan.csv",
        mime="text/csv",
        type="primary",
    )
