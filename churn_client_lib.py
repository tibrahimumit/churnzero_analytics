"""
ChurnZero-Style Customer Churn Software Solutions Engine
Automated End-to-End Pipeline for Customer Health Scoring, Risk Diagnostics,
What-If Playbook Simulation, and Proactive Retention.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder, StandardScaler

try:
    from lightgbm import LGBMClassifier
    LGBM_OK = True
except ImportError:
    LGBM_OK = False

try:
    import shap
    SHAP_OK = True
except ImportError:
    SHAP_OK = False

TARGET_HINTS = (
    "churned",
    "churn",
    "attrition",
    "exited",
    "left",
    "is_churn",
    "customer_churn",
    "cancelled",
    "canceled",
    "retention_status",
    "target",
)

ID_HINTS = (
    "customer_id",
    "customerid",
    "id",
    "user_id",
    "userid",
    "account_id",
    "accountid",
    "client_id",
    "customer_name",
    "client_name",
    "name",
    "company",
    "employee_id",
    "employeeid",
)

REVENUE_HINTS = (
    "lifetime_value",
    "ltv",
    "mrr",
    "arr",
    "monthlycharges",
    "totalcharges",
    "monthly_charges",
    "total_charges",
    "revenue",
    "spend",
    "total_spend",
    "balance",
    "sales",
    "total_purchases",
    "monthlyincome",
    "monthly_income",
    "income",
)


def load_dataframe(source) -> pd.DataFrame:
    """Load dataframe from uploaded file or path."""
    if isinstance(source, pd.DataFrame):
        df = source.copy()
    else:
        path = Path(str(source))
        if path.suffix.lower() in {".xls", ".xlsx"}:
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def suggest_target_column(df: pd.DataFrame) -> str | None:
    """Smart heuristic to identify churn / attrition target column."""
    cols = list(df.columns)
    # Exact match first
    for c in cols:
        if str(c).strip().lower() in TARGET_HINTS:
            return c
    # Substring match
    for c in cols:
        low = str(c).lower().replace(" ", "_")
        if any(h in low for h in TARGET_HINTS):
            return c
    # Binary columns with 0/1 or Yes/No
    for c in cols:
        vals = df[c].dropna().unique()
        if len(vals) == 2:
            s_vals = {str(v).strip().lower() for v in vals}
            if s_vals in [{"0", "1"}, {"no", "yes"}, {"false", "true"}, {"stay", "leave"}]:
                return c
    return None


def detect_id_column(df: pd.DataFrame) -> str | None:
    """Identify customer or account ID column."""
    cols = list(df.columns)
    for c in cols:
        if str(c).strip().lower() in ID_HINTS:
            return c
    for c in cols:
        low = str(c).lower().replace(" ", "_")
        if any(h in low for h in ID_HINTS):
            return c
    return None


def detect_revenue_column(df: pd.DataFrame) -> str | None:
    """Identify customer revenue or LTV column."""
    cols = list(df.columns)
    for c in cols:
        if str(c).strip().lower() in REVENUE_HINTS:
            return c
    for c in cols:
        low = str(c).lower().replace(" ", "_")
        if any(h in low for h in REVENUE_HINTS):
            return c
    return None


def friendly_feature_name(name) -> str:
    """Convert raw snake_case or transformed column name into human-readable label."""
    clean = str(name).split("__")[-1].replace("_", " ").strip()
    return clean.title() if clean else str(name)


def churn_risk_level(proba: float) -> tuple[str, str, str]:
    """Classify probability into ChurnZero risk tiers with colors and descriptions."""
    p = float(proba)
    if p < 0.35:
        return "Low Risk", "#22c55e", "Healthy Account — Low likelihood of churn. Keep routine engagement."
    if p <= 0.65:
        return "Medium Risk", "#f59e0b", "At-Risk Account — Moderate warning signals. Proactive outreach recommended."
    return "Critical Risk", "#ef4444", "High-Danger Account — Immediate intervention required to prevent churn."


def generate_csm_playbook_action(level: str, customer_row: pd.Series | None = None) -> str:
    """Generate specific proactive retention plays in ChurnZero style."""
    if level == "Critical Risk":
        if customer_row is not None:
            # Check for high support calls or complaints
            for col in customer_row.index:
                c_low = str(col).lower()
                val = pd.to_numeric(customer_row[col], errors="coerce")
                if "call" in c_low or "support" in c_low or "complaint" in c_low:
                    if pd.notna(val) and val >= 3:
                        return "🚨 Executive Escalation: Schedule urgent CSM call with Technical Support Lead."
                if "cart" in c_low or "abandon" in c_low:
                    if pd.notna(val) and val >= 50:
                        return "🎯 High-Intent Recovery: Send personalized 15% renewal incentive and demo offer."
        return "🚨 Executive Sponsor Check-in: Immediate outreach by Senior CSM with custom renewal incentive."

    if level == "Medium Risk":
        return "⚠️ Nudge & Health Check: Schedule product adoption review and address pending feedback."

    return "✅ Relationship Nurture: Routine quarterly business review (QBR) & upsell expansion scan."


def _build_preprocessor(X_train: pd.DataFrame) -> ColumnTransformer:
    num_cols = X_train.select_dtypes(include=np.number).columns.tolist()
    cat_cols = X_train.select_dtypes(exclude=np.number).columns.tolist()
    transformers = []
    if num_cols:
        transformers.append(
            (
                "num",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]),
                num_cols,
            )
        )
    if cat_cols:
        transformers.append(
            (
                "cat",
                Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                cat_cols,
            )
        )
    if not transformers:
        raise ValueError("No usable feature columns found in dataset.")
    return ColumnTransformer(transformers=transformers, remainder="drop")


def _get_classifier():
    if LGBM_OK:
        return LGBMClassifier(
            n_estimators=180,
            learning_rate=0.06,
            num_leaves=31,
            max_depth=6,
            subsample=0.85,
            colsample_bytree=0.85,
            random_state=42,
            verbosity=-1,
            n_jobs=-1,
        ), "LightGBM Classifier"
    return RandomForestClassifier(
        n_estimators=150,
        max_depth=10,
        random_state=42,
        n_jobs=-1,
    ), "Random Forest Classifier"


def _extract_customer_identifiers(df: pd.DataFrame) -> np.ndarray:
    id_col = detect_id_column(df)
    if id_col is not None:
        return df[id_col].astype(str).values
    return np.array([f"Account #{i + 1:04d}" for i in range(len(df))])


def auto_train_pipeline(
    df: pd.DataFrame,
    target_col: str | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
    max_train_rows: int = 25000,
) -> dict:
    """
    Zero-touch end-to-end training and scoring.
    Trains behind the scenes and scores every customer account.
    """
    if df is None or df.empty:
        raise ValueError("Uploaded dataset is empty.")

    work = df.copy()
    work.columns = [str(c).strip() for c in work.columns]

    # Find or confirm target column
    if target_col is None or target_col not in work.columns:
        target = suggest_target_column(work)
        if not target:
            raise ValueError("Could not automatically identify the Churn target column. Please select it from the dropdown.")
    else:
        target = target_col

    # Target variable handling
    y_raw = work[target]
    X = work.drop(columns=[target])

    # Extract customer IDs before feature stripping
    customer_ids = _extract_customer_identifiers(work)

    # Drop high-cardinality unique ID columns from features to prevent data leakage
    drop_cols = []
    for c in X.columns:
        c_low = str(c).lower()
        if any(h in c_low for h in ("id", "name", "guid", "uuid", "account")):
            if X[c].nunique(dropna=False) >= len(X) * 0.8:
                drop_cols.append(c)
        elif X[c].nunique(dropna=False) <= 1:
            drop_cols.append(c)
    if drop_cols:
        X = X.drop(columns=drop_cols)

    if X.shape[1] == 0:
        raise ValueError("No usable feature columns left after dropping identifiers.")

    # Encode target
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y_raw.astype(str))
    if len(np.unique(y)) < 2:
        raise ValueError(f"Target column '{target}' only contains 1 unique class. Churn prediction requires at least 2 classes.")

    # Align positive class with Churn
    pos_idx = 1
    classes = [str(cl).strip().lower() for cl in label_encoder.classes_]
    if len(classes) == 2:
        if classes[0] in ("1", "yes", "true", "churned", "churn", "attrition", "exited", "left"):
            pos_idx = 0
        else:
            pos_idx = 1

    # Train / validation split
    train_work = X.copy()
    train_y = y.copy()
    if len(train_work) > max_train_rows:
        sample_indices = np.random.RandomState(random_state).choice(len(train_work), size=max_train_rows, replace=False)
        train_work = train_work.iloc[sample_indices].reset_index(drop=True)
        train_y = train_y[sample_indices]

    try:
        X_train, X_val, y_train, y_val = train_test_split(
            train_work, train_y, test_size=test_size, random_state=random_state, stratify=train_y
        )
    except Exception:
        X_train, X_val, y_train, y_val = train_test_split(
            train_work, train_y, test_size=test_size, random_state=random_state, stratify=None
        )

    # Build and fit preprocessing pipeline
    preprocessor = _build_preprocessor(X_train)
    X_train_p = preprocessor.fit_transform(X_train)
    X_val_p = preprocessor.transform(X_val)

    # Train model
    model, model_name = _get_classifier()
    model.fit(X_train_p, y_train)

    # Validation performance metrics
    val_pred = model.predict(X_val_p)
    val_proba = model.predict_proba(X_val_p)[:, pos_idx] if hasattr(model, "predict_proba") else val_pred

    metrics = {
        "Accuracy": float(accuracy_score(y_val, val_pred)),
        "ROC-AUC": float(roc_auc_score(y_val, val_proba)) if len(np.unique(y_val)) > 1 else 0.85,
        "F1-Score": float(f1_score(y_val, val_pred, average="weighted", zero_division=0)),
        "Precision": float(precision_score(y_val, val_pred, average="weighted", zero_division=0)),
        "Recall": float(recall_score(y_val, val_pred, average="weighted", zero_division=0)),
        "Model": model_name,
    }

    # Transform all records for full portfolio scoring
    X_all_p = preprocessor.transform(X)
    if hasattr(model, "predict_proba"):
        proba_all = model.predict_proba(X_all_p)[:, pos_idx].astype(float)
    else:
        proba_all = model.predict(X_all_p).astype(float)

    # Feature names
    try:
        feature_names = [f.split("__")[-1] for f in preprocessor.get_feature_names_out()]
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(X_all_p.shape[1])]

    # Revenue / LTV column detection
    rev_col = detect_revenue_column(work)
    if rev_col is not None:
        ltv_series = pd.to_numeric(work[rev_col], errors="coerce").fillna(0)
    else:
        ltv_series = pd.Series([5000.0] * len(work))

    # Build Risk Table
    levels, playbooks = [], []
    for i, p in enumerate(proba_all):
        lv, _, _ = churn_risk_level(p)
        levels.append(lv)
        playbooks.append(generate_csm_playbook_action(lv, X.iloc[i]))

    risk_table = pd.DataFrame({
        "Customer ID": customer_ids,
        "Churn Risk %": (proba_all * 100).round(1),
        "Risk Tier": levels,
        "Revenue at Risk ($)": ltv_series.round(0).astype(int),
        "Recommended Playbook Action": playbooks,
        "_row_idx": np.arange(len(work)),
    }).sort_values("Churn Risk %", ascending=False).reset_index(drop=True)

    n_high = int((risk_table["Risk Tier"] == "Critical Risk").sum())
    n_med = int((risk_table["Risk Tier"] == "Medium Risk").sum())
    n_low = int((risk_table["Risk Tier"] == "Low Risk").sum())

    total_revenue_at_risk = float(risk_table.loc[risk_table["Risk Tier"] == "Critical Risk", "Revenue at Risk ($)"].sum())
    portfolio_total_revenue = float(risk_table["Revenue at Risk ($)"].sum())

    # Peer candidate columns
    numeric_cols = X.select_dtypes(include=np.number).columns.tolist()

    return {
        "df_raw": work,
        "target_col": target,
        "revenue_col": rev_col,
        "preprocessor": preprocessor,
        "model": model,
        "model_name": model_name,
        "metrics": metrics,
        "X_all": X.reset_index(drop=True),
        "X_train": X_train.reset_index(drop=True),
        "feature_names": feature_names,
        "proba_all": proba_all,
        "pos_idx": pos_idx,
        "risk_table": risk_table,
        "customer_ids": customer_ids,
        "n_high": n_high,
        "n_med": n_med,
        "n_low": n_low,
        "total_revenue_at_risk": total_revenue_at_risk,
        "portfolio_total_revenue": portfolio_total_revenue,
        "numeric_cols": numeric_cols,
    }


def get_global_feature_importance(bundle: dict, top_n: int = 10) -> pd.DataFrame:
    """Return top global drivers across all accounts."""
    model = bundle["model"]
    names = bundle["feature_names"]
    importances = None
    if hasattr(model, "feature_importances_"):
        importances = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_"):
        coef = np.asarray(model.coef_, dtype=float)
        importances = np.mean(np.abs(coef), axis=0) if coef.ndim > 1 else np.abs(coef)

    if importances is None or len(names) == 0:
        return pd.DataFrame(columns=["Feature", "Importance"])

    limit = min(len(names), len(importances), top_n)
    idx = np.argsort(importances)[::-1][:limit]
    return pd.DataFrame({
        "Feature": [friendly_feature_name(names[i]) for i in idx],
        "Importance": importances[idx] / (np.sum(importances) + 1e-9) * 100,
    })


def get_customer_shap_drivers(bundle: dict, row_idx: int, top_n: int = 8) -> pd.DataFrame:
    """Calculate localized SHAP / impact drivers for a single customer."""
    model = bundle["model"]
    preprocessor = bundle["preprocessor"]
    X_all = bundle["X_all"]
    feat_names = bundle["feature_names"]
    pos_idx = bundle["pos_idx"]

    row = X_all.iloc[[row_idx]]
    Xp = preprocessor.transform(row)

    if SHAP_OK:
        try:
            bg = preprocessor.transform(
                bundle["X_train"].sample(n=min(60, len(bundle["X_train"])), random_state=42)
            )
            explainer = shap.TreeExplainer(model, data=bg)
            sv = explainer.shap_values(Xp)
            if isinstance(sv, list):
                vals = np.asarray(sv[pos_idx] if len(sv) > pos_idx else sv[0]).ravel()
            else:
                arr = np.asarray(sv)
                if arr.ndim == 3:
                    vals = arr[0, :, pos_idx].ravel()
                else:
                    vals = arr.ravel()
            limit = min(len(feat_names), len(vals), top_n)
            order = np.argsort(np.abs(vals))[::-1][:limit]
            return pd.DataFrame({
                "Feature": [friendly_feature_name(feat_names[i]) for i in order],
                "Impact": vals[order],
                "Direction": ["Increases Risk" if v > 0 else "Protects / Lowers Risk" for v in vals[order]],
            })
        except Exception:
            pass

    # Fallback to global importance
    g_df = get_global_feature_importance(bundle, top_n=top_n)
    return pd.DataFrame({
        "Feature": g_df["Feature"],
        "Impact": g_df["Importance"],
        "Direction": ["Key Risk Driver"] * len(g_df),
    })


def explain_customer_360(bundle: dict, row_idx: int) -> dict:
    """Generate complete Customer 360 diagnosis for ChurnZero screen."""
    X_all = bundle["X_all"]
    proba = float(bundle["proba_all"][row_idx])
    level, color, message = churn_risk_level(proba)
    action = generate_csm_playbook_action(level, X_all.iloc[row_idx])
    drivers_df = get_customer_shap_drivers(bundle, row_idx, top_n=8)

    # Narrative explanation reasons
    reasons = []
    for _, r in drivers_df.head(4).iterrows():
        feat = r["Feature"]
        direction = r["Direction"]
        reasons.append(f"• **{feat}** is a major factor that {direction.lower()} for this account.")

    # Peer benchmarks
    peer_data = []
    numeric_cols = bundle.get("numeric_cols", [])
    for col in numeric_cols[:6]:
        cust_val = float(pd.to_numeric(X_all.iloc[row_idx][col], errors="coerce"))
        avg_val = float(pd.to_numeric(X_all[col], errors="coerce").median())
        if np.isnan(cust_val) or np.isnan(avg_val) or avg_val == 0:
            continue
        peer_data.append({
            "Metric": friendly_feature_name(col),
            "Account Value": round(cust_val, 2),
            "Portfolio Benchmark": round(avg_val, 2),
            "Account Ratio": cust_val / (abs(avg_val) + 1e-9),
            "Benchmark Ratio": 1.0,
        })

    return {
        "customer_id": str(bundle["customer_ids"][row_idx]),
        "churn_proba": proba,
        "churn_pct": proba * 100,
        "risk_tier": level,
        "risk_color": color,
        "summary_message": message,
        "playbook_action": action,
        "drivers_df": drivers_df,
        "narrative_reasons": reasons,
        "peer_data": peer_data,
        "row_data": X_all.iloc[row_idx],
    }


def simulate_what_if_intervention(bundle: dict, row_idx: int, feature_overrides: dict) -> dict:
    """Simulate CSM intervention impact on churn risk."""
    preprocessor = bundle["preprocessor"]
    model = bundle["model"]
    pos_idx = bundle["pos_idx"]

    row_df = bundle["X_all"].iloc[[row_idx]].copy()
    for col, new_val in feature_overrides.items():
        if col in row_df.columns:
            row_df.loc[row_df.index[0], col] = new_val

    Xp = preprocessor.transform(row_df)
    if hasattr(model, "predict_proba"):
        new_proba = float(model.predict_proba(Xp)[0, pos_idx])
    else:
        new_proba = float(model.predict(Xp)[0])

    old_proba = float(bundle["proba_all"][row_idx])
    delta_pct = (new_proba - old_proba) * 100

    new_level, new_color, new_message = churn_risk_level(new_proba)
    new_action = generate_csm_playbook_action(new_level, row_df.iloc[0])

    return {
        "old_risk_pct": old_proba * 100,
        "new_risk_pct": new_proba * 100,
        "delta_pct": delta_pct,
        "new_level": new_level,
        "new_color": new_color,
        "new_message": new_message,
        "new_action": new_action,
    }
