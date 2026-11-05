from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from .config import RANDOM_STATE
from .preprocessing import CatBoostPreprocessor, make_preprocessor


def build_models(numerical_columns, categorical_columns):
    tree_preprocessor = make_preprocessor(numerical_columns, categorical_columns)
    return {
        "Logistic Regression": Pipeline([
            ("preprocessor", make_preprocessor(numerical_columns, categorical_columns, True)),
            ("model", LogisticRegression(C=0.5, max_iter=500, solver="liblinear", random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", RandomForestClassifier(
                n_estimators=350, max_depth=14, min_samples_leaf=5,
                max_features="sqrt", n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", XGBClassifier(
                n_estimators=400, max_depth=4, learning_rate=0.05,
                min_child_weight=5, subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=5.0, objective="binary:logistic",
                eval_metric="logloss", tree_method="hist", n_jobs=-1,
                random_state=RANDOM_STATE,
            )),
        ]),
        "LightGBM": Pipeline([
            ("preprocessor", tree_preprocessor),
            ("model", LGBMClassifier(
                n_estimators=300, num_leaves=15, max_depth=6,
                learning_rate=0.05, min_child_samples=50, reg_alpha=0.1,
                reg_lambda=5.0, objective="binary", verbosity=-1,
                n_jobs=-1, random_state=RANDOM_STATE,
            )),
        ]),
        "CatBoost": Pipeline([
            ("preprocessor", CatBoostPreprocessor(numerical_columns, categorical_columns)),
            ("model", CatBoostClassifier(
                iterations=350, depth=6, learning_rate=0.05,
                l2_leaf_reg=8.0, loss_function="Logloss", eval_metric="AUC",
                cat_features=list(range(len(numerical_columns), len(numerical_columns) + len(categorical_columns))),
                random_seed=RANDOM_STATE, thread_count=-1, verbose=False,
                allow_writing_files=False,
            )),
        ]),
    }
