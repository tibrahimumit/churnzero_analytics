import numpy as np
import pandas as pd

from .config import CATEGORICAL_COLUMNS, RAW_DATA_PATH, TARGET


def load_modeling_data():
    data = pd.read_csv(RAW_DATA_PATH, keep_default_na=False, na_values=[""])
    numeric_columns = [column for column in data if column not in CATEGORICAL_COLUMNS]
    data[numeric_columns] = data[numeric_columns].apply(pd.to_numeric, errors="coerce")

    data["Age"] = data["Age"].where(data["Age"].between(13, 65))
    data = data[data["Age"].notna() & data["Session_Duration_Avg"].notna()].copy()
    data.loc[~data["Cart_Abandonment_Rate"].between(0, 100), "Cart_Abandonment_Rate"] = np.nan
    data.loc[~data["Discount_Usage_Rate"].between(0, 100), "Discount_Usage_Rate"] = np.nan
    data.loc[data["Total_Purchases"] < 0, "Total_Purchases"] = np.nan
    data[TARGET] = data[TARGET].astype(int)
    for column in CATEGORICAL_COLUMNS:
        data[column] = data[column].astype("string")
    return data


def split_features_target(data):
    return data.drop(columns=[TARGET]), data[TARGET]
