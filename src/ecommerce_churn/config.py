from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "ecommerce_customer_churn_dataset.csv"
TARGET = "Churned"
RANDOM_STATE = 42
TEST_SIZE = 0.20
MODEL_DIR = PROJECT_ROOT / "artifacts" / "models"
RESULT_DIR = PROJECT_ROOT / "artifacts" / "results"
PLOT_DIR = PROJECT_ROOT / "artifacts" / "plots"
CATEGORICAL_COLUMNS = {"Gender", "Country", "City", "Signup_Quarter"}
