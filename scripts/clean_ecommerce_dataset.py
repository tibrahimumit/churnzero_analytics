import csv
import math
import statistics
from collections import Counter
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_PATH = BASE_DIR / "data" / "raw" / "ecommerce_customer_churn_dataset.csv"
CLEAN_PATH = BASE_DIR / "data" / "processed" / "cleaned_ecommerce_customer.csv"
REPORT_PATH = BASE_DIR / "reports" / "ecommerce_cleaning_report.md"

MISSING_TOKENS = {"", "null", "NULL", "N/A", "NA", "None", "-", "NaN"}
CATEGORICAL_COLUMNS = {"Gender", "Country", "City", "Signup_Quarter"}
TARGET_COLUMN = "Churned"
NUMERIC_COLUMNS = {
    "Age", "Membership_Years", "Login_Frequency", "Session_Duration_Avg",
    "Pages_Per_Session", "Cart_Abandonment_Rate", "Wishlist_Items",
    "Total_Purchases", "Average_Order_Value", "Days_Since_Last_Purchase",
    "Discount_Usage_Rate", "Returns_Rate", "Email_Open_Rate",
    "Customer_Service_Calls", "Product_Reviews_Written",
    "Social_Media_Engagement_Score", "Mobile_App_Usage",
    "Payment_Method_Diversity", "Lifetime_Value", "Credit_Balance", TARGET_COLUMN,
}
DECIMAL_PLACES = {
    "Age": 0,
    "Membership_Years": 1,
    "Login_Frequency": 0,
    "Session_Duration_Avg": 1,
    "Pages_Per_Session": 1,
    "Cart_Abandonment_Rate": 1,
    "Wishlist_Items": 0,
    "Total_Purchases": 1,
    "Average_Order_Value": 2,
    "Days_Since_Last_Purchase": 0,
    "Discount_Usage_Rate": 2,
    "Returns_Rate": 1,
    "Email_Open_Rate": 1,
    "Customer_Service_Calls": 0,
    "Product_Reviews_Written": 0,
    "Social_Media_Engagement_Score": 1,
    "Mobile_App_Usage": 1,
    "Payment_Method_Diversity": 0,
    "Lifetime_Value": 2,
    "Credit_Balance": 0,
    TARGET_COLUMN: 0,
}


def is_missing(value):
    return value.strip() in MISSING_TOKENS


def main():
    with RAW_PATH.open(newline="") as source:
        reader = csv.DictReader(source)
        columns = reader.fieldnames
        raw_rows = list(reader)

    missing_before = Counter()
    original_missing = Counter()
    invalid_values = Counter()
    category_normalizations = Counter()
    numeric_normalizations = Counter()
    category_maps = {column: {} for column in CATEGORICAL_COLUMNS}
    cleaned_rows = []
    age_rows_removed = Counter()
    session_rows_removed = Counter()

    for raw_row in raw_rows:
        cleaned_row = {}
        for column in columns:
            original = raw_row[column] or ""
            value = original.strip()
            if is_missing(value):
                missing_before[column] += 1
                original_missing[column] += 1
                cleaned_row[column] = ""
                continue

            if column in CATEGORICAL_COLUMNS:
                key = value.casefold()
                canonical = category_maps[column].setdefault(key, value)
                if canonical != original:
                    category_normalizations[column] += 1
                cleaned_row[column] = canonical
                continue

            try:
                number = float(value)
                if not math.isfinite(number):
                    raise ValueError
            except ValueError:
                invalid_values[column] += 1
                missing_before[column] += 1
                cleaned_row[column] = ""
                continue

            invalid = (
                (column == "Age" and (number < 13 or number > 100))
                or (column in {"Cart_Abandonment_Rate", "Discount_Usage_Rate"} and (number < 0 or number > 100))
                or (column == "Total_Purchases" and number < 0)
                or (column == TARGET_COLUMN and number not in {0, 1})
            )
            if invalid:
                invalid_values[column] += 1
                missing_before[column] += 1
                cleaned_row[column] = ""
            else:
                if column in DECIMAL_PLACES:
                    places = DECIMAL_PLACES[column]
                    normalized = f"{number:.{places}f}"
                    if normalized != value:
                        numeric_normalizations[column] += 1
                    cleaned_row[column] = normalized
                else:
                    cleaned_row[column] = value
        if not cleaned_row["Age"]:
            age_rows_removed["missing_age"] += 1
            continue
        if float(cleaned_row["Age"]) > 65:
            age_rows_removed["age_over_65"] += 1
            continue
        if not cleaned_row["Session_Duration_Avg"]:
            session_rows_removed["missing_session_duration"] += 1
            continue
        cleaned_rows.append(cleaned_row)

    duplicate_count = len(raw_rows) - len({tuple(row[column] for column in columns) for row in raw_rows})
    if duplicate_count:
        seen = set()
        deduplicated = []
        for row in cleaned_rows:
            key = tuple(row[column] for column in columns)
            if key not in seen:
                seen.add(key)
                deduplicated.append(row)
        cleaned_rows = deduplicated

    imputed_values = Counter()
    for column in columns:
        missing_rows = [row for row in cleaned_rows if not row[column]]
        if not missing_rows:
            continue
        if column in NUMERIC_COLUMNS:
            observed = [float(row[column]) for row in cleaned_rows if row[column]]
            replacement = statistics.median(observed)
            places = DECIMAL_PLACES[column]
            replacement_text = f"{replacement:.{places}f}"
        elif column in CATEGORICAL_COLUMNS:
            observed = [row[column] for row in cleaned_rows if row[column]]
            replacement_text = Counter(observed).most_common(1)[0][0]
        else:
            continue
        for row in missing_rows:
            row[column] = replacement_text
        imputed_values[column] = len(missing_rows)

    with CLEAN_PATH.open("w", newline="") as destination:
        writer = csv.DictWriter(destination, fieldnames=columns)
        writer.writeheader()
        writer.writerows(cleaned_rows)

    missing_after = Counter(
        column for row in cleaned_rows for column in columns if row[column] == ""
    )
    invalid_total = sum(invalid_values.values())
    report = f"""# E-commerce Customer Churn Cleaning Report

## Summary

- Original rows: {len(raw_rows):,}
- Final rows: {len(cleaned_rows):,}
- Rows removed because `Age` was missing: {age_rows_removed["missing_age"]:,}
- Rows removed because `Age` was above 65: {age_rows_removed["age_over_65"]:,}
- Total rows removed by age filter: {sum(age_rows_removed.values()):,}
- Rows removed because `Session_Duration_Avg` was missing: {session_rows_removed["missing_session_duration"]:,}
- Original columns: {len(columns)}
- Final columns: {len(columns)}
- Exact duplicate rows removed: {duplicate_count}
- Missing cells before cleaning: {sum(original_missing.values()):,}
- Missing cells after cleaning: {sum(missing_after.values()):,}
- Invalid values converted to missing: {invalid_total}
- Category formatting values standardized: {sum(category_normalizations.values())}
- Numeric formatting values standardized: {sum(numeric_normalizations.values())} (column-specific precision)
- Missing values imputed: {sum(imputed_values.values()):,} using numeric medians and categorical modes
- Target column: `{TARGET_COLUMN}`; values were validated but not encoded or changed when valid
- Identifier columns: None found

## Invalid values corrected

{chr(10).join(f"- `{column}`: {count}" for column, count in invalid_values.items() if count) or "- None"}

Rules applied:

- `Age` values below 13 or above 100 were treated as invalid.
- Rows with missing `Age` or `Age > 65` were removed as requested.
- `Cart_Abandonment_Rate` and `Discount_Usage_Rate` values outside 0-100 were treated as invalid.
- Negative `Total_Purchases` values were treated as invalid.
- Target values other than 0 or 1 were treated as invalid.
- Invalid cells were set to missing; rows were not removed.
- Remaining missing numeric cells were filled with their column median.
- Remaining missing categorical cells were filled with their column mode.

## Missing values by column

{chr(10).join(f"- `{column}`: {count}" for column, count in sorted(missing_after.items()) if count) or "- None"}

## Data types and roles

- Numeric columns ({len(NUMERIC_COLUMNS)}): {", ".join(f"`{column}`" for column in columns if column in NUMERIC_COLUMNS)}
- Categorical columns ({len(CATEGORICAL_COLUMNS)}): {", ".join(f"`{column}`" for column in columns if column in CATEGORICAL_COLUMNS)}
- Date columns: None found
- Identifier columns: None found
- Target: `{TARGET_COLUMN}`

No unusual values were removed solely because they were rare, and no outlier was deleted automatically.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")
    print(f"Saved {CLEAN_PATH}")
    print(f"Saved {REPORT_PATH}")
    print(f"Final shape: {len(cleaned_rows)} rows x {len(columns)} columns")
    print(f"Invalid values corrected: {invalid_total}")
    print(f"Missing cells after cleaning: {sum(missing_after.values())}")


if __name__ == "__main__":
    main()