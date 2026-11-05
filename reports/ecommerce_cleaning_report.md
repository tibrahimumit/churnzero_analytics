# E-commerce Customer Churn Cleaning Report

## Summary

- Original rows: 50,000
- Final rows: 43,815
- Rows removed because `Age` was missing: 2,545
- Rows removed because `Age` was above 65: 458
- Total rows removed by age filter: 3,003
- Rows removed because `Session_Duration_Avg` was missing: 3,182
- Original columns: 25
- Final columns: 25
- Exact duplicate rows removed: 0
- Missing cells before cleaning: 49,081
- Missing cells after cleaning: 0
- Invalid values converted to missing: 327
- Category formatting values standardized: 0
- Numeric formatting values standardized: 432892 (column-specific precision)
- Missing values imputed: 38,025 using numeric medians and categorical modes
- Target column: `Churned`; values were validated but not encoded or changed when valid
- Identifier columns: None found

## Invalid values corrected

- `Discount_Usage_Rate`: 207
- `Age`: 50
- `Total_Purchases`: 40
- `Cart_Abandonment_Rate`: 30

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

- None

## Data types and roles

- Numeric columns (21): `Age`, `Membership_Years`, `Login_Frequency`, `Session_Duration_Avg`, `Pages_Per_Session`, `Cart_Abandonment_Rate`, `Wishlist_Items`, `Total_Purchases`, `Average_Order_Value`, `Days_Since_Last_Purchase`, `Discount_Usage_Rate`, `Returns_Rate`, `Email_Open_Rate`, `Customer_Service_Calls`, `Product_Reviews_Written`, `Social_Media_Engagement_Score`, `Mobile_App_Usage`, `Payment_Method_Diversity`, `Lifetime_Value`, `Credit_Balance`, `Churned`
- Categorical columns (4): `Gender`, `Country`, `City`, `Signup_Quarter`
- Date columns: None found
- Identifier columns: None found
- Target: `Churned`

No unusual values were removed solely because they were rare, and no outlier was deleted automatically.
