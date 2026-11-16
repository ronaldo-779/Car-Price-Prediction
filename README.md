# Used Car Price Prediction

## Project structure

- `original_data/`: source CSV files copied from the uploaded ZIP archives.
- `processed_data/combined_used_cars_2000.csv`: standardized 2,000-record dataset.
- `notebooks/Used_Car_Price_Prediction.ipynb`: starter notebook for EDA, preprocessing, model comparison and evaluation.
- `models/`: saved trained models (initially empty).
- `visualizations/`: charts and plots (initially empty).
- `app/`: prediction interface files (initially empty).

## Dataset construction

The combined dataset contains exactly:
- 1,000 records from `UsedCars_Combined.csv` (Used Car Sales source).
- 1,000 records from `Cars24_Used_Cars_Clean_Data.xls` (CSV-formatted Cars24 source).

The two sources are both Indian used-car listing datasets. The original source files are retained unchanged in `original_data/`.

### Standardized columns

`Record_ID`, `Brand`, `Model`, `Year`, `City`, `Kilometers_Driven`, `Fuel_Type`, `Transmission`, `Selling_Price`, `Source`, `Source_Record_ID`.

`Selling_Price` is in INR according to the source listing data. Confirm this against the original source documentation before final academic submission.

### Preparation applied

1. Converted source columns to a shared schema.
2. Required a non-empty brand, model, year, city, fuel type, odometer and selling price.
3. Kept model years from 1990 through 2025.
4. Kept odometer readings from 0 through 600,000 km.
5. Kept positive selling prices.
6. Standardized common fuel/transmission labels where the mapping was unambiguous.
7. Checked exact cross-source duplicates using normalized brand, model, year, city, odometer and price fields; no exact key overlaps were found.
8. Selected 1,000 records from each source using fixed-seed sampling stratified across selling-price deciles.

The Used Car Sales source had 1,725 records initially; 2 failed the stated numeric plausibility filters, leaving 1,723 eligible records. The Cars24 cleaned source had 3,073 eligible records. The 2,000 selected rows are a reproducible sample using random seeds 42 and 2026.

### Important data caveats

- The Used Car Sales source does not provide transmission, so `Transmission` is set to `Not available` for those records. This is missing source information, not an inferred vehicle attribute.
- The Cars24 source has an `EMI` column with missing values, but EMI is not included in the standardized modelling dataset.
- `Source` and `Source_Record_ID` are retained for auditability and must not be used as model input features.
- Exact cross-source matching does not rule out near-duplicate listings with different spelling or formatting.
- Review outliers and category distributions during EDA. Do not automatically delete genuine expensive vehicles.
- Confirm source attribution, data collection details and dataset licenses before publication or academic submission.

## Model plan

The notebook compares:
1. Linear Regression
2. Decision Tree Regressor
3. Random Forest Regressor
4. Support Vector Regression
5. K-Nearest Neighbors Regressor
6. Gradient Boosting Regressor
7. XGBoost Regressor
8. Extra Trees Regressor

Model selection uses cross-validation on the training partition. The held-out test set is reserved for final evaluation. Metrics: MAE, MSE, RMSE and R².

## Environment

Recommended: Google Colab / Python 3.

Install packages if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost joblib gradio
```
