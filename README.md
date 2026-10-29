Food Delivery Time Prediction
A reproducible machine-learning project that predicts food-delivery time in minutes from order, rider, route, traffic, and weather information.

The project trains and compares regression models, tunes XGBoost, evaluates the selected model on a held-out test split, and saves a single deployable artifact containing the complete preprocessing pipeline and model.

Features
Predicts delivery time in minutes.

Cleans raw delivery records and engineers features such as delivery distance, order hour, and day of week.

Uses Haversine distance between restaurant and delivery coordinates.

Handles missing values, invalid ratings, invalid coordinates, and inconsistent categorical values.

Compares Linear Regression, Random Forest, and XGBoost using identical cross-validation folds.

Performs randomized hyperparameter tuning for XGBoost.

Selects the final model using cross-validation MAE, then evaluates it once on the test set.

Supports either a random train/test split or a time-based split using order dates.

Generates EDA charts: delivery-time distribution, delivery time by hour and traffic level, and numeric-feature correlations.

Saves one portable .joblib artifact containing the pipeline, feature metadata, model choice, CV comparison, test metrics, and training provenance.

Provides a CLI for training and batch prediction.

Project structure
text
.
├── food_delivery_pipeline.py       # Training, evaluation, and inference pipeline
├── ui.py                           # Streamlit user interface
├── main.ipynb                      # Exploratory analysis notebook
├── Food delivery.csv               # Training dataset
├── new_orders.csv                  # Example inference input
├── preds.csv                       # Example prediction output
├── requirements.txt                # Python dependencies
├── models/
│   └── delivery_time_model.joblib  # Saved model artifact
└── reports/
    ├── correlation.png
    ├── target_distribution.png
    └── time_by_hour_and_traffic.png
Machine-learning design
Stateless feature preparation
prepare_features() is the only place where raw data is cleaned. It performs stateless transformations such as:

Removing whitespace and converting missing-value markers to NaN.

Parsing and validating numeric fields.

Cleaning weather-condition labels.

Computing Haversine distance between restaurant and delivery coordinates.

Extracting order hour and day of week.

Treating invalid ratings and unrealistic distances as missing values.

Because this function does not learn any statistics from the data, it can safely run before the train/test split and again during inference.

Leakage-safe preprocessing
All fitted preprocessing steps—imputation, categorical encoding, and scaling—are stored inside a single scikit-learn Pipeline. This ensures that:

Imputation statistics are learned only from training data.

One-hot encoding does not cause column drift at inference time.

The complete preprocessing and model behavior is saved in one artifact.

There is no separate scaler file to manage.

Model comparison
The following models are compared using the same K-fold cross-validation splits on the training data:

Model	Purpose
Linear Regression	Simple interpretable baseline
Random Forest Regressor	Nonlinear tree-based baseline
XGBoost Regressor	Main candidate for tuned performance
Tuned XGBoost Regressor	XGBoost with randomized hyperparameter search
The final model is selected using cross-validation MAE. The test set is used only once, after model selection.

Evaluation metrics
The project reports:

MAE — Mean Absolute Error, in minutes.

RMSE — Root Mean Squared Error, in minutes.

R² — Coefficient of determination.

It also logs MAE breakdowns by traffic density and city type to identify where the model performs poorly.

Requirements
Python 3.10+

pandas

NumPy

scikit-learn

XGBoost

Joblib

Matplotlib

Streamlit

Install dependencies:

bash
pip install -r requirements.txt
Training the model
Run training with the default dataset and save the model artifact:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv"
Generate EDA charts while training:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --eda
Use a time-based split, which holds out the most recent orders for testing:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --split time
Use a random split instead:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --split random
Increase the number of XGBoost hyperparameter-search candidates:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --n-iter 50
Refit the selected model on all available data before saving:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --refit-full
Specify a custom model output path:

bash
python food_delivery_pipeline.py train --data "Food delivery.csv" --model models/my_model.joblib
Making predictions
Create a CSV containing raw order records with the required input columns, then run:

bash
python food_delivery_pipeline.py predict \
  --model models/delivery_time_model.joblib \
  --input new_orders.csv \
  --output preds.csv
The output file contains:

text
ID,predicted_minutes
If the input file does not contain an ID column, only predicted_minutes is written.

Required input columns
The prediction input must include the following raw columns:

text
Delivery_person_Age
Delivery_person_Ratings
Restaurant_latitude
Restaurant_longitude
Delivery_location_latitude
Delivery_location_longitude
Time_Orderd
Order_Date
Weatherconditions
Road_traffic_density
Vehicle_condition
Type_of_order
Type_of_vehicle
multiple_deliveries
Festival
City
The target column used during training is:

text
Time_taken(min)
Streamlit app
Launch the local prediction interface:

bash
streamlit run ui.py
Model artifact
The saved artifact is a dictionary stored at:

text
models/delivery_time_model.joblib
It contains:

Key	Description
Key	Description
pipeline	Full preprocessing and regression pipeline
features	Final model feature names
raw_columns	Required raw input columns
selected_model	Name of the selected model
cv_comparison	Cross-validation metrics for all candidates
test_metrics	MAE, RMSE, and R² on the held-out test set
meta	Training timestamp, split type, library versions, and refit status
Data-quality rules
Ratings outside the range 0–5 are treated as missing.

Delivery distances of 100 km or more are treated as invalid coordinates.

Missing numeric values are imputed using the median.

Missing categorical values are imputed using the most frequent value.

Traffic density is ordinal-encoded in the order: Low, Medium, High, Jam.

Unknown categorical values at inference time are safely handled by the encoder.

Possible improvements
Add geospatial features such as city-level average delivery time.

Use more sophisticated date features, including holidays and peak-hour indicators.

Add model explainability using SHAP values.

Deploy the model behind a FastAPI service.

Add automated tests for feature engineering and inference.

Track experiments with MLflow or Weights & Biases.

Monitor prediction drift and data quality in production.

License
This project is available for educational and portfolio use.
