# app.py - Model Training and Saving

import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import pickle

# Load your dataset
data = pd.read_csv("Boralesgamuwa_2012_2018_with_features.csv")

# Data preprocessing as described earlier
data['ticket_date'] = pd.to_datetime(data['ticket_date'], errors='coerce')
data.set_index('ticket_date', inplace=True)
data.fillna(method='ffill', inplace=True)
data.fillna(method='bfill', inplace=True)
non_numeric_columns = data.select_dtypes(exclude=['number']).columns
data.drop(non_numeric_columns, axis=1, inplace=True)
data['week_of_year'] = data.index.isocalendar().week
data['day_of_year'] = data.index.dayofyear
data['is_month_start'] = data.index.is_month_start.astype(int)
data['is_month_end'] = data.index.is_month_end.astype(int)

# Define features and target variable
features = ['lag_1', 'lag_2', 'rolling_7', 'day_of_week', 'month', 'week_of_year', 'day_of_year', 'is_month_start', 'is_month_end']
target = 'net_weight_kg'

# One-hot encode categorical variables
X = pd.get_dummies(data[features], columns=['day_of_week', 'month'])
y = data[target]

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define base models and meta-model for stacking
base_models = [
    ('svm', SVR()),
    ('knn', KNeighborsRegressor()),
    ('rf', RandomForestRegressor()),
    ('lgb', lgb.LGBMRegressor()),
    ('xgb', xgb.XGBRegressor())
]
meta_model = ElasticNet()

# Create pipeline with stacking
pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("stacking", StackingRegressor(
        estimators=base_models,
        final_estimator=meta_model,
        cv=5
    ))
])

# Define parameter grid and perform RandomizedSearchCV
param_grid = {
    'stacking__svm__C': [0.01, 0.1, 1, 10, 1000],
    'stacking__svm__epsilon': [0.01, 0.1, 0.5, 1.0],
    'stacking__svm__kernel': ['linear', 'rbf', 'poly'],
    'stacking__knn__n_neighbors': [3, 5, 7, 9],
    'stacking__rf__n_estimators': [100, 200, 300],
    'stacking__rf__max_depth': [None, 10, 20, 30],
    'stacking__lgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'stacking__lgb__num_leaves': [31, 62, 93],
    'stacking__xgb__learning_rate': [0.01, 0.05, 0.1, 0.2],
    'stacking__xgb__max_depth': [3, 6, 9],
    'stacking__xgb__n_estimators': [100, 200, 300],
    'stacking__final_estimator__alpha': [0.1, 1, 10]
}

random_search = RandomizedSearchCV(
    pipeline,
    param_distributions=param_grid,
    n_iter=100,
    scoring='neg_mean_squared_error',
    cv=5,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

# Fit RandomizedSearchCV
random_search.fit(X_train_scaled, y_train)

# Save the best model
with open('best_model.pkl', 'wb') as f:
    pickle.dump(random_search.best_estimator_, f)

print("Model training and saving complete.")

