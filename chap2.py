import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from zlib import crc32
import c2func
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LinearRegression

c2func.fetch_housing_data()
housing=c2func.load_housing_data()

strat_train_set, strat_test_set = c2func.Stratify(housing=housing.copy())
print(len(strat_train_set), len(strat_test_set))
# housing = data for training, housing_labels = target values
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', c2func.CombinedAttributesAdder()),
        ('std_scaler', StandardScaler())
    ])
num_attribs = list(housing.drop("ocean_proximity", axis=1))
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
    
housing_prepared =  full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)

# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Predictions: ", lin_reg.predict(some_data_prepared))
# print("Labels: ", list(some_labels))

# from sklearn.metrics import mean_squared_error
# housing_predictions = lin_reg.predict(housing_prepared)
# lin_mse = mean_squared_error(housing_labels, housing_predictions)
# lin_rmse = np.sqrt(lin_mse)
# print(lin_rmse)

# from sklearn.tree import DecisionTreeRegressor
# tree_reg = DecisionTreeRegressor()
# tree_reg.fit(housing_prepared, housing_labels)

# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_rmse = np.sqrt(tree_mse)
# print(tree_rmse)

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

# from sklearn.model_selection import cross_val_score

# scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# tree_rmse_scores = np.sqrt(-scores)
# display_scores(tree_rmse_scores)

# scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# lin_rmse_scores = np.sqrt(-scores)
# display_scores(lin_rmse_scores)

# this takes a long time ... 
from sklearn.ensemble import RandomForestRegressor
# forest_reg = RandomForestRegressor()
# forest_reg.fit(housing_prepared, housing_labels)
# scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
#                          scoring="neg_mean_squared_error", cv=10)
# forest_rmse_scores = np.sqrt(-scores)
# display_scores(forest_rmse_scores)

from sklearn.model_selection import GridSearchCV

param_grid = [
    { 'n_estimators': [3,10,30], 'max_features': [2,4,6,8]},
    { 'bootstrap': [False], 'n_estimators': [3,10], 'max_features': [2,3,4]}
]

forest_reg = RandomForestRegressor()
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
    scoring='neg_mean_squared_error', 
    return_train_score=True)

grid_search.fit(housing_prepared, housing_labels)

print(grid_search.best_params_)
print(grid_search.best_estimator_)
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)

