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

# housing = data for training, housing_labels = target values
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

add_bedrooms_per_room = False
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', c2func.CombinedAttributesAdder(add_bedrooms_per_room)),
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

some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)
print("Predictions: ", lin_reg.predict(some_data_prepared))


