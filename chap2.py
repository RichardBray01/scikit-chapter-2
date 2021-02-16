import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from zlib import crc32
import c2func

c2func.fetch_housing_data()
housing=c2func.load_housing_data()

strat_train_set, strat_test_set = c2func.Stratify(housing=housing.copy())

# housing = data for training, housing_labels = target values
housing = strat_train_set.drop("median_house_value", axis=1)
housing_labels = strat_train_set["median_house_value"].copy()

housing = c2func.CleanupData(housing)

for column in housing:
    print(housing[column].describe())


