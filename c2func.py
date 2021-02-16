import os
import tarfile
import urllib
import urllib.request
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
from sklearn.model_selection import train_test_split  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, TransformerMixin

from zlib import crc32

DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    os.makedirs(housing_path, exist_ok=True)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()
    
def load_housing_data(housing_path=HOUSING_PATH):
    csv_path = os.path.join(housing_path, "housing.csv")
    return pd.read_csv(csv_path)

# def messing(data):
    print(data.info())
    print(data["ocean_proximity"].value_counts())
    print(data.describe())
    print(data.head())
    data.hist(bins=50, figsize=(20,15))
    plt.show(block=False)
# example use ...
# chap2_functions.messing(data = housing)
# plt.show()


def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column ):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
# example use ...
# housing_with_id = housing.reset_index() # adds an 'index' column
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
# print(len(train_set), len(test_set))

from sklearn.model_selection import StratifiedShuffleSplit

def Stratify(housing):
    housing["income_cat"]= pd.cut(
        housing["median_income"], 
        bins=[0.0,1.5,3.0,4.5,6.0, np.inf],
        labels=[1,2,3,4,5])
    # housing["income_cat"].hist()
    # plt.show()

    split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    for train_index, test_index in split.split(housing, housing["income_cat"]):
        strat_train_set = housing.loc[train_index]
        strat_test_set = housing.loc[test_index]
    for set_ in (strat_train_set, strat_test_set):
        set_.drop("income_cat", axis=1, inplace=True)

    return strat_train_set, strat_test_set 

def ShowInitialCharts(housing):
    housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
        s=housing["population"]/100, label="population", figsize=(10,7),
        c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,)
    plt.legend()
    corr_matrix = housing.corr()
    print(corr_matrix["median_house_value"].sort_values(ascending=False))
    from pandas.plotting import scatter_matrix
    attributes = ["median_house_value", "median_income", "total_rooms",
        "housing_median_age"]
    scatter_matrix(housing[attributes], figsize=(12,8))
    housing.plot(kind="scatter", x="median_income", y="median_house_value",
        alpha=0.1)

from sklearn.base import BaseEstimator, TransformerMixin
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True):
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self # nothing else to do
    def transform(self, X):
        rooms_ix, bedrooms_ix, population_ix, households_ix = 3,4,5,6
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


def CleanupData(housing):
    # convert textual data to numeric 
    housing["ocean_proximity"] = OrdinalEncoder().fit_transform(housing[["ocean_proximity"]])

    add_bedrooms_per_room = False
    # imputer : convert na to median
    # attributes : generate value-added columns
    # scaler : values ~ N(0,1) 
    num_pipeline = Pipeline([
            ('imputer', SimpleImputer(strategy="median")),
            ('attribs_adder', CombinedAttributesAdder(add_bedrooms_per_room)),
            ('std_scaler', StandardScaler())
        ])

    if add_bedrooms_per_room:
        extracols  = ['add1', 'add2', 'add3'] 
    else:
        extracols = ['add1', 'add2'] # need add3 if passing True

    housing = pd.DataFrame(data=num_pipeline.fit_transform(housing),
        columns= housing.columns.union(extracols) )

    return housing  

   




