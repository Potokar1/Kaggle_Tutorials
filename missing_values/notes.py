# Just Notes, No Working Program
"""
# Python libraries represent missing numbers as nan which is short for "not a number".
# You can detect which cells have missing values, and then count how many there are in each column with the command:

missing_val_count_by_column = (data.isnull().sum())
print(missing_val_count_by_column[missing_val_count_by_column > 0

# Most libraries will give you errors if you try to build a model with missing values


# Solution 1: Drop columns with missing values

# If your data is in a DataFrame called original_data, you can drop columns with missing values. One way to do that is
data_without_missing_values = original_data.dropna(axis=1)

# In many cases, you'll have both a training dataset and a test dataset. You will want to drop the same columns in both DataFrames.
# In that case, you would write

cols_with_missing = [col for col in original_data.columns
                                 if original_data[col].isnull().any()]
redued_original_data = original_data.drop(cols_with_missing, axis=1)
reduced_test_data = test_data.drop(cols_with_missing, axis=1)

# Pretty much only good when most values in a column is missing


# Solution 2: Imputation

# Imputation fills in the missing value with some number.
# The imputed value won't be exactly right in most cases, but it usually gives more accurate models than dropping the column entirely.

from sklearn.impute import SimpleImputer
my_imputer = SimpleImputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)

# This gives the NAN values the mean value of the column which works prety well!
# Imputation can be used in a scikit-learn Pipeline


# Solution 3: An extention on Imputation


# Imputed values may by systematically above or below their actual values (which weren't collected in the dataset).
# Or rows with missing values may be unique in some other way.
# In that case, your model would make better predictions by considering which values were originally missing.

# make copy to avoid changing original data (when Imputing)
new_data = original_data.copy()

# make new columns indicating what will be imputed
cols_with_missing = (col for col in new_data.columns
                                 if new_data[col].isnull().any())
for col in cols_with_missing:
    new_data[col + '_was_missing'] = new_data[col].isnull()

# Imputation
my_imputer = SimpleImputer()
new_data = pd.DataFrame(my_imputer.fit_transform(new_data))
new_data.columns = original_data.columns

"""

# Actual Working Program

# Set Up
import pandas as pd

# Load data
melb_data = pd.read_csv('Kaggle_Tutorials\\missing_values\\melb_data.csv')

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

melb_target = melb_data.Price
# using all other columns except price?
melb_predictors = melb_data.drop(['Price'], axis=1)

# For the sake of keeping the example simple, we'll use only numeric predictors.
melb_numeric_predictors = melb_predictors.select_dtypes(exclude=['object'])

# splitting the data into test and train
X_train, X_test, y_train, y_test = train_test_split(
    melb_numeric_predictors, melb_target, train_size=0.7, test_size=0.3, random_state=0)

# Used to compare the quality of diffrent approaches to missing values. This function reports the out-of-sample MAE score from a RandomForest.


def score_dataset(X_train, X_test, y_train, y_test):
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return mean_absolute_error(y_test, preds)


# Get Model Score from Dropping Columns with Missing Values
cols_with_missing = [col for col in X_train.columns if X_train[col].isnull().any()]
reduced_X_train = X_train.drop(cols_with_missing, axis=1)
reduced_X_test = X_test.drop(cols_with_missing, axis=1)
print("Mean Absolute Error from dropping columns with Missing Values:")
print(score_dataset(reduced_X_train, reduced_X_test, y_train, y_test))

# Get Model Score from Imputation
from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)
print("Mean Absolute Error from Imputation:")
print(score_dataset(imputed_X_train, imputed_X_test, y_train, y_test))

# Get Score from Imputation with Extra Columns Showing What Was Imputed
imputed_X_train_plus = X_train.copy()
imputed_X_test_plus = X_test.copy()

cols_with_missing = (col for col in X_train.columns if X_train[col].isnull().any())
for col in cols_with_missing:
    imputed_X_train_plus[col + '_was_missing'] = imputed_X_train_plus[col].isnull()
    imputed_X_test_plus[col + '_was_missing'] = imputed_X_test_plus[col].isnull()

# Imputation
my_imputer = SimpleImputer()
imputed_X_train_plus = my_imputer.fit_transform(imputed_X_train_plus)
imputed_X_test_plus = my_imputer.transform(imputed_X_test_plus)

print("Mean Absolute Error from Imputation while Track What Was Imputed:")
print(score_dataset(imputed_X_train_plus, imputed_X_test_plus, y_train, y_test))


'''
# This article is missing quite a few steps in describing what imputation does in the context of SciKit. You cannot just do what the directions say in section 2:

from sklearn.preprocessing import Imputer
my_imputer = Imputer()
data_with_imputed_values = my_imputer.fit_transform(original_data)

# In this example, data_with_imputed_values will just be a ndarray of ndarrays with the imputed values and will not be structured like a Pandas DataFrame.
# You need to re-cast that last line:

data_with_imputed_values = pd.DataFrame(my_imputer.fit_transform(original_data))

# However, this means you will lose the column titles. Since the order of the columns does not change after imputation, you can add the titles back like this:

data_with_imputed_values.columns = original_data.columns

# This way you preserve the structure of the original data that gets lost when you follow the steps in this article.
'''
