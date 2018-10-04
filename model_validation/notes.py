# There are many metrics for summarizing model quality, but we'll start with one called Mean Absolute Error (also called MAE).
# Let's break down this metric starting with the last word, error.

# The prediction error for each house is:
#   error = actual − predicted

# With the MAE metric, we take the absolute value of each error. This converts each error to a positive number.
# We then take the average of those absolute errors. This is our measure of model quality. In plain English, it can be said as

#   On average, our predictions are off by about X.

# This is setting up the model
import pandas as pd

# Load data
melbourne_file_path = 'Kaggle_Tutorials\\model_validation\\melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing price values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.tree import DecisionTreeRegressor
# Define model
melbourne_model = DecisionTreeRegressor()
# Fit model
melbourne_model.fit(X, y)


# Once we have a model, here is how we calculate the mean absolute error:
from sklearn.metrics import mean_absolute_error

predicted_home_prices = melbourne_model.predict(X)
print(mean_absolute_error(y, predicted_home_prices))

# What's the problem with this?
# The model can pick up on false patters 'built into' the data set
# This makes the model 'less good' when predicting novel data
# A work around is to exclude some data. This is called validation data

# The scikit-learn library has a function train_test_split to break up the data into two pieces.
# We'll use some of that data as training data to fit the model, and we'll use the other data as validation data to calculate mean_absolute_error.

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
# The split is based on a random number generator. Supplying a numeric value to
# the random_state argument guarantees we get the same split every time we
# run this script.
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)
# Define model
melbourne_model = DecisionTreeRegressor()  # you could also put a random_state=# here!
# Fit model
melbourne_model.fit(train_X, train_y)

# get predicted prices on validation data
val_predictions = melbourne_model.predict(val_X)
print(mean_absolute_error(val_y, val_predictions))
