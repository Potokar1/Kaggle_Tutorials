import pandas as pd

melbourne_file_path = 'Kaggle_Tutorials\\first_model\\melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
print(melbourne_data.columns)

# The Melbourne data has some missing values (some houses for which some variables weren't recorded.)
# We'll learn to handle missing values in a later tutorial.
# Your Iowa data doesn't have missing values in the columns you use.
# So we will take the simplest option for now, and drop houses from our data.
# Don't worry about this much for now, though the code is:

# dropna drops missing values (think of na as "not available")
melbourne_data = melbourne_data.dropna(axis=0)

# SELECTING DATA FOR MODELING

# You can pull out a variable with dot-notation. This single column is stored in a Series, which is broadly like a DataFrame with only a single column of data.
# We'll use the dot notation to select the column we want to predict, which is called the prediction target. By convention, the prediction target is called y.
# So the code we need to save the house prices in the Melbourne data is
y = melbourne_data.Price

# SELECTING 'FEATURES'

# The columns that are inputted into our model (and later used to make predictions) are called "features."
# You can use all the other columns except the 'y' data to model, but sometimes fewer is better!

# We select multiple features by providing a list of column names inside brackets. Each item in that list should be a string (with quotes).
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'Lattitude', 'Longtitude']

# By conventions, this data is called X
X = melbourne_data[melbourne_features]

print(X.describe().to_string())

print(X.head())
print('\n\n')

# The steps to building and using a model are:
# Define: What type of model will it be? A decision tree? Some other type of model? Some other parameters of the model type are specified too.
# Fit: Capture patterns from provided data. This is the heart of modeling.
# Predict: Just what it sounds like
# Evaluate: Determine how accurate the model's predictions are.

from sklearn.tree import DecisionTreeRegressor

# Define model. Specify a number for random_state to ensure same results each run
melbourne_model = DecisionTreeRegressor(random_state=1)

# Fit model
print(melbourne_model.fit(X, y))

# Many machine learning models allow some randomness in model training.
# Specifying a number for random_state ensures you get the same results in each run.
# This is good practice, helps with consistancy maybe? analysis?

# We now have a fitted model that we can use to make predictions.

print("Making predictions for the following 5 houses:")
print(X.head())
print("The predictions are")
print(melbourne_model.predict(X.head()))
