# Code you have previously used to load data
import pandas as pd

# Path of the file to read (right click .csv file and choose project path)
# gotta add extra \ slashes to make sure python doesn't read it wrong
iowa_file_path = 'Kaggle_Tutorials\\first_model\\train.csv'

home_data = pd.read_csv(iowa_file_path)

# Step 1: Specify Prediction Target

# print the list of columns in the dataset to find the name of the prediction target
print(home_data.columns)

y = home_data.SalePrice

# Step 2: Create X

# Create the list of features below
feature_names = ["LotArea", "YearBuilt", "1stFlrSF",
                 "2ndFlrSF", "FullBath", "BedroomAbvGr", "TotRmsAbvGrd"]

# select data corresponding to features in feature_names
X = home_data[feature_names]

# Review Data - make sure X is good

# print description or statistics from X
print(X.describe().to_string())

# print the top few lines
print(X.head())

# Step 3: Specify and Fit Model
# Then fit the model you just created using the data in `X` and `y` that you saved above.
from sklearn.tree import DecisionTreeRegressor

# specify the model.
# For model reproducibility, set a numeric value for random_state when specifying the model
iowa_model = DecisionTreeRegressor(random_state=2)

# Fit the model
print(iowa_model.fit(X, y))

# Step 4: Make Predictions
# Make predictions with the model's `predict` command using `X` as the data. Save the results to a variable called `predictions`.

predictions = iowa_model.predict(X)
print(predictions)


print("Making predictions for the following 5 houses:")
print(X.head().to_string())
print("The predictions are")
print(iowa_model.predict(X.head()))
