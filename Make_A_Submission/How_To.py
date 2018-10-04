
# EXAMPLE CODE
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Read the data
train = pd.read_csv('Kaggle_Tutorials\\Make_A_Submission\\train.csv')

# pull data into target (y) and predictors (X)
train_y = train.SalePrice
predictor_cols = ['LotArea', 'OverallQual', 'YearBuilt', 'TotRmsAbvGrd']

# Create training predictors data
train_X = train[predictor_cols]

my_model = RandomForestRegressor()
my_model.fit(train_X, train_y)

# Read the test data
test = pd.read_csv('Kaggle_Tutorials\\Make_A_Submission\\test.csv')
# Treat the test data in the same way as training data. In this case, pull same columns.
test_X = test[predictor_cols]
# Use the model to make predictions
predicted_prices = my_model.predict(test_X)
# We will look at the predicted prices to ensure we have something sensible.
print(predicted_prices)

# We make submissions in CSV files. Your submissions usually have two columns: an ID column and a prediction column.
# The ID field comes from the test data (keeping whatever name the ID field had in that data, which for the housing data is the string 'Id'). The prediction column will use the name of the target field.

# We will create a DataFrame with this data, and then use the dataframe's to_csv method to write our submission file.
# Explicitly include the argument index=False to prevent pandas from adding another column in our csv file.

my_submission = pd.DataFrame({'Id': test.Id, 'SalePrice': predicted_prices})
# you could use any filename. We choose submission here
my_submission.to_csv('submission.csv', index=False)

# Hit the blue Publish button at the top of your notebook screen. It will take some time for your kernel to run.
# When it has finished your navigation bar at the top of the screen will have a tab for Output.
# This only shows up if you have written an output file (like we did in the Prepare Submission File step).

# LAST STEPS

# Click on the Output button. This will bring you to a screen with an option to Submit to Competition. Hit that and you will see how your model performed.

# If you want to go back to improve your model, click the Edit button, which re-opens the kernel. You'll need to re-run all the cells when you re-open the kernel.
