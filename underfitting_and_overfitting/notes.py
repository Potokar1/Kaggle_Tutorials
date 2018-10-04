# for more info on scikit-learn's documentation http://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html

# OVERFITTING
# So, in the context of decision trees, if we have way too many leafs, aka a lot of splits (tree height = very large and maybe 2^n nodes),
# then they may make very unreliable predictions for novel data (because each prediction is based on only a few [data elements])

# UNDERFITTING
# So, in the context of decision trees, if we have very little leafs, aka only a few splits (tree height < 3 and maybe 8 nodes),
# then they fail to capture important distinctions and patters in the data. So it performs poorly even on the training data.

# The max_leaf_nodes argument provides a very sensible way to control overfitting vs underfitting.
# The more leaves we allow the model to make, the more we move from the underfitting area in the above graph to the overfitting area.

# We can use a utility function to help compare MAE scores from different values for max_leaf_nodes:

from sklearn.metrics import mean_absolute_error
from sklearn.tree import DecisionTreeRegressor

# gets the mae for the given input


def get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y):
    model = DecisionTreeRegressor(max_leaf_nodes=max_leaf_nodes, random_state=0)
    model.fit(train_X, train_y)
    preds_val = model.predict(val_X)
    mae = mean_absolute_error(val_y, preds_val)
    return(mae)

# The data is loaded into train_X, val_X, train_y and val_y using the code you've already seen (and which you've already written).


# Data Loading Code Runs At This Point
import pandas as pd

# Load data
melbourne_file_path = 'Kaggle_Tutorials\\underfitting_and_overfitting\\melb_data.csv'
melbourne_data = pd.read_csv(melbourne_file_path)
# Filter rows with missing values
filtered_melbourne_data = melbourne_data.dropna(axis=0)
# Choose target and features
y = filtered_melbourne_data.Price
melbourne_features = ['Rooms', 'Bathroom', 'Landsize', 'BuildingArea',
                      'YearBuilt', 'Lattitude', 'Longtitude']
X = filtered_melbourne_data[melbourne_features]

from sklearn.model_selection import train_test_split

# split data into training and validation data, for both features and target
train_X, val_X, train_y, val_y = train_test_split(X, y, random_state=0)

# We can use a for-loop to compare the accuracy of models built with different values for max_leaf_nodes.

# compare MAE with differing values of max_leaf_nodes
for max_leaf_nodes in [5, 50, 500, 5000]:
    my_mae = get_mae(max_leaf_nodes, train_X, val_X, train_y, val_y)
    print("Max leaf nodes: %d  \t\t Mean Absolute Error:  %d" % (max_leaf_nodes, my_mae))

# Based on output, 500 is the optimal number of leaves

'''
Here's the takeaway: Models can suffer from either:

Overfitting: capturing spurious patterns that won't recur in the future, leading to less accurate predictions, or
Underfitting: failing to capture relevant patterns, again leading to less accurate predictions.
We use validation data, which isn't used in model training, to measure a candidate model's accuracy. This lets us try many candidate models and keep the best one.
'''
