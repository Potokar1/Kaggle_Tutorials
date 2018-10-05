'''
Categorical data is data that takes only a limited number of values.

For example, if you people responded to a survey about which what brand of car they owned,
the result would be categorical (because the answers would be things like Honda, Toyota, Ford, None, etc.).
Responses fall into a fixed set of categories.

You will get an error if you try to plug these variables into most machine learning models in Python without "encoding" them first.
Here we'll show the most popular method for encoding categorical variables.
'''
# One Hot Encoding

# works very well unless your categorical variable takes on a large number of values (>15)
'''
Creates a new table with a unique value for each catagory. Like this
[catagory]      [A] [B] [C] [D]
    A           [1] [ ] [ ] [ ]
    B           [ ] [1] [ ] [ ]
    C           [ ] [ ] [1] [ ]
    D           [ ] [ ] [ ] [1]
    C           [ ] [ ] [1] [ ]
    D           [ ] [ ] [ ] [1]
    A           [1] [ ] [ ] [ ]
    .                  .
    .                  .
    .                  .
'''

# Read the data
import pandas as pd
train_data = pd.read_csv('Kaggle_Tutorials\\catagorical_data_with_one_hot_encoding\\train.csv')
test_data = pd.read_csv('Kaggle_Tutorials\\catagorical_data_with_one_hot_encoding\\test.csv')

# Drop houses where the target is missing. We are not doing this with the test data because,
# the test data does not have a SalePrice Column
train_data.dropna(axis=0, subset=['SalePrice'], inplace=True)

target = train_data.SalePrice

# Since missing values isn't the focus of this tutorial, we use the simplest
# possible approach, which is to drops these columns.
cols_with_missing = [col for col in train_data.columns
                     if train_data[col].isnull().any()]
candidate_train_predictors = train_data.drop(['Id', 'SalePrice'] + cols_with_missing, axis=1)
candidate_test_predictors = test_data.drop(['Id'] + cols_with_missing, axis=1)

# "cardinality" means the number of unique values in a column.
# We use it as our only way to select categorical columns here. This is convenient, though
# a little arbitrary.
low_cardinality_cols = [cname for cname in candidate_train_predictors.columns if
                        candidate_train_predictors[cname].nunique() < 10 and
                        candidate_train_predictors[cname].dtype == "object"]
numeric_cols = [cname for cname in candidate_train_predictors.columns if
                candidate_train_predictors[cname].dtype in ['int64', 'float64']]
my_cols = low_cardinality_cols + numeric_cols
train_predictors = candidate_train_predictors[my_cols]
test_predictors = candidate_test_predictors[my_cols]

# To check which types our predictor columns are, we can use pandas assigned data type for each columm (dtype)
# this is a random sample! not the top 10 or anything like that
print(train_predictors.dtypes.sample(10))

# Object indicates that the column has text. Those are the ones we want to one-hot encode
# Pandas offers a convenient function called get_dummies to get one-hot encodings. Call it like this:

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)

'''
Alternatively, you could have dropped the categoricals.
To see how the approaches compare, we can calculate the mean absolute error of models built with two alternative sets of predictors:

1. One-hot encoded categoricals as well as numeric predictors
2. Numerical predictors, where we drop categoricals.

One-hot encoding usually helps, but it varies on a case-by-case basis.
In this case, there doesn't appear to be any meaningful benefit from using the one-hot encoded variables.
'''

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor


def get_mae(X, y):
    # multiple by -1 to make positive MAE score instead of neg value returned as sklearn convention
    return -1 * cross_val_score(RandomForestRegressor(50), X, y, scoring='neg_mean_absolute_error').mean()


predictors_without_categoricals = train_predictors.select_dtypes(exclude=['object'])

mae_without_categoricals = get_mae(predictors_without_categoricals, target)

mae_one_hot_encoded = get_mae(one_hot_encoded_training_predictors, target)

print('Mean Absolute Error when Dropping Categoricals: ' + str(int(mae_without_categoricals)))
print('Mean Abslute Error with One-Hot Encoding: ' + str(int(mae_one_hot_encoded)))

# Applying to Multiple Files

'''
So far, you've one-hot-encoded your training data. What about when you have multiple files (e.g. a test dataset, or some other data that you'd like to make predictions for)?
Scikit-learn is sensitive to the ordering of columns, so if the training dataset and test datasets get misaligned, your results will be nonsense.
This could happen if a categorical had a different number of values in the training data vs the test data.
'''

# Ensure the test data is encoded in the same manner as the training data with the align command:

one_hot_encoded_training_predictors = pd.get_dummies(train_predictors)
one_hot_encoded_test_predictors = pd.get_dummies(test_predictors)
final_train, final_test = one_hot_encoded_training_predictors.align(
    one_hot_encoded_test_predictors, join='left', axis=1)

'''
The align command makes sure the columns show up in the same order in both datasets (it uses column names to identify which columns line up in each dataset.)
The argument join='left' specifies that we will do the equivalent of SQL's left join.
That means, if there are ever columns that show up in one dataset and not the other, we will keep exactly the columns from our training data.
The argument join='inner' would do what SQL databases call an inner join, keeping only the columns showing up in both datasets. That's also a sensible choice.
'''

'''
Conclusion
The world is filled with categorical data. You will be a much more effective data scientist if you know how to use this data.
Here are resources that will be useful as you start doing more sophisticated work with cateogircal data.

Pipelines: Deploying models into production ready systems is a topic unto itself.
While one-hot encoding is still a great approach, your code will need to built in an especially robust way.
Scikit-learn pipelines are a great tool for this. Scikit-learn offers a class for one-hot encoding and this can be added to a Pipeline.
Unfortunately, it doesn't handle text or object values, which is a common use case.

Applications To Text for Deep Learning: Keras and TensorFlow have fuctionality for one-hot encoding, which is useful for working with text.

Categoricals with Many Values: Scikit-learn's FeatureHasher uses the hashing trick to store high-dimensional data. This will add some complexity to your modeling code.
'''
