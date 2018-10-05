import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

train_data = pd.read_csv('Kaggle_Tutorials\\missing_values\\train.csv')

# target
y = train_data.SalePrice
# predictors
X = train_data.drop(['SalePrice'], axis=1)
X_numeric_only = X.select_dtypes(exclude=['object'])

# Instead of dropping and getting MAE = 17667.30819634703...
# cols_with_missing = [col for col in X_numeric_only.columns if X_numeric_only[col].isnull().any()]
# reduced_X = X_numeric_only.drop(cols_with_missing, axis=1)

X_train, X_test, y_train, y_test = train_test_split(
    X_numeric_only, y, train_size=0.7, test_size=0.3, random_state=0)

from sklearn.impute import SimpleImputer

my_imputer = SimpleImputer()
imputed_X_train = my_imputer.fit_transform(X_train)
imputed_X_test = my_imputer.transform(X_test)

model = RandomForestRegressor(n_estimators=100)
model.fit(imputed_X_train, y_train)
predictions = model.predict(imputed_X_test)
print(mean_absolute_error(y_test, predictions))

# We get MAE = 17596.963401826484 ... 100+ improvement huh (this changes every runtime)
