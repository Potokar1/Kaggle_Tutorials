import pandas as pd

# Path of the file to read
iowa_file_path = 'C:\\Users\\Jack Dubbs\\Desktop\\Kaggle\\Kaggle_Tutorials\\explore_your_data\\train.csv'

# reading in the file to a varaible home_data
home_data = pd.read_csv(iowa_file_path)

# look at our data so we can get a better idea about what our data is like!
print(home_data.describe().to_string())
