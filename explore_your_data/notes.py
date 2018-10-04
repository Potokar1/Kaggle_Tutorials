# pandas is a data analysis tool
import pandas as pd

# this can allow us to see all the data that is outputted from describe
# pd.set_option('display.max_columns', None)  # or 1000
# print(home_data.describe().to_string()) <- this prints everything!!!! YASSS


# we want the data frame part of pandas, think of the dataframe like a table!
# simlar to a sheet in excel or a table in a sql database

# As an example, we'll looking at data about home prices in Melbourne,Australia.
# In the hands-on exercises, you will apply the same processes to a new dataset,
# which has home prices in Iowa.

# save filepath to variable for easier access
melbourne_file_path = 'C:\\Users\\Jack Dubbs\\Desktop\\Kaggle\\Kaggle_Tutorials\\explore_your_data\\melb_data.csv'
# read the data and store data in DataFrame titled melbourne_data
melbourne_data = pd.read_csv(melbourne_file_path)
# print a summary of the data in Melbourne data
print(melbourne_data.describe().to_string())
# print(melbourne_data.head(20)) This can show up the top 20 in the datafile
