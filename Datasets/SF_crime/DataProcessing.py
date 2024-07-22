import pandas as pd
import numpy as np
import time
import pickle
from datetime import datetime
from sklearn.preprocessing import LabelEncoder

# Load the dataset (adjust the path to your dataset)
sf_crime = pd.read_csv('/content/STHSL/Datasets/SF_crime/SFCrime.csv')

sf_crime = sf_crime[['Category', 'Descript', 'DayOfWeek', 'Date', 'Time', 'PdDistrict','Resolution', 'Address', 'X','Y']]

#Data Cleaning
# Drop rows with any missing values
sf_crime.dropna(inplace=True)

# Create a mapping dictionary for the new categories
category_mapping = {
    'ROBBERY': 'Violent Crimes',
    'VEHICLE THEFT': 'Property Crimes',
    'ARSON': 'Violent Crimes',
    'ASSAULT': 'Violent Crimes',
    'TRESPASS': 'Miscellaneous',
    'BURGLARY': 'Property Crimes',
    'LARCENY/THEFT': 'Property Crimes',
    'WARRANTS': 'Administrative and Other Offenses',
    'OTHER OFFENSES': 'Administrative and Other Offenses',
    'DRUG/NARCOTIC': 'Drug and Alcohol Related',
    'SUSPICIOUS OCC': 'Miscellaneous',
    'LIQUOR LAWS': 'Drug and Alcohol Related',
    'VANDALISM': 'Property Crimes',
    'WEAPON LAWS': 'Violent Crimes',
    'NON-CRIMINAL': 'Non-Criminal and Special Cases',
    'MISSING PERSON': 'Non-Criminal and Special Cases',
    'FRAUD': 'Property Crimes',
    'SEX OFFENSES, FORCIBLE': 'Violent Crimes',
    'SECONDARY CODES': 'Administrative and Other Offenses',
    'DISORDERLY CONDUCT': 'Public Order Crimes',
    'RECOVERED VEHICLE': 'Non-Criminal and Special Cases',
    'KIDNAPPING': 'Violent Crimes',
    'FORGERY/COUNTERFEITING': 'Property Crimes',
    'PROSTITUTION': 'Public Order Crimes',
    'DRUNKENNESS': 'Drug and Alcohol Related',
    'BAD CHECKS': 'White Collar Crimes',
    'DRIVING UNDER THE INFLUENCE': 'Drug and Alcohol Related',
    'LOITERING': 'Public Order Crimes',
    'STOLEN PROPERTY': 'Property Crimes',
    'SUICIDE': 'Non-Criminal and Special Cases',
    'BRIBERY': 'White Collar Crimes',
    'EXTORTION': 'White Collar Crimes',
    'EMBEZZLEMENT': 'Property Crimes',
    'GAMBLING': 'Public Order Crimes',
    'PORNOGRAPHY/OBSCENE MAT': 'Non-Criminal and Special Cases',
    'SEX OFFENSES, NON FORCIBLE': 'Violent Crimes',
    'TREA': 'Non-Criminal and Special Cases'
}

# Replace the original 'Category' column with the new categories
sf_crime['Category'] = sf_crime['Category'].map(category_mapping)

# Convert 'Date' and 'Time' columns to datetime because we can only use .dt accessor with datetimelike values
sf_crime['datetime'] = pd.to_datetime(sf_crime['Date'] + ' ' + sf_crime['Time'])

# Extract relevant features
sf_crime['year'] = sf_crime['datetime'].dt.year
sf_crime['hour'] = sf_crime['datetime'].dt.hour
sf_crime['month'] = sf_crime['datetime'].dt.month
sf_crime['day_of_week'] = sf_crime['datetime'].dt.dayofweek
sf_crime['day'] = pd.to_datetime(sf_crime['datetime']).dt.day
sf_crime['minute'] = pd.to_datetime(sf_crime['datetime']).dt.minute
# Filter data for years before 2018 for training and use 2018 for testing
train_df = sf_crime[sf_crime['year'] < 2018]
test_df = sf_crime[sf_crime['year'] == 2018]

label_encoder = LabelEncoder()

# Encode 'Category' (target variable)
train_df['Category'] = label_encoder.fit_transform(train_df['Category'])
test_df['Category'] = label_encoder.transform(test_df['Category'])

# Ensure 'Category' is integer type in both DataFrames
train_df['Category'] = train_df['Category'].astype(int)
test_df['Category'] = test_df['Category'].astype(int)

features = ['PdDistrict','hour','day','minute','day_of_week', 'month', 'X', 'Y']
X_train = train_df[features]
y_train = train_df['Category']
X_test = test_df[features]
y_test = test_df['Category']

# One-hot encode the 'PdDistrict' feature
X_train = pd.get_dummies(X_train, columns=['PdDistrict'])
X_test = pd.get_dummies(X_test, columns=['PdDistrict'])

# Ensure the columns match between train and test sets
X_train, X_test = X_train.align(X_test, join='left', axis=1, fill_value=0)

# Combine train and test data to ensure all unique regions are captured
all_data = pd.concat([train_df, test_df])
unique_regions = all_data[['X', 'Y']].drop_duplicates().reset_index(drop=True)

# Create a mapping for regions
region_map = {tuple(region): idx for idx, region in unique_regions.iterrows()}

# Function to create a region ID
def get_region_id(row):
    return region_map.get((row['X'], row['Y']), -1)

X_train['region'] = X_train.apply(get_region_id, axis=1)
X_test['region'] = X_test.apply(get_region_id, axis=1)

# Convert 'datetime' to timestamp
all_data['timestamp'] = all_data['datetime'].apply(lambda x: time.mktime(x.timetuple()))

# Define grid dimensions
min_lat, max_lat = all_data['Y'].min(), all_data['Y'].max()
min_lon, max_lon = all_data['X'].min(), all_data['X'].max()
lat_div, lon_div = 111 / 3, 84 / 3
lat_num = int((max_lat - min_lat) * lat_div) + 1
lon_num = int((max_lon - min_lon) * lon_div) + 1

# Calculate total training days
total_train_days = (2018 - 2015) * 365 + sum(1 for year in range(2015, 2018) if year % 4 == 0)

# Initialize tensors
trnTensor = np.zeros((lat_num, lon_num, total_train_days, len(label_encoder.classes_)))
valTensor = np.zeros((lat_num, lon_num, len(pd.date_range(start='2017-06-01', end=sf_crime['datetime'].max())), len(label_encoder.classes_)))
tstTensor = np.zeros((lat_num, lon_num, len(pd.date_range(start='2018-01-01', end=sf_crime['datetime'].max())), len(label_encoder.classes_)))

# Populate tensors
day_counter = 0
for i, row in all_data.iterrows():
    timestamp = row['timestamp']
    temT = time.localtime(timestamp)
    if 2015 <= temT.tm_year < 2018:
        day = day_counter
        day_counter += 1  # Increment day counter for training data
        tensor = trnTensor
    elif temT.tm_year >= 2018:
        day = (pd.to_datetime(time.strftime('%Y-%m-%d', temT)) - pd.to_datetime('2018-01-01')).days
        tensor = tstTensor
    else:
        continue  # Skip the iteration if none of the conditions are met
    
    row_id = int((row['Y'] - min_lat) * lat_div)
    col_id = int((row['X'] - min_lon) * lon_div)
    offense = row['Category']
    tensor[row_id][col_id][day][offense] += 1

# Save the tensors as .pkl files
names = ['trn.pkl','val.pkl','tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]  
for i in range(len(names)):
    if tensors[i] is not None:
        with open('/content/STHSL/Datasets/SF_crime/' + names[i], 'wb') as fs:
            pickle.dump(tensors[i], fs)
