import pandas as pd
!pip install google.colab
from google.colab import drive
import pandas as pd

#mounting drive
drive.mount('/content/drive')
# reading data
df=pd.read_csv("/content/drive/MyDrive/internship/Datasets/San Francisco Crime/SFCrime.csv")
df = df[['Date','Time','Category','X','Y']]

#Data Cleaning

# Drop rows with any missing values
df.dropna(inplace=True)

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
df['Category'] = df['Category'].map(category_mapping)

# Ensure the 'Time' column has the format ':00' for seconds
df['Time'] = df['Time'].str[:5] + ':00'
df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M:%S')

# Extract the AM/PM designation
df['AM_PM'] = df['Datetime'].dt.strftime('%p')

df = df[['Datetime','Category','X','Y']]

# Display the DataFrame
print(df)

# Save the DataFrame to a CSV file
csv_file_path = 'SF_crime_with_am_pm.csv'
df.to_csv(csv_file_path, index=False)

print(f"CSV file saved at {csv_file_path}")

# Uncomment the following lines if you are using Google Colab to download the file
from google.colab import files
files.download(csv_file_path)
