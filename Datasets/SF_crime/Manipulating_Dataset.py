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

df['Datetime'] = pd.to_datetime(df['Date'] + ' ' + df['Time'], format='%m/%d/%Y %H:%M') 

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
