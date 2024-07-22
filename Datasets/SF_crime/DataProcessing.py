import time
import numpy as np
import pickle

# Mapping for offense categories
offenseMap = {'Violent Crimes': 0, 'Property Crimes': 1, 'Miscellaneous': 2, 'Administrative and Other Offenses': 3, 
              'Drug and Alcohol Related': 4, 'Non-Criminal and Special Cases': 5, 'Public Order Crimes': 6, 'White Collar Crimes': 7}

latSet = set()
lonSet = set()
data = []

# Load and parse the dataset
with open('/teamspace/studios/this_studio/STHSL/Datasets/SF_crime/SFCrime.csv', 'r') as fs:
    fs.readline()  # Skip header
    for line in fs:
        arr = line.strip().split(',')
        
        timeArray = time.strptime(arr[0], '%m/%d/%Y %I:%M:%S %p')  # Adjusted the format
        timestamp = time.mktime(timeArray)
        offense = offenseMap[arr[1]]
        lat = float(arr[2])
        lon = float(arr[3])

        latSet.add(lat)
        lonSet.add(lon)

        data.append({
            'time': timestamp,
            'offense': offense,
            'lat': lat,
            'lon': lon
        })

print('Length of data', len(data), '\n')

minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)
latDiv = 111 / 80  # Adjusted to create a larger grid cell
lonDiv = 84 / 64  # Adjusted to create a larger grid cell

latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1

# Initialize tensors
trnTensor = np.zeros((latNum, lonNum, 3*365, len(offenseMap)))  # Training data: 3 years (2015 to mid-2017)
valTensor = np.zeros((latNum, lonNum, 184, len(offenseMap)))  # Validation data: mid-2017 to end of 2017 (184 days)
tstTensor = np.zeros((latNum, lonNum, 365, len(offenseMap)))  # Test data: 1 year initially, will extend if needed

# Populate tensors
for tup in data:
    temT = time.localtime(tup['time'])
    if temT.tm_year == 2015 or (temT.tm_year == 2016) or (temT.tm_year == 2017 and temT.tm_mon < 7):
        day = temT.tm_yday + (0 if temT.tm_year == 2015 else (365 if temT.tm_year == 2016 else 730)) - 1
        tensor = trnTensor
    elif temT.tm_year == 2017 and temT.tm_mon >= 7 and temT.tm_year < 2018:
        day = temT.tm_yday - 182 - 1
        tensor = valTensor
    elif temT.tm_year >= 2018:
        day = temT.tm_yday - 1
        tensor = tstTensor
    else:
        continue
    
    row = int((tup['lat'] - minLat) * latDiv)
    col = int((tup['lon'] - minLon) * lonDiv)
    offense = tup['offense']
    tensor[row][col][day][offense] += 1

# Save tensors to files
names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]
for i in range(len(names)):
    with open('/teamspace/studios/this_studio/STHSL/Datasets/SF_crime/' + names[i], 'wb') as fs:
        pickle.dump(tensors[i], fs)
