import time
import numpy as np
import pickle

offenseMap = {
    'Violent Crimes': 0, 
    'Property Crimes': 1, 
    'Miscellaneous': 2, 
    'Administrative and Other Offenses': 3, 
    'Drug and Alcohol Related': 4,
    'Non-Criminal and Special Cases': 5, 
    'Public Order Crimes': 6,
    'White Collar Crimes': 7,
    'ROBBERY': 8,  # Example of adding missing offense types
    'VEHICLE THEFT': 9  # Example of adding missing offense types
    # Add other offenses as needed
}

offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []

# Load and parse the dataset
with open('/teamspace/studios/this_studio/STHSL/Datasets/SF_crime/SFCrime.csv', 'r') as fs:
    fs.readline()  # Skip header
    for line in fs:
        arr = line.strip().split(',')
        print(arr)

        timeArray = time.strptime(arr[0], '%d/%m/%Y %H:%M')  # Adjusted the format
        timestamp = time.mktime(timeArray)
        if arr[1] in offenseMap:
            offense = offenseMap[arr[1]]
        else:
            print(f"Offense {arr[1]} not found in offenseMap")
            continue
        lat = float(arr[2])
        lon = float(arr[3])

        latSet.add(lat)
        lonSet.add(lon)
        timeSet.add(timestamp)
        offenseSet.add(offense)

        data.append({
            'time': timestamp,
            'offense': offense,
            'lat': lat,
            'lon': lon
        })

minTime = min(timeSet)
maxTime = max(timeSet)
minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)
latDiv = 111 / 3  # 1 degree of latitude is approximately 111 km
lonDiv = 84 / 3  # 1 degree of longitude varies, but let's use an average value

latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1

# Define the size of the tensors
trnTensor = np.zeros((latNum, lonNum, 2*365 + 181, len(offenseMap)))  # 2.5 years for training
valTensor = np.zeros((latNum, lonNum, 184, len(offenseMap)))  # 0.5 year for validation
tstTensor = np.zeros((latNum, lonNum, 365, len(offenseMap)))  # 1 year for testing

# Populate tensors
for i in range(len(data)):
    tup = data[i]
    temT = time.localtime(tup['time'])
    
    if (temT.tm_year == 2015) or (temT.tm_year == 2016) or (temT.tm_year == 2017 and temT.tm_mon < 7):
        day = temT.tm_yday + (0 if temT.tm_year == 2015 else (365 if temT.tm_year == 2016 else 730)) - 1
        tensor = trnTensor
    elif (temT.tm_year == 2017 and temT.tm_mon >= 7) and (temT.tm_year < 2018):
        day = temT.tm_yday - 182 + (365 if temT.tm_mon < 7 else 0) - 1
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

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]

for i in range(len(names)):
    with open('/teamspace/studios/this_studio/STHSL/Datasets/SF_crime/' + names[i], 'wb') as fs:
        pickle.dump(tensors[i], fs)
