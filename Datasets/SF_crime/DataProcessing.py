import time
import numpy as np
import pickle

offenseMap = {'Violent Crimes':0, 'Property Crimes':1, 'Miscellaneous':2, 'Administrative and Other Offenses':3, 'Drug and Alcohol Related':4,
 'Non-Criminal and Special Cases':5, 'Public Order Crimes':6,'White Collar Crimes':7}
offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []
with open('Datasets/SF_crime/SFCrime.csv', 'r') as fs:
	fs.readline()
	for line in fs:
		arr = line.strip().split(',')
		print(arr)

		timeArray = time.strptime(arr[0], '%m/%d/%Y %I:%M:%S %p')
		timestamp = time.mktime(timeArray)
		offense = offenseMap[arr[1]]
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
print('Length of data', len(data), '\n')
print('Offense:', offenseSet, '\n')
print('Latitude:', min(latSet), max(latSet))
print('Longtitude:', min(lonSet), max(lonSet))
print('Latitude:', min(latSet), max(latSet), (max(latSet) - min(latSet)) / (1 / 111), '\n')
print('Longtitude:', min(lonSet), max(lonSet), (max(lonSet) - min(lonSet)) / (1 / 84), '\n')
print('Time:')
minTime = min(timeSet)
maxTime = max(timeSet)
print(time.localtime(minTime))
print(time.localtime(maxTime))

minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)
latDiv = 111 / 3 #1
lonDiv = 84 / 3 #1
latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1
trnTensor = np.zeros((latNum, lonNum, 366+365-92-30, len(offenseSet)))
valTensor = np.zeros((latNum, lonNum, 30, len(offenseSet)))
tstTensor = np.zeros((latNum, lonNum, 92, len(offenseSet)))
for i in range(len(data)):
    tup = data[i]
    temT = time.localtime(tup['time'])
    
    # Determine if the data point should go to training or testing tensor
    if 2015 <= temT.tm_year < 2018:
        # Calculate day index for training data
        day = (temT.tm_yday - 1) + (temT.tm_year - 2015) * 365
        tensor = trnTensor
    elif temT.tm_year >= 2018:
        # Calculate day index for testing data
        day = (temT.tm_yday - 1) + (temT.tm_year - 2018) * 365
        tensor = tstTensor
    else:
        continue  # Skip the iteration if the year is not in the desired range
    
    # Calculate the row and column indices based on latitude and longitude
    row = int((tup['lat'] - minLat) * latDiv)
    col = int((tup['lon'] - minLon) * lonDiv)
    
    # Get the offense type
    offense = tup['offense']
    
    # Increment the count in the appropriate tensor
    tensor[row][col][day][offense] += 1

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]
for i in range(len(names)):
	with open('Datasets/CHI_crime/' + names[i], 'wb') as fs:
		pickle.dump(tensors[i], fs)
