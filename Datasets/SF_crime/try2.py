import time
import numpy as np
import pickle

offenseMap = {'Violent Crimes': 0, 'Property Crimes': 1, 'Miscellaneous': 2, 'Administrative and Other Offenses': 3, 'Drug and Alcohol Related': 4, 'Non-Criminal and Special Cases': 5, 'Public Order Crimes': 6, 'White Collar Crimes': 7}
offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []
with open('Datasets/SF_crime/SFCrime.csv', 'r') as fs:
	fs.readline()
	for line in fs:
		arr = line.strip().split(',')
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

minTime = min(timeSet)
maxTime = max(timeSet)
minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)
latDiv = 111 / 80 #1
lonDiv = 84 / 64 #1
latNum = int((maxLat - minLat) * latDiv) + 1
lonNum = int((maxLon - minLon) * lonDiv) + 1
trnTensor = np.zeros((latNum, lonNum, 366+365-92-30, len(offenseSet)))
valTensor = np.zeros((latNum, lonNum, 30, len(offenseSet)))
tstTensor = np.zeros((latNum, lonNum, 92, len(offenseSet)))
for i in range(len(data)):
  day=0
  tup = data[i]
  temT = time.localtime(tup['time'])
  if temT.tm_year == 2016 or temT.tm_year == 2017 and temT.tm_mon < 7:
    day = temT.tm_yday + (0 if temT.tm_year == 2016 else 366) - 2
    tensor = trnTensor
  elif temT.tm_year == 2017 and temT.tm_mon >= 7:
    day = temT.tm_mday - 1
    tensor = valTensor
  elif temT.tm_year == 2018 :
    day = temT.tm_yday - (365 - 92) - 2
    tensor = tstTensor
  row = int((tup['lat'] - minLat) * latDiv)
  col = int((tup['lon'] - minLon) * lonDiv)
  offense = tup['offense']
  print(f"Type of 'tensor': {type(tensor)}") 
  print(f"row: {row}, col: {col}, day: {day}, offense: {offense}")
  print(tensor.shape)
  tensor[row][col][day][offense] += 1

names = ['trn.pkl', 'val.pkl', 'tst.pkl']
tensors = [trnTensor, valTensor, tstTensor]
for i in range(len(names)):
	with open('Datasets/SF_crime/' + names[i], 'wb') as fs:
		pickle.dump(tensors[i], fs)
