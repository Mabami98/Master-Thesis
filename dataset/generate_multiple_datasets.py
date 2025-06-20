import time
import numpy as np
import pickle

offenseMap = {'THEFT': 0, 'BATTERY': 1, 'ASSAULT': 2, 'CRIMINAL DAMAGE': 3}
offenseSet = set()
latSet = set()
lonSet = set()
timeSet = set()
data = []
with open('/Users/martin/Desktop/STHSL/Datasets/CHI_crime/CHI_Crime.csv', 'r') as fs:
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


km_sizes = [1, 2, 3, 4, 5]

minLat = min(latSet)
minLon = min(lonSet)
maxLat = max(latSet)
maxLon = max(lonSet)

for km in km_sizes:
    latDiv = 111 / km
    lonDiv = 84 / km
    latNum = int((maxLat - minLat) * latDiv) + 1
    lonNum = int((maxLon - minLon) * lonDiv) + 1

    trnTensor = np.zeros((latNum, lonNum, 366+365-92-30, len(offenseSet)))
    valTensor = np.zeros((latNum, lonNum, 30, len(offenseSet)))
    tstTensor = np.zeros((latNum, lonNum, 92, len(offenseSet)))

    for tup in data:
        temT = time.localtime(tup['time'])

        if temT.tm_year == 2016 or (temT.tm_year == 2017 and temT.tm_mon < 9):
            day = temT.tm_yday + (0 if temT.tm_year == 2016 else 366) - 1
            tensor = trnTensor
        elif temT.tm_year == 2017 and temT.tm_mon == 9:
            day = temT.tm_mday - 1
            tensor = valTensor
        elif temT.tm_year == 2017 and temT.tm_mon > 9:
            day = temT.tm_yday - (365 - 92) - 1
            tensor = tstTensor
        else:
            continue

        row = int((tup['lat'] - minLat) * latDiv)
        col = int((tup['lon'] - minLon) * lonDiv)
        offense = tup['offense']
        tensor[row][col][day][offense] += 1

    names = [f'trn_{km}km.pkl', f'val_{km}km.pkl', f'tst_{km}km.pkl']
    tensors = [trnTensor, valTensor, tstTensor]

    for name, tensor in zip(names, tensors):
        with open(f'{name}', 'wb') as fs:
            pickle.dump(tensor, fs)
