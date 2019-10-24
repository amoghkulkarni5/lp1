import sys
import csv
import operator
from datetime import datetime
import random,time

import math


def loadCSV(filename):
    lines = csv.reader(open(filename, "rt"))
    dataset = list(lines)
    #the original file contains 3,74,116 entries. If all entries are considered then program will take lot of time
    #hence we will do a trick and select limited entries , number of entries upto 2000 is feasible
    #You may change both start and end position in the line below but of course start<end.
    return dataset[1:2000]

def splitDataset(dataset,splitratio):
	trainSize = int(len(dataset)*splitratio)
	trainSet = []
	copy = list(dataset)
	while (len(trainSet)<trainSize):
		index = random.randrange(len(copy))
		trainSet.append(copy.pop(index))
	return [trainSet,copy]


def clean_and_transform_dataset(dataset):
    '''
"Duration","Start date","End date","Start station number","Start station","End station number","End station","Bike number","Member type"
679,"2018-05-01 00:00:00","2018-05-01 00:11:19",31302,"Wisconsin Ave & Newark St NW",31307,"3000 Connecticut Ave NW / National Zoo","W22771","Member"
578,"2018-05-01 00:00:20","2018-05-01 00:09:59",31232,"7th & F St NW / National Portrait Gallery",31609,"Maine Ave & 7th St SW","W21320","Casual"
580,"2018-05-01 00:00:28","2018-05-01 00:10:09",31232,"7th & F St NW / National Portrait Gallery",31609,"Maine Ave & 7th St SW","W20863","Casual"
606,"2018-05-01 00:01:22","2018-05-01 00:11:29",31104,"Adams Mill & Columbia Rd NW",31509,"New Jersey Ave & R St NW","W00822","Member"
582,"2018-05-01 00:04:52","2018-05-01 00:14:34",31129,"15th St & Pennsylvania Ave NW/Pershing Park",31118,"3rd & Elm St NW","W21846","Member"
175,"2018-05-01 00:06:39","2018-05-01 00:09:34",31104,"Adams Mill & Columbia Rd NW",31117,"15th & Euclid St  NW","W22986","Member"
'''
    # column1 contains duration field which will be untouched
    # column 2 and 3 contains startTime and endTime respectively which will be used to calculate duration of trip/time of trip
    transformed_dataset = []
    for i in range(1, len(dataset)):
        new_row = []
        row = dataset[i]
        new_row.append(row[0])  # add first field as it is
        startTime = datetime.strptime(row[1], '%Y-%m-%d %H:%M:%S')
        endTime = datetime.strptime(row[2], '%Y-%m-%d %H:%M:%S')
        tripTime = endTime.timestamp() - startTime.timestamp()
        new_row.append(tripTime)
        new_row.append(row[3]) #instead of adding source name add uique identifier
        new_row.append(row[5])  #instead of adding destination name add uique identifier
        new_row.append(row[8]) #main thing class
        transformed_dataset.append(new_row)
    return transformed_dataset

def euclideanDistance(instance1,instance2,length):
    distance = 0.0
    for i in range(length):
        #print(instance1[i])
        distance += pow((float(instance1[i]) - float(instance2[i])), 2)
    return math.sqrt(distance)

def getNeighbours(trainingSet,testInstance,K):
    distances = []
    length = len(trainingSet[0])-1
    for x in range(len(trainingSet)):
        dist = euclideanDistance(trainingSet[x],testInstance,length)
        distances.append((trainingSet[x],dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(K):
        neighbors.append(distances[x][0])
    return neighbors

def getAnswer(neighbours):
    classVotes = {}
    for x in range(len(neighbours)):
        response = neighbours[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
    sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]

def getAccuracy(testData,predictions):
    correct = 0
    for i in range(len(testData)):
        if predictions[i] == testData[i][-1]:
            correct+=1
    accuracy = correct/len(testData)*100;
    return accuracy


if __name__ == "__main__":
    filename = sys.argv[1]
    dataset = loadCSV(filename)
    dataset = clean_and_transform_dataset(dataset)

    splitRatio = 0.65

    trainingSet,testSet = splitDataset(dataset,splitRatio)

    predictions = []


    start = time.time()

    for item in testSet:
        neighbours = getNeighbours(trainingSet,item,4)
        answer = getAnswer(neighbours)
        predictions.append(answer)

    accuracy = getAccuracy(testSet,predictions)
    print('Accuracy = ',accuracy)

    end = time.time()

    print('program took = ',end-start,' seconds time')



