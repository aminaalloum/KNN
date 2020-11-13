#!/usr/bin/env python
# coding: utf-8

# In[5]:


import pandas as pd
import csv
df=pd.read_csv("./Downloads/iris.data.txt")
df.columns=["sepal length", "sepal width", "petal length" ," petal width","species"]
df.head(10)


# In[6]:


import csv
with open('./Downloads/iris.data.txt', 'r') as csvfile:
    lines = csv.reader(csvfile)
    for row in lines:
        print (', '.join(row))


# In[67]:


import csv

import random

def loadDataset(filename, split, trainingSet=[] , testSet=[]):
    with open(filename, 'r') as csvfile:
        lines = csv.reader(csvfile)
        dataset = list(lines)
        for x in range(len(dataset)-1):
            for y in range(4):
                dataset[x][y] = float(dataset[x][y])
                if random.random() < split:
                    trainingSet.append(dataset[x])
                else:
                    testSet.append(dataset[x])

            


# In[68]:


trainingSet=[]

testSet=[]

loadDataset('./Downloads/iris.data.txt', 0.5, trainingSet, testSet)

print ('Train: ' + repr(len(trainingSet)))

print ('Test: ' + repr(len(testSet)) )


# In[69]:


import math
def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)


# In[ ]:





# In[70]:


import operator
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

def getNeighbors(trainingSet, testInstance, k):
    distances = []
    length = len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance, trainingSet[x], length)
        distances.append((trainingSet[x], dist))
        distances.sort(key=operator.itemgetter(1))
        neighbors = []
        for x in range(k):
            neighbors.append(distances[x][0])
            return neighbors
        testInstance = [5, 5, 5]
        k = 1
        neighbors = getNeighbors(trainingSet, testInstance, 1)
        print(neighbors)


# In[71]:


import operator
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
def getResponse(neighbors):
    classVotes = {}
    for x in range(len(neighbors)):
        response = neighbors[x][-1]
        if response in classVotes:
            classVotes[response] += 1
        else:
            classVotes[response] = 1
        sortedVotes = sorted(classVotes.items(), key=operator.itemgetter(1), reverse=True)
    return sortedVotes[0][0]
neighbors = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
response = getResponse(neighbors)
print("response")


# In[72]:


def getAccuracy(testSet, predictions):
    correct =0
    for x in range(len(testSet)-1):
        if testSet[x][-1] is  predictions[x]:
            correct += 1
    return (correct/float(len(testSet))) * 100.0
testSet = [[1,1,1,'a'], [2,2,2,'a'], [3,3,3,'b']]
predictions = ['a', 'a', 'a']
accuracy = getAccuracy(testSet, predictions)
print(accuracy)
    


# In[77]:


trainingSet=[]

testSet=[]

split = 0.67

loadDataset('./Downloads/iris.data.txt', split, trainingSet, testSet)

print ('Train set: ' + repr(len(trainingSet)))

print ('Test set: ' + repr(len(testSet)))


# In[78]:


predictions=[]

k = 3

for x in range(len(testSet)):
    neighbors = getNeighbors(trainingSet, testSet[x], k)
    print("---- neighbors : ",neighbors)
    result = getResponse(neighbors)
    print("------- reult : ", result)
    predictions.append(result)
    print('> predicted=' + repr(result) + ', actual=' + repr(testSet[x][-1]))
accuracy = getAccuracy(testSet, predictions)
print('Accuracy: ' + repr(accuracy) + '%')


# In[79]:


def ManhattanDist(isntance1,instance2,length):
    distance=0
    for x in range (length-1):
        ditance+=abs((instance1[x]-instance2[x]))
    return (distance)


# In[89]:



def GetNeighbors(trainingSet,testInstance,k):
    distance=[]
    length=len(testInstance)-1
    for x in range(len(trainingSet)):
        dist=euclideanDistance(testInstance,trainingSet[x],length)
        distance.append((training[x],dist))
        distance.sort(key=operator.itemgetter(1))
        neighbors=[]
        for x in range (k):
            neightbors.append(distance[x][0])
        return neighbors
    testInstance=[5,5,5]
    k=1
    neighbors=getNeighbors(trainingSet,testInstance,1)
    print(neightbors)


# In[ ]:





# In[ ]:




