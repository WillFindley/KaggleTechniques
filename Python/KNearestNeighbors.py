import csv
import sys
import math
import numpy as np
from collections import Counter

def KNN(trainDirectory,testdirectory):

    with open(trainDirectory) as train, open(testdirectory) as test:
        reader = csv.reader(train,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        # convert this to a numpy array for convenient matrix math
        memory = [np.array([item for item in row],dtype='i') for row in reader]
        train.close()

        # this choice has roughly as many effective parameters as image dimensions * categories
        k = len(memory)/(10*len(memory[0]));
        k = k - (k % 2) + 1;

        reader = csv.reader(test,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        # convert for convenient matrix math, as above
        probes = [np.array([item for item in row],dtype='i') for row in reader]
        test.close()

        # run the k-nearest neighbors
        runKNN(memory,k,probes)


def runKNN(memory,k,probes):

    with open('answers', 'w') as answerFile:
        answerFile.write("ImageId,Label")
        for toProbe in xrange(len(probes)):
            # get an answer for each of the test images
            answer = knnVote(memory,k,probes[toProbe]);
            answerFile.write(str(toProbe+1) + ",\"" + str(answer) + '\"\n')
            print str(answer) + '\t' + str(float(1+toProbe)/len(probes))
        answerFile.flush()
        answerFile.close()


# this should probably be implemented in a more vectorized manner, but I wanted to try out python objects and heaps
def knnVote(memory,k,probe):

    # a max heap on distance for the k nearest neighbors will automatically pop off the k+1 closest for each comparison in memory 
    q = MaxHeap(k)
    for row in xrange(len(memory)):
        # how far away is this memory location from the test probe
        distance = np.linalg.norm(probe - memory[row][1:])
        # check if it can go in the heap and pop for the most distant in the heap if it can
        q.pushPop([distance,memory[row][0]])
    # make a list of all of the k labels and return the most common one
    return Counter([item[1] for item in q.data]).most_common(1)[0][0]

# max heap object, pretty standard
class MaxHeap:
    def __init__(self,k):
        # initialize high so closer values will be placed in
        self.data = [[sys.maxint, 15] for i in xrange(k)]
    def pushPop(self,entry):
        # only add to the heap if this distance is smaller than the current largest (root) in the heap
        if (entry[0] < self.data[0][0]):
            self.addToHeap(entry,1)
    def addToHeap(self,entry,vertex):
        # standard heap updating
        if 2*vertex-1 < len(self.data) and self.data[2*vertex-1][0] > entry[0]:
            self.data[vertex-1] = self.data[2*vertex-1]
            self.addToHeap(entry,2*vertex)
        elif 2*vertex < len(self.data) and self.data[2*vertex][0] > entry[0]:
            self.data[vertex-1] = self.data[2*vertex]
            self.addToHeap(entry,2*vertex+1)
        else:
            self.data[vertex-1] = entry

KNN("../Data/train.csv","../Data/test.csv")
