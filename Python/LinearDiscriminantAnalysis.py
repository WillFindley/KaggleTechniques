import csv
import sys
import math
import numpy as np
from collections import namedtuple
from collections import Counter


def LDA(trainDirectory,testdirectory):

    with open(trainDirectory) as train, open(testdirectory) as test:
        reader = csv.reader(train,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        memory = [np.array([item for item in row],dtype='i') for row in reader]
        train.close()

        parameters = calculateLinearDiscriminantFunctions(memory)

        reader = csv.reader(test,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        probes = [np.array([item for item in row],dtype='i') for row in reader]
        test.close()

        runLDA(parameters,probes)


def calculateLinearDiscriminantFunctions(memory):

    Params = namedtuple('Params', ['piK','meanK','covariance','probeIndependentTerm'])

    N = len(memory)
    K = 10

    piK = np.array([0 for i in xrange(K)])
    meanK = [np.array([0 for i in xrange(len(memory[0][1:]))]) for i in xrange(K)]
    for row in xrange(N):
        piK[memory[row][0]] += 1
        meanK[memory[row][0]] += memory[row][1:]

    print str(N)

    for whichK in xrange(K):
        meanK[whichK] /= float(piK[whichK])
    piK /= float(N)

    covariance = 0
    for row in xrange(N):
        distanceFromCentroid = memory[row][1:] - meanK[memory[row][0]]
        covariance += np.dot(distanceFromCentroid,distanceFromCentroid)
    covariance /= float(N-K)

    probeIndependentTerm = [math.log(piK[whichK]) - (np.dot(meanK[whichK],meanK[whichK]) / (2*covariance)) for whichK in xrange(K)]

    parameters = Params(piK,meanK,covariance,probeIndependentTerm)
    return parameters


def runLDA(parameters,probes):

    with open('answersLDA', 'w') as answerFile:
        answerFile.write("ImageId,Label")
        for toProbe in xrange(len(probes)):
            answer = maxLDFunction(parameters,probes[toProbe]);
            answerFile.write(str(toProbe+1) + ",\"" + str(answer) + '\"\n')
            print str(answer) + '\t' + str(float(1+toProbe)/len(probes))
        answerFile.flush()
        answerFile.close()


def maxLDFunction(parameters,probe):

    answer = [-1, -inf]
    for whichK in xrange(len(parameters.piK)):
        thisLDFunction = np.dot(probe,parameters.meanK[whichK])/parameters.covariance + parameters.probeIndependentTerm
        if thisLDFunction >= answer[1]:
            answer = [whichK,thisLDFunction]
    return answer[0]

LDA("../Data/train.csv","../Data/test.csv")
