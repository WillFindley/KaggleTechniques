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
        # make into arrays for easier matrix math
        memory = [np.array([item for item in row],dtype='i') for row in reader]
        train.close()

        # determine the correct parameters for the gaussian distributions from the training set data
        parameters = calculateLinearDiscriminantFunctions(memory)

        reader = csv.reader(test,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        # make into arrays for easier matrix math
        probes = [np.array([item for item in row],dtype='i') for row in reader]
        test.close()

        # classify based on the training set parameters
        runLDA(parameters,probes)


def calculateLinearDiscriminantFunctions(memory):

    # seems to be the best way to store and return this information
    Params = namedtuple('Params', ['piK','meanK','covariance','probeIndependentTerm'])
    # pi0 is the bayesian prior for each category
    # meanK is the average value location in pixel space for each category
    # covariance is the common covariance for all categories' gaussian distributions
    # probIndependentTerm is the non-test-probe-dependent value to be added to the linear discriminant function

    N = len(memory)     # number of training images
    K = 10              # number of categories

    # initialize to 0 so that values can be added in
    piK = np.array([0.0 for i in xrange(K)])
    meanK = [np.array([0.0 for i in xrange(len(memory[0][1:]))]) for i in xrange(K)]
    for row in xrange(N):
        # count the number of images for each category
        piK[memory[row][0]] += 1
        # add all of the images for each category
        meanK[memory[row][0]] += memory[row][1:]

    for whichK in xrange(K):
        # calculate the average category image
        meanK[whichK] /= float(piK[whichK])
    # estimate the bayesian prior for all of the categories
    piK /= float(N)

    covariance = 0.0
    for row in xrange(N):
        # calculate the squared distance from the category centroid and add them together for all images
        distanceFromCentroid = memory[row][1:] - meanK[memory[row][0]]
        covariance += np.dot(distanceFromCentroid,distanceFromCentroid)
    # unbiased average for the covariance
    covariance /= float(N-K)

    # with all parameters calculated, go ahead and calculate the linear discriminant function terms that do not rely on the test probe image
    probeIndependentTerm = [math.log(piK[whichK]) - (np.dot(meanK[whichK],meanK[whichK]) / (2*covariance)) for whichK in xrange(K)]

    # make the named tuple for storage of parameters and return it
    parameters = Params(piK,meanK,covariance,probeIndependentTerm)
    return parameters


def runLDA(parameters,probes):

    with open('answersLDA.csv', 'w') as answerFile:
        answerFile.write("ImageId,Label\n")
        for toProbe in xrange(len(probes)):
            # for each test image probe, find the maximum likelihood linear discriminant function's category
            answer = maxLDFunction(parameters,probes[toProbe]);
            answerFile.write(str(toProbe+1) + ",\"" + str(answer) + '\"\n')
            print str(answer) + '\t' + str(float(1+toProbe)/len(probes))
        answerFile.flush()
        answerFile.close()


def maxLDFunction(parameters,probe):

    # spot 0 is the category and spot 1 is the linear discriminant function likelihood
    # initialize to a small amount so that all categories are higher and will replace it
    answer = [-1, float('-inf')]
    for whichK in xrange(len(parameters.piK)):
        # calculate the linear discriminant function likelihood
        thisLDFunction = np.dot(probe,parameters.meanK[whichK])/parameters.covariance + parameters.probeIndependentTerm[whichK]
        # if this category has a higher likelihood than the current best, update the best answer
        if thisLDFunction >= answer[1]:
            answer = [whichK,thisLDFunction]
    # return the category of the maximum likelihood linear discriminant function
    return answer[0]

LDA("../Data/train.csv","../Data/test.csv")
