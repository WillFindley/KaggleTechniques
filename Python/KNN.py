import csv
import random
import numpy as np

def KNN(dataDirectory):
    with open(dataDirectory) as f:
        reader = csv.reader(f,csv.QUOTE_NONNUMERIC)
        next(reader) # skip the labels line
        memory = [np.array([item for item in row],dtype='i') for row in reader]
        N = stochasticGradientDescent(memory)

def stochasticGradientDescent(memory):
    v = 0;
    N = 1;
    momentum = 0.5;
    learningRate = len(memory)/10;
    while True:
        newV = int(round(momentum*v - (1-momentum)*learningRate*(leaveOneOutCrossValidation(memory,N+1) - leaveOneOutCrossValidation(memory,N))))
        if newV == 0:
            break
        
        N = N + newV


def leaveOneOutCrossValidation(memory,N):
        for whichOut in xrange(len(memory)):
            leftOut = np.array(memory[whichOut],dtype='i')
            for row in xrange(3):
                if row != whichOut:
                    print np.linalg.norm(toCheck[1:] - memory[row][1:])

KNN("../Data/train.csv")
