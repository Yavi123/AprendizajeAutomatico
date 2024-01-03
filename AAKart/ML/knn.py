import numpy as np
import math
import sys

#data [n][m], dataOutput [n], input [m]
def predict(data, dataOutput, input):

    if data.shape[0] != dataOutput.shape[0]:
        return 0

    if data.shape[1] != input.shape[0]:
        return 0

    difference = sys.float_info.max
    output = 0

    for i in range(len(dataOutput)):
        currentDiff = euclideanDist(data[i], input)

        if currentDiff < difference:
            output = dataOutput[i]
            difference = currentDiff

    return output

def euclideanDist(a, b):
    return math.sqrt(np.sum((a - b) ** 2))
