import pandas as pd
import numpy as np
import math
import random

# This function calculates mean and amplitude of the given weight matrix
def calc_mean_amplitude(random_weights):
    mean_list = []
    amplitude_list = []
    
    for i in range(0, 3, 1):
        sum_weights = 0.0
        mean = [0.0, 0.0]
        for j in range(0, len(data_array), 1):
            mean = mean + data_array[j]*random_weights[j][i]
            sum_weights = sum_weights+random_weights[j][i]
        mean_list.append(mean/sum_weights)
        amplitude_list.append(sum_weights/150.0)
    return mean_list, amplitude_list

# This function calculates mean matrices (X-mean)


def calculate_meanmatrices(mean_list):
    mean_matrices = []
    for i in range(0, 3, 1):
        data_mean_array = df.subtract(mean_list[i]).to_numpy()
        mean_matrices.append(data_mean_array)
    return mean_matrices


#  This function calculatescovariant matrices

def covariant(mean_matrices):
    covariant_list = []
    for i in range(len(mean_matrices)):
        covariant_matrix = np.matrix([[0.0, 0.0], [0.0, 0.0]])
        weights = 0.0
        for j in range(0, 150, 1):
            covariant_matrix = covariant_matrix + \
                (random_weights[j][i] * np.matrix(mean_matrices[i]
                                                  [j]).T * np.matrix(mean_matrices[i][j]))
            weights = weights+random_weights[j][i]
        covariant_list.append(covariant_matrix/weights)

    return covariant_list


#  This function calculates the gaussian 

def guassian(x, mean, covariantmat):
    det = np.linalg.det(covariantmat)
    inv = np.linalg.inv(covariantmat)
    epower = math.exp((-0.5)*np.matrix(x)*np.matrix(inv)*np.matrix(x).T)
    return (((det)**(-0.5))*(epower))/(2*math.pi)

# Finding weights



# this functins takes in the random weights and generates mean, amplitude.
# after which it genrates new weights and return all of it 
def runEMAlgo(random_weights):

    mean_list, amplitude_list = calc_mean_amplitude(random_weights)
    mean_matrices = calculate_meanmatrices(mean_list)
    covariant_list = covariant(mean_matrices)
    act_weight_list = []
    convergence_list = []
    for i in range(150):
        weight_list = []
        sum_weight = 0.0
        for c in range(3):
            weight_list.append(
                amplitude_list[c]*guassian(mean_matrices[c][i], mean_list[c], covariant_list[c]))
            sum_weight = sum_weight + weight_list[c]
        convergence_list.append(weight_list)
        weight_list = weight_list/sum_weight
        act_weight_list.append(weight_list)

    return mean_list, amplitude_list, covariant_list, act_weight_list, convergence_list

# Convergence Criteria


def find_convergence(convergence_list):
    total = 0
    for i in range(150):
        total = total + \
            math.log(convergence_list[i][0] +
                     convergence_list[i][1]+convergence_list[i][2])
    return total

def printOutput(result):
    for i in range(3):
        print("\n\n***Cluster "+str(i+1)+"***")
        for j in range(3):
            if(j==0):print( "Mean: ", result[j][i])
            if(j==1):print( "Amplitude: ", result[j][i])
            if(j==2):print( "Co variance Matrix \n", result[j][i])


if __name__ == "__main__":
    result = []
    random_weights = []
    count = 0
    curr_convergence = 0
    pre_convergence = 0

    # read matrix from the file and store it in a numpy array for faster computation  
    df = pd.read_csv('clusters.txt', header=None)
    data_array = df.to_numpy()
 
    # assigning random weights and normalising so that sum of each row comes to 1
    random_weights = np.random.rand(150,3)
    random_weights=random_weights/random_weights.sum(axis=1,keepdims=1)
    

    # recursively call function unitill they converge
    while (True):
        result = runEMAlgo(random_weights)
        random_weights = result[3]
        pre_convergence = curr_convergence
        curr_convergence = find_convergence(result[4])
        if count != 0:
            if curr_convergence - pre_convergence < 0.005:
                break
        count = count + 1
    # fuction to print the result
    printOutput(result)


    


