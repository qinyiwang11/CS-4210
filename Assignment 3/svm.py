#-------------------------------------------------------------------------
# AUTHOR: Qinyi Wang
# FILENAME: svm.py
# SPECIFICATION: Simulate a grid search to find which combination of four SVM hyperparameters has the best prediction performance
# FOR: CS 4210- Assignment #3
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn import svm
import numpy as np
import pandas as pd
import csv

accuracy = 0
highestAcc = 0

# defining the hyperparameter values
c = [1, 5, 10, 100]
degree = [1, 2, 3]
kernel = ["linear", "poly", "rbf"]
decision_function_shape = ["ovo", "ovr"]

df = pd.read_csv('optdigits.tra', sep=',', header=None) # reading the training data by using Pandas library

x_training = np.array(df.values)[:,:64] # getting the first 64 fields to create the feature training data and convert them to NumPy array
y_training = np.array(df.values)[:,-1] # getting the last field to create the class training data and convert them to NumPy array

df = pd.read_csv('optdigits.tes', sep=',', header=None) # reading the training data by using Pandas library

x_test = np.array(df.values)[:,:64] # getting the first 64 fields to create the feature testing data and convert them to NumPy array
y_test = np.array(df.values)[:,-1] # getting the last field to create the class testing data and convert them to NumPy array

# created 4 nested for loops that will iterate through the values of c, degree, kernel, and decision_function_shape
for cVal in c:
    for dVal in degree:
        for kVal in kernel:
            for sVal in decision_function_shape:
                # Create an SVM classifier that will test all combinations of c, degree, kernel, and decision_function_shape
                # For instance svm.SVC(c=1, degree=1, kernel="linear", decision_function_shape = "ovo")
                clf = svm.SVC(C = cVal, degree = dVal, kernel = kVal, decision_function_shape = sVal)
                
                # Fit SVM to the training data
                clf.fit(x_training, y_training)

                # make the SVM prediction for each test sample and start computing its accuracy
                count = 0
                for (x_testSample, y_testSample) in zip(x_test, y_test):
                    if clf.predict([x_testSample]) == y_testSample:
                        count += 1
                accuracy = count / len(x_test)

                # check if the calculated accuracy is higher than the previously one calculated. If so, update the highest accuracy and print it together
                # with the SVM hyperparameters. Example: "Highest SVM accuracy so far: 0.92, Parameters: a=1, degree=2, kernel= poly, decision_function_shape = 'ovo'"
                if accuracy > highestAcc:
                    highestAcc = accuracy
                    best = [cVal, dVal, kVal, sVal]
                    print("Highest SVM accuracy so far: {:f}, Parameters: a = {:d}, degree = {:d}, kernel = {:s}, decision_function_shape = {:s}".format(highestAcc, cVal, dVal, kVal, sVal))

#print the final, highest accuracy found together with the SVM hyperparameters
#Example: "Highest SVM accuracy: 0.95, Parameters: a=10, degree=3, kernel= poly, decision_function_shape = 'ovr'"
print("\nHighest SVM accuracy: {:f}, Parameters: a = {:d}, degree = {:d}, kernel = {:s}, decision_function_shape = {:s}".format(highestAcc, best[0], best[1], best[2], best[3]))
