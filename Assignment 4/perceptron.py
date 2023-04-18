#-------------------------------------------------------------------------
# AUTHOR: Qinyi Wang
# FILENAME: perceptron.py
# SPECIFICATION: compare the prediction performance of perceptron and MLP classifiers
# FOR: CS 4210- Assignment #4
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

# IMPORTANT NOTE: YOU HAVE TO WORK WITH THE PYTHON LIBRARIES numpy AND pandas to complete this code.

# importing some Python libraries
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier #pip install scikit-learn==0.18.rc2 if needed
import numpy as np
import pandas as pd

n = [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0]
r = [True, False]

df = pd.read_csv('optdigits.tra', sep=',', header=None) # reading the data by using Pandas library

X_training = np.array(df.values)[:,:64] # getting the first 64 fields to form the feature data for training
y_training = np.array(df.values)[:,-1]  # getting the last field to form the class label for training

df = pd.read_csv('optdigits.tes', sep=',', header=None) # reading the data by using Pandas library

X_test = np.array(df.values)[:,:64]    # getting the first 64 fields to form the feature data for test
y_test = np.array(df.values)[:,-1]     # getting the last field to form the class label for test

perAcc = 0
mlpAcc = 0

for i in n: # iterates over n

    for j in r: # iterates over r

        for a in range(2): # iterates over the algorithms

            # Create a Neural Network classifier
            if a == 0:
                clf = Perceptron(eta0 = i, shuffle = j, max_iter = 1000)
            else:
                clf = MLPClassifier(activation = 'logistic', learning_rate_init = i, hidden_layer_sizes = (25,), shuffle = j, max_iter = 1000)

            # Fit the Neural Network to the training data
            clf.fit(X_training, y_training)

            # make the classifier prediction for each test sample and start computing its accuracy
            #hint: to iterate over two collections simultaneously with zip() Example:

            count = 0
            correct = 0
            wrong = 0
            for (x_testSample, y_testSample) in zip(X_test, y_test):
                prediction = clf.predict([x_testSample])
                count += 1
                if prediction == [y_testSample]:
                    correct += 1
                else:
                    wrong += 1

            accuracy = correct / wrong

            # check if the calculated accuracy is higher than the previously one calculated for each classifier
            # If so, update the highest accuracy and print it together with the network hyperparameters
            output = "Highest "
            if a == 0:
                output += "Perceptron"
            else:
                output += "MLP"
            output += " accuracy so far: " + "{:.3f}".format(accuracy) + ", Parameters: learning rate = " + str(i) + ", shuffle = " + str(j)
            if a == 0 and accuracy >= perAcc:
                perAcc = accuracy
                print(output)
            elif a == 1 and accuracy >= mlpAcc:
                mlpAcc = accuracy
                print(output)