#-------------------------------------------------------------------------
# AUTHOR: Qinyi Wang
# FILENAME: decision_tree_2.py
# SPECIFICATION: Train, test, and output the performace of the 3 models created using each training set on the test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 2 hrs
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn import tree
import csv

dataSets = ['contact_lens_training_1.csv', 'contact_lens_training_2.csv', 'contact_lens_training_3.csv']

for ds in dataSets:
    dbTraining = []
    X = []
    Y = []

    # reading the training data in a csv file
    with open(ds, 'r') as csvfile:
        reader = csv.reader(csvfile)
        for i, row in enumerate(reader):
            if i > 0: #skipping the header
                dbTraining.append(row)
    
    # transform the original categorical training features to numbers and add to the 4D array X. For instance Young = 1, Prepresbyopic = 2, Presbyopic = 3
    dict = {'Young': 1, 'Prepresbyopic': 2, 'Presbyopic': 3, 'Myope': 1, 'Hypermetrope': 2, 'Yes': 1, 'No': 2, 'Normal': 1, 'Reduced': 2}

    for i in range(len(dbTraining)):
        temp = []
        for j in range(0, 4):
            temp.append(int(dict[dbTraining[i][j]]))
        X.append(temp)
    
    # transform the original categorical training classes to numbers and add to the vector Y. For instance Yes = 1, No = 2, so Y = [1, 1, 2, 2, ...]
    for i in range(len(dbTraining)):
        Y.append(int(dict[dbTraining[i][4]]))
    
    # loop your training and test tasks 10 times here
    bestAcc = 1.0
    for i in range (10):
       
       # fitting the decision tree to the data setting max_depth=3
       clf = tree.DecisionTreeClassifier(criterion = 'entropy', max_depth = 3)
       clf = clf.fit(X, Y)

       # read the test data and add this data to dbTest
       dbTest = []
       testData = []
       with open('contact_lens_test.csv', 'r') as csvfile:
           reader = csv.reader(csvfile)
           for i, row in enumerate(reader):
               if i > 0:
                   testData.append(row)
        
       for i in range(len(testData)):
            temp = []
            for j in range(0, 5):
                temp.append(int(dict[testData[i][j]]))
            dbTest.append(temp)
        
       count = 0
       correct = 0
       for data in dbTest:
           # transform the features of the test instances to numbers following the same strategy done during training,
           # and then use the decision tree to make the class prediction. For instance: class_predicted = clf.predict([[3, 1, 2, 1]])[0]
           # where [0] is used to get an integer as the predicted class label so that you can compare it with the true label
           class_predicted = clf.predict([[data[0], data[1], data[2], data[3]]])[0]
           
           # compare the prediction with the true label (located at data[4]) of the test instance to start calculating the accuracy            
           if class_predicted == data[4]:
               correct = correct + 1
           count = count + 1
        
        # find the average of this model during the 10 runs (training and test set)
       currAccuracy = correct / count
       if(currAccuracy < bestAcc):
           bestAcc = currAccuracy

    # print the average accuracy of this model during the 10 runs (training and test set).
    # your output should be something like that: final accuracy when training on contact_lens_training_1.csv: 0.2
    print("Final accuracy when training on " + ds + ": " + str(bestAcc))