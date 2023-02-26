#-------------------------------------------------------------------------
# AUTHOR: Qinyi Wang
# FILENAME: naive_bayes.py
# SPECIFICATION: Read the training set and output the classification of each test instance from the test set
# FOR: CS 4210- Assignment #2
# TIME SPENT: 45 mins
#-----------------------------------------------------------*/

# IMPORTANT NOTE: DO NOT USE ANY ADVANCED PYTHON LIBRARY TO COMPLETE THIS CODE SUCH AS numpy OR pandas. You have to work here only with standard dictionaries, lists, and arrays

# importing some Python libraries
from sklearn.naive_bayes import GaussianNB
import csv

# reading the training data in a csv file
db = []
with open('weather_training.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:   # skipping the header
            db.append(row)

# transform the original training features to numbers and add them to the 4D array X
# transform the original training classes to numbers and add them to the vector Y.
# For instance Sunny = 1, Overcast = 2, Rain = 3, so X = [[3, 1, 1, 2], [1, 3, 2, 2], ...]]
dict = {'Sunny': 1, 'Overcast': 2, 'Rain': 3, 'Hot': 1, 'Mild': 2, 'Cool': 3, 'High': 1, 'Normal': 2, 'Weak': 1, 'Strong': 2, 'Yes': 1, 'No': 2}

X = []
Y = []
for i in range(len(db)):
    X.append([int(dict[db[i][1]]),
              int(dict[db[i][2]]),
              int(dict[db[i][3]]),
              int(dict[db[i][4]])])
    Y.append(int(dict[db[i][5]]))

# fitting the naive bayes to the data
clf = GaussianNB()
clf.fit(X, Y)

# reading the test data in a csv file
testData = []
with open('weather_training.csv') as csvfile:
    reader = csv.reader(csvfile)
    for i, row in enumerate(reader):
        if i > 0:   # skipping the header
            testData.append(row)

dbTest = []
for i in range(len(testData)):
    dbTest.append([int(dict[testData[i][1]]),
                   int(dict[testData[i][2]]),
                   int(dict[testData[i][3]]),
                   int(dict[testData[i][4]])])

# printing the header os the solution
print("{:<8}{:<15}{:<15}{:<12}{:<10}{:<15}{:<10}".format("Day", "Outlook", "Temperature", "Humidity", "Wind", "PlayTennis", "Confidence"))

# use your test samples to make probabilistic predictions. For instance: clf.predict_proba([[3, 1, 2, 1]])[0]
predict = []
for i in range(len(testData)):
    predict.append(clf.predict_proba([dbTest[i]])[0])
    
    if predict[i][0] >= 0.75:
        confidence = predict[i][0]
        resultClass = "Yes"
    else:
        confidence = predict[i][1]
        resultClass = "No"

    if predict[i][0] >= 0.75 or predict[i][1] >= 0.75:        
        print("{:<8}{:<15}{:<15}{:<12}{:<10}{:<15}{:<10}".format(testData[i][0], testData[i][1], testData[i][2], testData[i][3], testData[i][4], resultClass, round(confidence, 2)))