#-------------------------------------------------------------------------
# AUTHOR: Qinyi Wang
# FILENAME: clustering.p
# SPECIFICATION: determine which k value maximizes Silhouette coefficient
# FOR: CS 4210- Assignment #5
# TIME SPENT: 1 hr
#-----------------------------------------------------------*/

# importing some Python libraries
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score
from sklearn import metrics

df = pd.read_csv('training_data.csv', sep=',', header=None) # reading the data by using Pandas library

# assign your training data to X_training feature matrix
X_training = df

maxCoef = 0
bestK = 0
# run kmeans testing different k values from 2 until 20 clusters
for k in range (2, 21):
     kmeans = KMeans(n_clusters = k, random_state=0)
     kmeans.fit(X_training)

     # for each k, calculate the silhouette_coefficient by using: silhouette_score(X_training, kmeans.labels_)
     silhouette_coefficient = silhouette_score(X_training, kmeans.labels_)

     # find which k maximizes the silhouette_coefficient
     if silhouette_coefficient > maxCoef:
          maxCoef = silhouette_coefficient
          bestK = k

     #plot the value of the silhouette_coefficient for each k value of kmeans so that we can see the best k
     plt.scatter(k, silhouette_coefficient)
     
plt.show()

# reading the test data (clusters) by using Pandas library
df = pd.read_csv('testing_data.csv', sep=',', header=None)

# assign your data labels to vector labels (you might need to reshape the row vector to a column vector)
labels = np.array(df.values).reshape(1, len(df))[0]

# Calculate and print the Homogeneity of this kmeans clustering
print("K-Means Homogeneity Score = " + metrics.homogeneity_score(labels, kmeans.labels_).__str__())
