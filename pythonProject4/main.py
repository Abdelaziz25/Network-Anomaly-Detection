from pandas.core.arrays.timedeltas import precision_from_unit
import pandas as pd
from sklearn.preprocessing import LabelEncoder
# upload data
df_Data = pd.read_csv('kddcup.data.gz')
df_test = pd.read_csv('corrected.gz')

le = LabelEncoder()

#Handling categorical features to numerical tcp column in train data and udp column in test data 
all_categories = set(df_Data['tcp'].unique()).union(set(df_test['udp'].unique()))
le.fit(list(all_categories))
df_Data['tcp'] = le.transform(df_Data['tcp'])
df_test['udp'] = le.transform(df_test['udp'])



#Handling categorical features to numerical http column in train data and private column in test data 
all_categories = set(df_Data['http'].unique()).union(set(df_test['private'].unique()))
le.fit(list(all_categories))
df_Data['http'] = le.transform(df_Data['http'])
df_test['private'] = le.transform(df_test['private'])


#Handling categorical features to numerical SF column in train data and SF column in test data 
all_categories = set(df_Data['SF'].unique()).union(set(df_test['SF'].unique()))
le.fit(list(all_categories))
df_Data['SF'] = le.transform(df_Data['SF'])
df_test['SF'] = le.transform(df_test['SF'])


#Handling categorical features to numerical normal column in train data and normal column in test data 
all_categories = set(df_Data['normal.'].unique()).union(set(df_test['normal.'].unique()))
le.fit(list(all_categories))
df_Data['normal.'] = le.transform(df_Data['normal.'])
df_test['normal.'] = le.transform(df_test['normal.'])



train_data = df_Data
test_data = df_test

# getting labels of train data and test. 
y_train = train_data['normal.'].values
y_test = test_data['normal.'].values

import numpy as np

def kmeans(X, k, epsilon):
    n_samples, n_features = X.shape
    # Randomly choose k data points as the initial centroids
    centroids = X[np.random.choice(n_samples, k, replace=False)]
    distances = np.zeros((n_samples, k))
    labels = np.zeros(n_samples)
    old_centroids = np.zeros((k, n_features))

   # Continue until the centroids don't change by more than epsilon
    while np.linalg.norm(centroids - old_centroids) > epsilon:
        old_centroids = centroids.copy()
         # Calculate the Euclidean distances from each sample to each centroid
        for i in range(k):
            distances[:, i] = np.linalg.norm(X - centroids[i], axis=1)
         # Assign each sample to the nearest centroid
        labels = np.argmin(distances, axis=1)
        # Update the centroids to be the mean of the samples assigned to them
        for i in range(k):
            centroids[i] = np.mean(X[labels == i], axis=0)

    return labels, centroids



#Evalution
from sklearn.metrics import precision_score, recall_score, f1_score
from scipy.stats import entropy

precision = precision_score(y_train, labels)
recall = recall_score(y_train, labels)
f1 = f1_score(y_train, labels)
ce = entropy(train_data.drop(["normal."],axis=1), labels)