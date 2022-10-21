# -*- coding: utf-8 -*-
"""
Created on Fri Oct 21 20:44:19 2022

@author: 47483
"""

# Importing the library
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('Customer_Data.csv')

#Matrix of features (Independent Variables) 
X = dataset.iloc[:, [2, 3]].values

#Vector of the dependent variable
y = dataset.iloc[:, -1].values

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

# Training the K-NN model on the Training set
from sklearn.neighbors import KNeighborsClassifier
""" Defining the model/ classifier as k-nearest neighbors vote and 
Specify number neighbors = 5 , Metric for distance computation is “minkowski”
for  Euclidean distance when p = 2. """
classifier_model = KNeighborsClassifier(n_neighbors = 5, metric = 'minkowski', p = 2)
#fit = Capturing the patterns from the provided data
classifier_model.fit(X_train, y_train)


# Predicting the Test set results
predicted_y = classifier_model.predict(X_test)

# Making the Confusion Matrix for visualization of the performance 
#Determining how accurate the model's predictions are.
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, predicted_y)
print(cm)