#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 15:45:27 2018

@author: harkirat
"""

#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import metrics

#load dataset
df = pd.read_csv('ChurnData.csv')

#selection of features
data = df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip',   'callcard', 'wireless']]
labels = df[['churn']]

np.random.seed(1)
#split of dataset
mask = np.random.rand(len(data))<=0.8
train_data_X = data[mask]
test_data_X = data[~mask]
train_data_Y = labels[mask]
test_data_Y = labels[~mask]

#conversion of data from dataframe to matrix
train_data_X = train_data_X.values.T
train_data_Y = train_data_Y.values.T
test_data_X = test_data_X.values.T
test_data_Y = test_data_Y.values.T

#normalisation of data
mean = np.mean(train_data_X,axis=1).reshape(train_data_X.shape[0],1)
std = np.std(train_data_X,axis=1).reshape(train_data_X.shape[0],1)
train_data_X = (train_data_X-mean)/std
test_data_X = (test_data_X-mean)/std

#initialisation of weights
weights = np.zeros((train_data_X.shape[0],1))
bias = 0

#defining sigmoid function
def sigmoid(x):
    '''
    This function is used sigmoid value of input x
    '''
    z = 1/(1+np.exp(-x))
    return z

def costFunction(data,labels,weights,bias):
    '''
    This function is used for calculating loss incurred due to model
    Input is data, labels and weights along with bias
    Return cost/loss
    '''
    #no of examples
    m = data.shape[1]
    predictions = sigmoid(np.dot(weights.T,data)+bias)
    cost = (-1/m)*(np.sum(labels*np.log(predictions)+(1-labels)*np.log(1-predictions)))
    return cost

def learningModel(data,labels,weights,bias,learning_rate=0.001,num_iterations=20000):
    '''
    learningModel is used to learn parameters related to model
    Input is data to be trained on, corrected labels to compare, weights and bias, learning rate and number of iterations for training\
    Return updated parameters of the model
    '''
    m = data.shape[1]
    cost = []
    for i in range(num_iterations):
        predictions = sigmoid(np.dot(weights.T,data)+bias)
        dz = (1/m)*(predictions - labels)
        db = np.sum(dz)
        dw = np.dot(data,dz.T)
        bias = bias - learning_rate*db
        weights = weights - learning_rate*dw
        cost.append(costFunction(data,labels,weights,bias))
        
    plt.plot(range(num_iterations),cost)
    return {
            'weights':weights,
            'bias':bias
            }

print('Testing Learning Rate on Training Data')
#learning rate will vary between 10^-5 to 10^-1
learning = -4 * np.random.rand(11)-1
learning = np.power(10,learning)

#used for collecting f1score and parameters of each testing so to choose afterwards
modelSelection=[]
for i in learning:
    parameters = learningModel(train_data_X,train_data_Y,weights,bias,i)
    train_cost = costFunction(train_data_X,train_data_Y,parameters['weights'],parameters['bias'])
    test_cost = costFunction(test_data_X,test_data_Y,parameters['weights'],parameters['bias'])
    print('For Learning rate',i)
    print('Training Cost',train_cost)
    print('Test Cost',test_cost)
    print('F1 Score')
    predictions = sigmoid(np.dot(parameters['weights'].T,test_data_X)+bias) >= 0.5
    f1 = metrics.f1_score(test_data_Y.T,predictions.T)
    print(f1)
    print('')
    modelSelection.append([parameters,f1])
#model selection on basis of r2_score
max_f1 = 0
for i,model in enumerate(modelSelection):
    if model[1]>max_f1:
        max_f1 = model[1]
        index = i
    
#final Model selection
print('Final Model Selection')
print('Learning rate: ',learning[index])
print('Corresponding F1_score: ',modelSelection[index][1])
print('Corresponding Parameters: ',modelSelection[index][0])
print('Test cost: ',costFunction(test_data_X,test_data_Y,modelSelection[index][0]['weights'],modelSelection[index][0]['bias']))
selectedWeights = modelSelection[index][0]['weights']
selectedBias = modelSelection[index][0]['bias']
predictedOutput = sigmoid(np.dot(selectedWeights.T,test_data_X)+selectedBias) >= 0.5
print(' ')
print('Accuracy score')
print(metrics.accuracy_score(test_data_Y.T,predictedOutput.T))
print('Jacard score')
print(metrics.jaccard_similarity_score(test_data_Y.T,predictedOutput.T))
print('Confusion Matrix')
print(metrics.confusion_matrix(test_data_Y.T,predictedOutput.T))
print('Classification Report')
print(metrics.classification_report(test_data_Y.T,predictedOutput.T))
