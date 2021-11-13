#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 01:41:22 2021

@author: shahan
"""

"""
Optimizer list

** From Keras : **
SGD 
    No momentum
    Momentum
    Nesterov Momentum
RMSprop
Adam
    ADAM
    ADAMW
    AMSGrad
Adadelta
Adagrad
Adamax
Nadam
Ftrl

** Custom : **

ADABOUND
    Adabound
    AMSBOUND
ADABELIEF
RADAM

"""
"""
1. ALL Optimizers list
2. ALL Model Architectures - 3 models , 
3. AUC curve
4. HeatMaps


"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# Optimizers
import tensorflow as tf 
import tensorflow_addons as tfa
from tensorflow.keras.optimizers import Adam, SGD, RMSprop, Adagrad, Nadam, Adadelta, Adamax, Ftrl
from adabelief_tf import AdaBeliefOptimizer
from keras_adabound import AdaBound

from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Lambda, Dense, Embedding, Bidirectional
from tensorflow.keras.layers import Dropout, Input,InputLayer, ReLU, LSTM
from tensorflow.keras.layers import GRU, SimpleRNN

#metrics
from sklearn.metrics import f1_score , recall_score, accuracy_score, precision_score, confusion_matrix
from sklearn.metrics import roc_curve, auc


import warnings
warnings.filterwarnings('ignore')

#Mark Down print
from IPython.display import Markdown, display

def printmd(string):
    # Print with Markdowns    
    display(Markdown(string))


# 1. ALL Optimizer List
# Adam weight decay

AdamW = tfa.optimizers.AdamW(learning_rate=0.001, weight_decay=0.001)
AMSGrad = Adam(amsgrad=True)
Rectified_Adam = tfa.optimizers.RectifiedAdam(learning_rate=0.001)

SGD_momentum = SGD(momentum=0.9)
Nesterov_SGD_momentum = SGD(momentum=0.9, nesterov=True)
#Adam #Nadam #SGD #RMSprop #Adadelta #Adagrad #Adamax #Ftrl
AdaBelief = AdaBeliefOptimizer(learning_rate=1e-3, epsilon=1e-14, rectify=False)
Adabound = AdaBound(lr=1e-3, final_lr=0.1)
AMSbound = AdaBound(lr=1e-3, final_lr=0.1, amsgrad=True)

def SIMPLE_RNN_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength) (inp)
    X = SimpleRNN(64) (X)
    X = Dense(16, activation='relu') (X)
    X = Dropout(0.1) (X)
    X = Dense(1, activation='sigmoid') (X)
    # Create Model instance which converts sentence_indices into X.
    rnn = Model(inputs=inp, outputs=X)
    return rnn

def BI_RNN_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength)(inp)
    X = Bidirectional(SimpleRNN(64))(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Create Model instance which converts sentence_indices into X.
    brnn = Model(inputs=inp, outputs=X)
    return brnn

# def BI_RNN_ARCHITECTURE_SPAMBASE(numSamples, numFeatures):
#     inp = Input((numFeatures,), dtype="float64")
#     X = Bidirectional(SimpleRNN(64, input_shape=(numSamples, numFeatures)))(inp)
#     X = Dense(16, activation='relu')(X)
#     X = Dropout(0.1)(X)
#     X = Dense(1, activation='sigmoid')(X)
#     # Create Model instance which converts sentence_indices into X.
#     brnn = Model(inputs=inp, outputs=X)
#     return brnn

def LSTM_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength)(inp)
    X = LSTM(64)(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Create Model instance which converts sentence_indices into X.
    lstm = Model(inputs=inp, outputs=X)
    return lstm


def BI_LSTM_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength) (inp)
    X = Bidirectional(LSTM(64))(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Create Model instance which converts sentence_indices into X.
    bilstm = Model(inputs=inp, outputs=X)
    return bilstm

# def BI_LSTM_ARCHITECTURE_SPAMBASE(numSamples, numFeatures):
    
#     inp = Input((numFeatures,), dtype="float64")
#     X = Bidirectional(LSTM(64, input_shape=(numSamples, numFeatures)))(inp)
#     X = Dense(16, activation='relu')(X)
#     X = Dropout(0.1)(X)
#     X = Dense(1, activation='sigmoid')(X)
#     # Create Model instance which converts sentence_indices into X.
#     bilstm = Model(inputs=inp, outputs=X)
#     return bilstm



def GRU_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength)(inp)
    X = GRU(64)(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Create Model instance which converts sentence_indices into X.
    gru = Model(inputs=inp, outputs=X)
    return gru


def BI_GRU_ARCHITECTURE(maxLength, maxFeature, embedding_vecor_length=100):
    
    inp = Input((maxLength,), dtype="int32")
    X = Embedding(maxFeature, embedding_vecor_length, input_length=maxLength)(inp)
    X = Bidirectional(GRU(64))(X)
    X = Dense(16, activation='relu')(X)
    X = Dropout(0.1)(X)
    X = Dense(1, activation='sigmoid')(X)
    # Create Model instance which converts sentence_indices into X.
    gru_bi = Model(inputs=inp, outputs=X)
    return gru_bi

# def BI_GRU_ARCHITECTURE_SPAMBASE(numSamples, numFeatures):
    
#     inp = Input((numFeatures), dtype="float64")
#     X = Bidirectional(GRU(64, input_shape=(numSamples, numFeatures)))(inp)
#     X = Dense(16, activation='relu')(X)
#     X = Dropout(0.1)(X)
#     X = Dense(1, activation='sigmoid')(X)
#     # Create Model instance which converts sentence_indices into X.
#     gru_bi = Model(inputs=inp, outputs=X)
#     return gru_bi



def plot_graphs(history, metric):
    plt.plot(history.history[metric])
    plt.plot(history.history['val_'+metric], '')
    plt.xlabel("Epochs")
    plt.ylabel(metric)
    plt.legend([metric, 'val_'+metric])
    
def get_Metrics(y_test, y_pred, average="macro"):
    
    lr_fpr, lr_tpr, _ = roc_curve(y_test, y_pred)
    # find area under curve score
    lr_auc = auc(lr_fpr, lr_tpr)
    precision = precision_score(y_test, y_pred, average = average)
    recall = recall_score(y_test, y_pred, average = average)
    f1_score_ = f1_score(y_test, y_pred, average = average)
    accuracy = accuracy_score(y_test, y_pred)
    #print(f"precision : {precision} recall : {recall} f1_score : {f1_score_} accuracy : {accuracy}")
    return precision, recall, f1_score_, accuracy, lr_auc

def Heatmap_ConfusionMatrix(y_test,y_pred,fName= 0):
    # Confusion Matrix
    # sklearn builtin function to calculate confusion matrix values using true labels and predictions
    CF = confusion_matrix(y_test,y_pred.round())
    # list of labels that will be displayed on the image boxes
    labels = ['True Neg','False Pos','False Neg','True Pos']
    # list of all possible label values
    categories = ['Spam', 'Ham']
    group_names = ['True Neg','False Pos','False Neg','True Pos']
    # count total values present in each cell of the matrix
    group_counts = ["{0:0.0f}".format(value) for value in CF.flatten()]
    # count percentage of total values present in each cell of the matrix
    group_percentages = ["{0:.2%}".format(value) for value in CF.flatten()/np.sum(CF)]
    # group the labels to plot in graph
    labels = [f"{v1}\n{v2}\n{v3}"for v1, v2, v3 in zip(group_names,group_counts,group_percentages)]
    # reshape true label values according to the requirement
    labels = np.asarray(labels).reshape(2,2)
    # declare graph using heatmap function
    heatmap=sns.heatmap(CF, annot=labels, fmt='', cmap='Blues')
    # plot confusion matrix
    # fig = heatmap.get_figure()
    # save confusion matrix as image in results folder
    # fig.savefig('drive/MyDrive/SPAM classification deep learning/heatmaps/'+fName)
    # fig.savefig(f"{fName}.jpeg")
    # display confusion matrix as numeric values
    print(CF)
    return heatmap

def ROC_AUC(y_test, y_pred, fname= 0):
    # evluate true positive rate and false positive rate using sklearn builtin function
    fpr, tpr, _ = roc_curve(y_test, y_pred)
    # find area under curve score
    auc_ = auc(fpr, tpr)

    # display auc score
    print("AUC:", auc_)
    # plot linear line with no learning
    plt.plot([0, 1], [0, 1], 'k--')
    # plot tpr and fpr ratio
    plt.plot(fpr, tpr, marker='.', label='(auc = %0.3f)' % auc_)
    # assign labels
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.title('Receiver Operating Characterisics')
    plt.legend(loc='lower right')
    # plt.savefig(f"drive/MyDrive/SPAM classification deep learning/Visuals/{fname}")
    # plt.savefig(f"{fname}.jpeg")
    return auc_

































































 
 


