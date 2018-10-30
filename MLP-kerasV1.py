# -*- coding: utf-8 -*-
"""
Created on Fri Oct 26 01:41:20 2018
@author: vivenkyan
"""

import os, re, timeit
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.layers.embeddings import Embedding
from keras.optimizers import SGD



def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
        return v
    return v / norm


def build_data(train_input_, train_label_index):
    x = []
    y = []
    
    for i in range(len(train_input_)):
        for j in range(len(train_input_[i])):
            #x.append(normalize(train_input_[i][j]))
            x.append(train_input_[i][j])
            y.append(train_label_index[i][j])
                    
    return (np.array(x), np.array(y))



data_path = 'dataV2/'

#Load Embeddings
    
#Train data embeddings
train_input_ = np.load(data_path + 'train_input.npy')

#Train labels as indexes
train_label_index_ = np.load(data_path + 'train_label_index.npy')


#Test data embeddings
test_input_ = np.load(data_path + 'test_input.npy')

#Test labels as indexes
test_label_index_ = np.load(data_path + 'test_label_index.npy')


print("Data Loaded!!!\n")


TRAIN_SIZE = len(train_input_)
TEST_SIZE = len(test_input_)

x_train, y_train = build_data(train_input_[:TRAIN_SIZE], train_label_index_[:TRAIN_SIZE])

print("Train data len:", len(x_train))
assert len(x_train) == len(y_train), "**Size Mismatch!!!***"


x_test, y_test = build_data(test_input_[:TEST_SIZE], test_label_index_[:TEST_SIZE])

print("Train data len:", len(x_test))
assert len(x_test) == len(y_test), "**Size Mismatch!!!***"

print("\nData formatting done!!!")


# transform labels into one hot representation
y_train_one_hot = (np.arange(np.max(y_train) + 1) == y_train[:, None]).astype(float)

y_test_one_hot = (np.arange(np.max(y_test) + 1) == y_test[:, None]).astype(float)

#lr = np.arange(OP_DIM)
#test_labels_one_hot = (lr==test_labels).astype(np.float)

"""
#removing zeroes and ones from the labels:
y_train_one_hot[y_train_one_hot==0] = 0.01
y_train_one_hot[y_train_one_hot==1] = 0.99

y_test_one_hot[y_test_one_hot==0] = 0.01
y_test_one_hot[y_test_one_hot==1] = 0.99
"""

print("\nx_train shape", x_train.shape)
print("y_train shape", y_train.shape)
print("y_train_one_hot shape", y_train_one_hot.shape)


print("\nx_test shape", x_test.shape)
print("y_test shape", y_test.shape)
print("y_test_one_hot shape", y_test_one_hot.shape)


#MLP Network architecture

IP_DIM = 25 #x_train.shape[1]
OP_DIM = 8  #y_train_one_hot.shape[1]


print('Building model...')
model_mlp = Sequential()


##Dense(64) is a fully-connected layer with 64 hidden units.
#In the first layer, we must specify the expected input data shape
model_mlp.add(Dense(100, input_dim= IP_DIM, activation='relu'))
model_mlp.add(Dropout(0.5))

model_mlp.add(Dense(50, activation='relu'))
model_mlp.add(Dropout(0.5))

model_mlp.add(Dense(OP_DIM, activation='softmax'))

#sgd = SGD(lr=0.05, decay=1e-6, momentum=0.95, nesterov=True)
model_mlp.compile(loss='categorical_crossentropy', 
                  optimizer='adam',     
                  #optimizer = sgd,
                  metrics=['accuracy'])


print("Train data len:", len(x_train))

EPOCHS = 10
BATCH_SIZE = 64

print('\n\nTraining Model...')

start = timeit.default_timer()

#batch_size: Integer or None. Number of samples per gradient update. 
#If unspecified, batch_size will default to 32.
model_mlp.fit(x_train, y_train_one_hot,
              batch_size = BATCH_SIZE,
              epochs = EPOCHS,
              validation_data=(x_test, y_test_one_hot))

#model_lstm.fit(data, np.array(labels), validation_split=0.2, epochs=3)

print("\n\nTotal training time: %.4f seconds." % (timeit.default_timer() - start))

start = timeit.default_timer()
score, acc = model_mlp.evaluate(x_test, y_test_one_hot, batch_size = BATCH_SIZE)

print("\nTesting time: %.4f seconds." % (timeit.default_timer() - start))

print('\nTest score:', score)
print('Test accuracy:', acc)




