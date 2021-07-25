#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 13 12:20:54 2021

@author: scottleroux
"""

import tensorflow as tf
import pandas as pd
import numpy as np 
from matplotlib import pyplot 
from tensorflow.keras.layers import Dense

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

from tensorflow.keras.wrappers.scikit_learn import KerasClassifier

df = pd.read_csv("Data/circle_square.data")

dfx = df.iloc[::3]
dfy = df.iloc[1::3]
dfc = df.iloc[2::3]

dfy = dfy.rename(index=lambda s:s-1)
dfc = dfc.rename(index=lambda s:s-2)
dfnew = pd.DataFrame()
dfnew['x'] = dfx
dfnew['y'] = dfy 

dfc = dfc.astype(int)
#uncomment line below to transform data for hinge loss
#dfc = dfc.applymap(lambda x: x-1 if x == 0 else x)

n_train,n_test = 8000, 1800

train_x, test_x, train_y, test_y = train_test_split(dfnew, dfc, test_size= 1800, random_state=4)
train_x, val_x, train_y, val_y = train_test_split(train_x, train_y, test_size = 200, random_state=3)

opts = ['sgd', 'adagrad', 'adam', 'rmsprop']
#test different optimizers 
def create_mod(opt):
    model = tf.keras.models.Sequential()
    model.add(Dense(50, input_dim = 2, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    return model


for i in range(len(opts)):
    plot_no = 420 + (i+1)
    pyplot.subplot(plot_no)
    mod_i = create_mod(opts[i])
    history = mod_i.fit(train_x, train_y, validation_data = (val_x,val_y), epochs = 100, verbose = 0)
    
    _, train_acc = mod_i.evaluate(train_x,train_y, verbose=0)
    _, test_acc = mod_i.evaluate(test_x, test_y, verbose=0)
    print('Train: %.3f, Test: %.3f' % (train_acc, test_acc))
    # plot training history
    pyplot.plot(history.history['accuracy'], label='train')
    pyplot.plot(history.history['val_accuracy'], label='test')
    pyplot.title('optimizer = ' + opts[i], pad = -40)

pyplot.legend()
pyplot.show()



def create_model(learn_rate = 0.001, momentum = 0.9):
    model = tf.keras.models.Sequential()
    model.add(Dense(50, input_dim = 2, activation = 'relu', kernel_initializer = 'he_uniform'))
    model.add(Dense(12, activation = 'relu'))
    model.add(Dense(1, activation='sigmoid'))
    opt = tf.keras.optimizers.Adam(learning_rate = learn_rate)
    
    model.compile(loss = 'binary_crossentropy', optimizer = opt, metrics = ['accuracy'])
    
    return model

seed = 8 

model = KerasClassifier(build_fn = create_model, epochs = 100, batch_size = 32, verbose = 1)


learn_rate = [0.0001, .001, .002, .005, .01, .02, .05, .1, .2, .5]
mo = [2.0, 1.0, .9, .5, .1, .01, .05, .001]


param_grid = dict(learn_rate = learn_rate)


grid = GridSearchCV(model, param_grid, cv = 3)
grid_result = grid.fit(train_x[:1000], train_y[:1000])

print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))

print(type(grid_result.best_params_))

model = create_model(learn_rate = grid_result.best_params_['learn_rate'])


history = model.fit(train_x, train_y, validation_data = (val_x,val_y), epochs = 100, verbose = 1)


_, train_acc = model.evaluate(train_x, train_y, verbose = 0)
_, test_acc = model.evaluate(test_x, test_y, verbose = 0)

print("train accuary = %.3f,\n test accuracy = %.3f" % (train_acc,test_acc))


pyplot.subplot(211)
pyplot.title('Loss ')
pyplot.plot(history.history['loss'],label='train')
pyplot.plot(history.history['val_loss'],label='test')
pyplot.legend()

pyplot.subplot(212)
pyplot.title('Accuracy ')
pyplot.plot(history.history['accuracy'],label='train')
pyplot.plot(history.history['val_accuracy'],label='test')
pyplot.legend()
pyplot.show()


pyplot.plot(learn_rate,grid_result.cv_results_['mean_test_score'], label = 'mean')
pyplot.plot(learn_rate, grid_result.cv_results_['std_test_score'], label = 'std')
pyplot.legend()
pyplot.show()






















