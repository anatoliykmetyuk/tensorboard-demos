#!/usr/bin/env python

import os
os.environ['KERAS_BACKEND'      ] = 'tensorflow'
os.environ['MKL_THREADING_LAYER'] = 'GNU'

import keras as ks
from keras.models    import Sequential
from keras.layers    import Dense
from keras.callbacks import TensorBoard

from common import new_run_log_dir, train_X, train_Y, samples_num, print_summary

log_dir = new_run_log_dir('keras-demo')

# Parameters
learning_rate   = 0.000001
training_epochs = 500
display_step    = 10
hidden_size     = 128

# Model
model = Sequential()
model.add(Dense(hidden_size, activation='relu', input_shape=[1]))
model.add(Dense(1))

model.compile(loss      = ks.losses.mean_squared_error,
              optimizer = ks.optimizers.Adadelta())

model.summary()

model.fit(train_X, train_Y,
          epochs          = training_epochs,
          verbose         = 1,
          validation_data = (train_X, train_Y),
          callbacks       = [TensorBoard(log_dir        = log_dir,
                                         histogram_freq = 50)])

print_summary(train_Y, model.predict(train_X))
