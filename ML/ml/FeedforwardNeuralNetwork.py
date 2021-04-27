import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers
from tensorflow.keras.callbacks import ModelCheckpoint

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import data_loading_and_preprocessing



def checkpoint(filename):

    checkpoint_callback = ModelCheckpoint(filename,
                                      monitor='val_mse',
                                      save_best_only=True,
                                      verbose=1)
    return checkpoint_callback

def compile_model(train_X):
    # Network
    model = Sequential()
    model.add(Dense(22, activation='relu', input_shape=(train_X.shape[1],)))
    model.add(Dropout(.2))
    model.add(Dense(22, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(11, activation='relu'))
    model.add(Dropout(.2))
    model.add(Dense(1))

    opt = optimizers.Adam(learning_rate=0.001)
    model.compile(optimizer=opt, loss='mae', metrics=['mse'])

    return model


def fit_model(model, train_X, train_y):
    print('Fitting model...')
    history = model.fit(train_X,
                        train_y,
                        epochs=20,
                        validation_split=0.05,
                        batch_size=1,
                        callbacks=[checkpoint_callback])

    return  history

# Load data prepared data
data = data_loading_and_preprocessing.make_prepared_data()

# Find unique courses:
course = data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].unique()

for example in course:
    x = data.loc[data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] == example]
    x = x.drop(['НАПРАВЛЕНИЕ_ПОДГОТОВКИ', 'keyID'], axis=1)

    print('#################################')
    print('INFO:')
    print('Training ', example, ' course..')
    print('#################################')

    filename = '../models/FNN_models/' + example + '-model'

    checkpoint_callback = checkpoint(filename)

    print('Dividing data...')
    train_data = np.array(x.drop('avg_mark', axis=1))
    train_label = x['avg_mark']

    norm = Normalizer()

    print('Preprocessing data...')
    train_data = norm.fit_transform(train_data)

    train_X, test_X, train_y, test_y = train_test_split(train_data, train_label, test_size=0.3, random_state=1)

    print(train_X.shape)
    train_y = np.asarray(train_y).astype(np.float)
    test_y = np.asarray(test_y).astype(np.float)

    model = compile_model(train_X)
    history = fit_model(model, train_X, train_y)

    mae, mse = model.evaluate(test_X, test_y)
    print('MSE on test data : ', mse)
    print('MAE on test data : ', mae)

    print('END INFO')
    print('#################################')
    print()


