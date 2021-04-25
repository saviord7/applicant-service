import numpy as np
import pandas as pd

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras import optimizers

from sklearn.preprocessing import Normalizer
from sklearn.model_selection import train_test_split
import data_loading_and_preprocessing



def compile_model(train_X):
    # Network
    model = Sequential()
    model.add(Dense(22, activation='relu', input_shape=(train_X.shape[1],)))
    model.add(Dropout(0.2))
    model.add(Dense(11, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(22, activation='relu'))
    model.add(Dropout(0.2))

    model.add(Dense(11, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1))

    opt = optimizers.SGD(learning_rate=0.01)
    # Compile network
    model.compile(optimizer=opt, loss='mse', metrics=['mae'])

    return model


def fit_model(model, train_X, train_y):
    print('Fitting model...')
    history = model.fit(train_X, train_y, epochs=7, batch_size=1)

    return  history

# Load data prepared data
data = data_loading_and_preprocessing.make_prepared_data()

# Find unique courses:
course = data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].unique()

for example in course:
    x = data.loc[data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] == example]
    x = x.drop(['НАПРАВЛЕНИЕ_ПОДГОТОВКИ', 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ_ЕГЭ_С_ОЛИМПИАДОЙ',
                'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЛИМПИАДА_ЗА_100_БАЛЛОВ', 'keyID'], axis=1)

    print('#################################')
    print('INFO:')
    print('Training ', example, ' course..')
    print('#################################')

    print('Dividing data...')
    train_data = np.array(x.drop('avg_mark', axis=1))
    train_label = x['avg_mark']

    norm = Normalizer()

    print('Preprocessing data...')
    train_data = norm.fit_transform(train_data)

    train_X, test_X, train_y, test_y = train_test_split(train_data, train_label, test_size=0.1, random_state=1)

    print(train_X.shape)
    train_y = np.array(train_y)
    test_y = np.array(test_y)

    model = compile_model(train_X)
    history = fit_model(model, train_X, train_y)

    mse, mae = model.evaluate(test_X, test_y)
    print('MSE on test data : ', mse)
    print('MAE on test data : ', mae)

    print('Saving model ...')
    filename = '../models/FNN_models/' + example + '-model'
    model.save(filename)
    print('END INFO')
    print('#################################')
    print()


