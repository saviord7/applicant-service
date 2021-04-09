import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from joblib import dump
import  data_loading_and_preprocessing

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

    sc = StandardScaler()

    print('Preprocessing data...')
    train_data = sc.fit_transform(train_data)

    print('Splitting data for test / train ...')
    train_X, test_X, train_y, test_y = train_test_split(train_data, train_label, test_size=0.1, random_state=1)

    train_y = np.array(train_y)
    test_y = np.array(test_y)

    model = LinearRegression()

    print('Fitting model...')
    model.fit(train_X, train_y)

    print('Model score: ', model.score(train_X, train_y))

    print('Fitting model with test data...')
    y_pred = model.predict(test_X)

    mse = mean_squared_error(test_y, y_pred)
    mae = mean_absolute_error(test_y, y_pred)
    print('Score on test data: mse: %.3f, mae: %.3f' % (mse, mae))

    filename = '../models/LR_models/' + example + '-model.joblib'
    print('Saving model...')
    dump(model, filename)
    print('END INFO')
    print('#################################')
    print()
