import numpy as np
import pandas as pd

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from joblib import dump

# Reading the dataset


data_start = pd.read_csv('../data/etu_dataset/data.csv', sep=',')
marks = pd.read_csv('../data/etu_dataset/avg_marks.csv', sep=',')

# Take useful columns


data = data_start[[
      'БАЛЛ_ЗА_ДОСТИЖЕНИЯ',
      'БАЛЛ_ОЛИМПИАДЫ_И_КОНКУРСЫ',
      'ЕСТЬ_АТТЕСТАТ_С_ОТЛИЧИЕМ',
      'ЕСТЬ_ДИПЛОМ_С_ОТЛИЧИЕМ',
      'ОЦЕНКА_ЗА_СОЧИНЕНИЕ',
      'УРОВЕНЬ_ДИПЛОМА',
      'ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ_ЕГЭ_С_ОЛИМПИАДОЙ',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЛИМПИАДА_ЗА_100_БАЛЛОВ',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2',
      'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3',
      'НАПРАВЛЕНИЕ_ПОДГОТОВКИ',
      'ИНДИВИДУАЛЬНЫЙ_НОМЕР'
]]

data = data.rename({'ИНДИВИДУАЛЬНЫЙ_НОМЕР': 'keyID'}, axis=1)

# Preprocessing


data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] = data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].str[:8]
data['УРОВЕНЬ_ДИПЛОМА'] = LabelEncoder().fit_transform(data['УРОВЕНЬ_ДИПЛОМА'])
data['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = LabelEncoder().fit_transform(data['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'])

# Check for duplicates


if (len(marks)) != (len(marks['keyID'].unique())):
    duplicateRows_Labels = marks[marks.duplicated(['keyID'], keep=False)]
    for x in range(len(duplicateRows_Labels)):
        marks.drop(duplicateRows_Labels.index[x], inplace=True)

#  Merge the data


all_data = pd.merge(data, marks, on='keyID', right_index=False, sort=False)

# Take data where avg_mark not 0


all_data = all_data.loc[all_data['avg_mark'] != 0]

# Find unique courses:
course = all_data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].unique()

# Split data,  create LR model, train, watch score, save models


for example in course:
    x = all_data.loc[all_data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] == example]
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
