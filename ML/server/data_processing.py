import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

arr_field_names = [
 'БАЛЛ_ЗА_ДОСТИЖЕНИЯ',
 'БАЛЛ_ОЛИМПИАДЫ_И_КОНКУРСЫ',
 'ЕСТЬ_АТТЕСТАТ_С_ОТЛИЧИЕМ',
 'ЕСТЬ_ДИПЛОМ_С_ОТЛИЧИЕМ',
 'ОЦЕНКА_ЗА_СОЧИНЕНИЕ',
 'УРОВЕНЬ_ДИПЛОМА',
 'ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3'
]

data = np.array([[0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                 [10, 10, 10, 0, 0, 2, 0, 230, 70, 60, 100],
                 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
                 [0, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0]])


def preprocessing(data=data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data


def predict_list(data=data, models_dir='../models/LR_models/'):
    #print(data)
    data = preprocessing(data)
    prediction_list = []
    all_models = os.listdir(models_dir)
    for item in all_models:
        # print(str(item))
        model = joblib.load(models_dir + item)
        prediction = model.predict(data)
        #print(prediction)
        prediction = np.append(prediction, item[:8])
        prediction_list.append(prediction)

    return prediction_list
