import numpy as np
import os
import joblib
import random
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
data_row = {'УРОВЕНЬ_ДИПЛОМА': 'Среднее', 'БАЛЛ_ЗА_ДОСТИЖЕНИЯ': '5','ЕСТЬ_АТТЕСТАТ_С_ОТЛИЧИЕМ': 'on',
            'ЕСТЬ_ДИПЛОМ_С_ОТЛИЧИЕМ': 'on',
            'БАЛЛ_ОЛИМПИАДЫ_И_КОНКУРСЫ': '5', 'ОЦЕНКА_ЗА_СОЧИНЕНИЕ': '10',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_1_ПРИОРИТЕТ': 'Английский язык',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1': '55',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_2_ПРИОРИТЕТ': 'Английский язык',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2': '75',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_3_ПРИОРИТЕТ': 'Английский язык',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3': '90', 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ': '210',
            'ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ': 'Не требуется'}


def normalize(data):
    sc = StandardScaler()
    data = sc.fit_transform(data)
    return data


def predict_list(data, models_dir='../models/LR_models/'):
    #print(data)
    data = preprocessing(data)
    data = normalize(data)
    prediction_list = []
    all_models = os.listdir(models_dir)
    for item in all_models:
        # print(str(item))
        model = joblib.load(models_dir + item)
        prediction = model.predict(data)
        #print(prediction)
        prediction = np.append(prediction, item[:8])
        prediction_list.append(prediction)
    prediction_list.sort(key=lambda x: x[0], reverse=True)
    return prediction_list



def preprocessing(data_row):
    data = []
    for i in arr_field_names:
        try:
           example = int(data_row[i])
        except:
           example = random.randint(0,1)
  #      print(data_row[i])
        data.append(example)
    data = np.array([data])
    return data
