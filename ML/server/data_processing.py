import numpy as np
import os
import joblib
import random
from sklearn.preprocessing import Normalizer
import os
os.environ ['TF_CPP_MIN_LOG_LEVEL'] = '2'
from tensorflow import keras

arr_field_names = [
 'БАЛЛ_ЗА_ДОСТИЖЕНИЯ',
 'БАЛЛ_ОЛИМПИАДЫ_И_КОНКУРСЫ',
 'ЕСТЬ_АТТЕСТАТ_С_ОТЛИЧИЕМ',
 'ЕСТЬ_ДИПЛОМ_С_ОТЛИЧИЕМ',
 'ОЦЕНКА_ЗА_СОЧИНЕНИЕ',
 'УРОВЕНЬ_ДИПЛОМА',
 'ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_1_ПРИОРИТЕТ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_2_ПРИОРИТЕТ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_3_ПРИОРИТЕТ',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2',
 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3'
]
data_row = {'УРОВЕНЬ_ДИПЛОМА': 'Среднее', 'БАЛЛ_ЗА_ДОСТИЖЕНИЯ': '15','ЕСТЬ_АТТЕСТАТ_С_ОТЛИЧИЕМ': 'on',
            'ЕСТЬ_ДИПЛОМ_С_ОТЛИЧИЕМ': 'on',
            'БАЛЛ_ОЛИМПИАДЫ_И_КОНКУРСЫ': '5', 'ОЦЕНКА_ЗА_СОЧИНЕНИЕ': '5',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_1_ПРИОРИТЕТ': 'Английский язык',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1': '95',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_2_ПРИОРИТЕТ': 'Русский язык',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2': '95',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_3_ПРИОРИТЕТ': 'Обществознание',
            'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3': '90', 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ': '280',
            'ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ': 'Не требуется'}



def normalize(data):
    norm = Normalizer()
    data = norm.fit_transform(data)
    return data

def predict_list_FNN(data, models_dir='../models/FNN_models/'):
    #print(data)
    data = preprocessing(data)
    data = normalize(data)
    prediction_list = []
    all_models = os.listdir(models_dir)
    print(all_models)
    for item in all_models:
        model = keras.models.load_model(models_dir + str(item))
        prediction = model.predict(data)
        print(prediction)
        prediction = np.append(prediction, item[:8])
        prediction_list.append(prediction)
    prediction_list.sort(key=lambda x: x[0], reverse=True)
    return prediction_list


def predict_list_LR(data, models_dir='../models/LR_models/'):
    #print(data)
    data = preprocessing(data)
    #data = normalize(data)
    print(data)
    prediction_list = []
    all_models = os.listdir(models_dir)
    for item in all_models:
        print(str(item))
        model = joblib.load(models_dir + item)
        prediction = model.predict(data)
        print(prediction)
        prediction = np.append(prediction, item[:8])
        prediction_list.append(prediction)
    prediction_list.sort(key=lambda x: x[0], reverse=True)
    return prediction_list

def check_data_text(example):

    check_flag = False
    subject = {'Математика': 0, 'Английский язык': 1, 'Информатика': 2, 'Физика': 3, 'Обществознание': 4,
               'Русский язык': 5, 'Химия': 6}
    education = {'Среднее': 0, 'Начальное профессиональное': 1, 'Среднее профессиональное': 2,
                 'Высшее профессиональное': 3}
    loc_student = {'Не требуется': 0, 'Общежитие': 1, 'Только регистрация': 2}


    for key in subject:
        if key == example:
            numbered_data = int(subject[key])
            check_flag = True

    if not check_flag:
        for key in education:
            if key == example:
                numbered_data = int(education[key])
                check_flag = True

    if not check_flag:
        for key in loc_student:
            if key == example:
                numbered_data = int(loc_student[key])
                check_flag = True

    if not check_flag:
        print('There is not the same text data, the value will be set as 0')
        numbered_data = 0


    print('Your text data: ', example)
    print('Your numbered data: ', numbered_data)
    return numbered_data



def preprocessing(data_row):
    data = []
    for i in arr_field_names:
        print(i)
        try:
           example = int(data_row[i])
        except:
            try:
                example = check_data_text(data_row[i])
            except:
                example = 0
  #      print(data_row[i])
        data.append(example)
    data = np.array([data])
    print(data)
    return data


