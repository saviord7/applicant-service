import numpy as np
import pandas as pd


data_link_1 = '../data/etu_dataset/data.csv'
data_link_2 ='../data/etu_dataset/avg_marks.csv'




def load_data():
    data_start = pd.read_csv(data_link_1, sep=',')
    marks = pd.read_csv(data_link_2, sep=',')

    return data_start, marks


def work_with_students_data(dataFrame):
    data = dataFrame[[
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
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_1_ПРИОРИТЕТ',
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_2_ПРИОРИТЕТ',
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_3_ПРИОРИТЕТ',
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_1',
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_2',
        'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ОЦЕНКА_3',
        'НАПРАВЛЕНИЕ_ПОДГОТОВКИ',
        'ИНДИВИДУАЛЬНЫЙ_НОМЕР'
    ]]

    data = data.rename({'ИНДИВИДУАЛЬНЫЙ_НОМЕР': 'keyID'}, axis=1)

    return data


def preprocessing_and_encoding(dataFrame):
    # Preprocessing and encoding

    subject = {'Математика': 0, 'Английский язык': 1, 'Информатика': 2, 'Физика': 3, 'Обществознание': 4,
               'Русский язык': 5, 'Химия': 6}
    education = {'Среднее': 0, 'Начальное профессиональное': 1, 'Среднее профессиональное': 2,
                 'Высшее профессиональное': 3}
    loc_student = {'Общежитие': 1, 'Только регистрация': 2}
    prioritet = ['НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_1_ПРИОРИТЕТ', 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_2_ПРИОРИТЕТ',
                 'НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_ПРЕДМЕТ_3_ПРИОРИТЕТ']

    for example in education:
        dataFrame['УРОВЕНЬ_ДИПЛОМА'] = np.where(dataFrame['УРОВЕНЬ_ДИПЛОМА'] == example, education[example],
                                           dataFrame['УРОВЕНЬ_ДИПЛОМА'])

    for example in loc_student:
        dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = np.where(dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] == example, loc_student[example],
                                                   dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'])
    dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'].replace(np.nan, 0)

    for attr in prioritet:
        for example in subject:
            dataFrame[attr] = np.where(dataFrame[attr] == example, subject[example], dataFrame[attr])



    data = dataFrame[dataFrame['НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ'] != 0]
    data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] = data['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].str[:8]

    return data



def checking_for_duplicates(dataFrame, key='keyID'):
    if (len(dataFrame)) != (len(dataFrame[key].unique())):
        duplicateRows_Labels = dataFrame[dataFrame.duplicated(['keyID'], keep=False)]
        for x in range(len(duplicateRows_Labels)):
            dataFrame.drop(duplicateRows_Labels.index[x], inplace=True)

    return dataFrame

def merging(dataFame1, dataFrame2, key='keyID'):
    all_data = pd.merge(dataFame1, dataFrame2, on=key, right_index=False, sort=False)
    all_data = all_data.loc[all_data['avg_mark'] != 0]

    return all_data



def make_prepared_data():
    student_data, marks = load_data()
    student_data = work_with_students_data(student_data)
    student_data = preprocessing_and_encoding(student_data)
    student_data = checking_for_duplicates(student_data, 'keyID')
    marks = checking_for_duplicates(marks, 'keyID')

    merging_data = merging(student_data,marks)

    return merging_data


