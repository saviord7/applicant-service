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

    dataFrame['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'] = dataFrame['НАПРАВЛЕНИЕ_ПОДГОТОВКИ'].str[:8]

    dataFrame['УРОВЕНЬ_ДИПЛОМА'] = np.where(dataFrame['УРОВЕНЬ_ДИПЛОМА'] == 'Среднее', 0,
                                       dataFrame['УРОВЕНЬ_ДИПЛОМА'])
    dataFrame['УРОВЕНЬ_ДИПЛОМА'] = np.where(dataFrame['УРОВЕНЬ_ДИПЛОМА'] == 'Начальное профессиональное', 1,
                                       dataFrame['УРОВЕНЬ_ДИПЛОМА'])
    dataFrame['УРОВЕНЬ_ДИПЛОМА'] = np.where(dataFrame['УРОВЕНЬ_ДИПЛОМА'] == 'Среднее профессиональное', 2,
                                       dataFrame['УРОВЕНЬ_ДИПЛОМА'])
    dataFrame['УРОВЕНЬ_ДИПЛОМА'] = np.where(dataFrame['УРОВЕНЬ_ДИПЛОМА'] == 'Высшее профессиональное', 3,
                                       dataFrame['УРОВЕНЬ_ДИПЛОМА'])

    dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = np.where(dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] == 'Общежитие', 1,
                                               dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'])
    dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = np.where(dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] == 'Только регистрация', 2,
                                               dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'])
    dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'] = dataFrame['ОБЩЕЖИТИЕ_ТИП_ЗАСЕЛЕНИЯ'].replace(np.nan, 0)

    data = dataFrame[dataFrame['НАПРАВЛЕНИЕ_В_ПРИКАЗЕ_БАЛЛ'] != 0]

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


