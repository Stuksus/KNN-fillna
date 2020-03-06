import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


class fillna_knn(object):
    def __init__(self, dataframe, flag):
        self.inplace = flag
        if self.inplace:
            self.dataset = dataframe.copy()
        else:
            self.dataset = dataframe
        self.unique_cat = []
        self.encoder = LabelEncoder()

    def fill(self):
        self.__encoding()  # При создании объекта кодируем датафрейм
        self.count_nan = sum(self.dataset.isnull().sum())
        print(self.count_nan)
        while self.count_nan > 0:
            for column in self.dataset.columns.values:
                for row in range(len(self.dataset)):
                    if pd.isnull(self.dataset[column][row]):
                        df = self.__new_dataset(column)
                        total, target = self.__preparing_data(df, row, column)
                        self.dataset[column][row] = self.__nearest_neighbor(total, target)
                        self.count_nan = sum(self.dataset.isnull().sum())
                        print(self.count_nan, 'count_nan')
        self.dataset[self.object_col] = self.dataset[self.object_col].astype('int')
        self.__decoding()
        return self.dataset

    def __dist(self, first_vector, second_vector):
        first_vector = np.array(first_vector)
        distance = (sum((first_vector - second_vector) ** 2)) ** 0.5
        return distance

    def __without_nan(self, column_val):  # пропуск NaN при кодировании
        final_column_val = [0] * len(column_val)
        for index, element in enumerate(column_val):
            if element is not np.nan:
                final_column_val[index] = self.encoder.transform([column_val[index]])[0]
            else:
                final_column_val[index] = element
        return final_column_val

    def __encoding(self):
        # сбор категориальных признаков
        self.object_col = self.dataset.describe(include=['O']).columns.values
        for col in self.object_col:
            self.unique_cat += list(self.dataset[col].unique())
        self.unique_cat = list(set(self.unique_cat))
        index_del_nan = self.unique_cat.index(np.nan)
        self.unique_cat.pop(index_del_nan)
        # кодирование полученных признаков
        self.encoder.fit(self.unique_cat)
        self.dataset[self.object_col] = self.dataset[self.object_col].apply(lambda col: self.__without_nan(col))

    def __new_dataset(self, column):
        dataframe = self.dataset.copy()
        for index, val in enumerate(dataframe[column]):
            if pd.isnull(val):
                dataframe.drop([index], axis=0, inplace=True)
        dataframe.dropna(inplace=True)
        return dataframe

    def __preparing_data(self, data, row, column):
        # выделяем целевую строчку
        target_row = self.dataset.iloc[row].drop([column], inplace=False)
        # выделяем целевой столбец
        target_column = data[column].copy()
        # удаляем целевой столбец из датасета
        data = data.drop([column], axis=1)
        # объединяем строку  и класс(значение целевого столбца) соответсвующий ей
        total = []
        if len(target_column) == len(data):
            for i in range(len(data)):
                total.append([data.iloc[i], target_column.iloc[i]])
        else:
            raise SystemExit('Целевая колонка не равна по длине основный данным')
        return total, target_row

    def __nearest_neighbor(self, total, test_point):
        test_dist = [[self.__dist(test_point, data[0]), data[1]] for data in total]
        return sorted(test_dist)[0:1][0][1]

    def __decoding(self):
        self.dataset[self.object_col] = self.dataset[self.object_col].apply(
            lambda col: self.encoder.inverse_transform(col))
