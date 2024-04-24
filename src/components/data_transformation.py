import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
import sys
from dataclasses import dataclass

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataPreprocessingConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "proprocessor.pkl")


class DataPreprocessing:
    def __init__(self, file_path):
        self.file_path = file_path
        self.df = pd.read_csv(file_path)
        self.le = LabelEncoder()
        self.data_preprocessing_config = DataPreprocessingConfig()
        self.train = self.df.iloc[:-100]
        self.test = self.df.iloc[-100:]

    def preprocess(self):
        self.add_column_names()
        self.add_date_features()
        self.convert_to_numeric()
        self.add_target()
        self.add_technical_indicators()
        self.drop_sundays()
        self.encode_categoricals()
        self.add_new_features()
        self.drop_na()
        self.drop_column_Tomorrow()
        self.displayData()

        save_object(
            file_path=self.data_preprocessing_config.preprocessor_obj_file_path, obj=self.test)
        return {
            'df': self.df,
            'predictors': ['Open', 'High', 'Low', 'Close', 'Volume', 'Day_enc', 'Month_enc', 'Moving Average', 'MACD', 'Signal Line', 'RSI', 'Close_Ratio_2',
                           'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250', 'Close_Ratio_1000', 'Trend_1000'],
            'target': 'Target'
        }

    def add_column_names(self):
        line = self.df.columns
        self.df.columns = ['Time', 'Open', 'High', 'Low', 'Close', 'Volume']
        new_line = pd.DataFrame({'Time': [line[0]], 'Open': [line[1]], 'High': [
                                line[2]], 'Low': [line[3]], 'Close': [line[4]], 'Volume': [line[5]]})
        self.df = pd.concat([new_line, self.df]).reset_index(drop=True)

    def add_date_features(self):
        try:
            self.df['Date'] = pd.to_datetime(
                self.df['Time'], format='%Y-%m-%d ')
            self.df['Day'] = self.df['Date'].dt.day_name()
            self.df['Month'] = self.df['Date'].dt.month_name()
            self.df['Quarter'] = self.df['Date'].dt.quarter
            self.df['Year'] = self.df['Date'].dt.year
            self.df['Tomorrow'] = self.df['Close'].shift(-1)
            logging.info('Date features added successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def convert_to_numeric(self):
        try:
            cols = self.df[['Open', 'High', 'Low',
                            'Close', 'Volume', 'Tomorrow']]
            for col in cols:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce')
                logging.info('Converted to numeric successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def add_target(self):
        try:
            self.df['Target'] = (self.df['Tomorrow'] >
                                 self.df['Close']).astype(int)
            logging.info('Target added successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def add_technical_indicators(self):
        try:
            self.df['Moving Average'] = self.df['Close'].rolling(
                window=20).mean()
            self.df['MACD'] = self.df['Close'].ewm(span=12, adjust=False).mean(
            ) - self.df['Close'].ewm(span=26, adjust=False).mean()
            self.df['Signal Line'] = self.df['MACD'].ewm(
                span=9, adjust=False).mean()
            self.df['RSI'] = 100 - (100 / (1 + (self.df['Close'].diff().fillna(0).rolling(
                window=14).apply(lambda x: x[x > 0].mean() / -x[x < 0].mean()))))
            logging.info('Technical indicators added successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def drop_na(self):
        try:
            missing_values_pourcentage = self.df.isnull().sum() / len(self.df)
            if missing_values_pourcentage.max() > 0.0000001:
                self.df = self.df.dropna().reset_index(drop=True)
                logging.info(
                    'Dropped columns with missing values successfully')

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def drop_sundays(self):
        try:
            self.df = self.df[self.df['Day'] !=
                              'Sunday'].reset_index(drop=True)
            logging.info('Dropped Sundays successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def encode_categoricals(self):
        try:
            self.df['Day_enc'] = self.le.fit_transform(self.df['Day'])
            self.df['Month_enc'] = self.le.fit_transform(self.df['Month'])
            logging.info('Encoded categoricals successfully')
        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)

    def add_new_features(self):
        self.df = self.df.drop(columns=['Time', 'Day', 'Month'])
        horizons = [2, 5, 60, 250, 1000]
        new_predictors = []
        for horizon in horizons:
            rolling_average = self.df.rolling(horizon).mean()
            ratio_column = f"Close_Ratio_{horizon}"
            self.df[ratio_column] = self.df["Close"] / rolling_average["Close"]
            trend_column = f"Trend_{horizon}"
            self.df[trend_column] = self.df.shift(
                1).rolling(horizon).sum()["Target"]
            new_predictors += [ratio_column, trend_column]

    def drop_column_Tomorrow(self):
        self.df = self.df.drop(columns=['Tomorrow'])

    def displayData(self):
        print(f"Shape: {self.df.shape}")
        print(f"Columns: {self.df.columns}")

    def transform(self):
        try:
            data_preprocessing = DataPreprocessing(
                file_path='artifacts/data.csv')
            preprocessing_result = data_preprocessing.preprocess()
            logging.info('Data preprocessing completed')

            save_object(
                file_path='artifacts/preprocessor.pkl', obj=data_preprocessing)

            return preprocessing_result

        except Exception as e:
            logging.error(e)
            raise CustomException(e, sys)


if __name__ == "__main__":
    data_preprocessing = DataPreprocessing(
        file_path='artifacts/data.csv')
    preprocessing_result = data_preprocessing.preprocess()
    logging.info('Data preprocessing completed')
    save_object(
        file_path='artifacts/preprocessor2.pkl', obj=data_preprocessing)
