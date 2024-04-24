import os
import sys
import pandas as pd
from datetime import date
from src.exception import CustomException
from src.utils import load_object
import argparse


class PredictPipeline:

    def __init__(self):
        pass

    def predict(self):
        try:
            # Load the trained model and preprocessor
            model_path = os.path.join("artifacts", "model.pkl")
            test_path = os.path.join("artifacts", "test.csv")
            test = pd.read_csv(test_path)
            model = load_object(model_path)
            predictors = ['Open', 'High', 'Low', 'Close', 'Volume', 'Day_enc', 'Month_enc', 'Moving Average', 'MACD', 'Signal Line', 'RSI', 'Close_Ratio_2',
                          'Trend_2', 'Close_Ratio_5', 'Trend_5', 'Close_Ratio_60', 'Trend_60', 'Close_Ratio_250', 'Trend_250', 'Close_Ratio_1000', 'Trend_1000']

            today = test.iloc[-1]
            today_Df = pd.DataFrame(today).T
            today_date = today['Date']
            today = today.drop('Date')
            for col in today_Df.columns:
                if today_Df[col].dtype == 'object':
                    today_Df[col] = pd.to_numeric(
                        today_Df[col], errors='coerce')
            today_Df = today_Df.drop(columns=['Target', 'Date'])
            tommorow_pred = model.predict(today_Df[predictors])
            bias = 'Bullish' if tommorow_pred == 1 else 'Bearish'
            # calculate tommorow date based on today date
            today_date = pd.to_datetime(today_date)
            tommorow_date = today_date + pd.DateOffset(days=1)
            print(f'today_date : {today_date}')
            print(f'Date : {tommorow_date}, Daily Bias Prediction : {bias}')
            return tommorow_date, bias

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    def main(args):
        print(f"Welcome {args.name}!")
        print("Running the Daily Bias Prediction for EUR/USD...")
        obj = PredictPipeline()
        obj.predict()
        print("Prediction completed successfully!")

    if __name__ == "__main__":
        parser = argparse.ArgumentParser(
            description='Daily Bias Prediction for EUR/USD')
        parser.add_argument('--name', type=str,
                            default='Bilal', help='Your name')
        args = parser.parse_args()
        main(args)
