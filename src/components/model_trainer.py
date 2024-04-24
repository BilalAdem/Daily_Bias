import numpy as np
import pandas as pd
from sklearn.metrics import precision_score, classification_report, confusion_matrix, accuracy_score
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier, XGBRFClassifier
import sys
from dataclasses import dataclass

from sklearn.ensemble import AdaBoostClassifier, ExtraTreesClassifier, GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import precision_score, classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    train_data_path: str = os.path.join('artifacts', "train.csv")
    test_data_path: str = os.path.join('artifacts', "test.csv")


class ModelTraining:
    def __init__(self, df, predictors, target):
        self.model_trainer_config = ModelTrainerConfig()

        self.train = df.iloc[:-100]
        self.test = df.iloc[-100:]

        os.makedirs(os.path.dirname(
            self.model_trainer_config.train_data_path), exist_ok=True)

        self.train.to_csv(self.model_trainer_config.train_data_path,
                          index=False, header=True)

        os.makedirs(os.path.dirname(
            self.model_trainer_config.test_data_path), exist_ok=True)

        self.test.to_csv(self.model_trainer_config.test_data_path,
                         index=False, header=True)
        logging.info('Saved the raw data')
        self.predictors = predictors
        self.target = target

    def train_model(self):
        try:

            best_model = XGBClassifier(learning_rate=0.02, n_estimators=600, objective='binary:logistic',
                                       nthread=1, subsample=0.6, min_child_weight=1, max_depth=5, gamma=1.5, colsample_bytree=0.8)

            logging.info(
                f"Best found model on both training and testing dataset is {best_model}")
            print('Best Model:', best_model)
            best_model.fit(self.train[self.predictors],
                           self.train[self.target])
            predicted = best_model.predict(self.test[self.predictors])
            predicted_df = pd.DataFrame(predicted, columns=[self.target])
            predicted_df.to_csv(
                'artifacts/predicted.csv', index=False, header=True)
            # compare the predicted values with the actual values in cross tab
            corss_tab = pd.crosstab(
                self.test[self.target], predicted, rownames=['Actual'], colnames=['Predicted'])
            print(corss_tab)

            precision = precision_score(self.test[self.target], predicted)
            classification = classification_report(
                self.test[self.target], predicted)
            confusion = confusion_matrix(self.test[self.target], predicted)
            accuracy = accuracy_score(self.test[self.target], predicted)

            save_object(
                self.model_trainer_config.trained_model_file_path, best_model)
            logging.info(
                f"Model saved at {self.model_trainer_config.trained_model_file_path}")
            print('Accuracy:', accuracy)
            print('Precision:', precision)
            print('Classification Report:\n', classification)
            print('Confusion Matrix:\n', confusion)
        except Exception as e:
            raise CustomException(e, sys)
