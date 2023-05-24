import sys
import os
from dataclasses import dataclass
from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor
)
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_model

@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info('Splitting the training and testing input data')
            X_train, y_train, X_test, y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                train_array[:,:-1],
                train_array[:,-1]
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Nearest Neighbours": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "catboots Regressor": CatBoostRegressor(verbose = False),
                "Adaboost Regressor": AdaBoostRegressor() }

            model_report:dict=evaluate_model(X_train=X_train, y_train=y_train, X_test= X_test, y_test= y_test, models= models)

            # To get the best model score

            best_model_score = max(sorted(model_report.values()))

            # To get the best model name from the dict 
            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score<0.6:
                raise CustomException("No Best Model Found")
            
            logging.info(f"Best model found on both training and testing data")

            save_object(file_path=ModelTrainerConfig.trained_model_file_path, obj=best_model)

            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_pred=predicted, y_true=y_test)
            return r2_square

        
        
        except Exception as e:
            raise CustomException(e, sys)
        