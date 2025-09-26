import os
import sys
from dataclasses import dataclass

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object, evaluate_models
from src.components.data_transformation import DataTransformation


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")


class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test input data")
            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "XGB Regressor": XGBRegressor(),
                "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                "AdaBoost Regressor": AdaBoostRegressor(),
            }

            parameters = {
                "Decision Tree": {
                    "criterion": [
                        "squared_error",
                        "friedman_mse",
                        "absolute_error",
                        "poisson",
                    ],
                },
                "Random Forest": {
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [2, 8, 32, None],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.001, 0.5, 0.3],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                    "max_depth": [2, 8, 32, None],
                },
                "Linear Regression": {},
                "XGB Regressor": {
                    "learning_rate": [0.1, 0.001, 0.5, 0.3],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "CatBoosting Regressor": {
                    "depth": [6, 8, 10],
                    "learning_rate": [0.01, 0.05, 0.1],
                    "iterations": [30, 50, 100],
                },
                "AdaBoost Regressor": {
                    "learning_rate": [0.1, 0.001, 0.5, 0.3],
                    "n_estimators": [8, 16, 32, 64, 128, 256],
                },
                "K-Neighbors Regressor": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"],
                    "metric": ["euclidean", "manhattan", "minkowski"],
                },
            }

            model_report: dict = evaluate_models(
                X_train, y_train, X_test, y_test, models, parameters
            )

            # To get the best model score from the dictionary
            best_model_score = max(model_report.values(), key=lambda x: x["test_r2"])[
                "test_r2"
            ]

            # To get the best model name from the dictionary
            best_model_name = max(
                model_report, key=lambda k: model_report[k]["test_r2"]
            )

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")

            logging.info(
                f"Best model found: {best_model_name} with R2 = {best_model_score}"
            )

            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            predicted = best_model.predict(X_test)

            r2_square = r2_score(y_test, predicted)
            mae = mean_absolute_error(y_test, predicted)
            mse = mean_squared_error(y_test, predicted)

            return r2_square, mae, mse

        except Exception as e:
            raise CustomException(e, sys)
