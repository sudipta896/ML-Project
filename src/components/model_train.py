import os
import sys
from dataclasses import dataclass

import numpy as np
from sklearn.metrics import r2_score

from catboost import CatBoostRegressor
from sklearn.ensemble import (
    AdaBoostRegressor,
    GradientBoostingRegressor,
    RandomForestRegressor,
)
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor

from src.exception import CustomException
from src.logger import logging
from src.utils import save_object, evaluate_models


# ==============================
# Config
# ==============================
@dataclass
class ModelTrainerConfig:
    trained_model_file_path: str = os.path.join("artifacts", "model.pkl")


# ==============================
# Trainer Class
# ==============================
class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array, test_array):
        try:
            logging.info("Splitting training and test data")

            X_train, y_train, X_test, y_test = (
                train_array[:, :-1],
                train_array[:, -1],
                test_array[:, :-1],
                test_array[:, -1],
            )

            # ==============================
            # Models
            # ==============================
            models = {
                "Random Forest": RandomForestRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Gradient Boosting": GradientBoostingRegressor(),
                "Linear Regression": LinearRegression(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False, allow_writing_files=False),
                "AdaBoost": AdaBoostRegressor(),
            }

            # ==============================
            # Hyperparameters
            # ==============================
            params = {
                "Decision Tree": {
                    "criterion": ["squared_error", "friedman_mse", "absolute_error"],
                },
                "Random Forest": {
                    "n_estimators": [16, 32, 64, 128],
                },
                "Gradient Boosting": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [32, 64, 128],
                },
                "Linear Regression": {},
                "XGBRegressor": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [32, 64, 128],
                },
                "CatBoost": {
                    "depth": [6, 8],
                    "learning_rate": [0.01, 0.1],
                    "iterations": [50, 100],
                },
                "AdaBoost": {
                    "learning_rate": [0.1, 0.01],
                    "n_estimators": [32, 64, 128],
                },
            }

            # ==============================
            # Model Evaluation
            # ==============================
            model_report = evaluate_models(
                X_train=X_train,
                y_train=y_train,
                X_test=X_test,
                y_test=y_test,
                models=models,
                param=params,
            )

            logging.info(f"Model Report: {model_report}")

            # ==============================
            # Best Model Selection
            # ==============================
            best_model_score = max(model_report.values())
            best_model_name = max(model_report, key=model_report.get)

            logging.info(f"Best Model: {best_model_name}")
            logging.info(f"Best Score: {best_model_score}")

            # ==============================
            # Safety Check
            # ==============================
            if best_model_score < 0.6:
                raise CustomException("No good model found", sys)

            # ==============================
            # IMPORTANT FIX: Train Best Model
            # ==============================
            best_model = models[best_model_name]
            best_model.fit(X_train, y_train)   # ✅ FIXED HERE

            # ==============================
            # Save Model
            # ==============================
            save_object(
                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model,
            )

            logging.info("Best model saved successfully")

            # ==============================
            # Final Evaluation
            # ==============================
            predicted = best_model.predict(X_test)
            r2_square = r2_score(y_test, predicted)

            logging.info(f"Final R2 Score: {r2_square}")

            return r2_square

        except Exception as e:
            raise CustomException(e, sys)


# ==============================
# Run block
# ==============================
# if __name__ == "__main__":
#     from src.components.data_transformation import DataTransformation

#     data_transformation = DataTransformation()

#     train_arr, test_arr, _ = data_transformation.initiate_data_transformation(
#         train_path="artifacts/train.csv",
#         test_path="artifacts/test.csv",
#     )

#     trainer = ModelTrainer()
#     score = trainer.initiate_model_trainer(train_arr, test_arr)

#     print(f"Model Training Completed. R2 Score: {score}")