import sys
import pandas as pd
import numpy as np
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        try:
            logging.info("Loading the preprocessor and model")
            preprocessor_path = "artifacts/preprocessor.pkl"
            model_path = "artifacts/model.pkl"

            preprocessor = load_object(file_path=preprocessor_path)
            model = load_object(file_path=model_path)

            logging.info("Transforming the features using the preprocessor")
            data_scaled = preprocessor.transform(features)

            logging.info("Making predictions using the trained model")
            preds = model.predict(data_scaled)

            return preds
        except Exception as e:
            logging.error("Error occurred during prediction")
            raise CustomException(e, sys)


class CustomData:
    def __init__(
        self,
        gender: str,
        ethnicity: str,
        parental_level_of_education: str,
        lunch: str,
        test_preparation_course: str,
        reading_score: float,
        writing_score: float,
    ):
        self.gender = gender
        self.ethnicity = ethnicity
        self.parental_level_of_education = parental_level_of_education
        self.lunch = lunch
        self.test_preparation_course = test_preparation_course
        self.reading_score = reading_score
        self.writing_score = writing_score
        logging.info("CustomData class initialized")

    def get_data_as_dataframe(self):
        try:
            custom_data_input_dict = {
                "gender": [self.gender],
                "race_ethnicity": [self.ethnicity],
                "parental_level_of_education": [self.parental_level_of_education],
                "lunch": [self.lunch],
                "test_preparation_course": [self.test_preparation_course],
                "reading_score": [self.reading_score],
                "writing_score": [self.writing_score],
            }
            return pd.DataFrame(custom_data_input_dict, index=[0])
        except Exception as e:
            logging.error("Error in getting data as DataFrame")
            raise CustomException(e, sys)
