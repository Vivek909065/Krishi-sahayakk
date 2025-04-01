import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        '''
        This function is responsible for data transformation
        '''
        try:
            # Define categorical and numerical columns
            numerical_columns = ["Zn", "Cu", "Fe", "Mn", "B", "S"]
            categorical_columns = ["District"]  # Added this line to define categorical_columns

            logging.info(f"Categorical columns: {categorical_columns}")
            logging.info(f"Numerical columns: {numerical_columns}")

            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="mean")),  # Changed "average" to "mean"
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(sparse_output=False)),  # Dense output for scaler
                    ("scaler", StandardScaler())
                ]
            )

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline", num_pipeline, numerical_columns),
                    ("cat_pipeline", cat_pipeline, categorical_columns)
                ]
            )

            return preprocessor
        
        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # Verify file existence
            if not os.path.exists(train_path):
                raise FileNotFoundError(f"Training file not found at: {train_path}")
            if not os.path.exists(test_path):
                raise FileNotFoundError(f"Test file not found at: {test_path}")

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info(f"Read train data with columns: {train_df.columns.tolist()}")
            logging.info(f"Read test data with columns: {test_df.columns.tolist()}")

            logging.info("Obtaining preprocessing object")
            
            try:
                preprocessing_obj = self.get_data_transformer_object()
            except Exception as e:
                raise CustomException(e, sys)
            
            target_column_name = "Zn"  # Update this based on soil.csv (e.g., "Zn (ppm)")

            # Check if target column exists
            if target_column_name not in train_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in training data. Available columns: {train_df.columns.tolist()}", sys)
            if target_column_name not in test_df.columns:
                raise CustomException(f"Target column '{target_column_name}' not found in test data. Available columns: {test_df.columns.tolist()}", sys)

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df = train_df[target_column_name]

            input_feature_test_df = test_df.drop(columns=[target_column_name], axis=1)
            target_feature_test_df = test_df[target_column_name]

            logging.info("Applying preprocessing object on training and testing dataframes")

            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, np.array(target_feature_train_df)]
            test_arr = np.c_[input_feature_test_arr, np.array(target_feature_test_df)]

            logging.info("Saved preprocessing object")
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )

            return (
                train_arr,
                test_arr,
                self.data_transformation_config.preprocessor_obj_file_path,
            )
        
        except Exception as e:
            raise CustomException(e, sys)

if __name__ == "__main__":
    from src.components.data_ingestion import DataIngestion
    obj = DataIngestion()
    train_data, test_data = obj.initiate_data_ingestion()
    data_transformation = DataTransformation()
    train_arr, test_arr, preprocessor_path = data_transformation.initiate_data_transformation(train_data, test_data)
    logging.info(f"Transformation completed. Preprocessor saved at: {preprocessor_path}")