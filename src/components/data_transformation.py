"""import sys
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


@dataclass
class DatatransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")
    
class DataTransformation:

    def __init__(self):
        self.transformation_config = DatatransformationConfig()
        
    def get_data_transformer_object(self):  
        
        #THIS FUNCTION IS RESPONSIBLE FOR DATA TRANSFORMATION
        try:
            numerical_columns=["percentage of Zn","percentage of Cu","percentage of Fe","percentage of Mn","percentage of B","percentage of S"]
            catagorical_columns=["Zn","Fe",
                                 "Cu","Mn",
                                 "B",
                                 "S"
                                 
                                 ]
            
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='median')),
                    ("scaler",StandardScaler())
                ]
                
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",simpleImputer(strategy="most_Frequency")),
                    ("one_hot_encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            
            logging.info("Numerical columns standard scaling completed")
            
            logging.info("Categorical columns encoding completed")
            
            preprocessor=ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns)
                    ("cat_pipelines",cat_pipeline,categorical_columns)
                ]
            )
            
            return preprocessor
            
            )
        except Exception as e:
            
            raise CustomException(e,sys) 
    
    def initiate_data_transformation(self,train_path,text_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)
            
            logging.info("Read train and test data completed")
            
            logging.info("obtaining the preprocessor object")
            
            preprocessing_obj=self.get_data_transformer_object()
            
            target_column_name="Zn"
            
            numerical_columns=["Zn","Fe","Cu","Mn","B","S"]
            
            iniput_feature_train_df=train_df.drop[columns=target_column_name],axis=1)
        except:
            pass"""
            
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
class DatatransformationConfig:
    preprocessor_obj_file_path = os.path.join('artifacts', "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.transformation_config = DatatransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns = [
                "percentage of Zn", "percentage of Cu", "percentage of Fe", 
                "percentage of Mn", "percentage of B", "percentage of S"
            ]
            categorical_columns = ["Zn", "Fe", "Cu", "Mn", "B", "S"]

            num_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler())
            ])

            logging.info("Numerical columns standard scaling completed")
            logging.info("Categorical columns encoding completed")

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipelines", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")
            logging.info("obtaining the preprocessor object")

            preprocessing_obj = self.get_data_transformer_object()

            target_column_name = "Zn"
            numerical_columns = ["Zn", "Fe", "Cu", "Mn", "B", "S"]

            input_feature_train_df = train_df.drop(columns=[target_column_name], axis=1)
            target_feature_train_df=train_df[target_column_name]
            
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            
            logging.info(f"Applying processing objecton training dataframe and testing dataframe.")
            
            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)
            
            train_arr=np.c[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]
            
            logging.info(f"Saved preprocessing object.")
            
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
                
            )
            return(
                train_arr,
                test_arr,
                self.initiate_data_transformation_config.preprocessor_obj_file_path,
            )
            
            # Add rest of the method implementation
            return preprocessing_obj, input_feature_train_df

        except Exception as e:
            raise CustomException(e, sys)