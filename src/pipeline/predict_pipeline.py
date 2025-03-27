import os
import sys
import pandas as pd
from src.exception import CustomException
from src.utils import load_object


class PredictPipeline:
    def __init__(self):
        pass

    def predict(self,features):
        try:
            model_path=os.path.join("artifacts","model.pkl")
            preprocessor_path=os.path.join('artifacts','preprocessor.pkl')
            print("Before Loading")
            model=load_object(file_path=model_path)
            preprocessor=load_object(file_path=preprocessor_path)
            print("After Loading")
            data_scaled=preprocessor.transform(features)
            preds=model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e,sys)



class CustomData:
    def __init__(  self,
        District: str,
        Zn: int,
        
        Cu: int,
        Fe: int,
        Mn: int,
        B: int,
        S: int):

        self.District = District

        self.Zn = Zn

        self.Cu = Cu

        self.Fe = Fe

        self.Mn = Mn

        self.B = B

        self.S = S

    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "District": [self.District],
                "Zn": [self.Zn],
                "Cu": [self.Cu],
                "Fe": [self.Fe],
                "Mn": [self.Mn],
                "B": [self.B],
                "S": [self.S],
            }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:
            raise CustomException(e, sys)