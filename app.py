"""from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application=Flask(__name__)

app=application

## Route for a home page

@app.route('/')
def index():
    return render_template('main.html') 

@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    if request.method=='GET':
        return render_template('home.html')
    else:
        data=CustomData(
            District=request.form.get('District '),
            Zn=float(request.form.get('Zn')),
            Fe=float(request.form.get('Fe')),
            Cu=float(request.form.get('Cu')),
            Mn=float(request.form.get('Mn')),
            B=float(request.form.get('writing_score')),
            S=float(request.form.get('reading_score'))

        )
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
        print("Before Prediction")

        predict_pipeline=PredictPipeline()
        print("Mid Prediction")
        results=predict_pipeline.predict(pred_df)
        print("after Prediction")
        return render_template('home.html',results=results[0])
    

if __name__=="__main__":
    app.run(host="0.0.0.0")"""
    
from flask import Flask, render_template, request, jsonify
import pandas as pd
from src.exception import CustomException
from src.utils import load_object
import os
import sys

app = Flask(__name__)

# Your existing classes
class PredictPipeline:
    def __init__(self):
        pass

    def predict(self, features):
        try:
            model_path = os.path.join("artifacts", "model.pkl")
            preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            model = load_object(file_path=model_path)
            preprocessor = load_object(file_path=preprocessor_path)
            data_scaled = preprocessor.transform(features)
            preds = model.predict(data_scaled)
            return preds
        
        except Exception as e:
            raise CustomException(e, sys)

class CustomData:
    def __init__(self, District: str, Zn: int, Cu: int, Fe: int, Mn: int, B: int, S: int):
        self.District = District
        self.Zn  = Zn 
        self.Cu  = Cu 
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

# Routes
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = CustomData(
            District=request.form['District'],
            Zn=int(request.form['Zn']),
            Cu=int(request.form['Cu']),
            Fe=int(request.form['Fe']),
            Mn=int(request.form['Mn']),
            B=int(request.form['B']),
            S=int(request.form['S'])
        )
        
        # Convert to dataframe
        df = data.get_data_as_data_frame()
        
        # Make prediction
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df)
        
        return jsonify({'prediction': float(prediction[0])})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True)