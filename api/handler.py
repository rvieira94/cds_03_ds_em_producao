from flask import Flask, request, Response
from rossmann.rossmann import Rossmann
import pandas as pd
import pickle

# loading model
model = pickle.load( open('/PATH/model/model_rossmann.pkl', 'rb') )

app = Flask(__name__)

@app.route( '/rossmann/predict', methods=['POST'] )
def rossmann_predict():
    
    test_json = request.get_json()
    
    if test_json: #there's data
        
        if isinstance(test_json,dict): #unique example
            test_raw = pd.DataFrame(test_json, index=[0])
        else: #multiple exemple
            test_raw = pd.DataFrame(test_json, columns=test_json[0].keys())
            
        # instantiate rossmann class
        pipeline = Rossmann()
        
        # data cleaning
        df1 = pipeline.data_cleaning(test_raw)
        
        # feature engineering
        df2 = pipeline.feature_engineering(df1)
        
        # data preparation
        df3 = pipeline.data_preparation(df2)
        
        # prediction
        df_response = pipeline.prediction(model, test_raw, df3)
        
        Response('{}',status=200,mimetype='application/json')

        return df_response
        
    else:
        return Response('{}',status=200,mimetype='application/json')
    
    
if __name__ == '__main__':
    app.run('0.0.0.0')