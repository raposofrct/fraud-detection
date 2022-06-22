from flask import Flask,request,Response
from fraud import Fraud
import pickle as pkl
import pandas as pd
import os

app = Flask(__name__)
@app.route('/predict',methods=['POST'])

def predict():
    json = request.get_json()
    
    if json:
        dados = pd.DataFrame(json)
        model = pkl.load(open('pkl/model.pkl','rb'))
        
        dados = Fraud().data_cleaning(dados)
        dados = Fraud().feature_engineering(dados)
        dados = Fraud().data_filtering(dados)
        dados = Fraud().data_preparation(dados)
        return Fraud().get_predictions(model,dados)
        
    else:
        Response('{}',status=200)
    
if __name__=='__main__':
    app.run(host='0.0.0.0',port=os.environ.get('PORT',8080),debug=False)