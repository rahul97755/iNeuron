#from flask import Flask,request,render_template,jsonify
from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn 

#from src.pipeline.prediction_pipeline import CustomData,PredictPipeline


### for local uncomment the filepath and model   ###

filepath = r'C:\mystuff\git hub repo\iNeuron\ML project\cc_default_prediction\artifacts\model.pkl'
model = pickle.load(open(filepath,'rb'))



app=Flask(__name__)
@app.route('/')

def index():
      return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict_target():
      BILL_AMT3=float(request.form.get('BILL_AMT3'))
      AGE =float(request.form.get('AGE'))
      PAY_2 =float(request.form.get('PAY_2'))
      PAY_AMT4 =float(request.form.get('PAY_AMT4'))
      PAY_3 =float(request.form.get('PAY_3'))
      PAY_0 =float(request.form.get('PAY_0'))
      BILL_AMT1 =float(request.form.get('BILL_AMT1'))
      PAY_AMT3 =float(request.form.get('PAY_AMT3'))
      PAY_6 =float(request.form.get('PAY_6'))
      PAY_4 =float(request.form.get('PAY_4'))
      PAY_5 =float(request.form.get('PAY_5'))
      PAY_AMT1 =float(request.form.get('PAY_AMT1'))
      LIMIT_BAL =float(request.form.get('LIMIT_BAL'))
      PAY_AMT6 =float(request.form.get('PAY_AMT6'))
      PAY_AMT2 =float(request.form.get('PAY_AMT2'))
      

      ##prediction
      result = model.predict_proba(np.array([BILL_AMT3,AGE,PAY_2,PAY_AMT4,PAY_3,PAY_0,BILL_AMT1,PAY_AMT3,
                                             PAY_6,PAY_4,PAY_5,PAY_AMT1,LIMIT_BAL,PAY_AMT6,PAY_AMT2]).reshape(1,15))
      
      return render_template('index.html',result=result[:,1])



################## for server ##############
### for server uncomment below abd comment local ##
#if __name__=='__main__':
 #   app.run(host='0.0.0.0',port=8080)    


############# for local #####################
#### for local uncomment below ##
if __name__=='__main__':
      app.run(debug=True)
