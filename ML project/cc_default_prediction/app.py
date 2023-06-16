#from flask import Flask,request,render_template,jsonify
from flask import Flask,render_template,request
from src.pipeline.prediction_pipeline import CustomData,PredictPipeline
from src.components.data_transformation import DataModifier



app=Flask(__name__)
@app.route('/')

def index():
      return render_template('index.html')

@app.route('/predict',methods=['GET','POST'])
def predict_target():
    if request.method=='GET':
        return render_template('form.html')
    else:
      data=CustomData(BILL_AMT3=float(request.form.get('BILL_AMT3')),
            AGE =float(request.form.get('AGE')),
            PAY_2 =float(request.form.get('PAY_2')),
            PAY_AMT4 =float(request.form.get('PAY_AMT4')),
            PAY_3 =float(request.form.get('PAY_3')),
            PAY_0 =float(request.form.get('PAY_0')),
            BILL_AMT1 =float(request.form.get('BILL_AMT1')),
            PAY_AMT3 =float(request.form.get('PAY_AMT3')),
            PAY_6 =float(request.form.get('PAY_6')),
            PAY_4 =float(request.form.get('PAY_4')),
            PAY_5 =float(request.form.get('PAY_5')),
            PAY_AMT1 =float(request.form.get('PAY_AMT1')),
            LIMIT_BAL =float(request.form.get('LIMIT_BAL')),
            PAY_AMT6 =float(request.form.get('PAY_AMT6')),
            PAY_AMT2 =float(request.form.get('PAY_AMT2'))
            )
      final_new_data=data.get_data_as_dataframe()   
      
      modifier = DataModifier(final_new_data)
      
      # Call the functions to modify the columns
      modifier.change_column1('BILL_AMT3_woe')
      modifier.change_column2('AGE_woe')
      modifier.change_column3('PAY_2_woe')
      modifier.change_column4('PAY_AMT4_woe')
      modifier.change_column5('PAY_3_woe')
      modifier.change_column6('PAY_0_woe')
      modifier.change_column7('BILL_AMT1_woe')
      modifier.change_column8('PAY_AMT3_woe')
      modifier.change_column9('PAY_6_woe')
      modifier.change_column10('PAY_4_woe')
      modifier.change_column11('PAY_5_woe')
      modifier.change_column12('PAY_AMT1_woe')
      modifier.change_column13('LIMIT_BAL_woe')
      modifier.change_column14('PAY_AMT6_woe')
      modifier.change_column15('PAY_AMT2_woe')


      predict_pipeline=PredictPipeline()
      pred_prob=predict_pipeline.predict(final_new_data)

      if pred_prob>0.26:
           pred='Customer is risky'
      else:
           pred='Customer is not risky' 
      


      return render_template('index.html',result=pred)

                        



################## for server ##############
### for server uncomment below abd comment local ##
#if __name__=='__main__':
 #   app.run(host='0.0.0.0',port=8080)    


############# for local #####################
#### for local uncomment below ##
if __name__=='__main__':
      app.run(debug=True)
