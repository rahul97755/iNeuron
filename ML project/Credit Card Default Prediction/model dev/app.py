
from flask import Flask,render_template,request
import pickle
import numpy as np
import sklearn 

### for local uncomment the filepath and model   ###

#filepath = r'C:\mystuff\git hub repo\rd\my python revision\model_development\liner_reg.pkl'
#model = pickle.load(open(filepath,'rb'))

#### for server uncomment the below code####

model = pickle.load(open('liner_reg.pkl','rb'))


app=Flask(__name__)
@app.route('/')

def index():
      return render_template('index.html')



@app.route('/predict',methods=['POST'])
def predict_price():
    crim =float(request.form.get('CRIM'))
    zn =float(request.form.get('ZN'))
    nox =float(request.form.get('NOX'))
    rm =float(request.form.get('RM'))
    dis =float(request.form.get('DIS'))
    rad = int(request.form.get('RAD'))
    paritio =float(request.form.get('PTRATIO'))
    lstat =float(request.form.get('LSTAT'))
    
    
    ##prediction
    result = model.predict(np.array([crim,zn,nox,rm,dis,rad,paritio,lstat]).reshape(1,8))

    #return str(result[0,0])
    return render_template('index.html',result=result[0,0])

################## for server ##############
### for server uncomment below abd comment local ##
if __name__=='__main__':
    app.run(host='0.0.0.0',port=8080)    


############# for local #####################
#### for local uncomment below ##
# if __name__=='__main__':
#     app.run(debug=True)

    