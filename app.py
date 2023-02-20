from flask import Flask,render_template,request
import pandas as pd
import pickle


app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')



@app.route('/predict', methods=["POST", "GET"])
def predict():
    test_text1 = request.form['hi']
  
    pred_args =[test_text1]
    ml_model = pickle.load(open('pipeline.pkl','rb'))
    pred = ml_model.predict(pred_args)

    predo = pred[0].astype("int32")
      
    return render_template('predict.html', prediction = predo)


if __name__ == "__main__":
    app.run()