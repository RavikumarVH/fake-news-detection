from flask import  Flask,request,jsonify
import pandas as pd
import pickle
from flask_cors import CORS, cross_origin

app=Flask(__name__)

def load_pickle(filename):
    f=open(filename,"rb")
    model = pickle.load(f)
    f.close()
    return  model

model=load_pickle("fakenewsdetection_model.pkl")
scaler=load_pickle("fakenewsdetection_scalar.pkl")

def output_lable(n):
    if n == 0:
        return "Fake News"
    elif n == 1:
        return "Not A Fake News"

def model_testing(text):
    news={"text":[text]}
    new_df=pd.DataFrame(news)
    new_x_test=new_df["text"]
    new_x_test=scaler.transform(new_x_test)
    pred=model.predict(new_x_test)
    return output_lable(pred[0])

@app.route("/")
@cross_origin()
def home():
    data={
        "WelcomeMsg":"Welcome to flask application"
    };
    return jsonify(data)

@app.route("/predictFakeNews", methods=["POST"])
@cross_origin()
def predictFakeNews():
    req_data=request.get_json(force=True)
    text = req_data['news']
    result = model_testing(text);
    data = {
        "Result": result
    };
    return jsonify(data)

if __name__ == "__main__":
  app.run()