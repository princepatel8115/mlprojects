from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

application =Flask(__name__)

app=application

#rout for home page

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predictdata",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        return render_template("home.html")
    else:
        data=CustomData(   #in this we are reading data from webpage or data which we wiill pass in home page 
            gender=request.form.get("gender"),
            race_ethnicity=request.form.get("ethnicity"),
            parental_level_of_education=request.form.get("parental_level_of_education"),
            lunch=request.form.get("lunch"),
            test_preparation_course=request.form.get("test_preparation_course"),
            reading_score=float(request.form.get("reading_score")),
            writing_score=float(request.form.get("writing_score"))

        )
        #by this function data input in converted into dataframe
        pred_df=data.get_data_as_data_frame()
        print(pred_df)
       
        #now we are defing predict pipeline
        predict_pipeline=PredictPipeline() #created object of predictpipeline
        results=predict_pipeline.predict(pred_df)  #now this object will go to predict function which we have created in predictpipeline
        return render_template("home.html",results=results[0])  #now we have to print this value on webpage whoch is our prediction

if __name__=="__main__":
    app.run(host="0.0.0.0",debug=True)