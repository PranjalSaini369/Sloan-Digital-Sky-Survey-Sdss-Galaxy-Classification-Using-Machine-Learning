#Impoting Libraries 
from flask import Flask , request  , render_template 
import pickle
import numpy as np 

flask_app = Flask(__name__)
model = pickle.load("rf_redshift_model.pkl" , "rb" )

@flask_app.route("/")
def Home():
    return render_template("index-main.html")
@flask_app.route("/predict" , method=["POST"])
def predict():
    float_feature=[float(x) for x in request.form.values()]
    features = [np.array(float_feature)]
    prediction = model.predict(features)
    return render_template("Index-main.html" , prediction_text="The Galaxy is {}".format(prediction))

if __name__ == "__main__":
    flask_app.run(debug=True)