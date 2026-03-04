from flask import Flask, render_template, request
import mlflow
from flask_app.preprocessing import normalize_text
from dotenv import load_dotenv
import os
import pickle

# Set up DagsHub credentials for MLflow tracking
dagshub_token = os.getenv("DAGSHUB_PAT")
if not dagshub_token:
    raise EnvironmentError("DAGSHUB_PAT environment variable is not set")

os.environ["MLFLOW_TRACKING_USERNAME"] = dagshub_token
os.environ["MLFLOW_TRACKING_PASSWORD"] = dagshub_token

dagshub_url = "https://dagshub.com"
repo_owner = "abhirupray14"
repo_name = "mlops-mini-project"

# Set up MLflow tracking URI
mlflow.set_tracking_uri(f'{dagshub_url}/{repo_owner}/{repo_name}.mlflow')



app = Flask(__name__)

#load model from model registry
model_name = "my_model"
model_version = 1

model_uri = f"models:/{model_name}/{model_version}"
model = mlflow.pyfunc.load_model(model_uri)

vectoriser = pickle.load(open('models/vectorizer.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html',result = None)

@app.route('/predict',methods=['POST'])

def predict():
    text = request.form['text']
   
    # clean
    text = normalize_text(text)

    # bow
    features = vectoriser.transform([text])

    #predict
    result = model.predict(features)

    return render_template('index.html', result=result[0]) 

if __name__ == '__main__':
    app.run(debug=True,host = "0.0.0.0")