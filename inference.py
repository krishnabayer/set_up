import numpy as np
import pandas as pd
from sklearn.datasets import  load_breast_cancer as cancer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix  
from joblib import load, dump


svc_model = load('svc_model.joblib')
data_scaler = load('scaler.joblib')



def predict_breast_cancer(feature_list):
    inference_data = feature_list
    scaled_data = data_scaler.transform([inference_data])
    prediction = svc_model.predict(scaled_data)
    if prediction[0]==1:
        print('breast cancer identified')
    else:
        print('breast cancer not present')
    return prediction[0]    