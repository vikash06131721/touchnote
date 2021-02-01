"""
Code for reading the model and predicting
"""

import datetime
import pandas as pd
import joblib
rel_dt = pd.read_csv('relationship.csv')
joblib_clf = joblib.load("best_gs_pipeline_brthday_svc.pkl")

def predict_(inp_json,model):
    user_id = inp_json['user_uuid']
    text = inp_json['text']
    all_json ={}
    
    date = rel_dt[rel_dt['user_uuid']==user_id]
    predicting_df = pd.DataFrame(columns=['all_text_new'], index=[0])
    predicting_df['all_text_new']=text
    pred_ = model.predict(predicting_df)[0]

    if pred_ == 1:
        all_json['birthday'] ='birthday'
        all_json['relationship']=rel_dt[rel_dt['user_uuid']==user_id]['relationship'].to_list()
    else:
        all_json['birthday'] ='not birthday'
        all_json['relationship']=rel_dt[rel_dt['user_uuid']==user_id]['relationship'].to_list()
    return all_json


     


    
