# -*- coding: utf-8 -*-
"""
Created on Sat Feb  5 09:46:47 2022

@author: kushp
"""
import pandas as pd, numpy as np

input_filename_pneumonia = "pneumonia_data.csv"
input_filename_symptoms = "symptoms_data.csv"

df_pneumonia = pd.read_csv(input_filename_pneumonia, sep=',', encoding='latin-1')
df_pneumonia = df_pneumonia.rename(columns={'Body temprature (C)': 'Fever'})

for i in range(len(df_pneumonia["Gender"])):
    
    if df_pneumonia["Gender"][i] == "Female": 
        
        df_pneumonia["Gender"][i] = 0
        
    else: 
        
        df_pneumonia["Gender"][i] = 1

for i in range(len(df_pneumonia["Age years (60 Yrs)"])):
    
    if df_pneumonia["Age years (60 Yrs)"][i] > 59: 
        
        df_pneumonia["Age years (60 Yrs)"][i] = 1
        
    else: 
        
        df_pneumonia["Age years (60 Yrs)"][i] = 0
    
for i in range(len(df_pneumonia["Cough Details"])):
    
    if df_pneumonia["Cough Details"][i] == "Dry":
        
        df_pneumonia["Cough Details"][i] = 1
        
    else:
        
        df_pneumonia["Cough Details"][i] = 0
    
for i in range(len(df_pneumonia["Shortness of Breath"])):
    
    if df_pneumonia["Shortness of Breath"][i] == "Yes":
        
        df_pneumonia["Shortness of Breath"][i] = 1
    
    else:
        
        df_pneumonia["Shortness of Breath"][i] = 0

for i in range(len(df_pneumonia["Headache"])):
    
    if df_pneumonia["Headache"][i] == "Yes":
        
        df_pneumonia["Headache"][i] = 1
    
    else:
        
        df_pneumonia["Headache"][i] = 0
        
for i in range(len(df_pneumonia["Sore Throat"])):
    
    if df_pneumonia["Sore Throat"][i] == "Yes":
        
        df_pneumonia["Sore Throat"][i] = 1
    
    else:
        
        df_pneumonia["Sore Throat"][i] = 0

for i in range(len(df_pneumonia["Fever"])):
    
    if df_pneumonia["Fever"][i] > 37.5:
        
        df_pneumonia["Fever"][i] = 1
    
    else:
        
        df_pneumonia["Fever"][i] = 0  

for i in range(len(df_pneumonia["Outcome"])):
    
    if df_pneumonia["Outcome"][i] == "COVID-19":
        
        df_pneumonia["Outcome"][i] = 2
    
    elif df_pneumonia["Outcome"][i] == "non COVID-19":
        
        df_pneumonia["Outcome"][i] = 1
        
    else:
        
        df_pneumonia["Outcome"][i] = 0

df_symptoms = pd.read_csv(input_filename_symptoms, sep=',', encoding='latin-1')

delete_rows_sypmtoms = []

for i in range(len(df_symptoms["age_60_and_above"])):
    
    if df_symptoms["age_60_and_above"][i] == "None":
        
        delete_rows_sypmtoms.append(i)
        
for i in range(len(df_symptoms["corona_result"])):
    
    if df_symptoms["corona_result"][i] == "positive":
        
        delete_rows_sypmtoms.append(i)

for i in range(len(df_symptoms["gender"])):
    
    if df_symptoms["gender"][i] == "None":
        
        delete_rows_sypmtoms.append(i)

for i in range(len(df_symptoms["corona_result"])):
    
    if df_symptoms["corona_result"][i] != "negative":
        
        delete_rows_sypmtoms.append(i)

delete_rows_sypmtoms = np.unique(delete_rows_sypmtoms)
        
df_symptoms = df_symptoms.drop(df_symptoms.index[delete_rows_sypmtoms])

df_symptoms = df_symptoms.reset_index(drop = True)

for i in range(len(df_symptoms["gender"])):
    
    if df_symptoms["gender"][i] == "female":
        
        df_symptoms["gender"][i] = 0
    
    else:
        
        df_symptoms["gender"][i] = 1

for i in range(len(df_symptoms["age_60_and_above"])):

    if df_symptoms["age_60_and_above"][i] == "Yes":
        
        df_symptoms["age_60_and_above"][i] = 1
    
    else:
        
        df_symptoms["age_60_and_above"][i] = 0
    
for i in range(len(df_symptoms["corona_result"])):

    if df_symptoms["corona_result"][i] == "negative":
        
        df_symptoms["corona_result"][i] = 0

df_pneumonia = df_pneumonia.append(df_symptoms, ignore_index=True)

df_pneumonia.to_csv('combined_cf_data.csv')