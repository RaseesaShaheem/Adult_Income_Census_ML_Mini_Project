from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd


app = Flask(__name__)

model=pickle.load(open('best_random_forest_model.pkl','rb'))
print("Model loaded successfully!")

Education_encoder=pickle.load(open('Education_encoder.pkl','rb'))
Marital_status_encoder=pickle.load(open('Marital_status_encoder.pkl','rb'))
Native_Country_encoder=pickle.load(open('Native_Country_encoder.pkl','rb'))
Occupation_encoder=pickle.load(open('Occupation_encoder.pkl','rb'))
Workclass_encoder=pickle.load(open('Workclass_encoder.pkl','rb'))
Relationship_encoder=pickle.load(open('Relationship_encoder.pkl','rb'))
scaler=pickle.load(open('scaler.pkl','rb'))

columns = [
    'Age', 'Workclass', 'Education', 'Education-num', 'Marital_status',
    'Occupation', 'Relationship', 'Sex', 'Capital_Gain', 'Capital_Loss',
    'Hours_Per_Week', 'Native_Country'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
           # Convert numeric inputs
           age = float(request.form['age'])
           Education_num = float(request.form['Education-num'])
           Capital_Gain = float(request.form['Capital_Gain'])
           Capital_Loss = float(request.form['Capital_Loss'])
           Hours_Per_Week = float(request.form['Hours_Per_Week'])

           # Get categorical inputs
           Workclass = request.form['Workclass']
           Education = request.form['Education']
           Marital_status = request.form['Marital_status']
           Occupation = request.form['Occupation']
           Relationship = request.form['Relationship']
           Sex = request.form['Sex']
           Native_Country = request.form['Native_Country']

           Workclass_val = Workclass_encoder.transform([Workclass])[0]
           Education_val = Education_encoder.transform([Education])[0]
           Marital_status_val = Marital_status_encoder.transform([Marital_status])[0]
           Occupation_val = Occupation_encoder.transform([Occupation])[0]
           Relationship_val = Relationship_encoder.transform([Relationship])[0]
           Native_Country_val = Native_Country_encoder.transform([Native_Country])[0]
           
           Sex_num = 1 if Sex == 'Male' else 0

           final_features = [
                age, Workclass_val, Education_val, Education_num, Marital_status_val,
                Occupation_val, Relationship_val, Sex_num,
                Capital_Gain, Capital_Loss, Hours_Per_Week, Native_Country_val
            ]
           
           data_out = pd.DataFrame([final_features], columns=columns)
           print(data_out)
           print(data_out.shape)
           final_scaled = scaler.transform(data_out)



           predicted_data = model.predict(final_scaled)
           output = "greater than 50" if predicted_data[0] == 1 else "less than or equal to 50"
           return render_template('index.html',prediction_text=f'Predicted Income: {output}')
        
        except Exception as e:
            print("Error:", e)
            return render_template('index.html', prediction_text=f'Error: {str(e)}')
        
if __name__ == "__main__":
    app.run(debug=True)        







 