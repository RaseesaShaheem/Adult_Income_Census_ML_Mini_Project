# Importing libraries
import pandas as pd
import numpy as np
from datetime import datetime

from sklearn.preprocessing import LabelEncoder, StandardScaler

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV

import pickle

# Loading the dataset
df=pd.read_csv('adult.csv')

# droping duplicates
df.drop_duplicates(inplace=True)

# if I remove the outliers the accurecy decreases so not handling ouutliers
outlier_cols=['Age','Final_Sampling_Weight','Education-num','Capital-gain','Capital-loss','Hours-per-week']

# removing unwanted cols
df.drop(columns=['Final_Sampling_Weight','Race'],inplace=True)

# Transformation
df.select_dtypes(include=['int64', 'float64']).skew()
df['Capital_Gain'] = np.log1p(df['Capital_Gain'])
df['Capital_Loss'] = np.log1p(df['Capital_Loss'])
df['Age'] = np.log1p(df['Age'])
df['Hours_Per_Week'] = np.log1p(df['Hours_Per_Week'])

# Feature Engineering
# label encoding
le_WC=LabelEncoder()
le_Edu=LabelEncoder()
le_MS=LabelEncoder()
le_Occ=LabelEncoder()
le_Rel=LabelEncoder()
le_NC=LabelEncoder()

df['Workclass']=le_WC.fit_transform(df['Workclass'])
df['Education']=le_Edu.fit_transform(df['Education'])
df['Marital_status']=le_MS.fit_transform(df['Marital_status'])
df['Occupation']=le_Occ.fit_transform(df['Occupation'])
df['Relationship']=le_Rel.fit_transform(df['Relationship'])
df['Native_Country']=le_NC.fit_transform(df['Native_Country'])

with open('Workclass_encoder.pkl', 'wb') as f:
    pickle.dump(le_WC, f)

with open('Education_encoder.pkl', 'wb') as f:    
    pickle.dump(le_Edu, f)

with open('Marital_status_encoder.pkl', 'wb') as f:
    pickle.dump(le_MS, f)

with open('Occupation_encoder.pkl', 'wb') as f:
    pickle.dump(le_Occ, f)

with open('Relationship_encoder.pkl', 'wb') as f:
    pickle.dump(le_Rel, f)

with open('Native_Country_encoder.pkl', 'wb') as f:
    pickle.dump(le_NC, f)   

# One hot encoding
df['Sex']=pd.get_dummies(df['Sex'], drop_first=True,dtype=int)
df['Income']=pd.get_dummies(df['Income'], drop_first=True,dtype=int)

#Splitting the data 
x=df.drop('Income', axis=1)
y=df['Income'] 

# scaling
scaler=StandardScaler()
x_scaled=scaler.fit_transform(x)

# Converting to dataframe
x=pd.DataFrame(x_scaled)

with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.33, random_state=42)

# Random forest with RandomizedSearchCV
rfc = RandomForestClassifier(random_state=42)

param_dist = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}

rand_search = RandomizedSearchCV(
    estimator=rfc, 
    param_distributions=param_dist,
    n_iter=10,            # try only 10 random combinations
    cv=3,                 # 3-fold CV is faster than 5
    scoring='accuracy',
    random_state=42,
    n_jobs=-1             # use all CPU cores
)

rand_search.fit(x_train, y_train)
best_model = rand_search.best_estimator_

with open('best_random_forest_model.pkl', 'wb') as f:
    pickle.dump(best_model, f)

print("Best model saved successfully!")    