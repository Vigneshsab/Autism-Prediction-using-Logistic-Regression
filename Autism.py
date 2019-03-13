# -*- coding: utf-8 -*-
"""
Created on Sat Aug 25 15:35:04 2018

@author: Vignesh
"""
#importing all the necessary modules
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
#importing dataset
data = pd.read_csv('Autism_data.csv')
#incase if dataset has unncessary rows
data = data.iloc[:, :21]
sns.distplot(data_featured['age_numeric'],bins=50,kde=False)
data.replace("?",np.nan,inplace=True)
data=data.drop('used_app_before',axis=1)
#to find whether all ages belong to the same range, converting to float
data['age_numeric']=data['age_numeric'].apply(lambda x:float(x))
data['age_numeric'].max()
#invalid age 383 is replaced by mean
data.loc[data.age_numeric == 383, 'age_numeric'] = 30
data['age_numeric']=data['age_numeric'].fillna(30)
#nan fields in age are filled with mean 
data=data.drop('ethnicity',axis=1)
data.drop(['country_of_res','relation'],axis=1,inplace=True)
#unncessary fields are dropped
#setting dummy variables for each attribute
sex=pd.get_dummies(data['gender'],drop_first=True)
jaund=pd.get_dummies(data['jaundice'],drop_first=True,prefix="Had_jaundice")
rel_autism=pd.get_dummies(data['autism'],drop_first=True,prefix="Rel_had")
detected=pd.get_dummies(data['Class/ASD'],drop_first=True,prefix="Detected")
data=data.drop(['gender','jaundice','autism','Class/ASD'],axis=1)
#concatenating the new features to new dataset
data_featured=pd.concat([data,sex,jaund,rel_autism,detected],axis=1)
data_featured.head()
#to visualize the number of counts of each age group
sns.distplot(data_featured['age_numeric'],bins=50,kde=False)
#grouping age groups to identify which grp has autism more
groups = data_featured.groupby(['age_desc','Detected_YES'])
groups.size()
#visualizing other features
sns.countplot(x='Detected_YES',data=data_featured)
sns.countplot(x='Detected_YES',hue="m",data=data_featured)

#Machine Learning Part
#Logistic Regression
X=data_featured[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score',
       'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'age_numeric', 'result_numeric', 'm',
       'Had_jaundice_yes', 'Rel_had_yes']]
y=data_featured['Detected_YES']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=101)
from sklearn.linear_model import LogisticRegression
lgr=LogisticRegression()
lgr.fit(X_train,y_train)

pred=lgr.predict(X_test)
from sklearn.metrics import classification_report
print(classification_report(y_true=y_test,y_pred=pred))
from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,pred)
cm