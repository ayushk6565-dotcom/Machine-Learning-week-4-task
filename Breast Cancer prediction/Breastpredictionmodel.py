import pandas as pd 
import numpy as np 
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

data = pd.read_csv("Breast Cancer prediction/data.csv")
df = pd.DataFrame(data)
df = df.drop("id",axis=1)
df.drop(columns=['Unnamed: 32'], inplace=True)
encoder = LabelEncoder()
df["diagnosis"] = encoder.fit_transform(df["diagnosis"])
X = df.drop("diagnosis",axis=1)
y = df["diagnosis"]
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train,X_test,y_train,y_test = train_test_split(X,y,train_size=0.8,random_state=42)
model = LogisticRegression()
model.fit(X_train,y_train)
prediction = model.predict(X_test)


print(classification_report(y_test, prediction))

print(confusion_matrix(y_test, prediction))


