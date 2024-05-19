import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

data = pd.read_csv("diabetes_data.csv")
x = data.drop(columns="Outcome")
y = data['Outcome']

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2, random_state=42)
scaler= StandardScaler()
x_train_scaled = scaler.fit_transform(x_train.values)
x_test_scaled = scaler.transform(x_test.values)

model = LogisticRegression()
model.fit(x_train_scaled, y_train)

y_pred_test = model.predict(x_test_scaled)
y_pred_train = model.predict(x_train_scaled)

accuracy_tarin = accuracy_score(y_train, y_pred_train)
accuracy_test = accuracy_score(y_test, y_pred_test)

print("Train accuracy: ", accuracy_tarin)
print("Test accuracy: ", accuracy_test)

print("Enter the data:")
preg = int(input("Month of pregenancy: "))
glucose = int(input("Enter glucose level: "))
blood = int(input("Enter blood pressure: "))
skin = int(input("Enter your skin thickness: "))
insulin = int(input("Enter your insulin level"))
bmi = float(input("Enter BMI: "))
dpf = float(input("Enter Diabetes Pefigree Function: "))
age = int(input("Enter Age: "))

new_data = np.array([
    preg,
    glucose,
    blood,
    skin,
    insulin,
    bmi,
    dpf,
    age
    ])

new_data_scaled= scaler.transform(new_data)
predictions = model.predict(new_data_scaled)
print("Outcome: ", predictions[0])
