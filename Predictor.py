import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.svm import SVC
import joblib
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


# Get data from csv file
df = pd.read_csv(
    'D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\dataset.csv')


df.dropna(inplace=True)
df = df.sample(frac=1).reset_index(drop=True)

# Separate Labels from features
print(df['label'])
x = df.drop(['label'], axis=1)
y = df['label']

# Split data for testing and training
x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=42)


scalar = StandardScaler()
x = scalar.fit_transform(x)
model = SVC(kernel='linear')

# Training Model
model.fit(x_train, y_train)

joblib.dump(model, "D:\\MCA Study Material\\Coding_Practice\\MachineLearning\\Udemy_Projects\\Hand-Written Digits Recognization\\model.lib")

y_pred = model.predict(x_test)
print("Accuracy Score", accuracy_score(y_test, y_pred))
print(list(y_test))
print(list(y_pred))
