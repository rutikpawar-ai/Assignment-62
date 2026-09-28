# Assignment 62
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt


# step 1 : load the dataset

df = pd.read_csv("Employee_Attrition.csv")

# step 2 : load first few data of dataset
print("first few data of dataset is :")

print(df.head())

# step 3 : check for missung values in dataset

print("check for missing values in dataset :")
print(df.isnull().sum())

# step 4 : identify the numerical and categorical columns in dataset
print("identify the numerical and categorical columns in dataset :")
print(df.dtypes)

# step 5 : convert categorical columns into numerical columns using one hot encoding
print("convert categorical columns into numerical columns using one hot encoding :")

df = pd.get_dummies(df, columns=["OverTime"],drop_first=True, dtype=int)
df["Attrition"] = df["Attrition"].map({"Yes":1, "No":0})


print(df.head())

# step 6 : split the dataset into independent and dependent variables
print("split the dataset into independent and dependent variables :")

x = df.drop(["Attrition"], axis=1)
y = df["Attrition"]

print("independent variables are :")
print(x.head())

print("dependent variable is :")
print(y.head())

# step 7 : split the dataset into traing and testting datase
print("split the dataset into traing and testting datase :")
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3, random_state=42)

# step 8 : apply appropriate feature scaling

print("apply appropriate feature scaling :")

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

# step 9 : design the mlp with at least 2 hidden layers 

print("design the mlp with at least 2 hidden layers :")

model = MLPClassifier(
    hidden_layer_sizes=(5,5),
    activation='relu',
    solver='adam',
    max_iter=1000,
    random_state=42
)

# step 10 : train the network

print("train the network :")

model = model.fit(x_train,y_train)

# step 11 : display the number of iteration required for training the model

print("display the number of iteration required for training the model :")
print(model.n_iter_)

# step 12 : calculate traing accurecy

print("calculate traing accurecy :")

y_train_pred = model.predict(x_train)

train_accurecy = accuracy_score(y_train, y_train_pred)

print("training accurecy is :", train_accurecy * 100)

# step 13 : calculate testing accurecy

print("calculate testing accurecy :")

y_test_pred = model.predict(x_test)

y_test_accurecy = accuracy_score(y_test, y_test_pred)
print("testing accurecy is :", y_test_accurecy * 100)

# step 14 : display confusion matrix

print("display confusion matrix :")
print(confusion_matrix(y_test, y_test_pred))

# step 15 : plot thr loss curve

print("plot thr loss curve :")
plt.plot(model.loss_curve_)
plt.xlabel("Iterations")
plt.ylabel("Loss")
plt.title("Training Loss Curve")
plt.show()

# step 16 : explain wether the model is overfitting or underfitting

print("explain wether the model is overfitting or underfitting :")
if train_accurecy > y_test_accurecy:
    print("The model is overfitting as training accuracy is higher than testing accuracy.")
elif train_accurecy < y_test_accurecy:
    print("The model is underfitting as testing accuracy is higher than training accuracy.")
else:
    print("The model is well-fitted as training and testing accuracies are similar.")


    


