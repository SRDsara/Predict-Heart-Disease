import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from seaborn import countplot
from sklearn import preprocessing, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

#1-Import dataset
df = pd.read_csv("F:\\tutorials\machine learning\machine_learning_with_python_jadi-main\heart_disease.csv")

#2- Cleaning dataset
#2-1- label encoding
le_gender = preprocessing.LabelEncoder().fit(["female","male"])
df["gender"] = le_gender.fit_transform(df["gender"])

#2-2- delete extra columns
df1 = df.drop('education', axis=1)

#2-3- drop Nan values
#print(df1.isnull().sum())
df1.dropna(inplace=True)
#print(df1.isnull().sum())

#3- plotting
fig, ax = plt.subplots(2)
sns.countplot(x='gender',hue='TenYearCHD',data=df, ax=ax[0])
sns.countplot(x='age',hue='TenYearCHD', data=df, ax=ax[1])

fig, ax = plt.subplots(2)
sns.countplot(x='cigsPerDay',hue='TenYearCHD',data=df, ax=ax[0])
sns.countplot(x='prevalentStroke',hue='TenYearCHD', data=df, ax=ax[1])
plt.show()

#4- define X(features) and y(label)
X = df1[["gender","age","currentSmoker","cigsPerDay","BPMeds","prevalentStroke","prevalentHyp","diabetes","totChol","sysBP","diaBP","BMI","heartRate","glucose"]]
y = df1[["TenYearCHD"]]

#5- normalize dataset
scaler = preprocessing.StandardScaler().fit(X)
X = scaler.fit_transform(X)

#6- train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

#7- Modeling
LR = LogisticRegression().fit(X_train, y_train.values.ravel())

#8- predict
y_pred = LR.predict(X_test)

#9- confusiosn matrix
confusion_matrix = metrics.confusion_matrix(y_test, y_pred)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
cm_display.plot()
plt.show()

#10- check accuracy
print(accuracy_score(y_test, y_pred))

#11- check the probablity of heart disease 
y_pred_prob = LR.predict_proba(X_test)