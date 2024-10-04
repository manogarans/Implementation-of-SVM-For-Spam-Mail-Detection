# Implementation-of-SVM-For-Spam-Mail-Detection

## AIM:
To write a program to implement the SVM For Spam Mail Detection.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
 1.Import the necessary python packages using import statements.

 2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

 3.Split the dataset using train_test_split.

 4.Calculate Y_Pred and accuracy.

 5.Print all the outputs.

 6.End the Program.

## Program:
Program to implement the SVM For Spam Mail Detection..
### Developed by: MANOGARAN S
### RegisterNumber:  212223240081
```
import chardet
file='spam.csv'
with open (file,'rb') as rawdata:
    result = chardet.detect(rawdata.read(100000))
result

import pandas as pd
data=pd.read_csv("spam.csv",encoding='windows-1252')

data.head()

data.info()

data.isnull().sum()

x=data["v1"].values
y=data["v2"].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer()

x_train=cv.fit_transform(x_train)
x_test=cv.transform(x_test)

from sklearn.svm import SVC
svc=SVC()
svc.fit(x_train,y_train)
y_pred=svc.predict(x_test)
y_pred

from sklearn import metrics
accuracy=metrics.accuracy_score(y_test,y_pred)
accuracy
```
## Output:
### ENCODING :
![Screenshot 2024-10-04 103858](https://github.com/user-attachments/assets/0906892c-f64a-4054-b8cb-9224435489d6)
### HEAD() :
![Screenshot 2024-10-04 103920](https://github.com/user-attachments/assets/2f88fb16-0115-409f-a025-4e8d3a070783)
### INFO() :
![Screenshot 2024-10-04 103930](https://github.com/user-attachments/assets/0e5e9d3b-9b5a-4345-aa05-2552e74cf619)
### Isnull().sum():
![Screenshot 2024-10-04 103938](https://github.com/user-attachments/assets/29b9531f-e923-4309-9816-80b68adfc36c)
### Prediction of y:
![Screenshot 2024-10-04 103948](https://github.com/user-attachments/assets/4dbad751-a1ec-4ca5-9ff2-f5f56992246c)
### Accuracy:
![Screenshot 2024-10-04 103959](https://github.com/user-attachments/assets/69c6aa1a-febd-4260-92e8-d265c947e7f6)


## Result:
Thus the program to implement the SVM For Spam Mail Detection is written and verified using python programming.
