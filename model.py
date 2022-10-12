import pandas as pd

import pickle
from sklearn.linear_model import LogisticRegression
from sklearn import svm
import matplotlib.pyplot as plt

df=pd.read_csv('marks.csv')
df.info()


df.isnull().sum()
df.describe()
df.dropna(axis=0,inplace=True)

df['gender'] = df['gender'].map({'male': 1 ,'female': 2})


cdf = df[['gender','internalmarks','internalmarks1','internalmarks2','study_hours','externalmarks']]

x = cdf.iloc[:, :5]
y = cdf.iloc[:, -1]


from sklearn.linear_model import LinearRegression
linearRegression = LinearRegression()
linearRegression.fit(x, y)
lracc = linearRegression.score(x,y)


from sklearn.tree import DecisionTreeRegressor
model = DecisionTreeRegressor(random_state=44)
model.fit(x, y)
dtacc = model.score(x,y)
          
from sklearn.svm import SVR
SVM = SVR()
SVM.fit(x, y)
SVMacc =SVM.score(x, y)


print(SVM.predict([[2,69,90,88,6.56]]))


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0, test_size=0.3, shuffle=False)
SVM.score(X_train, y_train)

svm_acc=round(SVM.score(x,y), 4)



data = {'LinearRegression':lracc*100, 'SVC':SVMacc*100, 'DecisionTree':dtacc*100}
courses = list(data.keys())
values = list(data.values())
  
fig = plt.figure(figsize = (10, 5))
 
# creating the bar plot
plt.bar(courses, values, color =['black', 'red', 'green', 'cyan'], 
        width = 0.4)
 
plt.xlabel("Algorithm")
plt.ylabel("Accuracy")
plt.title("Accuracy of Algorithms")
plt.show()


file=open('my_model.pkl','wb')
pickle.dump(model,file,protocol=2)