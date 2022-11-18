import pandas as py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

data = py.read_csv('Salary_Data.csv') #importing csv
x = data.iloc[:, :-1].values #getting independent variable
y = data.iloc[:, -1].values # getting dependent the variable

x_train, x_test, y_train ,y_test = train_test_split(x,y,test_size=0.2,random_state=1) #splitiing data into training and test set

reg = LinearRegression() #implementing LinearRegression class
reg.fit(x_train,y_train) #fitting both training that we split into the regression formula

plt.scatter(x_train,y_train,color = 'red') # using scatter plot to do visualization
plt.plot(x_train,reg.predict(x_train), color = 'blue') #plotting the scatter plot to show dot;s and regression line
plt.title('Salary Vs Exp')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show() # using show method to implement the plot

plt.scatter(x_test,y_test,color = 'red')
plt.plot(x_train,reg.predict(x_train), color = 'blue')
plt.title('Salary Vs Exp')
plt.xlabel('Years of exp')
plt.ylabel('Salary')
plt.show()