import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder,LabelEncoder,StandardScaler
from sklearn.model_selection import train_test_split

data = pd.read_csv('/Users/jaygupta/PycharmProjects/ML/Data Preprocessing/Data.csv') # this is used to get csv file
x = data.iloc[:, :-1].values # this is used to get independent variable
y = data.iloc[:, -1].values # this is used to get dependent varibale

imputer = SimpleImputer(missing_values=np.nan, strategy='mean') # this is used to convet null values into mean of the coloum
imputer.fit(x[:, 1:3]) # .fit is used because imputer class dont have a fir_transform method , fit will only fit the formual but will not trasform the data
x[:, 1:3] = imputer.transform(x[:, 1:3]) # . transform will transform the fir into the data

ct = ColumnTransformer(transformers=[('encoder', OneHotEncoder(), [0])], remainder='passthrough') # this is used to get categorical values
x = ct.fit_transform(x)

le = LabelEncoder()  # this is used to convert the dependent variable into 1,0
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state= 1) # this is used to split the data into training and test set

# feature scaling is done after the split because of data leakage

sc = StandardScaler() # this function does stardarization and is used for feature scaling
x_train[:, 3:] = sc.fit_transform(x_train[:, 3:]) # we have used only the numerical value not 1,0 because the country column does't require feature scaling
x_test[:, 3:] = sc.transform(x_test[:, 3:]) # here we have used the transform method because we want to use the older object
print(x_train)
#print(x_test)
#print(y_train)
#print(y_test)
