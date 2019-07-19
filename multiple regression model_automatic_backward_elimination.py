# multiple regression model

import numpy as np
#import matplotlib.pyplot as plt
import pandas as pd

#importing the dataset
dataset=pd.read_csv('50_Startups.csv')
X=dataset.iloc[:, :-1].values
y=dataset.iloc[:,4].values


#Encoding catagorical data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
label = LabelEncoder()
X[:,3]= label.fit_transform(X[:,3])
onehot = OneHotEncoder(categorical_features=[3])
X=onehot.fit_transform(X).toarray()

#avoiding the dummy variable trap
X=X[:, 1:]

#splitting thbe dataset into the Training set and test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test =  train_test_split(X, y, test_size = 0.1, random_state=0)

#featuring scaling
"""from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train  = sc_X.tranform(X_train)
X_test = sc_x.transform(x_test)
sc_y = StandardScaler()
y_train = sc_y.fit_transform(y_train)"""

#fitting the multiple linear regression to the training set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train,y_train)

#predicting the test set results
y_pred = regressor.predict(X_test)

#building the automatic optimal model using backward elimination
import statsmodels.formula.api as sm
def backwardeli(X,sl):
    num=len(X[0])
    for i in range(0 , num):
        regressor_ols = sm.OLS(endog = y,exog = X).fit()
        maxvar = max(regressor_ols.pvalues).astype(float)
        if maxvar > sl:
            for j in  range(0, num - i):
                if( regressor_ols.pvalues[j].astype(float) == maxvar):
                    X = np.delete(X,j,i)
    regressor_ols.summary()
    return X
sl = 0.05
X_opt = X[:, [0, 1, 2, 3, 4]]
X_modeled = backwardeli(X_opt, sl)
