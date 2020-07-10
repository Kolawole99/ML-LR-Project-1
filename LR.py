#===========================IMPORTING TREQUIRED LIBRARIES AND DATASETS==========================

#===========================importing libraries, modules and packages======================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
# %matplotlib inline #needed in jupyter notebooks

#==================Loading the data into the project=================================
churn_df = pd.read_csv("ChurnData.csv")
data = churn_df.head()
print(data)



#===================================DATA PREPROCESSING AND SELECTION==================================

#============================================Data selection=====================================
churn_df = churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip', 'callcard', 'wireless','churn']]
churn_df['churn'] = churn_df['churn'].astype('int')
data = churn_df.head()
print(data)

data_shape = churn_df.shape
print(data_shape)

#=======================================Features and Target selection===========================
X = np.asarray(churn_df[['tenure', 'age', 'address', 'income', 'ed', 'employ', 'equip']])
X[0:5]
print(X)

y = np.asarray(churn_df['churn'])
y [0:5]
print(y)

#======================================Normalize dataset===============================
X = preprocessing.StandardScaler().fit(X).transform(X)
X[0:5]
print(X)



#==============================================TRAIN/TEST SPLIT========================================

X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.2, random_state=4)
print ('Train set:', X_train.shape,  y_train.shape)
print ('Test set:', X_test.shape,  y_test.shape)



#=================================================MODELLING===========================================

#===================================Logistic Regression with SCIKIT-LEARN===============================
LR = LogisticRegression(C=0.01, solver='liblinear').fit(X_train,y_train)
    #other scikit-learn solvers are ‘newton-cg’, ‘lbfgs’, ‘liblinear’, ‘sag’, ‘saga’ solvers
    #the version of Logistic Regression in Scikit-learn, support regularization. Regularization is a technique used to solve the overfitting problem in machine learning models. C parameter indicates inverse of regularization strength which must be a positive float
LR



#============================================PREDICTION=====================================
yhat = LR.predict(X_test)
yhat

yhat_prob = LR.predict_proba(X_test)
    #predict_proba returns estimates for all classes, ordered by the label of classes. So, the first column is the probability of class 1, P(Y=1|X), and second column is probability of class 0, P(Y=0|X):
yhat_prob
print(yhat_prob)


#=========================
