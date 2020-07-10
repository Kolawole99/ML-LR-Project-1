#===========================IMPORTING TREQUIRED LIBRARIES AND DATASETS==========================

#===========================importing libraries, modules and packages======================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
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



