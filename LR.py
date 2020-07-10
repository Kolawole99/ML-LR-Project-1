#===========================IMPORTING TREQUIRED LIBRARIES AND DATASETS==========================

#===========================importing libraries, modules and packages======================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
from sklearn import preprocessing
import matplotlib.pyplot as plt
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