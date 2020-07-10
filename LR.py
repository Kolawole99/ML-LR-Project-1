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