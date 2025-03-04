#===========================IMPORTING TREQUIRED LIBRARIES AND DATASETS==========================

#===========================importing libraries, modules and packages======================
import pandas as pd
import pylab as pl
import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt
import itertools
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import log_loss
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


#============================================EVALUATION========================================

#===========================================Jaccard index========================================
jaccard_similarity_score = jaccard_score(yhat, y_test)
print("Jacccard Similarity Score: ", jaccard_similarity_score)


#=========================================Confusion Matrix======================================
def plot_confusion_matrix(cm, classes, normalize = False, title = 'Confusion matrix', cmap = plt.cm.Blues):
    
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
print(confusion_matrix(y_test, yhat, labels=[1,0]))

#=======================plotting the confusion matrix
# Compute confusion matrix
cnf_matrix = confusion_matrix(y_test, yhat, labels=[1,0])
np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
plt.figure()
plot_confusion_matrix(cnf_matrix, classes=['churn=1','churn=0'],normalize= False,  title='Confusion matrix')

#Report result and comparisons
print (classification_report(y_test, yhat))
    #Precision is a measure of the accuracy provided that a class label has been predicted. It is defined by: precision = TP / (TP + FP)
    #Recall is true positive rate. It is defined as: Recall =  TP / (TP + FN)
    #F1 score reaches its best value at 1 (perfect precision and recall) and worst at 0.


#===========================================Log Loss========================================
log_loss = log_loss(y_test, yhat_prob)
print(log_loss)


# #=======================SAMPLE MODEL USING DIFFERENT LOGISTIC REGRESSION PARAMETERS========================
# LR2 = LogisticRegression(C=0.01, solver='sag').fit(X_train, y_train)
# yhat_prob2 = LR2.predict_proba(X_test)
# print ("LogLoss for sample parameter: %.2f" % log_loss(y_test, yhat_prob2))


