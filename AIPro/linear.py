import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
#read data and remove any NaN
d=pd.read_csv("IMDB-Movie-Data.csv")
d=d.dropna(axis=0, how='any')
#Splitting and evzluating data
trainx,testx,trainy,testy=train_test_split(d[d.columns[6:32]],d.iloc[:,-1],test_size=0.25,random_state=0)
s = StandardScaler()

# trainx = np.array(trainx).reshape(len(trainx), -1)

trainx =s.fit_transform(trainx)
testx=s.transform(testx)
trainx = np.array(trainx).reshape(len(trainx), -1)
trainy = np.array(trainy).reshape(len(trainy), -1)
#Training model
lr = LinearRegression()
predy = lr.fit(trainx, trainy)

#Evaluation
confusion_mat = confusion_matrix(testy,predy)
accuracy = accuracy_score(testy,predy)
precision = precision_score(testy,predy)
recall = recall_score(testy,predy)
print('Confusion Matrix is :',confusion_mat)
print('\nAccuracy:',accuracy)
print('\nPrecision:',precision)
print('\nRecall: ',recall)
fpr, tpr, thresh = roc_curve(testy, predy)
roc_auc = auc(fpr, tpr)
plt.title('ROC')
plt.plot(fpr, tpr, 'k',label='AUC = %0.2f'% roc_auc)
plt.legend(loc='lower right')
plt.plot([0,1],[0,1],'b--')
plt.xlim([-0.1,1.2])
plt.ylim([-0.1,1.2])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


