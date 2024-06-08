import numpy as np
import pandas as pd
#import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.neural_network import MLPClassifier

import sys 
from scipy import stats

def norm(x):
	return (x - train_stats['mean']) / train_stats['std']

fr="set2dseqhalf.csv"
fr="set2dcb.csv"
fr=sys.argv[1]
dat_set=pd.read_csv(fr,header=0 )  
COLUMNS=dat_set.columns[:]
LABEL=COLUMNS[-1]
FEATURES=COLUMNS[:-1]
df=pd.DataFrame(dat_set,columns=COLUMNS)


df1=df[df.label==1]
df2=df[df.label==2]
df3=df[df.label==3]

dftrain=pd.concat([df1.iloc[:-5],df2.iloc[:-5],df3.iloc[:-5]],axis=0)
dftest=pd.concat([df1.iloc[-5:],df2.iloc[-5:],df3.iloc[-5:]],axis=0)

num=[]
nright=[]
aright=np.zeros((3,3))
maxl=[]
i=0

train_stats = dftrain.describe() #Generates descriptive statistics that summarize
train_stats.pop(LABEL)  # kick out 513 column from  train_stats
train_stats = train_stats.transpose()# for better view
train_labels = dftrain.pop(LABEL) # kick out label column from train_stats and give to train_labels
test_labels = dftest.pop(LABEL)
normed_train_data = norm(dftrain)
normed_test_data = norm(dftest)
x_train=normed_train_data
y_train=train_labels
x_test=normed_test_data
y_test =test_labels
numtmp=[len(y_test[y_test==1]),len(y_test[y_test==2]),len(y_test[y_test==3])]
right=numtmp.copy()
reg=MLPClassifier(max_iter=2000,solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(16), random_state=i) #
reg.fit(x_train, y_train)
y=reg.predict(x_test)
for ii in range(len(y)):
	aright[y_test.iloc[ii]-1,y[ii]-1]=aright[y_test.iloc[ii]-1,y[ii]-1]+1
	if (y_test.iloc[ii] != y[ii]):
		#print("Labeled %d, Predicted %d"%(y_test.iloc[ii],y[ii]))
		right[y_test.iloc[ii]-1]=right[y_test.iloc[ii]-1]-1

print(numtmp)
print(right)
#
'''
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN_rdm0.py set2dseqhalf.csv
[5, 5, 5]
[4, 5, 4]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN_rdm0.py set2dcb.csv
[5, 5, 5]
[5, 5, 5]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$
'''