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

fr="data2.csv"
fr=sys.argv[1]
dat_set=pd.read_csv(fr,header=0 )  
COLUMNS=dat_set.columns[:]
LABEL=COLUMNS[-1]
FEATURES=COLUMNS[:-1]
df=pd.DataFrame(dat_set,columns=COLUMNS)
data_set0=df.sample(frac=1.0,random_state=0)

num=[]
nright=[]
aright=np.zeros((3,3))
maxl=[]
for i in range(1000):
	data_set=data_set0.copy()
	train_dataset = data_set.sample(frac=0.9, random_state=i)  # 
	test_dataset = data_set.drop(train_dataset.index)
	train_stats = train_dataset.describe() #Generates descriptive statistics that summarize
	train_stats.pop(LABEL)  # kick out 513 column from  train_stats
	train_stats = train_stats.transpose()# for better view
	train_labels = train_dataset.pop(LABEL) # kick out label column from train_stats and give to train_labels
	test_labels = test_dataset.pop(LABEL)
	normed_train_data = norm(train_dataset)
	normed_test_data = norm(test_dataset)
	x_train=normed_train_data
	y_train=train_labels
	x_test=normed_test_data
	y_test =test_labels
	numtmp=[len(y_test[y_test==1]),len(y_test[y_test==2]),len(y_test[y_test==3])]
	right=numtmp.copy()
	reg=MLPClassifier(max_iter=2000,solver='lbfgs', alpha=1e-4,hidden_layer_sizes=(16), random_state=i) #
	reg.fit(x_train, y_train)
	y=reg.predict(x_test)
	for ii in range(len(y_test)):
		aright[y_test.iloc[ii]-1,y[ii]-1]=aright[y_test.iloc[ii]-1,y[ii]-1]+1
		if (y_test.iloc[ii] != y[ii]):
			#print("Labeled %d, Predicted %d"%(y_test.iloc[ii],y[ii]))
			right[y_test.iloc[ii]-1]=right[y_test.iloc[ii]-1]-1
	#print(numtmp)
	#print(right)
	#maxa=right[0]/numtmp[0]+right[1]/numtmp[1]+right[2]/numtmp[2]
	#print("prediction accuracy:",i,right[0]/numtmp[0],right[1]/numtmp[1],right[2]/numtmp[2],maxa)
	if len(num)==0:
		num=numtmp
		nright=right
		#maxl=[maxa]
	else:
		num=np.vstack((num,numtmp))
		nright=np.vstack((nright,right))
		#maxl.append(maxa)

print("Total tested:",num[:,0].sum(),num[:,1].sum(),num[:,2].sum())
print("Correct prediction:",nright[:,0].sum(),nright[:,1].sum(),nright[:,2].sum())
print(aright)
acc=[nright[:,0].sum()/num[:,0].sum(),nright[:,1].sum()/num[:,1].sum(),nright[:,2].sum()/num[:,2].sum()]
print("prediction accuracy:",acc)

#print("max",maxl.argmax(),maxl.max())
'''
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN.py set2dcb.csv
Total tested: 5044 4951 5005
Correct prediction: 5044 4951 5005
[[5044.    0.    0.]
 [   0. 4951.    0.]
 [   0.    0. 5005.]]
prediction accuracy: [1.0, 1.0, 1.0]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN.py set2dseqhalf.csv
Total tested: 5044 4951 5005
Correct prediction: 4556 4868 4660
[[4556.   44.  444.]
 [  83. 4868.    0.]
 [ 345.    0. 4660.]]
prediction accuracy: [0.9032513877874703, 0.9832357099575844, 0.9310689310689311]
'''
