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
dat_set=pd.read_csv(fr,header=0 )  
COLUMNS=dat_set.columns[:]
LABEL=COLUMNS[-1]
FEATURES=COLUMNS[:-1]
df=pd.DataFrame(dat_set,columns=COLUMNS)
data_set0=df.sample(frac=1.0,random_state=0)

num=[]
nright=[]
for i in range(100):
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
	for ii in range(len(y)):
		if (y_test.iloc[ii] != y[ii]):
			print("Labeled %d, Predicted %d"%(y_test.iloc[ii],y[ii]))
			right[y_test.iloc[ii]-1]=right[y_test.iloc[ii]-1]-1
	
	if len(num)==0:
		num=numtmp
		nright=right
	else:
		num=np.vstack((num,numtmp))
		nright=np.vstack((nright,right))


acc=[nright[:,0].sum()/num[:,0].sum(),nright[:,1].sum()/num[:,1].sum(),nright[:,2].sum()/num[:,2].sum()]

print("Total tested:",num[:,0].sum(),num[:,2].sum(),num[:,2].sum())
print("Correct prediction:",nright[:,0].sum(),nright[:,1].sum(),nright[:,2].sum())
print("prediction accuracy:",acc)
