# Flexible Artificial Tactility with Excellent Robustness and Temperature Tolerance Based on Organohydrogel Sensor Array for Robot Motion Detection and Object Shape Recognition
# Adv. Mater. 2024, 2408193


# Data

*.csv files are the recorded experiment data files. 

**For one grasping**

1-4 columns are the signal difference of the grasping

5 column is the object shape: 1 for ball, 2 for cone, 3 for cuboid

**For two grasping**

1-4 columns and 5-8 columns are the signal difference of the first grasping and the second grasping

9 column is the object shape: 1 for ball, 2 for cone, 3 for cuboid


# How to use

please install scikit-learn library

`pip install scikit-learn`

and then just 
```
python NN_rdm0.py onegrasp.csv    # For first 45 graspings of each object are used to train, last 5 of each object for testing.  
python NN_rdm0.py twograsp.csv    # For first 45 graspings of each object are used to train, last 5 of each object for testing.  

python NN.py onegrasp.csv    # mix all of them, and random split the whole set into traning and testing set with ratio 9:1   
python NN.py twograsp.csv     # mix all of them, and random split the whole set into traning and testing set with ratio 9:1   
```

Your run should be like this:
```
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN_rdm0.py  onegrasp.csv
[5, 5, 5]
[4, 5, 4]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN_rdm0.py  twograsp.csv 
[5, 5, 5]
[5, 5, 5]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN.py  twograsp.csv 
Total tested: 5044 4951 5005
Correct prediction: 5044 4951 5005
[[5044.    0.    0.]
 [   0. 4951.    0.]
 [   0.    0. 5005.]]
prediction accuracy: [1.0, 1.0, 1.0]
(autoskl) zhouych@DESKTOP-QCMF2L3:.../manuscript$ python NN.py onegrasp.csv
Total tested: 5044 4951 5005
Correct prediction: 4556 4868 4660
[[4556.   44.  444.]
 [  83. 4868.    0.]
 [ 345.    0. 4660.]]
prediction accuracy: [0.9032513877874703, 0.9832357099575844, 0.9310689310689311]
```

https://onlinelibrary.wiley.com/doi/10.1002/adma.202408193
