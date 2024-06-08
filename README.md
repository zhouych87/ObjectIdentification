# Data

*.csv files are the experiment data file. 

**For one grasping**

1-4 colums are the signal difference of the grasping

5 colum is the object shape: 1 for ball, 2 for cone, 3 for cuboid

**For two grasping**

1-4 colums and 5-8 colums are the signal difference of the first grasping and the second grasping

9 column is the object shape: 1 for ball, 2 for cone, 3 for cuboid


# How to use

please install scikit-learn library

`pip install scikit-learn`

and then just 
```
python NN_rdm0.py onegrasp.csv    # For first 45 graspings of each object are used to train, last 5 of each object for testing.  
python NN_rdm0.py twograsp.csv    # For first 45 graspings of each object are used to train, last 5 of each object for testing.  

python NN.py onegrasp.csv    # mix all of them, and randowm split the whole set into traning and testing set with ratio 9:1   
python NN.py twograsp.csv     # mix all of them, and randowm split the whole set into traning and testing set with ratio 9:1   
```


