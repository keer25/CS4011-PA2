import numpy as np
from sklearn.decomposition import PCA

# Importing the data from the dataset
train_features = np.loadtxt('../../Dataset/DS3/train.csv', delimiter=',')
print(train_features.shape)