import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Importing the data from the dataset
X = np.loadtxt('../../Dataset/DS3/train.csv', delimiter=',')
y = np.loadtxt('../../Dataset/DS3/train_labels.csv', delimiter=',')

pca = PCA(n_components=1)
X_new = pca.fit_transform(X, y)

np.savetxt('extracted_features.csv', X_new, delimiter=',')

lr = LinearRegression()
lr.fit(X_new, y)

np.savetxt('coeffs.csv', lr.coef_, delimiter=',')

X_test = np.loadtxt('../../Dataset/DS3/test.csv', delimiter=',')
y_test = np.loadtxt('../../Dataset/DS3/test_labels.csv', delimiter=',')
X_test_transformed = pca.transform(X_test)

y_pred = lr.predict(X_test_transformed)

y_labels = (y_pred > 1.5 )+ np.ones(y_pred.shape)

precision_1 = sum(np.logical_and(y_labels == 1, 1 == y_test))/sum(y_labels == 1)
recall_1 =  sum(np.logical_and(y_labels == 1, 1 == y_test))/sum(y_test==1)
fmeasure_1 = (2*precision_1*recall_1)/(precision_1 + recall_1)

precision_2 = sum(np.logical_and(y_labels == 2, 2 == y_test))/sum(y_labels == 2)
recall_2 =  sum(np.logical_and(y_labels == 2, 2 == y_test))/sum(y_test==2)
fmeasure_2 = (2*precision_2*recall_2)/(precision_2 + recall_2)

print("Class Precision       Recall         F measure")
print('1     ' + str(precision_1) + '  ' + str(recall_1) + '  ' + str(fmeasure_1))
print('2     ' + str(precision_2) + '  ' + str(recall_2) + '  ' + str(fmeasure_2))

f = open("results.txt", "w")
f.write("Class Precision       Recall         F measure\n")
f.write('1     ' + str(precision_1) + '  ' + str(recall_1) + '  ' + str(fmeasure_1) + '\n')
f.write('2     ' + str(precision_2) + '  ' + str(recall_2) + '  ' + str(fmeasure_2) + '\n')

cmap = []
for i in range(0,1000):
	cmap.append('r')
for i in range(0,1000):
	cmap.append('b')

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=cmap, marker='o')
ax.set_xlabel('1')
ax.set_ylabel('2')
ax.set_zlabel('3')

#z_line = lr.coef_ - lr.intercept_
# axb = fig.add_subplot(111, projection='3d')
# ax.line(X[:, 0], X[:, 1], )

plt.show()