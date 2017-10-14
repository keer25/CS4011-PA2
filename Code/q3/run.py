import numpy as np
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as QDA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

# Importing the Iris dataset
# Dataset 1 - Iris Setosa
#      		2 - Iris Versicolour
#      		3 - Iris Virginica
samples = np.loadtxt('../../Dataset/iris/iris.data', delimiter=',')
# Extracting only Petal length and width
X = samples[:, 2:4]
y = samples[:, 4]

# Performing LDA
lda = LDA()
lda.fit(X, y)

# Performing QDA
qda = QDA()
qda.fit(X, y)

reg_param = [0.001, 0.01, 0.1, 1, 10]
for r in reg_param:
	qda = QDA(reg_param = r)
	qda.fit(X, y)
