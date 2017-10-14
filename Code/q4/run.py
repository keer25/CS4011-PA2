from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import sklearn
import _pickle as pickle
import numpy as np

# Importing the image samples
samples = np.loadtxt('../../Dataset/DS2.csv', delimiter=',')
samples = sklearn.utils.shuffle(samples)

# Doing Model Selection with 500 data points
# X = samples[0:200, 0:96]
# y = samples[0:200, 96]

scaler = StandardScaler(copy=True)
# X_scaled = scaler.fit_transform(X)

# c_values = [0.01, 0.1, 1, 10]
# gamma_values = [0.1, 0.01, 0.001, 0.0001, 0.00001]
# rs = [0, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
# param_grid = [
		# {'C' : c_values, 'kernel' : ['linear'] },
		# {'C' : c_values, 'kernel' : ['poly'], 'degree' : [2, 3, 4, 5], 'gamma' : gamma_values, 'coef0' : rs },
		# {'C' : c_values, 'gamma' : gamma_values, 'kernel' : ['rbf'] },
		# {'C' : c_values, 'kernel' : ['sigmoid'], 'gamma' : gamma_values, 'coef0' : rs }
# 	]

# cv = StratifiedKFold(n_splits=10)
# svm = SVC()
# search = GridSearchCV(svm, param_grid, cv=cv.split(X_scaled, y))
# search.fit(X_scaled, y)

# print(search.cv_results_)
# print(search.best_params_)
# print(search.best_score_)


######################################################################
#                    Best parameters for all models                  #
######################################################################

X = samples[:, 0:96]
y = samples[:, 96]

X_scaled = scaler.fit_transform(X)
cv = StratifiedKFold(n_splits=10)
svm_linear = SVC(kernel='linear', C=10)
svm_poly = SVC(kernel='poly', C=1, degree=2, coef0=1, gamma=0.01)
svm_rbf = SVC(kernel='rbf', C=10, gamma=0.01)
svm_sigmoid = SVC(kernel='sigmoid', coef0=0, C=10, gamma=0.001)

score_linear = 0
score_poly = 0
score_rbf = 0
score_sigmoid = 0
for train_index, test_index in cv.split(X_scaled, y):
		X_train, X_test = X_scaled[train_index], X_scaled[test_index]
		y_train, y_test = y[train_index], y[test_index]
		svm_linear.fit(X_train, y_train)
		svm_poly.fit(X_train, y_train)
		svm_rbf.fit(X_train, y_train)
		svm_sigmoid.fit(X_train, y_train)
		score_linear += svm_linear.score(X_test, y_test)
		score_poly += svm_poly.score(X_test, y_test)
		score_rbf += svm_rbf.score(X_test, y_test)
		score_sigmoid += svm_sigmoid.score(X_test, y_test)

output = open('svm_model1.model', 'wb')
pickle.dump(svm_linear, output)
output = open('svm_model2.model', 'wb')
pickle.dump(svm_poly, output)
output = open('svm_model3.model', 'wb')
pickle.dump(svm_rbf, output)
output = open('svm_model4.model', 'wb')
pickle.dump(svm_sigmoid, output)

print(score_linear/10)
print(score_poly/10)
print(score_rbf/10)
print(score_sigmoid/10)