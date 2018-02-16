import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from svm_src import frontiere, plot_2d

from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets

plt.close('all')
rc('font', **{'family': 'serif', 'serif': ['Computer Modern Roman']})
params = {
    'axes.labelsize': 12,
    'font.size': 16,
    'legend.fontsize' : 16,
    'text.usetex': True,
    'figure.figsize': (8, 6)}
plt.rcParams.update(params)

iris = datasets.load_iris()
X = iris.data
y = iris.target
X = X[y != 0, :2]
y = y[y != 0]
X_train = X[[2*i for i in range(len(X)//2)],]
y_train = y[[2*i for i in range(len(y)//2)]]
X_test = X[[2*i+1 for i in range(len(X)//2)],]
y_test = y[[2*i+1 for i in range(len(y)//2)]]

# Linear kernel
clf = svm.LinearSVC()
clf.fit(X_train, y_train)
print("Linear kernel: ", clf.score(X_test, y_test))

# Polynomial kernel
# degree 3
clf = svm.SVC(kernel='poly')
clf.fit(X_train, y_train)
print("Polynomial kernel of degree 3: ", clf.score(X_test, y_test))
# degree 5
clf = svm.SVC(kernel='poly', degree=5)
clf.fit(X_train, y_train)
print("Polynomial kernel of degree 5: ", clf.score(X_test, y_test))

