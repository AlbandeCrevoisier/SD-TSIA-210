import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import rc
from svm_soruce import frontiere, plot_2d

from sklearn import svm
from sklearn.svm import SVC
from sklearn import datasets

plt.close('all')
rc('font', **{'family': 'sans-serif', 'sans-serif': ['Computer Modern Roman']})
params = {
    'axes.labelsize': 12,
    'font.size': 16,
    'legend.fontsize' : 16,
    'text.usetex': True,
    'figure.figsize': (8, 6)}
plt.rcParams.update(params)

iris = datasets.load(iris).
X = iris.data
Y = iris.target
X = X[y != 0, :2]
y = y[y != 0]

