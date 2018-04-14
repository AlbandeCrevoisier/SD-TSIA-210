import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
import datetime
sns.set()

products = pd.read_csv("products.csv")
customers = pd.read_csv("customers.csv")
orders_train = pd.read_csv("X_train.csv")
orders_test = pd.read_csv("X_test.csv")
orders = pd.concat(orders_train, orders_test, index=1)
returns = pd.read_csv("y_train.csv")
returns = pd.merge(returns, orders_train, "right", ["OrderNumber", "LineItem"])

orders = pd.merge(orders, products, "left", "VariantId")
orders = pd.merge(orders, customers, "left", "CustomerId")

orders_train = pd.merge(orders_train, products, "left", "VariantId")
orders_train = pd.merge(orders_train, customers, "left", "CustomerId")

