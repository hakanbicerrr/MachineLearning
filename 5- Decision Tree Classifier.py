import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics._regression import r2_score
from sklearn.tree import DecisionTreeClassifier
from adspy_shared_utilities import plot_decision_tree
from adspy_shared_utilities import plot_feature_importances
from sklearn import tree
mush_df = pd.read_csv("mushrooms.csv")
print(mush_df)
mush_df2 = pd.get_dummies(mush_df)
print(mush_df2)

X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]
print(X_mush)
print(y_mush)
X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

X_subset = X_test2
y_subset = y_test2

model = DecisionTreeClassifier(random_state=0)
clf = model.fit(X_train2, y_train2)
print(clf.score(X_train2,y_train2))

plt.figure(figsize=(10, 20))
plot_feature_importances(clf, X_train2.columns)
# plt.show()
# print(-np.sort(-clf.feature_importances_))
sorted = np.argsort(-clf.feature_importances_)
names=[]
# Most important 5 features
for i in range(5):
    names.append(X_train2.columns[sorted[:5]][i])
print(names)