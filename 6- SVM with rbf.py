import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.model_selection import validation_curve

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

gamma = np.logspace(-4,1,6)
print(gamma)

model = SVC(kernel="rbf", C=1, random_state=0)
train_scores, valid_scores = validation_curve(model, X_subset, y_subset, cv=3 , param_name="gamma", param_range=gamma, scoring="accuracy")

print(train_scores)
print(valid_scores)

train_mean = train_scores.mean(axis=1)
test_mean = valid_scores.mean(axis=1)

plt.plot(np.arange(6), train_mean, "-o")
plt.plot(np.arange(6), test_mean, "-o")
plt.show()