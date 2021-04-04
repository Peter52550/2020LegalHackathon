### gradient boosting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
# load data
train = pd.read_csv("new_data_more.csv")
# test = pd.read_csv("gb_test_no.csv")
print(train.info())
train.set_index("Id", inplace=True)
# test.set_index("Id", inplace=True)
y_train = train["結果"]
train.drop(labels="結果", axis=1, inplace=True)
# train.drop(labels="法院委託精神鑑定結果是否判定有19一二情形", axis=1, inplace=True)
# train.drop(labels="Id", axis=1, inplace=True)
# test.drop(labels="結果", axis=1, inplace=True)



train_test =  train


train_test_dummies = pd.get_dummies(train_test)
train_test_dummies.fillna(value=0.0, inplace=True)

# generate feature sets (X)
X_train = train_test_dummies.values[:]
# X_test = train_test_dummies.values[430:]
# transform data
print(X_train.shape)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
X_train_scale = scaler.fit_transform(X_train)
# X_test_scale = scaler.transform(X_test)
# split training feature and target sets into training and validation subsets
from sklearn.model_selection import train_test_split

X_train_sub, X_validation_sub, y_train_sub, y_validation_sub = train_test_split(X_train_scale, y_train, random_state=0)
print(X_train_sub.shape, y_train_sub.shape)
# import machine learning algorithms
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn import  metrics
# train with Gradient Boosting algorithm
# compute the accuracy scores on train and validation sets when training with different learning rates
train_acc = []
feature11 = []
feature2 = []
feature6 = []
feature8 = []
feature3 = []
learning_rates = [0.05, 0.1, 0.25, 0.5, 0.75, 1]
for learning_rate in learning_rates:
    gb = GradientBoostingClassifier(n_estimators=20, learning_rate = learning_rate, max_features=2, max_depth = 2, random_state = 0)
    gb.fit(X_train_sub, y_train_sub)
    print('feature importances: ', gb.feature_importances_)
    feature11.append(gb.feature_importances_[11])
    feature2.append(gb.feature_importances_[2])
    feature6.append(gb.feature_importances_[6])
    feature8.append(gb.feature_importances_[8])
    feature3.append(gb.feature_importances_[3])
    print("Learning rate: ", learning_rate)
    print("Accuracy score (training): {0:.3f}".format(gb.score(X_train_sub, y_train_sub)))
    print("Accuracy score (validation): {0:.3f}".format(gb.score(X_validation_sub, y_validation_sub)))
    train_acc.append(gb.score(X_train_sub, y_train_sub))
    print()
    
# Output confusion matrix and classification report of Gradient Boosting algorithm on validation set


gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 1, max_features=7, max_depth = 10,min_samples_leaf = 10, min_samples_split = 73,subsample = 0.8, random_state = 0)
gb.fit(X_train_sub, y_train_sub)
print('feature importances: ', gb.feature_importances_)

predictions = gb.predict(X_validation_sub)
prob = gb.predict_proba(X_validation_sub)[:, -1]
print (" Accuracy : %.4g " % metrics.accuracy_score( y_validation_sub.values, predictions))
# print  (" AUC Score (Tra, in): %f " % metrics.roc_auc_score(y_validation_sub, prob))

print('prediction: ', predictions)
print(y_validation_sub, y_validation_sub.shape)
print('prob: ', prob.shape)
print('decision function: ', gb.staged_decision_function(X_validation_sub))
print("Confusion Matrix:")
print(confusion_matrix(y_validation_sub, predictions))
print()
print("Classification Report")
print(classification_report(y_validation_sub, predictions))
plt.plot(train_acc)
# plt.plot(feature11)
# plt.plot(feature2)
# plt.plot(feature6)
# plt.plot(feature8)
# plt.plot(feature3)
print(type(plt))
# plt.legend(["training accuracy", "法院委託精神鑑定結果是否判定有19一二情形", "有無按時服藥","社會功能是否正常","是否有犯罪動機", "智商是否正常"])
# plt.legend('training_accuracy')
from sklearn.tree import export_graphviz
# from sklearn.externals.six import StringIO  
from IPython.display import Image
from sklearn import tree
import pydotplus
fn = list(train.columns)
cn = [1,2,3]
trees = gb.estimators_.ravel()
dot_data = tree.export_graphviz(trees[0], out_file=None, 
                                feature_names = fn,class_names=cn,
                         filled=True, rounded=True,  
                         special_characters=True)  
graph = pydotplus.graph_from_dot_data(dot_data)  
Image(graph.create_png())

# Precision is the ability of a classifier not to label an instance positive that is actually negative. 
# For each class it is defined as the ratio of true positives to the sum of true and false positives.
# Recall is the ability of a classifier to find all positive instances. 
# For each class it is defined as the ratio of true positives to the sum of true positives and false negatives.
# gb = GradientBoostingClassifier(n_estimators=20, learning_rate = 0.5, max_features=2, max_depth = 2, random_state = 0)
# gb.fit(X_train_sub, y_train_sub)
# predictions = gb.predict(X_validation_sub)

# ROC curve and Area-Under-Curve (AUC)


# y_scores_gb = gb.decision_function(X_validation_sub)
# fpr_gb, tpr_gb, _ = roc_curve(y_validation_sub, y_scores_gb)
# roc_auc_gb = auc(fpr_gb, tpr_gb)

# print("Area under ROC curve = {:0.2f}".format(roc_auc_gb))

