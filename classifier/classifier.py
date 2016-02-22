import numpy as np
import pandas as pd
from sklearn.cross_validation import KFold
from sklearn import ensemble
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB


np.random.seed(0)

# Loading data
df_train = pd.read_csv('dataset.csv')
labels = df_train['label'].values
df_train = df_train.drop(['label'], axis=1)


values = df_train.values
X = np.asarray(values)
y = np.asarray(labels)
X = StandardScaler().fit_transform(X) #normalization

# K-fold cross validation
# Data'yi 10a bol %90test %10train
kf = KFold(len(labels), n_folds=10)
accuracies = []
aucs = []

print("- Random Forest")
for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Random Forest
    clf = ensemble.RandomForestClassifier() #n_estimators=250
    clf.fit(X_train, y_train) #train ediyor, o vektorler uzerinden bi model olusturuyor
    predictions = clf.predict(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc = metrics.roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy Score is: ", accuracy)
    # print("AUC Score is:", auc)
    accuracies.append(accuracy)
    aucs.append(auc)

acc = sum(accuracies)/len(accuracies)
auc = sum(aucs)/len(aucs)
print("Average Accuracy : ", acc)
print("Average AUC      : ", auc)

print("- Decision Tree")
accuracies = []
aucs = []
for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Decision Tree
    clf = DecisionTreeClassifier(max_depth=5)
    clf.fit(X_train, y_train) #train ediyor, o vektorler uzerinden bi model olusturuyor
    predictions = clf.predict(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc = metrics.roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy Score is: ", accuracy)
    # print("AUC Score is:", auc)
    accuracies.append(accuracy)
    aucs.append(auc)

acc = sum(accuracies)/len(accuracies)
auc = sum(aucs)/len(aucs)
print("Average Accuracy : ", acc)
print("Average AUC      : ", auc)


print("- Quadric Discriminant Analysis")
accuracies = []
aucs = []
for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Quadric Discriminant Analysis
    clf = QuadraticDiscriminantAnalysis()
    clf.fit(X_train, y_train) #train ediyor, o vektorler uzerinden bi model olusturuyor
    predictions = clf.predict(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc = metrics.roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy Score is: ", accuracy)
    # print("AUC Score is:", auc)
    accuracies.append(accuracy)
    aucs.append(auc)

acc = sum(accuracies)/len(accuracies)
auc = sum(aucs)/len(aucs)
print("Average Accuracy : ", acc)
print("Average AUC      : ", auc)

print("- Naive Bayes")
accuracies = []
aucs = []
for train_index, test_index in kf:
    # print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    #Naive Bayes
    clf = GaussianNB()
    clf.fit(X_train, y_train) #train ediyor, o vektorler uzerinden bi model olusturuyor
    predictions = clf.predict(X_test)
    # fpr, tpr, thresholds = metrics.roc_curve(y_test, predictions, pos_label=1)
    auc = metrics.roc_auc_score(y_test, predictions)
    accuracy = accuracy_score(y_test, predictions)
    # print("Accuracy Score is: ", accuracy)
    # print("AUC Score is:", auc)
    accuracies.append(accuracy)
    aucs.append(auc)

acc = sum(accuracies)/len(accuracies)
auc = sum(aucs)/len(aucs)
print("Average Accuracy : ", acc)
print("Average AUC      : ", auc)

#print("#################Nearest Neighbors")
#print("#################Linear SVM")
#print("#################RBF SVM")
#print("#################AdaBoost")

#print("#################Linear Discriminant Analysis")