import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.linear_model import LogisticRegression
import pickle

parser = argparse.ArgumentParser(description = "Random Forest Learner Generator")
parser.add_argument('maxsamples', help="Maximum number of test samples")
parser.add_argument('dfs', nargs='+', help='List of paths to data files')

args = parser.parse_args()

progdata = []

noTrainingSamples = int(int(args.maxsamples)/(len(args.dfs)-1))

for i in range(len(args.dfs)):
  progdata.append(pd.read_csv(args.dfs[i],index_col=False))
  stride = int(len(progdata[i])/int(args.maxsamples))
  progdata[i] = progdata[i].iloc[::stride, :]

for i in range(len(args.dfs)):
  trainset = pd.DataFrame()
  for j in range(len(args.dfs)):
    if i == j:
      continue
    trainset = pd.concat([trainset, progdata[j].iloc[::len(args.dfs)-1]],ignore_index=True,axis=0)

  testset = pd.read_csv(args.dfs[i],index_col=False)

  del trainset['pcdiff']
  del testset['pcdiff']

  label_idx = len(trainset.columns) - 1

  X_train = trainset.iloc[:, 0:label_idx].values
  y_train = trainset.iloc[:, label_idx].values
  X_test  = testset.iloc[:, 0:label_idx].values
  y_test  = testset.iloc[:, label_idx].values

  classifier = RandomForestClassifier(n_estimators=20, random_state=0)
  #classifier = DecisionTreeClassifier(max_depth = 10)

  clf = classifier.fit(X_train, y_train)
  y_pred = classifier.predict(X_test)
  print("Testing " + args.dfs[i] + "'s performance... ")
  print("Accuracy when trained with all features: " + str(accuracy_score(y_test, y_pred)))
  feature_imp = pd.Series(clf.feature_importances_,index=trainset.columns[0:7]).sort_values(ascending=False)
  print(feature_imp)

  for j in range(len(trainset.columns)-1):
    X_train_partial = np.delete(X_train, j, 1)
    X_test_partial  = np.delete(X_test, j, 1)
    classifier.fit(X_train_partial, y_train)
    y_pred = classifier.predict(X_test_partial)
    print("Accuracy when trained without " + trainset.columns[j] + ": " + str(accuracy_score(y_test, y_pred)))
    X_train_partial = X_train[:, j].reshape(-1,1)
    X_test_partial  = X_test[:, j].reshape(-1,1)
    classifier.fit(X_train_partial, y_train)
    y_pred = classifier.predict(X_test_partial)
    print("Accuracy when trained only with " + trainset.columns[j] + ": " + str(accuracy_score(y_test, y_pred)))