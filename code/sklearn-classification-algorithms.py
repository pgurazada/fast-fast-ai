
import numpy as np

from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Perceptron, LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier

# data
X, y = datasets.make_classification(n_samples=1000, n_features=20)

X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                    test_size=0.2, 
                                                    random_state=20130810)

X_train = StandardScaler().fit(X_train).transform(X_train)
X_test = StandardScaler().fit(X_train).transform(X_test)

# Sequence of models
# Perceptron
model_perceptron = Perceptron(max_iter=40, eta0=0.1, random_state=20130810)
model_perceptron.fit(X_train, y_train)

y_pred_perceptron = model_perceptron.predict(X_test)
print('Number of misclassified samples by the Perceptron algorithm: ', 
      (y_test != y_pred_perceptron).sum())

# Logistic Regression
for c in np.arange(-5, 5, dtype=float):
    model_logit = LogisticRegression(C=10**c, random_state=20130810)
    model_logit.fit(X_train, y_train)
    y_pred_logit = model_logit.predict(X_test)
    print('Number of misclassified samples by Logistic Regression with C =', 
          10**c, 'is: ', (y_test != y_pred_logit).sum())

# Support Vector Machines
for kernel_str in ['linear', 'rbf']:
    model_svm = SVC(kernel='linear', C=1.0, random_state=20130810)
    model_svm.fit(X_train, y_train)
    y_pred_svm = model_svm.predict(X_test)
    print('Number of misclassified samples by SVM with kernel: ', kernel_str,
          'is: ', (y_test != y_pred_svm).sum())

# Decision Tree Classifier
for criterion_str in ['entropy', 'gini']:
    model_dtree = DecisionTreeClassifier(criterion=criterion_str, 
                                         max_depth=3, 
                                         random_state=20130810)

    model_dtree.fit(X_train, y_train)
    y_pred_dtree = model_dtree.predict(X_test)
    print('Number of misclassified samples by Decision Trees with criterion', 
           criterion_str, 'is: ', (y_test != y_pred_dtree).sum())

# Random Forest Classifier
for nb_estimators in [25, 50, 100, 150]:
    model_random_forest = RandomForestClassifier(n_estimators=nb_estimators, 
                                                 random_state=20130810,
                                                 n_jobs=-1)

    model_random_forest.fit(X_train, y_train)
    y_pred_random_forest = model_random_forest.predict(X_test)
    print('Number of misclassified samples by Random Forests with', 
           nb_estimators, ' estimators is: ', (y_test != y_pred_random_forest).sum())


