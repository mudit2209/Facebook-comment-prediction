# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 15:24:15 2019

@author: Mudit Sharma
"""
# Importing data

import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier 
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score  
from sklearn.ensemble import AdaBoostClassifier 
import itertools
from sklearn.model_selection import cross_val_score

dataset = pd.read_csv('Features_variant.csv')

#### EDA ####


dataset.head(5) # Viewing the Data
dataset.columns
dataset.describe() # Distribution of Numerical Variables

dataset = dataset.loc[:,['Popularity', 'checkin', 'daily interest', 'page category',
       'C before', 'C in last 24', 'C in L 48 to L 24', 'C in F 24',
       'Base time', 'Post length', 'share count', 'Promoted', 'H comments rec',
       'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'b sun', 'b mon',
       'b tue', 'b wed', 'b thu', 'b fri', 'b sat', 'comments in H'] ]

dataset.columns = ['Popularity', 'checkin', 'interest', 'page_category',
       'Comments_bb', 'Comments_last_24', 'Comments_l48_L24', 'Comments_First_24', 
       'Base_time', 'Post_length', 'share_count', 'Promoted', 'hours_H',
       'sun', 'mon', 'tue', 'wed', 'thu', 'fri', 'sat', 'b sun', 'b mon',
       'b tue', 'b wed', 'b thu', 'b fri', 'b sat', 'comments_nxt_H']

dataset['comments'] = np.where(dataset['comments_nxt_H']==0,0,1) #converting into binary classification
dataset = dataset.drop(columns = ['comments_nxt_H', 'mon', 'b mon'])
dataset.head(5)

dataset.comments.value_counts()

# Removing NaN
dataset.isna().any()
dataset.isna().sum()
dataset = dataset.dropna()

dataset.comments.value_counts()

# Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(dataset.drop(columns = 'comments'), dataset['comments'],
                                                    test_size = 0.3,
                                                    random_state = 0)

# Balancing the Training Set
y_train.value_counts()

pos_index = y_train[y_train.values == 1].index
neg_index = y_train[y_train.values == 0].index

if len(pos_index) > len(neg_index):
    higher = pos_index
    lower = neg_index
else:
    higher = neg_index
    lower = pos_index

random.seed(8)
higher = np.random.choice(higher, size=len(lower))
lower = np.asarray(lower)
new_indexes = np.concatenate((lower, higher))

X_train = X_train.loc[new_indexes,]
y_train = y_train[new_indexes]

# Feature Scaling
from sklearn.preprocessing import StandardScaler
sc_X = StandardScaler()
X_train2 = pd.DataFrame(sc_X.fit_transform(X_train))
X_test2 = pd.DataFrame(sc_X.transform(X_test))
X_train2.columns = X_train.columns.values
X_test2.columns = X_test.columns.values
X_train2.index = X_train.index.values
X_test2.index = X_test.index.values
X_train = X_train2
X_test = X_test2

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()

# Suppport vector machines

# RBF kernel
model = svm.SVC(decision_function_shape='ovr', kernel='rbf')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
print('classification report for train rbf', classification_report(y_train,y_pred_train))
print('classification report for test rbf', classification_report(y_test,y_pred))
print('confusion matrix for train rbf')
plot_confusion_matrix(confusion_matrix(y_train,y_pred_train.round()), classes=[0,1])
plt.show()
print('confusion matrix for test rbf')
plot_confusion_matrix(confusion_matrix(y_test,y_pred.round()), classes=[0,1])
plt.show()
print ("Accuracy Score for train rbf is:", accuracy_score(y_train, y_pred_train))
print ("Accuracy Score for test rbf is:", accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Decision tree Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# Linear kernel
model = svm.SVC(decision_function_shape='ovr', kernel='linear')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
print('classification report for train linear', classification_report(y_train,y_pred_train))
print('classification report for test linear', classification_report(y_test,y_pred))
print('confusion matrix for train linear')
plot_confusion_matrix(confusion_matrix(y_train,y_pred_train.round()), classes=[0,1])
plt.show()
print('confusion matrix for test linear')
plot_confusion_matrix(confusion_matrix(y_test,y_pred.round()), classes=[0,1])
plt.show()
print ("Accuracy Score for train linear is:", accuracy_score(y_train, y_pred_train))
print ("Accuracy Score for test linear is:", accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Decision tree Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

#Sigmoid kernel
model = svm.SVC(decision_function_shape='ovr', kernel='sigmoid')
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
print('classification report for train sigmoid', classification_report(y_train,y_pred_train))
print('classification report for test sigmoid', classification_report(y_test,y_pred))
print('confusion matrix for train sigmoid')
plot_confusion_matrix(confusion_matrix(y_train,y_pred_train.round()), classes=[0,1])
plt.show()
print('confusion matrix for test sigmoid')
plot_confusion_matrix(confusion_matrix(y_test,y_pred.round()), classes=[0,1])
plt.show()
print ("Accuracy Score for train sigmoid is:", accuracy_score(y_train, y_pred_train))
print ("Accuracy Score for test sigmoid is:", accuracy_score(y_test, y_pred))

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Decision tree Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

# # degree Polynomial kernel 
model = svm.SVC(decision_function_shape='ovr', kernel='poly', degree=3)
model.fit(X_train, y_train)
y_pred_train = model.predict(X_train)
y_pred = model.predict(X_test)
print('classification report for train poly', classification_report(y_train,y_pred_train))
print('classification report for test poly', classification_report(y_test,y_pred))
print('confusion matrix for train poly')
plot_confusion_matrix(confusion_matrix(y_train,y_pred_train.round()), classes=[0,1])
plt.show()
print('confusion matrix for test poly')
plot_confusion_matrix(confusion_matrix(y_test,y_pred.round()), classes=[0,1])
plt.show()
print ("Accuracy Score for train poly is:", accuracy_score(y_train, y_pred_train))
print ("Accuracy Score for test poly is:", accuracy_score(y_test, y_pred))

#Poly for various degree
test_acc = [] 
train_acc = []
degree = []

for i in range(1,5):
  model = svm.SVC(decision_function_shape='ovr', kernel='poly', degree=i)
  model.fit(X_train, y_train)
  y_pred_train = model.predict(X_train)
  y_pred = model.predict(X_test)
  test_acc.append(accuracy_score(y_test, y_pred)*100)
  train_acc.append(accuracy_score(y_train, y_pred_train)*100)
  degree.append(i)
        
plt.xlabel('Degree')
plt.ylabel('Accuracy')
plt.plot(degree,test_acc)
plt.plot(degree,train_acc)
plt.legend(['test_acc' , 'train_acc'],loc='upper center', ncol=3, fancybox=True, shadow=True)
plt.show()


# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = model, X = X_train, y = y_train, cv = 10)
print("Decision tree Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

Kernel = ['rbf', 'linear', 'sigmoid', 'poly']
train_acc = [0.8190576546740931,0.8075041396959205 ,0.7749887099202167 ,0.8018590998043053]
test_acc = [0.8058005562177195,0.8104092173222089 ,0.767103694874851 ,0.7880015891934843]

plt.xlabel('Kernel')
plt.ylabel('Accuracy')
plt.plot(Kernel,test_acc)
plt.plot(Kernel,train_acc)
plt.legend(['test_acc' , 'train_acc'],loc='upper center', ncol=3, fancybox=True, shadow=True)
plt.show()



###Decision Tree###

test_acc = [] 
train_acc = []
score = []
depth = []
cnf_matrix_train= []
cnf_matrix_test= []
max_dep = []
for i in range(1,20):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = i)
    clf_gini.fit(X_train, y_train)
    y_pred_tree = clf_gini.predict(X_test)
    y_pred_tree_train = clf_gini.predict(X_train)
    test_acc.append(accuracy_score(y_test, y_pred_tree)*100)
    train_acc.append(accuracy_score(y_train, y_pred_tree_train)*100)
    score.append(clf_gini.score(X_test,y_test))
    depth.append(i)
    
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(depth,test_acc)
plt.plot(depth,train_acc)
plt.legend(['test_acc' , 'train_acc'],loc='upper center', ncol=3, fancybox=True, shadow=True)
plt.show()

# For Depth = 15
clf_gini = DecisionTreeClassifier(max_depth =15, random_state = 100)
clf_gini.fit(X_train, y_train)
y_pred_tree = clf_gini.predict(X_test)
y_pred_tree_train = clf_gini.predict(X_train)
clf_gini.score(X_test,y_test)
print('Training confusion matrix for depth: 15')
print(classification_report(y_train,y_pred_tree_train))  
print(accuracy_score(y_train,y_pred_tree_train))
plot_confusion_matrix(confusion_matrix(y_train,y_pred_tree_train.round()), classes=[0,1])
plt.show()
print('Test confusion matrix for depth: 15')
print(classification_report(y_test,y_pred_tree))  
print(accuracy_score(y_test,y_pred_tree))
plot_confusion_matrix(confusion_matrix(y_test,y_pred_tree.round()), classes=[0,1])
plt.show()

# Decision tree pruning for Depth = 7
clf_gini = DecisionTreeClassifier(max_depth =7, random_state = 100)
clf_gini.fit(X_train, y_train)
y_pred_tree = clf_gini.predict(X_test)
y_pred_tree_train = clf_gini.predict(X_train)
clf_gini.score(X_test,y_test)
print('Training confusion matrix for depth: 7')
print(classification_report(y_train,y_pred_tree_train))  
print(accuracy_score(y_train,y_pred_tree_train))
plot_confusion_matrix(confusion_matrix(y_train,y_pred_tree_train.round()), classes=[0,1])
plt.show()
print('Test confusion matrix for depth: 7')
print(classification_report(y_test,y_pred_tree))  
print(accuracy_score(y_test,y_pred_tree))
plot_confusion_matrix(confusion_matrix(y_test,y_pred_tree.round()), classes=[0,1])
plt.show()

# Accuracy vs depth for max_depth = 7
test_acc = [] 
train_acc = []
score = []
depth = []
for i in range(1,7):
    clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100, max_depth = i)
    clf_gini.fit(X_train, y_train)
    y_pred_tree = clf_gini.predict(X_test)
    y_pred_tree_train = clf_gini.predict(X_train)
    test_acc.append(accuracy_score(y_test, y_pred_tree)*100)
    train_acc.append(accuracy_score(y_train, y_pred_tree_train)*100)
    score.append(clf_gini.score(X_test,y_test))
    depth.append(i)
    print(score)
    
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(depth,test_acc)
plt.plot(depth,train_acc)
plt.legend(['test_acc' , 'train_acc'],loc='upper center', ncol=3, fancybox=True, shadow=True)
plt.show()
    
print("*Boosting1*")

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = clf_gini, X = X_train, y = y_train, cv = 10)
print("Decision tree Accuracy: %0.3f (+/- %0.3f)" % (accuracies.mean(), accuracies.std() * 2))

#boosting
test_acc  = []
train_acc = []
depth = []
for i in range(1,20):
    dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=i) 
    clf = AdaBoostClassifier(n_estimators=200, base_estimator=dt,learning_rate=1)
    clf.fit(X_train,y_train)
    y_pred_boost = clf.predict(X_test)
    y_pred_boost_train = clf.predict(X_train)
    test_acc.append(accuracy_score(y_test,y_pred_boost)*100)
    train_acc.append(accuracy_score(y_train,y_pred_boost_train)*100)
    depth.append(i)
    
   
plt.xlabel('Depth')
plt.ylabel('Accuracy')
plt.plot(depth,test_acc)
plt.plot(depth,train_acc)
plt.legend(['test_acc' , 'train_acc'],loc='lower right', ncol=3, fancybox=True, shadow=True)
plt.show()


# Boosting after pruning

dt = DecisionTreeClassifier(criterion = "gini", random_state = 100,max_depth=2) 
clf = AdaBoostClassifier(n_estimators=200, base_estimator=dt,learning_rate=1)
clf.fit(X_train,y_train)
y_pred_boost = clf.predict(X_test)
y_pred_boost_train = clf.predict(X_train)
print('Training confusion matrix for depth: 2')
print(classification_report(y_train,y_pred_tree_train))  
print(accuracy_score(y_train,y_pred_tree_train))
plot_confusion_matrix(confusion_matrix(y_train,y_pred_tree_train.round()), classes=[0,1])
plt.show()
print('Test confusion matrix for depth: 2')
print(classification_report(y_test,y_pred_tree))  
print(accuracy_score(y_test,y_pred_tree))
plot_confusion_matrix(confusion_matrix(y_test,y_pred_tree.round()), classes=[0,1])
plt.show()
    
   
