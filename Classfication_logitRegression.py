import  pandas  as pd
from sklearn.preprocessing import scale  # 使用scikit-learn进行数据预处理
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn import svm
from sklearn import tree
from sklearn.model_selection import GridSearchCV
import  numpy  as np
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn import linear_model
from sklearn.neural_network import MLPClassifier

from sklearn import datasets

import warnings
warnings.filterwarnings("ignore")

def my_confusion_matrix(y_true, y_pred,algorithm):
    from sklearn.metrics import confusion_matrix
    labels = list(set(y_true))
    conf_mat = confusion_matrix(y_true, y_pred, labels=labels)
    print("[%s]confusion_matrix(left labels: y_true, up labels: y_pred):" % (algorithm))
    print(conf_mat)


def my_classification_report(y_true, y_pred, algorithm):
    from sklearn.metrics import classification_report
    print("[%s]classification_report:" % (algorithm))
    print(classification_report(y_true, y_pred))

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    # classes = classes[unique_labels(y_true, y_pred)]
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax

########################################################################################################################
## 读取iris数据
########################################################################################################################
iris  = datasets.load_iris()

########################################################################################################################
## 随机划分训练测试集
########################################################################################################################
feature = iris.data
label = iris.target
# 随机划分训练测试集
X_train,X_test, Y_train, Y_test = train_test_split(feature,label,test_size=0.3, random_state=0,stratify=label)


# 数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
# 将数据进行归一化
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

########################################################################################################################
## 逻辑回归分类
########################################################################################################################
classifier = linear_model.LogisticRegression(C=1e5)
classifier.fit(X_train,Y_train)

Y_train_predict = classifier.predict(X_train)
Y_test_predict = classifier.predict(X_test)
Y_train_predict = Y_train_predict.astype('int32')
Y_test_predict = Y_test_predict.astype('int32')
Y_train = Y_train.astype('int32')
Y_test = Y_test.astype('int32')

# 计算KNN分类器的准确率
print("LogisticRegression训练集：",classifier.score(X_train,Y_train))
print("LogisticRegression测试集：",classifier.score(X_test,Y_test))
# 分类报告
my_classification_report(Y_test, Y_test_predict ,'LogisticRegression')
# 混淆矩阵
my_confusion_matrix(Y_test, Y_test_predict, 'LogisticRegression')
# 画混淆矩阵
np.set_printoptions(precision=2)
class_names = np.array(['0', '1', '2'])
# Plot non-normalized confusion matrix
plot_confusion_matrix(Y_test, Y_test_predict, classes=class_names,
                      title='[LogisticRegression]Confusion matrix, without normalization')
# # Plot normalized confusion matrix
# plot_confusion_matrix(Y_test, Y_test_predict, classes=class_names, normalize=True,
#                       title='[LogisticRegression]Normalized confusion matrix')

print('=================================================================================================')



plt.show()