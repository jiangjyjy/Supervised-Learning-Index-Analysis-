from sklearn.preprocessing import scale  # 使用scikit-learn进行数据预处理
import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsRegressor
from sklearn import svm
from sklearn import tree
from sklearn.ensemble import RandomForestRegressor

import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from sklearn import linear_model
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

from sklearn.model_selection import GridSearchCV

from sklearn import datasets
from sklearn.neural_network import MLPRegressor

########################################################################################################################
## 加载数据集并随机划分训练测试集
########################################################################################################################
# 读取数据文件
boston  = datasets.load_boston()
# 提取feature
feature = boston.data
# 提取label
label = boston.target

# 随机划分训练测试集，其中80%训练，20%测试
X_train,X_test, Y_train, Y_test = train_test_split(feature,label,test_size=0.2, random_state=0)
# 数据归一化
min_max_scaler = preprocessing.MinMaxScaler()
# 将数据进行归一化
X_train = min_max_scaler.fit_transform(X_train)
X_test = min_max_scaler.transform(X_test)

Y_trainTemp = np.zeros((Y_train.shape[0],1),Y_train.dtype)
Y_trainTemp[:,0] = Y_train

Y_testTemp = np.zeros((Y_test.shape[0],1),Y_test.dtype)
Y_testTemp[:,0] = Y_test

Y_train = min_max_scaler.fit_transform(Y_trainTemp)
Y_test = min_max_scaler.transform(Y_testTemp)


########################################################################################################################
# decison tree
########################################################################################################################
tree_regressor = tree.DecisionTreeRegressor(max_depth=20)
tree_regressor.fit(X_train,Y_train)
predict_y_train = tree_regressor.predict(X_train)
predict_y_test = tree_regressor.predict(X_test)
scores_train = tree_regressor.score(X_train,Y_train)
scores_test = tree_regressor.score(X_test,Y_test)
print('Decision Tree train r2:{0} test r2:{1}'.format(scores_train,scores_test))

# The mean squared error
print('[Train][Decision Tree][RMSE]: %.8f'
      % np.sqrt(mean_squared_error(Y_train, predict_y_train)))
# Explained variance score: 1 is perfect prediction
#  R2 决定系数（拟合优度）
print('[Train][Decision Tree][R2]: %.8f' % r2_score(Y_train, predict_y_train))
# MAE
print('[Train][Decision Tree][MAE]: %.8f' % mean_absolute_error(Y_train, predict_y_train))
# The mean squared error
print('[Test][Decision Tree][RMSE]: %.8f'
      % np.sqrt(mean_squared_error(Y_test, predict_y_test)))
# Explained variance score: 1 is perfect prediction
#  R2 决定系数（拟合优度）
print('[Test][Decision Tree][R2]: %.8f' % r2_score(Y_test, predict_y_test))
# MAE
print('[Test][Decision Tree][MAE]: %.8f' % mean_absolute_error(Y_test, predict_y_test))

print('===================================================================================')

plt.figure()
plt.subplot(211)
f1 = plt.plot(Y_train,'ro-')
f2 = plt.plot(predict_y_train,'g^-')
plt.axis('tight')
plt.title("[Train][Decision Tree]Comparison of True Value and Predicted Value")
plt.legend(labels=['True','Predict'],loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Value')
plt.subplot(212)
f1 = plt.plot(Y_test,'ro-')
f2 = plt.plot(predict_y_test,'g^-')
plt.axis('tight')
plt.title("[Test][Decision Tree]Comparison of True Value and Predicted Value")
plt.legend(labels=['True','Predict'],loc='upper right')
plt.xlabel('Sample')
plt.ylabel('Value')





plt.show()
