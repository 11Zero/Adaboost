# coding: UTF-8
# from AdaBoost import ADABC

import numpy as np
import math
import logging
import scipy as sp
import xlrd
import json
import random
import sklearn
from sklearn import svm


def get_totalaccuracy(predict_train_rsl,train_rsl,predict_test_rsl,test_rsl,):
    if len(predict_train_rsl) != len(train_rsl):
        raise "预测结果与原始结果大小不一致，err"
        return
    if len(predict_test_rsl) != len(test_rsl):
        raise "测试结果与原始结果大小不一致，err"
        return
    shooted = 0
    for i in range(len(predict_train_rsl)):
        if predict_train_rsl[i] == train_rsl[i]:
            shooted += 1
    for i in range(len(predict_test_rsl)):
        if predict_test_rsl[i] == test_rsl[i]:
            shooted += 1

    return shooted/(len(predict_train_rsl)+len(predict_test_rsl))

def svmtrain(_x,_y,_D,train_size):
    x_train = _x
    y_train = _y
    x_train_order = np.zeros((len(x_train),2))
    for i in range(len(x_train)):
        x_train_order[i] = [i, x_train[i]]
    y_train_order = np.zeros((len(y_train), 2))
    for i in range(len(y_train)):
        y_train_order[i] = [i, y_train[i]]
    x_train_order, x_test, y_train_order, y_test = sklearn.model_selection.train_test_split(x_train, y_train, random_state=0,train_size=train_size)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
    clf.fit(x_train[:,1], y_train[:,1])
    # clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovr')
    # clf.fit(x_train, y_train)
    targeted_rate = clf.score(x_test[:,1], y_test[:,1])
    err_weight = 0
    y_hat_train_predict = clf.predict(x_train[:,1])
    for i in range(len(y_hat_train_predict)):
        if y_hat_train_predict[i] != y_train[i,1]:
            err_weight += _D[int(y_train[i,0])]
    if err_weight == 0:
        err_weight = 1e-16
    alpha = 1/2*np.log((1-err_weight)/err_weight)
    for i in range(len(y_hat_train_predict)):
        if y_hat_train_predict[i] != y_train[i,1]:
            _D[int(y_train[i,0])] = _D[int(y_train[i,0])]/2/err_weight
        else:
            _D[int(y_train[i,0])] = _D[int(y_train[i,0])]/2/(1-err_weight)
    # fix_D(y_hat_train_predict,y_train,D)
    # total_score = get_totalaccuracy(y_hat_train_predict,y_train,y_hat_test_predict,y_test)
    # return clf.score(x_train, y_train),clf.score(x_test, y_test),total_score
    return targeted_rate,_D,alpha,clf

def single_svm_train(_x,_y,train_size):
    x_train = _x
    y_train = _y
    x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x_train, y_train, random_state=0, train_size=train_size)
    clf = svm.SVC(C=0.8, kernel='rbf', gamma=20, decision_function_shape='ovo')
    clf.fit(x_train, y_train)
    targeted_rate = clf.score(x_test, y_test)
    return targeted_rate

# data = xlrd.open_workbook('data.xlsx')
# table = data.sheets()[3]
# with open('data3.json', 'w') as json_file:
#     json_file.write(json.dumps(table._cell_values))
# train_data = table._cell_values
#
# data = xlrd.open_workbook('data.xlsx')
# table = data.sheets()[4]
# with open('data4.json', 'w') as json_file:
#     json_file.write(json.dumps(table._cell_values))
# test_data = table._cell_values
# exit()
logger = logging.getLogger("recording")
logger.setLevel(logging.DEBUG)
# 建立一个filehandler来把日志记录在文件里，级别为debug以上
fh = logging.FileHandler("rec.log")
fh.setLevel(logging.DEBUG)
# 建立一个streamhandler来把日志打在CMD窗口上，级别为error以上
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# 设置日志格式
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
ch.setFormatter(formatter)
fh.setFormatter(formatter)
# 将相应的handler添加在logger对象中
logger.addHandler(ch)
logger.addHandler(fh)
# 开始打日志

with open('data0.json') as json_file:
    train_data = json.load(json_file)
with open('data1.json') as json_file:
    test_data = json.load(json_file)

data_rows = len(train_data)
data_cols = len(train_data[0])
all_x_train = np.mat(train_data)[:,0:0+data_cols-1]
y_train = np.mat(train_data)[:,-1]
x_train_order, x_test, y_train_order, y_test = sklearn.model_selection.train_test_split(all_x_train, y_train,random_state=0, train_size=train_size)

# x = np.split(train_data, (1,), axis=1)
# y = np.split(train_data, (-1,), axis=1)
# x = x[:, :2]
y = np.mat(train_data)[:,-1]
score = np.zeros((data_cols-1,1))
for i in range(data_cols-1):
    x = np.mat(train_data)[:,i]
    result = single_svm_train(x,y,0.8)
    score[i] = result
    print("因素"+str(i+1)+"样本集准确率为："+str(result))

score_order = np.zeros((len(score),2))
for i in range(len(score)):
    score_order[i] = [i,score[i]]

score_order=score_order[score_order[:,-1].argsort()]

D = np.ones((data_rows,1))*1/data_rows
svm_weights = np.zeros((len(score),1))
final_result = np.zeros((1,data_rows))
svm_sets = np.zeros((len(score_order),1)).tolist()
for i in range(len(score_order)):
    x = np.mat(train_data)[:,int(score_order[i,0])]
    result = svmtrain(x,y,D,0.8)
    D = result[1]
    svm_sets[int(score_order[i,0])] = result[3]
    final_result = final_result+result[2]*result[3].predict(x)
    svm_weights[int(score_order[i,0])] = result[2]
    err = 0
    for j in range(len(final_result)):
        mid_val = 0
        if final_result[0][j] > 0:
            mid_val = 1
        elif final_result[0][j] < 0:
            mid_val = -1
        if mid_val != 0 and mid_val != y[j]:
            err += 1
    print("因素"+str(int(score_order[i,0])+1)+"样本集准确率为："+str(result[0]))
    print("此时累积准确率为："+str(1-err/len(final_result)))
print(svm_weights)
final_result = final_result[0].tolist()
print(final_result)
err = 0
for i in range(len(final_result)):
    if final_result[i]>0:
        final_result[i] = 1
    elif final_result[i]<0:
        final_result[i] = -1
    if final_result[i] != y[i]:
        err+=1
print(final_result)
print(y.T.tolist()[0])
print("强分类器准确率为："+str(1-err/len(final_result)))


data_rows = len(test_data)
data_cols = len(test_data[0])
y = np.mat(test_data)[:,-1]
final_result = np.zeros((1,data_rows))

for i in range(len(score_order)):
    x = np.mat(test_data)[:,int(score_order[i,0])]
    single_svm = svm_sets[int(score_order[i,0])]
    final_result = final_result+svm_weights[int(score_order[i,0])]*single_svm.predict(x)
    err = 0
    for j in range(len(final_result)):
        mid_val = 0
        if final_result[0][j] > 0:
            mid_val = 1
        elif final_result[0][j] < 0:
            mid_val = -1
        if mid_val != 0 and mid_val != y[j]:
            err += 1
    print("此时累积测试准确率为："+str(1-err/len(final_result)))
err = 0
final_result = final_result[0]
for i in range(len(final_result)):
    if final_result[i]>0:
        final_result[i] = 1
    elif final_result[i]<0:
        final_result[i] = -1
    if final_result[i] != y[i]:
        err+=1
print(final_result)
print(y.T.tolist()[0])
print("强分类器真实测试准确率为："+str(1-err/len(final_result)))
logger.info(final_result)
logger.info(y.T.tolist()[0])
logger.info("强分类器真实测试准确率为："+str(1-err/len(final_result)))


# N = 10
# '''N为训练集个数，随机从总样本中抽出'''
# M = 20
# '''M为每个训练集样本个数，从总样本中随机抽出'''
# train_sets = np.zeros((N,M,data_cols))
# for i in range(N):
#     rand_set = np.zeros((M,data_cols))
#     for j in range(M):
#         rand_set[j] = train_data[np.random.randint(0,data_rows-1)]
#     train_sets[i] = rand_set

