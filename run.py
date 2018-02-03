# coding: UTF-8
# from AdaBoost import ADABC

import numpy as np
import datetime
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

    return shooted/(np.size(predict_train_rsl)+np.size(predict_test_rsl))

def get_weight(singal_svm,x_test,y_test,_D):
    max = np.zeros((1, np.size(x_test[0, :])))
    min = np.zeros((1, np.size(x_test[0, :])))
    test_x_normal = np.mat(np.zeros(np.shape(x_test)))
    for i in range(np.size(x_test[0, :])):
        max[0, i] = x_test[:, i].max()
        min[0, i] = x_test[:, i].min()
        if max[0, i] == min[0, i]:
            test_x_normal[:, i] = np.mat(np.ones(np.shape(x_test[:, i])))
        else:
            test_x_normal[:, i] = (x_test[:, i] - min[0, i]) / (max[0, i] - min[0, i])
    targeted_rate = singal_svm.score(test_x_normal, y_test)
    err_weight = 0
    y_hat_test_predict = singal_svm.predict(test_x_normal)
    for i in range(np.size(y_hat_test_predict)):
        if y_hat_test_predict[i] != y_test[i]:
            err_weight += _D[i]
    if err_weight == 0:
        err_weight = 1e-16
    alpha = 1/2*np.log((1-err_weight)/err_weight)
    for i in range(np.size(y_hat_test_predict)):
        if y_hat_test_predict[i] != y_test[i]:
            _D[i] = _D[i]/2/err_weight
        else:
            _D[i] = _D[i]/2/(1-err_weight)
    return targeted_rate,_D,alpha

def single_svm_train(train_x,train_y,test_x,test_y):

    clf = svm.SVC(C=0.5, kernel='linear', gamma=40, decision_function_shape='ovr')
    max = np.zeros((1,np.size(train_x[0,:])))
    min = np.zeros((1,np.size(train_x[0,:])))
    train_x_normal = np.mat(np.zeros(np.shape(train_x)))
    for i in range(np.size(train_x[0,:])):
        max[0,i] = train_x[:,i].max()
        min[0, i] = train_x[:, i].min()
        if max[0, i] == min[0, i]:
            train_x_normal[:, i] = np.mat(np.ones(np.shape(train_x_normal[:, i])))
        else:
            train_x_normal[:, i] = (train_x[:, i] - min[0, i]) / (max[0, i] - min[0, i])
    clf.fit(train_x_normal, train_y)
    test_x_normal = np.mat(np.zeros(np.shape(test_x)))
    for i in range(np.size(test_x[0, :])):
        max[0, i] = test_x[:, i].max()
        min[0, i] = test_x[:, i].min()
        if max[0, i] == min[0, i]:
            test_x_normal[:, i] = np.mat(np.ones(np.shape(test_x_normal[:, i])))
        else:
            test_x_normal[:, i] = (x_test[:, i] - min[0, i]) / (max[0, i] - min[0, i])
    targeted_rate = clf.score(test_x_normal, test_y)
    return targeted_rate,clf

# data = xlrd.open_workbook('data.xlsx')
# table = data.sheets()[6]
# with open('data7.json', 'w') as json_file:
#     json_file.write(json.dumps(table._cell_values))
# train_data = table._cell_values
#
# data = xlrd.open_workbook('data.xlsx')
# table = data.sheets()[7]
# with open('data8.json', 'w') as json_file:
#     json_file.write(json.dumps(table._cell_values))
# test_data = table._cell_values
# exit()
train_name = 'data1.json'
test_name = 'data2.json'
train_size = 0.7
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

with open(train_name) as json_file:
    train_data = json.load(json_file)

data_rows = len(train_data)
data_cols = len(train_data[0])
all_x_data = np.mat(train_data)[:,0:0+data_cols-1]
all_y_data = np.mat(train_data)[:,-1]
random_seed = int(datetime.datetime.now().strftime("%H%M%S"))
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(all_x_data, all_y_data,random_state=random_seed, train_size=train_size)
test_data_rows = len(y_test)
train_data_rows = len(y_train)

score = np.zeros((data_cols-2,1))
svm_sets = np.zeros((len(score),1)).tolist()
for i in range(data_cols-2):
    result = single_svm_train(x_train[:,i:i+2],y_train,x_test[:,i:i+2],y_test)
    score[i] = result[0]
    svm_sets[i] = result[1]
    print("因素"+str(i+1)+"样本集准确率为："+str(result[0]))

score_order = np.zeros((len(score),2))
for i in range(len(score)):
    score_order[i] = [i,score[i]]

score_order=score_order[score_order[:,-1].argsort()]

D = np.ones((test_data_rows,1))*1/test_data_rows
svm_weights = np.zeros((len(score),1))
final_result = np.zeros((1,test_data_rows))
for i in range(len(score_order)):
    x = x_test[:,int(score_order[i,0]):int(score_order[i,0])+2]
    svm = svm_sets[int(score_order[i,0])]
    print("D = " + str(D.T))
    result = get_weight(svm,x,y_test,D)
    D = result[1]
    x_normal = np.mat(np.zeros(np.shape(x)))
    max = np.zeros((1, np.size(x[0, :])))
    min = np.zeros((1, np.size(x[0, :])))
    for j in range(np.size(x[0, :])):
        max[0, j] = x[:, j].max()
        min[0, j] = x[:, j].min()
        if max[0, j] == min[0, j]:
            x_normal[:, j] = np.mat(np.ones(np.shape(x_normal[:, j])))
        else:
            x_normal[:, j] = (x[:, j] - min[0, j]) / (max[0, j] - min[0, j])
    # max = x.max()
    # min = x.min()
    # x_normal = (x - min) / (max - min)
    final_result = final_result+result[2]*svm.predict(x_normal)
    svm_weights[int(score_order[i,0])] = result[2]
    err = 0
    for j in range(len(final_result)):
        mid_val = 0
        if final_result[0][j] > 0:
            mid_val = 1
        elif final_result[0][j] < 0:
            mid_val = -1
        if mid_val != y_test[j]:
            err += 1
        if mid_val == 0:
            print("mid_val = "+str(mid_val))
    print("因素"+str(int(score_order[i,0])+1)+"样本集准确率为："+str(result[0]))
    print("此时累积准确率为："+str(1-err/len(final_result)))
print("权重为"+str(svm_weights.T.tolist()))
final_result = final_result[0].tolist()
print(final_result)
err = 0
for i in range(len(final_result)):
    if final_result[i]>0:
        final_result[i] = 1
    elif final_result[i]<0:
        final_result[i] = -1
    if final_result[i] != y_test[i]:
        err = err + 1
print(final_result)
print(y_test.T.tolist()[0])
print("强分类器准确率为："+str(1-err/len(final_result)))

with open(test_name) as json_file:
    test_data = json.load(json_file)

print("----------------------------------------------------")
data_rows = len(test_data)
data_cols = len(test_data[0])
y = np.mat(test_data)[:,-1]
final_result = np.zeros((1,data_rows))
for i in range(len(svm_weights)):
    x = np.mat(test_data)[:,i:i+2]
    x_normal = np.mat(np.zeros(np.shape(x)))
    max = np.zeros((1, np.size(x[0, :])))
    min = np.zeros((1, np.size(x[0, :])))
    for j in range(np.size(x[0, :])):
        max[0, j] = x[:, j].max()
        min[0, j] = x[:, j].min()
        if max[0, j] == min[0, j]:
            x_normal[:, j] = np.mat(np.ones(np.shape(x_normal[:, j])))
        else:
            x_normal[:, j] = (x[:, j] - min[0, j]) / (max[0, j] - min[0, j])
    single_svm = svm_sets[i]
    forecast = single_svm.predict(x)
    final_result = final_result+svm_weights[i]*forecast
    print(str(i+1)+"因素：weight = "+str(svm_weights[i])+"fore = "+str(forecast))
    print(final_result)
    err = 0
    for j in range(len(final_result)):
        mid_val = 0
        if final_result[0][j] > 0:
            mid_val = 1
        elif final_result[0][j] < 0:
            mid_val = -1
        if mid_val != y[j]:
            err += 1
        if mid_val == 0:
            print("累积第"+str(i+1)+"个svm时出现异常")
    print("此时累积测试准确率为："+str(1-err/len(final_result)))
print(final_result)
err = 0
final_result = final_result[0]
for i in range(len(final_result)):
    if final_result[i]>0:
        final_result[i] = 1
    elif final_result[i]<0:
        final_result[i] = -1
    if final_result[i] != y[i]:
        err+=1
    if final_result[i] == 0:
        final_result[i] = 0
print(final_result)
print(y.T.tolist()[0])
print("强分类器真实测试准确率为："+str(1-err/len(final_result)))
logger.info(final_result)
logger.info(y.T.tolist()[0])
logger.info("强分类器真实测试准确率为："+str(1-err/len(final_result)))

