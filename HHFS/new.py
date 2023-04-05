import numpy as np
import math
import random
import matplotlib.pyplot as plt
import pandas as pd
import copy
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import jaccard_score
from sklearn.metrics import f1_score
from sklearn.feature_selection import mutual_info_classif
from sklearn.neighbors import KNeighborsClassifier 

#函数
def Eager(trainData):
    global pathList
    leafNodes = []
    returnList = []
    for go in DAG.columns:
        if(DAG.loc[go,:].sum() == 0):
            leafNodes.append(go)
    for leaf in leafNodes:
        pathList = []
        getPathsToRoot([],leaf)
        for path in pathList:
            
            #path = list(filter(lambda x:x in DAG.columns,path))
            pathIG = IGWeight[path]
            highIG = pathIG[pathIG >= pathIG.mean()]
            for index in highIG.index:
                returnList.append(index)
    
    return list(set(returnList))

def Lazy(x):
    #step1
    dropList = []
    for name in DAG.columns:
        if(x[name] == 1):
            #遍历当前节点的所有祖先节点
            for anc in AncDict[name]:
                if(Relevance1[anc] < Relevance1[name]):
                    dropList.append(anc)
        else:
            dropList.append(name)
    return list(set(DAG.columns)-set(dropList))

def test(x,trainData):
    global pathList
    #找到所有叶子结点
    leafNodes = []
    dropList = []
    returnList = []
    for go in DAG.columns:
        if(DAG.loc[go,:].sum() == 0):
            leafNodes.append(go)
    #step1
    dropList = []
    for name in DAG.columns:
        if(x[name] == 1):
            #遍历当前节点的所有祖先节点
            for anc in AncDict[name]:
                if(Relevance1[anc] < Relevance1[name]):
                    dropList.append(anc)
        else:
            dropList.append(name)
    selectFeatures = list(set(DAG.columns)-set(dropList))

    print(len(selectFeatures))
    #step2
    leafNodes = []
    for go in DAG.columns:
        if(DAG.loc[go,:].sum() == 0):
            leafNodes.append(go)
    for leaf in leafNodes:
        pathList = []
        getPathsToRoot([],leaf)
        for path in pathList:
            path = list(filter(lambda x:x in selectFeatures,path))
            pathIG = IGWeight[path]
            highIG = pathIG[pathIG >= pathIG.mean()]
            for index in highIG.index:
                returnList.append(index)
    return list(set(returnList))

def GM_score(x,y): #计算GM 传入numpy
    cm1 = confusion_matrix(x,y)
    sensitivity1 = cm1[0,0]/(cm1[0,0]+cm1[0,1])
    specificity1 = cm1[1,1]/(cm1[1,0]+cm1[1,1])
    return sensitivity1,specificity1

def getPathsToRoot(path,name):#传入list 方便递归
    global pathList
    flag = 0
    path.append(name)
    for anc in DAG.columns:
        if(DAG.loc[anc,name] == 1):
            getPathsToRoot(copy.copy(path),anc)
            flag = 1
    if(flag == 0):
        pathList.append(path)
        
def RPV(x):
    dropList = []
    for name in DAG.columns:
        if(x[name] == 1):
            #遍历当前节点的所有祖先节点
            for anc in AncDict[name]:
                if(Relevance1[anc] < Relevance1[name]):
                    dropList.append(anc)
        else:
            dropList.append(name)
    return list(set(DAG.columns)-set(dropList))

def MR(x):
    global pathList
    dropSet = set()
    for name in DAG.columns:
        pathList = []
        if(x[name] == 1):#找到所有至根节点路径
            getPathsToRoot([],name)
            for path in pathList:
                #保留相关性最高特征，其它特征加入dropSet
                dropSet.update(Relevance1[path].sort_values(ascending=False).index[1:])
        else:
            dropSet.add(name)
#             getPathsToLeft([],name)
#             for path in pathList:
#                 #保留相关性最高特征，其它特征加入dropSet
#                 dropSet.update(Relevance0[path].sort_values(ascending=False).index[1:])
        #print(pathList)
    return list(set(DAG.columns)-dropSet)

def myknn(train,label,x_test):
    predictList = []
    for j in range(x_test.shape[0]):
        test = x_test.iloc[j,:]
        maxJcd = 0
        prediction = 0
        for i in range(train.shape[0]):
            temp = jaccard_score(train.iloc[i,:],test)
            if(temp > maxJcd):
                prediction = label[i]
                maxJcd = temp
        predictList.append(prediction)
    return predictList

def getAncestors(name):
    global relativeList
    for relative in DAG.columns:
        if(DAG.loc[relative,name] == 1):
            relativeList.append(relative)
            getAncestors(relative)
def getDecestors(name):
    global relativeList
    for relative in DAG.columns:
        if(DAG.loc[name,relative] == 1):
            relativeList.append(relative)
            getDecestors(relative)
def getPathsToRoot(path,name):#传入list 方便递归
    global pathList
    flag = 0
    path.append(name)
    for anc in DAG.columns:
        if(DAG.loc[anc,name] == 1):
            getPathsToRoot(copy.copy(path),anc)
            flag = 1
    if(flag == 0):
        pathList.append(path)
def getPathsToLeft(path,name):#传入list 方便递归
    global pathList
    flag = 0
    path.append(name)
    for dec in DAG.columns:
        if(DAG.loc[name,dec] == 1):
            flag = 1
            getPathsToLeft(copy.copy(path),dec)
    if(flag == 0):
        pathList.append(path)

#加载数据 预处理
data = pd.read_csv("../data/real_world_datasets/Datasets/DATA_SportsTweetsc.csv",index_col=0)
label = data["label"]
#删除无用列
data = data.drop(["label"],axis=1)

#初始化基因本体的DAG
DAG = pd.read_csv("../data/real_world_datasets/Hierarchy/DAG_SportsTweetsc.csv",index_col=0)

threshold = 0.999
AncDict = {}
DecDict = {}
relativeList = []
sparseList = []
sensitivity = []
specificity = []
F1 = []
AUC = []
featureNum = []
pathList = []
numList = []

#过滤低纬度特征
for c in data.columns:
    data[c] = data[c].astype('int')
    if(data[c].sum() < 3):
        data.drop(c,axis=1,inplace=True)
DAG = DAG.loc[data.columns,data.columns]

#计算相关性
IGWeight = mutual_info_classif(data,label,discrete_features=True)
IGWeight = pd.Series(IGWeight,index=DAG.columns)

#计算相关性
data["label"] = label
Relevance1 = pd.Series(0.0,index=DAG.columns)
for name in DAG.columns:
    temp = data[data[name] == 1]
    prob = temp['label'].mean()
    Relevance1[name] = (prob-0.5)*(prob-0.5)+(0.5-prob)*(0.5-prob)
    
Relevance0 = pd.Series(0.0,index=DAG.columns)
for name in DAG.columns:
    temp = data[data[name] == 0]
    prob = temp['label'].mean()
    Relevance0[name] = (prob-0.5)*(prob-0.5)+(0.5-prob)*(0.5-prob)  
data = data.drop(["label"],axis=1)

#计算所有节点的祖先节点
for name in DAG.columns:
    global relativeList
    relativeList = []
    getAncestors(name)
    AncDict[name] = relativeList
#计算所有节点的孩子节点
for name in DAG.columns:
    global relativeList
    relativeList = []
    getDecestors(name)
    DecDict[name] = relativeList

#10折交叉验证
kf = KFold(n_splits=3,shuffle=True)
for train_index ,test_index in kf.split(data):
    trainData = data.iloc[train_index,:]
    testData = data.iloc[test_index,:]
    Y_train = label.values[train_index] #用于训练模型
    Y_test = label.values[test_index]#用于交叉验证
    predicetList = []
    
    feat1 = Eager(trainData)
    for i in range(testData.shape[0]):  #对每一个样本进行特征选择和预测
        
        feat2 = Lazy(testData.iloc[i,:]) #RPV特征选择
        #selectFeatures = list(set(feat1).union(set(feat2))) #并集
        selectFeatures = list(set(feat1).intersection(set(feat2)))
        
        print(len(feat1),len(feat2),len(selectFeatures))
        
        if(len(selectFeatures) == 0):
            predicetList.append(pd.Series(Y_train).mode()[0])
            continue
            #selectFeatures = DAG.columns
        else:
            numList.append(len(selectFeatures))
        X_train = trainData.loc[:,selectFeatures]
        X_test = testData.loc[:,selectFeatures].iloc[i,:]
        
        gnb = GaussianNB()
        predicetList.append(gnb.fit(X_train, Y_train).predict(X_test.values.reshape(1,len(selectFeatures)))[0])#把当前样本预测结果加入List
        #knn=KNeighborsClassifier()
        #knn.fit(X_train,Y_train)
        #predicetList.append(knn.predict(X_test.values.reshape(1,len(selectFeatures)))[0])
    print(predicetList)
    print(Y_test)
    sensi,speci= GM_score(np.array(predicetList),Y_test)
    sensitivity.append(sensi)
    specificity.append(speci)
    F1.append(f1_score(np.array(predicetList),Y_test))
    try:
        AUC.append(roc_auc_score(np.array(predicetList),Y_test))
    except ValueError:
        pass

#新方法
a = np.nanmean(sensitivity)
b = np.nanmean(specificity)
print(a)
print(b)
print("GM")
print(round(math.sqrt(a*b)*100,1))
print("f1")
print(round(np.nanmean(F1)*100,1))
print("AUC")
print(round(np.nanmean(AUC)*100,1))
print("选择特征百分比")
print(round(np.nanmean(numList)/DAG.shape[0]*100,1))