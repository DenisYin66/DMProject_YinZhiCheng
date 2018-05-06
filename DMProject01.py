# -*- coding: utf-8 -*-
#coding=utf-8
#@author: Yinzhicheng// yinzhicheng@cug.edu.cn
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import matplotlib.pyplot as plt
import math as ma
import scipy.stats as st
data = []
#下载数据集magic04
def loadDataSet(filename):
    fData = open(filename)
    for line in fData.readlines():
        curLine = line.strip().split(',')
        curLine.pop()
        curLine = list(map(eval, curLine))
        data.append(curLine)
#获取多元均值向量
def getMultivariateMeanVector(dataSet):
    Mean=np.mean(dataSet,axis=0)
    MeanMatric = np.matrix(Mean)
    return MeanMatric.transpose()
#获取样本协方差the sample covariance
def getSampleCovariance(dataSet):
    return  np.cov(np.array(dataSet).T)
#获取样本协方差（中心矩阵内积公式）
def getSampleCovarianceByInner(dataSet):
    Z = getZ(dataSet)
    len = Z.shape[0]
    lenz = dataSet.__len__()
    print(len)
    print(lenz)
    return   np.dot(Z.T,Z)/len
#获取样本协方差（中心矩阵外积公式）
def getSampleCovarianceByOuter(dataSet):
    Cov = 0
    Z = getZ(dataSet)
    for index in range(Z.shape[0]):
        Cov += np.dot(Z[index,:].T,Z[index,:])/Z.shape[0]
    return Cov
#获取某条属性的方差
def getOneAttrVar(DataVector):
    try:
        Var = np.var(DataVector)
        return Var
    except e:
        if hasattr(e,"code"):
            print(e.code)
        if hasattr(e,"reason"):
            print(e.reason)
        return 0
#获取Centered Data matrix矩阵Z
def getZ(dataSet):
    u = getMultivariateMeanVector(dataSet)
    D = np.matrix(dataSet)
    Z = D - np.matrix(np.repeat(u.transpose(),D.shape[0],axis=0))
    return  Z
#获取属性1和属性2相关性，以角度表示
def getAngleCos(dataSet):
    Z = getZ(dataSet)
    attr1 = np.matrix(Z[:,0])
    attr2 = np.matrix(Z[:,1])
    aLen = ma.sqrt(np.dot(attr1.transpose(),attr1))
    bLen = ma.sqrt(np.dot(attr2.transpose(),attr2))
    return np.dot(attr1.transpose(),attr2)/aLen/bLen
#获取属性1的PDF（概率密度函数）
def getPDFofAttr1(DataSet):
    mean= np.mean(DataSet,axis=0)[0]
    var = np.var(DataSet[0])
    xs = np.linspace(-2000, 2000, 1000)
    plt.title("My Project01 learning")
    plt.plot(xs, st.norm.pdf(xs, loc=mean, scale=var) ,'b',label = 'line1')
    plt.ylabel('Probability density function')
    plt.show()
#获取所有属性中最大的方差值与最小值
def getMaxAndMinVar(DataSet):
    VarArr = np.var(DataSet,axis=0)
    VarArr.sort()
    VarMin = VarArr[0]
    VarMax = VarArr[len(VarArr)-1]
    return VarMax,VarMin

loadDataSet('E:\DataMining\Data\magic04.txt')
DataMean = getMaxAndMinVar(data)
a=getAngleCos(data)
print(DataMean)
