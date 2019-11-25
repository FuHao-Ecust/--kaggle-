
# coding: utf-8

from pandas import read_csv
from os.path import join as pjoin
from sklearn.model_selection import train_test_split  
import utils
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

y_train=df_train.pop('SalePrice')

#删除并返回数据集中SalePrice标签列
all_df=pd.concat((df_train,df_test),axis=0) #要处理的整体数据集

garage_obj=['GarageType','GarageFinish','GarageQual','GarageCond'] #列出车库这一类
for garage in garage_obj:
   all_df[garage].fillna('missing',inplace=True)

#把1900标签填入空缺处表示年代久远
all_df['GarageYrBlt'].fillna(1900.,inplace=True) 
all_df['MasVnrType'].fillna('missing',inplace=True)  #用missing标签表示没装修过
all_df['MasVnrArea'].fillna(0,inplace=True)   #用0表示没装修过的装修面积


#均值补齐LotFrontage列
all_df['LotFrontage'].fillna(all_df['LotFrontage'].mean(),inplace=True)
#还有部分少量的缺失值，不是很重要，可以用one-hotd转变离散值，然后均值补齐
all_df = all_df.drop(['MiscFeature','Alley','Fence','FireplaceQu','Id'],axis=1)



all_dummies_df=pd.get_dummies(all_df)
mean_col=all_dummies_df.mean()
all_dummies_df.fillna(mean_col,inplace=True)


a=all_dummies_df.columns[all_dummies_df.dtypes=='int64'] #数值为int型
b=all_dummies_df.columns[all_dummies_df.dtypes=='float64'] #数值为float型

a_mean=all_dummies_df.loc[:,a].mean()
a_std=all_dummies_df.loc[:,a].std()
all_dummies_df.loc[:,a]=(all_dummies_df.loc[:,a]-a_mean)/a_std #使数值型为int的所有列标准化
b_mean=all_dummies_df.loc[:,b].mean()
b_std=all_dummies_df.loc[:,b].std()
all_dummies_df.loc[:,b]=(all_dummies_df.loc[:,b]-b_mean)/b_std #使数值型为float的所有列标准化

df_train1=all_dummies_df.iloc[:1460,:]
df_test1=all_dummies_df.iloc[1460:,:] 

traindata = pd.concat((df_train1,y_train),axis = 1)



#生成数据集。数据集包括标签，全包含在返回值的dataset上
def get_Datasets():
    from sklearn.datasets import make_classification
    dataSet,classLabels=make_classification(n_samples=200,n_features=100,n_classes=2)
    #print(dataSet.shape,classLabels.shape)
    return np.concatenate((dataSet,classLabels.reshape((-1,1))),axis=1)

#切分数据集，实现交叉验证。可以利用它来选择决策树个数。但本例没有实现其代码。
#原理如下：
#第一步，将训练集划分为大小相同的K份；
#第二步，我们选择其中的K-1分训练模型，将用余下的那一份计算模型的预测值，
#这一份通常被称为交叉验证集；第三步，我们对所有考虑使用的参数建立模型
#并做出预测，然后使用不同的K值重复这一过程。
#然后是关键，我们利用在不同的K下平均准确率最高所对应的决策树个数
#作为算法决策树个数
def splitDataSet(dataSet,n_folds):
    fold_size=len(dataSet)/n_folds
    data_split=[]
    begin=0
    end=fold_size
    for i in range(n_folds):
        data_split.append(dataSet[begin:end,:])
        begin=end
        end+=fold_size
    return data_split


#构建n个子集
def get_subsamples(dataSet,n):
    subDataSet=[]
    for i in range(n):
        index=[]
        for k in range(len(dataSet)):
            index.append(np.random.randint(len(dataSet)))
        subDataSet.append(dataSet[index,:])
    return subDataSet

#划分数据集
def binSplitDataSet(dataSet,feature,value):
    mat0=dataSet[nonzero(dataSet[:,feature]>value)[0],:]
    mat1=dataSet[nonzero(dataSet[:,feature]<value)[0],:]
    return mat0,mat1

#计算方差，回归时使用
def regErr(dataSet):
    return np.var(dataSet[:,-1])*shape(dataSet)[0]
#计算平均值，回归时使用
def regLeaf(dataSet):
    return np.mean(dataSet[:,-1])
def MostNumber(dataSet):  #返回多类
    #number=set(dataSet[:,-1])
    len0=len(np.nonzero(dataSet[:,-1]==0)[0])
    len1=len(np.nonzero(dataSet[:,-1]==1)[0])
    if len0>len1:
        return 0
    else:
        return 1

#选取任意的m个特征，在这m个特征中，选取分割时的最优特征  
def select_best_feature(dataSet,m,alpha="huigui"):
    f=dataSet.shape[1]
    index=[]
    bestS=inf;bestfeature=0;bestValue=0;
    if alpha=="huigui":
        S=regErr(dataSet)
    else:
        S=gini(dataSet)
    for i in range(m):
        index.append(np.random.randint(f))
    for feature in index:
        #for splitVal in set(dataSet[:,feature]):
        for splitVal in set(i[0] for i in dataSet[:,index].tolist()):
            mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
            if alpha=="huigui":  newS=regErr(mat0)+regErr(mat1)
            else:
                newS=gini(mat0)+gini(mat1)
            if bestS>newS:
                bestfeature=feature
                bestValue=splitVal
                bestS=newS
    if (S-bestS)<0.001 and alpha=="huigui":    #如果误差不大就退出
        return None,regLeaf(dataSet)
    elif (S-bestS)<0.001:
        #print(S,bestS)
        return None,MostNumber(dataSet)
    #mat0,mat1=binSplitDataSet(dataSet,feature,splitVal)
    return bestfeature,bestValue

def createTree(dataSet,alpha="huigui",m=20,max_level=10):   #实现决策树，使用20个特征，深度为10
    bestfeature,bestValue=select_best_feature(dataSet,m,alpha=alpha)
    if bestfeature==None:
        return bestValue
    retTree={}
    max_level-=1
    if max_level<0:   #控制深度
        return regLeaf(dataSet)
    retTree['bestFeature']=bestfeature
    retTree['bestVal']=bestValue
    lSet,rSet=binSplitDataSet(dataSet,bestfeature,bestValue)
    retTree['right']=createTree(rSet,alpha,m,max_level)
    retTree['left']=createTree(lSet,alpha,m,max_level)
    return retTree

def RondomForest(dataSet,n,alpha="huigui"):   #树的个数
    #dataSet=get_Datasets()
    Trees=[]
    for i in range(n):
        X_train, X_test, y_train, y_test = train_test_split(dataSet[:,:-1], dataSet[:,-1], test_size=0.33, random_state=42)
        X_train=np.concatenate((X_train,y_train.reshape((-1,1))),axis=1)
        Trees.append(createTree(X_train,alpha=alpha))
    return Trees

#预测单个数据样本
def treeForecast(tree,data,alpha="huigui"):
    bestfeature,bestValue=select_best_feature(data,m=20,alpha=alpha)
    if alpha=="huigui":
        if not isinstance(tree,dict):
            return float(tree)
        if data[tree['bestFeature']]>tree['bestVal']:
            if type(tree['left'])=='float':
                return tree['left']
            else:
                return treeForecast(tree['left'],data,alpha)
        else:
            if type(tree['right'])=='float':
                return tree['right']
            else:
                return treeForecast(tree['right'],data,alpha)   
    else:
        if not isinstance(tree,dict):
            return int(tree)
        if data[tree['bestFeature']]>tree['bestVal']:
            if type(tree['left'])=='int':
                return tree['left']
            else:
                return treeForecast(tree['left'],data,alpha)
        else:
            if type(tree['right'])=='int':
                return tree['right']
            else:
                return treeForecast(tree['right'],data,alpha)   
#单棵树预测测试集               
def createForeCast(tree,dataSet,alpha="huigui"):
    m=len(dataSet)
    yhat=np.mat(zeros((m,1)))
    for i in range(m):
        yhat[i,0]=treeForecast(tree,dataSet[i,:],alpha)
    return yhat

#随机森林预测
def predictTree(Trees,dataSet,alpha="huigui"):
    m=len(dataSet)
    yhat=np.mat(zeros((m,1)))
    for tree in Trees:
        yhat+=createForeCast(tree,dataSet,alpha)
    if alpha=="huigui": yhat/=len(Trees)
    else:
        for i in range(len(yhat)):
            if yhat[i,0]>len(Trees)/2:
                yhat[i,0]=1
            else:
                yhat[i,0]=0
    return yhat

if __name__ == '__main__' :
    dataSet = traindata.values
    RomdomTrees=RondomForest(dataSet,4,alpha="huigui")   #4棵树，分类。
    print("---------------------RomdomTrees------------------------")
    testdata = df_test1.values
    yhat=predictTree(RomdomTrees,testdata,alpha="huigui")
    print(yhat.T)


