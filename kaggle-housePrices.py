
from pandas import read_csv
from os.path import join as pjoin
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  
import utils
import numpy as np

LOG_DIR = '../result'
logger = utils.setup_logger(LOG_DIR, "house-p-", save_log=True)
MODEL_CP_DIR = '../result'
RESULT_DIR = '../result'

#读取数据集
df_train=pd.read_csv('train.csv')
df_test=pd.read_csv('test.csv')

y_train=df_train.pop('SalePrice')  #删除并返回数据集中SalePrice标签列
all_df=pd.concat((df_train,df_test),axis=0) #要处理的整体数据集


total=all_df.isnull().sum().sort_values(ascending=False)  #每列缺失数量
percent=(all_df.isnull().sum()/len(all_df)).sort_values(ascending=False) #每列缺失率
miss_data=pd.concat([total,percent],axis=1,keys=['total','percent'])



garage_obj=['GarageType','GarageFinish','GarageQual','GarageCond'] #列出车库这一类
for garage in garage_obj:
   all_df[garage].fillna('missing',inplace=True)

all_df['GarageYrBlt'].fillna(1900.,inplace=True) 



all_df['MasVnrType'].fillna('missing',inplace=True)  #用missing标签表示没装修过
all_df['MasVnrArea'].fillna(0,inplace=True)   #用0表示没装修过的装修面积

(all_df.isnull().sum()/len(all_df)).sort_values(ascending=False) 

#均值补齐LotFrontage列
all_df['LotFrontage'].fillna(all_df['LotFrontage'].mean(),inplace=True) #还有部分少量的缺失值，不是很重要，可以用one-hotd转变离散值，然后均值补齐
all_df

all_df.drop(['MiscFeature','Alley','Fence','FireplaceQu'],axis=1)

all_dummies_df=pd.get_dummies(all_df)
mean_col=all_dummies_df.mean()
all_dummies_df.fillna(mean_col,inplace=True)




#数据集中数值类型为int和float
a=all_dummies_df.columns[all_dummies_df.dtypes=='int64'] #数值为int型
b=all_dummies_df.columns[all_dummies_df.dtypes=='float64'] #数值为float型

#进行标准化处理，符合0-1分布
a_mean=all_dummies_df.loc[:,a].mean()
a_std=all_dummies_df.loc[:,a].std()
all_dummies_df.loc[:,a]=(all_dummies_df.loc[:,a]-a_mean)/a_std #使数值型为int的所有列标准化
b_mean=all_dummies_df.loc[:,b].mean()
b_std=all_dummies_df.loc[:,b].std()
all_dummies_df.loc[:,b]=(all_dummies_df.loc[:,b]-b_mean)/b_std #使数值型为float的所有列标准化




#处理后的训练集(不含Saleprice)
df_train1=all_dummies_df.iloc[:1460,:]    

df_train_train=df_train1.iloc[0:int(0.8*len(df_train1)),:]  #train中的训练集(不含Saleprice)
df_train_test=df_train1.iloc[int(0.8*len(df_train1)):,:]    #train中的测试集(不含Saleprice)

df_train_train_y=y_train.iloc[0:int(0.8*len(y_train))]     #train中训练集的target
df_train_test_y=y_train.iloc[int(0.8*len(df_train1)):]     #train中测试集的target

#处理后的测试集
df_test1=all_dummies_df.iloc[1460:,:] 
df_test1


x_train = df_train_train
y_train = df_train_train_y
x_test = df_train_test
y_test = df_train_test_y


from sklearn.linear_model import LinearRegression
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor





cv_params = {'n_estimators': [550, 575, 600, 650, 675]}
other_params = {'learning_rate': 0.1, 'n_estimators': 600, 'max_depth': 5, 'min_child_weight': 1, 'seed': 0, 'subsample': 0.8, 'colsample_bytree': 0.8, 'gamma': 0, 'reg_alpha': 0, 'reg_lambda': 1}

clfs = [DecisionTreeRegressor(), 
        RandomForestRegressor(),
        AdaBoostRegressor(), 
        GradientBoostingRegressor(), 
       ]




# plot_feature_importances
def plot_feature_importances(clf, X_train, y_train, X_test, y_test, top_n=10,
    figsize=(12,12), print_table=False, title="Feature Importances", logger = None):

    __name__ = "plot_feature_importances"
    
    import pandas as pd
    import numpy  as np
    import matplotlib.pyplot as plt
    from sklearn.externals import joblib
    
    from xgboost.core     import XGBoostError
    from lightgbm.sklearn import LightGBMError
    
    try: 
        if not hasattr(clf, 'feature_importances_'):
            clf.fit(X_train.values, y_train.values.ravel())

            if not hasattr(clf, 'feature_importances_'):
                raise AttributeError("{} does not have feature_importances_ attribute".
                                    format(clf.__class__.__name__))
                
    except (XGBoostError, LightGBMError, ValueError):
        clf.fit(X_train.values, y_train.values.ravel())
    
    model_name = clf.__class__.__name__
    
    # score
    logger.info("Model name: \n {}".format(clf.__class__.__name__))
        score = clf.score(X_test, y_test)
        y_pred = clf.predict(x_test)
        
    logger.info("Model score: \n {}".format(score))
    
    model_path = pjoin(MODEL_CP_DIR, clf.__class__.__name__ + '-'+str(score)+ '.pkl')
    logger.info("Model path: \n {}".format(model_path))
    
    joblib.dump(clf, model_path)
      
    # Feature Importance
    feat_imp = pd.DataFrame({'Importance':clf.feature_importances_})    
    feat_imp['Feature'] = X_train.columns
    feat_imp.sort_values(by='Importance', ascending=False, inplace=True)
    feat_imp = feat_imp.iloc[:top_n]
    
    feat_imp.sort_values(by='Importance', inplace=True)
    feat_imp = feat_imp.set_index('Feature', drop=True)
    feat_imp.plot.barh(title=title, figsize=figsize)
    plt.xlabel('Feature Importance Score')
    plt.savefig(pjoin(MODEL_CP_DIR, clf.__class__.__name__ + '-'+str(score)+"-Feature Importance.png"))
    plt.show()
        
    logger.info("Feature Importance Score: \n {} \n\n".format(feat_imp))
    
    
        
    if print_table:
        from IPython.display import display
        print("Top {} features in descending order of importance".format(top_n))
        display(feat_imp.sort_values(by='Importance', ascending=False))
        
    return feat_imp, score




best_score = 0
best_model = None
for clf in clfs:
    try:
        feat_imp, score = plot_feature_importances(
            clf, x_train, y_train, x_test, y_test,
            top_n=x_train.shape[1], title=clf.__class__.__name__, logger=logger)
        
        if best_score < score:
            best_score = score
            best_model = clf.__class__.__name__
            
         
    except AttributeError as e:
        print(e)
        
logger.info("Best model: \n {} : {}\n\n".format(best_model, best_score))




print(("Best model: \n  {}: {}\n".format(best_model, best_score)))


def predict_result(clf,x_train,y_train,x_test):
    rf = clf.fit(x_train,y_train)
    preds = rf.predict(x_test)
    result = pd.DataFrame({'SalePrice':preds})
    return result
    

submission = predict_result(GradientBoostingRegressor(),x_train,y_train,df_test1)
submission.to_csv('submission.csv',index=False)
submission.head(5)

