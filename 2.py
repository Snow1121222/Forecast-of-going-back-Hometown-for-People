# -*- coding: utf-8 -*-
"""
Created on Wed Oct  5 23:01:17 2022

@author: Lenovo
"""
"""
数据集数据包括位置类特征，互联网类特征，通话类特征
    位置类特特征：基于联通基站产生的用户信令数据；
    互联网类特征：基于联通用户上网产生的上网行为数据；
    通话类特征：基于联通用户日常通话、短信产生的数据
解题基本思路：对提供的特征进行探索后，选取合适的特征构建模型，最后进行二分类预测，主要是从数据清洗、特征构造和筛选以及模型融合的思路
注意：数据集中有一定比例的噪声，需要参赛选手甄别

该代码主要通过构造位置类和通话类数据（主要处理这两类数据）的特征进行训练预测，互联网数据并没有特意进行特征构造
"""
import numpy as np
import pandas as pd
#将离散型的数据转换成0到n-1之间的数，即编码与编码还原
from sklearn.preprocessing import LabelEncoder
#逻辑回归，这里应该是线性的，主要解决二分类问题，其实就是用来分类的，比如根据指标分成好瓜和坏瓜
from sklearn.linear_model import LogisticRegression
#基于GBDT的梯度提升树，用于分类问题 
from sklearn.ensemble import GradientBoostingClassifier
#xgboost每次迭代建立新树，但需要先对数据进行优化处理
from xgboost import XGBClassifier 
#lightgbm是优化xgboost，他支持并行
from lightgbm import LGBMClassifier
#catboost处理类别型特征的梯度提升算法，解决梯度偏差、预测偏移的问题，从而减少过拟合
from catboost import CatBoostClassifier
#stacking可以将多种分类模型集合，模型融合
from sklearn.ensemble import StackingClassifier
#交叉验证：数据集分成训练集和测试集，训练集对分类器进行训练，测试集验证。
#该函数采用分层分组，是每个分组中各类别的比例同整体数据中各类别的比例尽可能相同
from sklearn.model_selection import StratifiedKFold
#ROC曲线是以假正率FPR和真正率TPR为轴的曲线，ROC曲线下面的面积叫AUC，该函数计算AUC，auc越大说明性能越好
from sklearn.metrics import roc_auc_score
#划分测试集和训练集并返回
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')#忽略匹配的警告
'''
=======================一、读取数据=============================================
'''
#读取数据
train_data = pd.read_csv('./datas/dataTrain.csv')
test_data = pd.read_csv('./datas/dataA.csv')
submission = pd.read_csv('./datas/submit_example_A.csv')
data_NoLabel = pd.read_csv('./datas/dataNoLabel.csv')
''' 
======================二、特征构造（自己定义）===================================
'''
#特征衍生:遍历位置类特征(f3列非数字且是互联网类特征)
#过程：分别遍历训练集和测试集的位置类特征并返回对应集合中，如遍历训练集的f1，df存入i和i+1行进行四则运算的结果
loc_f = ['f1', 'f2', 'f4', 'f5', 'f6']
for df in [train_data, test_data]:
    for i in range(len(loc_f)):
        for j in range(i + 1, len(loc_f)):
            df[f'{loc_f[i]}+{loc_f[j]}'] = df[loc_f[i]] + df[loc_f[j]]
            df[f'{loc_f[i]}-{loc_f[j]}'] = df[loc_f[i]] - df[loc_f[j]]
            df[f'{loc_f[i]}*{loc_f[j]}'] = df[loc_f[i]] * df[loc_f[j]]
            df[f'{loc_f[i]}/{loc_f[j]}'] = df[loc_f[i]] / (df[loc_f[j]]+1)
#遍历通话类特征
com_f = ['f43', 'f44', 'f45', 'f46'] 
for df in [train_data, test_data]:
    for i in range(len(com_f)):
        for j in range(i + 1, len(com_f)):
            df[f'{com_f[i]}+{com_f[j]}'] = df[com_f[i]] + df[com_f[j]]
            df[f'{com_f[i]}-{com_f[j]}'] = df[com_f[i]] - df[com_f[j]]
            df[f'{com_f[i]}*{com_f[j]}'] = df[com_f[i]] * df[com_f[j]]
            df[f'{com_f[i]}/{com_f[j]}'] = df[com_f[i]] / (df[com_f[j]]+1)

# 特征离散化 位置类特征           
loc_f2 = [f'f{idx}' for idx in range(1, 7) if idx != 3]
for df in [train_data, test_data]:
    for col in loc_f2:
        df[f'{col}_log'] = df[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)

# 特征交叉 位置类特征       
log_f = [f'f{idx}_log' for idx in range(1, 7) if idx != 3]
for df in [train_data, test_data]:
    for i in range(len(log_f)):
        for j in range(i + 1, len(log_f)):
            df[f'{log_f[i]}_{log_f[j]}'] = df[log_f[i]]*100 + df[log_f[j]]

# 特征离散化   通话类特征         
com_f2 = [f'f{idx}' for idx in range(43, 47)]
for df in [train_data, test_data]:
    for col in com_f2:
        df[f'{col}_log'] = df[col].apply(lambda x: int(np.log(x)) if x > 0 else 0)

# 特征交叉  通话类特征      
log_f = [f'f{idx}_log' for idx in range(43, 47)]
for df in [train_data, test_data]:
    for i in range(len(log_f)):
        for j in range(i + 1, len(log_f)):
            df[f'{log_f[i]}_{log_f[j]}'] = df[log_f[i]]*100 + df[log_f[j]]       
#特征组合:f1,f2是位置类特征且由0,1组成，更方便一些，所以选这两列进行特征组合，f47=100*f1+f2
train_data['f47'] = train_data['f1'] * 100 + train_data['f2']  
test_data['f47'] = test_data['f1'] * 100 + test_data['f2']
'''
=======================三、f3数值化后，构造出训练集和测试集=======================
'''
#f3是互联网类特征，就是将f3转成数字
f3_columns = ['f3']
data = pd.concat([train_data, test_data]) #将两个数据集的数据拼成一张表
for col in f3_columns: #遍历第三列，将训练集和测试集的第三列的属性都进行编码转成数字 col是列，row是行
    le = LabelEncoder().fit(data[col]) #获取一个LabelEncoder并训练LabelEncoder
    train_data[col] = le.transform(train_data[col]) #使用训练好的LabelEncoder对原数据进行编码
    test_data[col] = le.transform(test_data[col])

#构造训练集和测试集，去掉原本训练集的这三列
feature_columns = [ col for col in train_data.columns if col not in ['id', 'label']]
#测试集和训练集是除了id,label列剩下的列，id是没用的，要根据现在的数据判断会不会返乡即label
train = train_data[feature_columns] 
label = train_data['label'] #target = 'label'
test = test_data[feature_columns]
'''
========================四、数据清洗============================================
'''
#去除干扰数据，即噪声，用集成学习梯度提升决策树。训练集大概60000行，这里训练60，方便剔除噪声
print("原始数据集用模型训练进行数据清洗：")
oof_preds = np.zeros(train.shape[0]) #获取一个长度数据集行数的全0数组
test_preds = np.zeros(test.shape[0])
skf = StratifiedKFold(n_splits=60)#把上面处理好的训练集和标签分成60份
model=GradientBoostingClassifier()
model_name="GradientBoostingClassifier"
print(f"model = {model_name}:")
for k, (train_index, test_index) in enumerate(skf.split(train, label)): #找到分成60份的行数
    data_train, data_test = train.iloc[train_index, :], train.iloc[test_index, :] #获取每行对应的元素值
    label_train, label_test = label.iloc[train_index], label.iloc[test_index]

    model.fit(data_train,label_train) #将数据集在模型中进行训练
    label_pred = model.predict_proba(data_test)[:,1] # 对数据进行预测，返回data_test中每行预测是1的概率
    oof_preds[test_index] = label_pred.ravel()#将label_pred数据拉成一维数组
    auc = roc_auc_score(label_test,label_pred)#计算曲线roc的面积，auc数值越高分类器越优秀
    print(" kfold = %d, val_auc = %.8f" % (k, auc))#第k轮，auc的值
print("model = %s, last_auc = %.8f" % (model_name, roc_auc_score(label, oof_preds)))#60次之后，用对应模型训练的auc的值
#剔除auc接近0.5的干扰数据
train = train[:50000] #训练后，50-59组auc接近0.5为干扰数据，去掉取前50000行元素
label = label[:50000]
'''
======================五、选取几个模型融合进行特征筛选============================
'''
gbc = GradientBoostingClassifier(
    learning_rate=0.1,
    max_depth=6, #它限制了树中结点的数量，调整它可获得更好的性能
    n_estimators=50 #要执行的提升次数，太小，容易欠拟合，n_estimators太大，又容易过拟合
)
xgbc = XGBClassifier(#回归预测
    objective='binary:logistic',#目标函数：二分类任务（类别）。基于此函数进行求解最优化回归树
    eval_metric='auc',#校验数据的评价指标，分类任务：auc--roc曲线下面积
    learning_rate=0.1, #学习率，控制每次迭代更新权重时的步长，值越小，训练越慢
    n_estimators=100, #总共迭代的次数，即决策树的个数,要执行的提升次数
    max_depth=7 #它限制了树中结点的数量。树的深度,这个值也是用来避免过拟合的。越大，模型会学到更具体更局部的样本，越容易过拟合；值越小，越容易欠拟合
)
lgbm = LGBMClassifier(
    objective='binary',#任务类型
    boosting_type='gbdt',#梯度提升决策树的类型，
    metrics='auc',#模型度量标准
    num_leaves=2 ** 6, #数的最大叶子数，用于控制模型复杂度即它的值的设置应该小于2^(max_depth)，否则会进行警告，可能会导致过拟合。
    max_depth=8,#每个弱学习器也就是决策树的最大深度
    learning_rate=0.05, 
    n_estimators=100, #训练轮数（拟合的树的棵树），弱学习器的个数，其中gbdt原理是利用通过梯度不断拟合新的弱学习器，直到达到设定的弱学习器的数量。
    colsample_bytree=0.8,#训练特征采样率，列
    subsample_freq=1,#子样本频率
    max_bin=255#分桶数   
)
cbc = CatBoostClassifier(
    loss_function='Logloss', #损失函数
    iterations=210, #最大树数
    depth=10, 
    learning_rate=0.03, 
    l2_leaf_reg=1, #正则参数
    verbose=0#显示日志，0就是什么也不显示
)
estimate_models = [('gbc', gbc) , ('xgbc', xgbc) , ('lgbm', lgbm) , ('cbc', cbc)]   
scf = StackingClassifier(#模式融合
    estimators=estimate_models, #先用上面的模型进行迭代训练：第一层
    final_estimator=LogisticRegression()#第二层进行逻辑回归训练
)

'''
特诊筛选思路：先将模型训练好，然后对验证集进行测试得到基础auc，循环遍历所有特征，在验证集上对单个特征进行标记后，
得到标记后的auc，评估两个auc的差值，差值越大，则说明特征重要性越高。
'''
#先将训练数据划分成训练集和验证集:方法的四个参数：待划分数据集，待划分的样本标签，用label分割数据，随机数种子
#划分出的训练集数据、测试集数据、训练集标签、测试集标签
#stratify=label，测试集和训练集中数据的分类比例和label一致。如label中0:1=3:7，测试集有700个数据，那210个是0,490个是1
Data_train, Data_test, Label_train, Label_test = train_test_split(train, label, 
                                                                  stratify=label, random_state=2022)
#组合模型进行训练和验证，先得到基础auc
scf.fit(Data_train, Label_train)#训练模型
Label_preds = scf.predict_proba(Data_test)[:, 1]
auc = roc_auc_score(Label_test, Label_preds)
print("用组合模型进行训练得到基础auc：")
print('auc = %.8f' % auc) #验证集进行测试得到基本的auc就是单个f1....,f2......（没有f3）,输出组合模型训练后的auc

#循环遍历特征，对验证集的单个特征进行标记:输出位置类特征、通信类特征、f3互联网特征
print("对对单个特征进行标记得到的auc，减去基础auc：")
masked_features = [] #选取auc1-auc为负的特征
data_test = Data_test.copy()
for col in feature_columns: #这里序号还包括暴力特征制造的序号
    data_test[col] = 0 #进行标记
    label_pred= scf.predict_proba(data_test)[:, 1]
    auc1 = roc_auc_score(Label_test, label_pred )
    if auc1 < auc:
        masked_features.append(col)
    #在验证集上对单个特征进行mask后，得到mask后的auc，评估两个auc的差值，相差越大
    print('%5s | %.8f | %.8f' % (col, auc1, auc1 - auc)) #包括上面特征构造出的特征列
    
#选取上面计算auc1 - auc差值为负的特征，对比特征筛选后的特征提升 
scf.fit(Data_train[masked_features], Label_train)#对上面auc1较低的进行auc提升
Label_preds = scf.predict_proba(Data_test[masked_features])[:, 1]#对数据进行预测，识别为1类的数据
auc = roc_auc_score(Label_test, Label_preds)#计算曲线roc的面积，auc数值越高分类器越优秀
print("选取上面计算auc1 - auc差值为负的重要性特征，进行auc训练提升：")
print('auc = %.8f' % auc)
'''
=====================六、对筛选后的特征进行组合模型训练===========================
'''
#模型训练
train = train[masked_features]
test = test[masked_features ]
print("选取auc1-auc为负的重要性特征，用组合模型进行模型训练：")
oof_preds2 = np.zeros(train.shape[0]) #获取一个长度数据集行数的全0数组
test_preds2 = np.zeros(test.shape[0])
model_name2="StackingClassifier"
model2=scf
skf = StratifiedKFold(n_splits=12) #数据分成12等分进行交叉验证
print(f"model = {model_name2}:")
#把上面处理好的训练集和标签分成5份
for k, (train_index, test_index) in enumerate(skf.split(train, label)): #找到分成12份的行数
    data_train, data_test = train.iloc[train_index, :], train.iloc[test_index, :] #获取每行对应的元素值
    label_train, label_test = label.iloc[train_index], label.iloc[test_index]

    model2.fit(data_train,label_train) #将数据集在模型中进行训练
    label_pred = model2.predict_proba(data_test)[:,1] # 对数据进行预测，返回data_test中每行预测是1的概率
    oof_preds2[test_index] = label_pred.ravel()#将label_pred2数据拉成一维数组
    auc = roc_auc_score(label_test,label_pred)#计算曲线roc的面积，auc数值越高分类器越优秀
    print(" kfold = %d, val_auc = %.8f" % (k, auc))#第k轮，auc的值
    test_kfold_preds = model2.predict_proba(test)[:, 1]#上面训练集训练完后，测试集在进行测试
    test_preds2 += test_kfold_preds.ravel()
print("model = %s, last_auc = %.8f" % (model_name2, roc_auc_score(label, oof_preds2)))#12次之后，用对应模型训练的auc的值
scf_train_preds = test_preds2 / 12
#work文件夹下，有之前运行出的submission数据文件，每轮数据是不一样的
'''
last_pred=scf_train_preds #这里将概率改成对应的0和1，最后再写入提交文件
for i in range(len(scf_train_preds)):
    if scf_train_preds[i] >= 0.5:
        last_pred[i]=1
    else:
        last_pred[i]=0
'''
submission['label'] = scf_train_preds #model_train返回值是返乡的概率
submission.to_csv('./datas/submission.csv', index=False)#将结果存到这个数据文档中 aasa
