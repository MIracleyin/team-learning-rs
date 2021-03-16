#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/15 下午4:07
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : DataLoadandPreprocessing.py.py

# import packages
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.model_selection import train_test_split

# import data
train_df = pd.read_csv('../../dataset/train.csv')
test_df = pd.read_csv('../../dataset/test.csv')

print(train_df.shape, test_df.shape)

# 进行数据合并
label = train_df['Label']
del train_df['Label']

data_df = pd.concat((train_df, test_df))

del data_df['Id']

print(data_df.columns)

# 特征分开类别
sparse_feas = [col for col in data_df.columns if col[0] == 'C']
dense_feas = [col for col in data_df.columns if col[0] == 'I']

# 填充缺失值
data_df[sparse_feas] = data_df[sparse_feas].fillna('-1')
data_df[dense_feas] = data_df[dense_feas].fillna(0)


# 进行编码  类别特征编码
for feat in sparse_feas:
    le = LabelEncoder()
    data_df[feat] = le.fit_transform(data_df[feat])

# 数值特征归一化
mms = MinMaxScaler()
data_df[dense_feas] = mms.fit_transform(data_df[dense_feas])


# 分开测试集和训练集
train = data_df[:train_df.shape[0]]
test = data_df[train_df.shape[0]:]

train['Label'] = label

train_set, val_set = train_test_split(train, test_size = 0.2, random_state=2020)
train_set['Label'].value_counts()

val_set['Label'].value_counts()


# 保存文件
train_set.reset_index(drop=True, inplace=True)
val_set.reset_index(drop=True, inplace=True)

train_set.to_csv('preprocessed_data/train_set.csv', index=0)
val_set.to_csv('preprocessed_data/val_set.csv', index=0)
test.to_csv('preprocessed_data/test.csv', index=0)