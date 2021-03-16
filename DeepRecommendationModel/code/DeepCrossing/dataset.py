#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/15 下午9:01
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : dataset.py
#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/15 下午8:25
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : utils.py.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import  MinMaxScaler, LabelEncoder
from  DeepRecommendationModel.code.utils import SparseFeat, DenseFeat, VarLenSparseFeat
def data_process(data_df, dense_features, sparse_features):
    """
    简单处理特征，包括填充缺失值，数值处理，类别编码
    param data_df: DataFrame格式的数据
    param dense_features: 数值特征名称列表
    param sparse_features: 类别特征名称列表
    """
    data_df[dense_features] = data_df[dense_features].fillna(0.0)
    for f in dense_features:
        data_df[f] = data_df[f].apply(lambda x: np.log(x + 1) if x > -1 else -1)

    data_df[sparse_features] = data_df[sparse_features].fillna("-1")
    for f in sparse_features:
        lbe = LabelEncoder()
        data_df[f] = lbe.fit_transform(data_df[f])

    return data_df[dense_features + sparse_features]

def data_load(data_df):
    """
    简单处理特征，包括填充缺失值，数值处理，类别编码
    param data_df: DataFrame格式的数据
    param dense_features: 数值特征名称列表
    param sparse_features: 类别特征名称列表
    """
    data = data_df
    # 划分dense和sparse特征
    columns = data.columns.values
    dense_features = [feat for feat in columns if 'I' in feat] #数值为稠密特征
    sparse_features = [feat for feat in columns if 'C' in feat] #

    # 简单的数据预处理
    train_data = data_process(data, dense_features, sparse_features)
    train_data['label'] = data['label']

    # 将特征做标记
    dnn_feature_columns = [SparseFeat(feat, vocabulary_size=data[feat].nunique(), embedding_dim=4)
                           for feat in sparse_features] + [DenseFeat(feat, 1, )
                                                           for feat in dense_features]
    return dnn_feature_columns

def sparesFeature(feat, feat_num, embed_dim=4):
    """
    create dictionary for sparse feature
    :param feat: feature name 特征名
    :param feat_num: the total number of sparse features that do not repeat 未重复的稀疏特征数量
    :param embed_dim: embedding dimension 嵌入维度
    :return:
    """
    return {'feat': feat, 'feat_num': feat_num, 'embed_dim': embed_dim}

if __name__ == "__main__":
    # 读取数据
    data = pd.read_csv('../data/criteo_sample.txt')
    dnn_feature_columns = data_load(data)

