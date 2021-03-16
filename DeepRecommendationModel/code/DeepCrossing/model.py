#!/usr/bin/env python
# -*- coding: utf-8 -*-
# !@Time    : 2021/3/15 下午8:24
# !@Author  : miracleyin @email: miracleyin@live.com
# !@File    : model.py

import torch
import torch.nn as nn
import torch.nn.functional as F

class Residual_block(nn.Module):
    def __init__(self, dim_stack, hidden_unit):
        super(Residual_block, self).__init__()
        self.linear1 = nn.Linear(dim_stack, hidden_unit)
        self.linear2 = nn.Linear(hidden_unit, dim_stack)
        self.relu = nn.ReLU()
    def forward(self, x):
        orig_x = x.clone()
        x = self.linear1(x)
        x = self.linear2(x)
        out = self.relu(x + orig_x)
        return out

class Deep_Crossing(nn.Module):
    def __init__(
            self,
            number_of_category_feature,
            number_of_total_feathre,
            list_of_nunique_category,
            hidden_size_for_residual_block,
            embedding_size):
        super(Deep_Crossing, self).__init__()
        self.number_of_category_feature = number_of_category_feature
        self.number_of_total_feature = number_of_total_feathre
        self.list_of_nunique_category = list_of_nunique_category

        total_size_for_single_sample = sum(
            i if i <= embedding_size else embedding_size for i in list_of_nunique_category
        ) + (number_of_total_feathre - number_of_category_feature)
        self.hidden_size_for_residual_block = hidden_size_for_residual_block
        self.embedding_size = embedding_size

        self.residual_blocks = nn.ModuleList([
            Residual_block(total_size_for_single_sample, size) for size in hidden_size_for_residual_block
        ])
        self.Full_connect_layer_after_residual_block = nn.Linear(total_size_for_single_sample, 1)
        self.Sigmoid = nn.Sigmoid()

    def forward(self, x):
        x_numeric, x_category = x[:, : -self.number_of_category_feature], x[:, :-self.number_of_category_feature]
        one_hot_list = []
        for i in range(x_category.shape[1]):
            embedded_feature = torch.zeros(
                x.shape[0], self.list_of_nunique_category[i]
            ).scatter_(
                1, x_category.T[i].reshape(-1,1).long(), 1)
            if embedded_feature.shape[-1] > self.embedding_sizeL:
                embedded_feature = nn.Linear(self.list_of_nunique_category[i], self.embedding_size)(embedded_feature)
            one_hot_list.append(embedded_feature)
        x_category = torch.cat(one_hot_list. -1)
        x = torch.cat([x_numeric, x_category], -1)
        for block in self.residual_blocks:
            x = block(x)
        x = self.Full_connect_layer_after_residual_block(x)
        out = self.Sigmoid(x)
        return out



if __name__ == "__main__":
