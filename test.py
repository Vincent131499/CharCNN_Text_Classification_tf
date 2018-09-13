# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     test
   Description :
   Author :       Stephen
   date：          2018/9/11
-------------------------------------------------
   Change Activity:
                   2018/9/11:
-------------------------------------------------
"""
__author__ = 'Stephen'
import tensorflow as tf
from tensorflow.python import debug as tf_debug
import time
import numpy as np
import os
import datetime
from data_helper import Dataset
from CharCNN_model import CharCNN
from config import config

#GPU设置
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ['CUDA_VISIBLE_DEVICES'] = "0"

#读取数据
print('正在载入数据......')

print('==========训练集================')
train_data = Dataset(config.train_data_source)
train_data.dataset_read()
"""
train_data:
样本维度： (120000, 1014)
标签维度： (120000, 4)
"""

print('==========测试集================')
dev_data = Dataset(config.dev_data_source)
dev_data.dataset_read()
"""
dev_data:
样本维度： (7600, 1014)
标签维度： (7600, 4)
"""

print(dev_data.doc_image)

print('====================')

print(dev_data.label_image)