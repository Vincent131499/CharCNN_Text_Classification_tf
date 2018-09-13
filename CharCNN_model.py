# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     charCNN_model
   Description :  构建基于字符的CNN模型
   Author :       Stephen
   date：          2018/9/12
-------------------------------------------------
   Change Activity:
                   2018/9/12:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf
from data_helper import Dataset
from math import sqrt
from config import config

class CharCNN(object):
    def __init__(self, l0, num_classes, conv_layers, fc_layers, l2_reg_lambda):

        #创建占位符placeholder
        self.input_x = tf.placeholder(tf.int32, [None, l0], name='input_x')
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, name='dropout_keep_prob')

        #保存L2正则项loss
        l2_loss = tf.constant(0.0)

        #Embedding-layer
        with tf.device('/cpu:0'), tf.name_scope('embedding'):
            train_data = Dataset(config.train_data_source)
            self.W, _ = train_data.onehot_dic_build()
            self.x_image = tf.nn.embedding_lookup(self.W, self.input_x)
            #将x转换为符合tensor格式的四维变量
            self.x_flat = tf.expand_dims(self.x_image, -1)
            print('嵌入层构建完成！')

        #conv-pool(6层)
        for i, c in enumerate(conv_layers):
            with tf.name_scope('conv_layer-%s'%(i + 1)):
                print('开始第 %d 卷积层的处理'%(i + 1))
                filter_width = self.x_flat.get_shape()[2].value #卷积滤波器的宽度
                filter_shape = [c[1], filter_width, 1, c[0]] #卷积滤波器的维度

                stdv = 1 / sqrt(c[0] * c[1])
                #均匀分布
                # w_conv = tf.Variable(tf.random_uniform(shape=filter_shape, minval=-stdv, maxval=stdv),
                #                      dtype='float32', name='W')
                # b_conv = tf.Variable(tf.random_uniform(shape=[c[0]], minval=-stdv, maxval=stdv),
                #                      dtype='float32', name='b')
                #高斯分布，即正态分布
                w_conv = tf.Variable(tf.random_normal(shape=filter_shape, mean=0.0, stddev=0.05),
                                     dtype='float32', name='W')
                b_conv = tf.Variable(tf.constant(0.1, shape=[c[0]]), name='b')

                conv = tf.nn.conv2d(self.x_flat,
                                    w_conv,
                                    strides=[1, 1, 1, 1],
                                    padding='VALID',
                                    name='conv')
                h_conv = tf.nn.bias_add(conv, b_conv)
                """忽然发现这里并没有用到激活函数，直接连接到最大池化"""

                #判断池化参数是否为空
                if not c[-1] is None:
                    ksize_shape = [1, c[2], 1, 1]
                    h_pool = tf.nn.max_pool(h_conv,
                                            ksize=ksize_shape,
                                            strides=ksize_shape,
                                            padding='VALID',
                                            name='pool')
                else:
                    h_pool = h_conv
                print('池化后的维度：', h_pool.get_shape())
                """此时输出的结果.shape=[batch, ]"""

                #将输出的结果转置为[batch, sentence_length, embedding_size, channels]
                self.x_flat = tf.transpose(h_pool, [0, 1, 3, 2], name='transpose')

                print(self.x_flat.get_shape())

        #将输出的维度转换为全连接层的二维输入
        with tf.name_scope('reshape'):
            fc_dim = self.x_flat.get_shape()[1].value * self.x_flat.get_shape()[2].value #全连接层维度：[batch, fc_dim]
            self.x_flat = tf.reshape(self.x_flat, shape = [-1, fc_dim])

        weights = [fc_dim] + fc_layers
        for i, fl in enumerate(fc_layers):
            with tf.name_scope('fc_layer-%s'%(i + 1)):
                print('开始第 %d 个全连接层的处理'%(i + 1))
                #均匀分布
                # stdv = 1 / sqrt(weights[i])
                # w_fc = tf.Variable(tf.random_uniform(shape=[weights[i], fl], minval=-stdv, maxval=stdv),
                #                    dtype='float32', name='W')
                # b_fc = tf.Variable(tf.random_uniform(shape=[fl], minval=-stdv, maxval=stdv),
                #                    dtype='float32', name='b')

                #高斯分布，即正态分布
                w_fc = tf.Variable(tf.random_normal(shape=[weights[i], fl], mean=0.0, stddev=0.05),
                                   dtype='float32', name='W')
                b_fc = tf.Variable(tf.constant(0.1, shape=[fl]), dtype='float32', name='b')

                """这个全连接层中间加了激活函数"""
                self.x_flat = tf.nn.relu(tf.matmul(self.x_flat, w_fc) + b_fc)

                with tf.name_scope('drop_out'):
                    self.x_flat = tf.nn.dropout(self.x_flat, self.dropout_keep_prob)

        with tf.name_scope('output_layer'):
            print('开始输出层的处理')
            #高斯分布，即正态分布
            w_out = tf.Variable(tf.random_normal(shape=[fc_layers[-1], num_classes], mean=0.0, stddev=0.05),
                                dtype='float32', name='W')
            b_out = tf.Variable(tf.constant(0.1, shape=[num_classes]), name='b')

            #均匀分布
            # stdv = 1 / sqrt(weights[-1])
            # w_out = tf.Variable(tf.random_uniform(shape=[fc_layers[-1], num_classes], minval=-stdv, maxval=stdv),
            #                     dtype='float32', name='W')
            # b_out = tf.Variable(tf.random_uniform(shape=[num_classes], minval=-stdv, maxval=stdv),
            #                     dtype='float32', name='b')

            self.y_pred = tf.nn.xw_plus_b(self.x_flat, w_out, b_out, name='y_pred')
            self.predictions = tf.argmax(self.y_pred, 1, name='predictions')

        #损失计算
        with tf.name_scope('loss'):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.y_pred, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda * l2_loss

        #精度计算
        with tf.name_scope('accuracy'):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, dtype='float'), name='accuracy')
