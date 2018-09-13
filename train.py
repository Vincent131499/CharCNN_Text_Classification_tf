# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     train
   Description :   模型训练
   Author :       Stephen
   date：          2018/9/12
-------------------------------------------------
   Change Activity:
                   2018/9/12:
-------------------------------------------------
"""
__author__ = 'Stephen'

import tensorflow as tf
# from tensorflow.python import debug as tf_debug
import time
import numpy as np
import os
import datetime
from data_helper import Dataset
from CharCNN_model import CharCNN
from config import config

#GPU设置
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0"

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

with tf.Graph().as_default():
    session_conf = tf.ConfigProto(allow_soft_placement = True,
                                  log_device_placement = False)
    session_conf.gpu_options.allow_growth = True
    session = tf.Session(config=session_conf)

    with session.as_default():
        cnn_model = CharCNN(l0 = config.l0,
                            num_classes = config.nums_classes,
                            conv_layers = config.model.conv_layers,
                            fc_layers = config.model.fc_layers,
                            l2_reg_lambda = 0)

        # 定义训练过程training procedure
        global_step = tf.Variable(0, name='global_step', trainable=False)
        # 1.定义优化器Adam
        optimizer = tf.train.AdamOptimizer(config.model.learning_rate)
        # 2.通过优化器Adam计算梯度
        grads_and_vars = optimizer.compute_gradients(cnn_model.loss)
        # 3.将梯度应用于变量并更新global_step(自增)
        train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        # Keep track of gradient values and sparsity (optional)
        grad_summaries = []
        for g, v in grads_and_vars:
            if g is not None:
                grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
                sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
                grad_summaries.append(grad_hist_summary)
                grad_summaries.append(sparsity_summary)
        grad_summaries_merged = tf.summary.merge(grad_summaries)

        # 将模型和记录summary写入本地文件
        timestamp = str(int(time.time()))
        # 获取完整路径
        out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs_Gussian', timestamp))
        print('Writing to {}\n'.format(out_dir))

        # 记录训练过程中的loss和accuracy (Summaries for loss and accuracy)
        loss_summary = tf.summary.scalar('loss', cnn_model.loss)
        accuracy_summary = tf.summary.scalar('accuracy', cnn_model.accuracy)

        """记录训练过程Train Summaries"""
        # tf.summary.merge将指定的summary组合在一起
        train_summary_op = tf.summary.merge([loss_summary, accuracy_summary, grad_summaries_merged])
        train_summary_dir = os.path.join(out_dir, "summaries", "train")
        train_summary_writer = tf.summary.FileWriter(train_summary_dir, session.graph_def)

        """记录测试过程Dev Summaries"""
        dev_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
        dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')
        dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, session.graph_def)

        """检查点Checkpointing"""
        checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
        checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
        # 在Tensorflow中假定路径已经存在，故我们需要判断该路径，不存在需要创建
        if os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)
        # 创建一个Saver对象用来保存模型
        saver = tf.train.Saver(tf.global_variables())

        # 初始化变量
        # session.run(tf.initialize_all_variables())
        session.run(tf.global_variables_initializer())
        # session.run(tf.local_variables_initializer())


        # 定义单个训练步骤
        def train_step(x_batch, y_batch):
            feed_dict = {cnn_model.input_x: x_batch,
                         cnn_model.input_y: y_batch,
                         cnn_model.dropout_keep_prob: config.model.dropout_keep_prob}
            _, step, summaries, loss, accuracy = session.run(
                [train_op, global_step, train_summary_op, cnn_model.loss, cnn_model.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

        """对dev-set进行评估"""
        def dev_step(x_dev, y_dev, writer=None):
            feed_dict = {cnn_model.input_x: x_dev,
                         cnn_model.input_y: y_dev,
                         cnn_model.dropout_keep_prob: 1.0}
            step, summaries, batch_loss, accuracy = session.run(
                [global_step, dev_summary_op, cnn_model.loss, cnn_model.accuracy],
                feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print('{}: step {}, loss {:g}, acc {:g}'.format(time_str, step, batch_loss, accuracy))
            train_summary_writer.add_summary(summaries, step)

            return batch_loss, accuracy

        print('模型初始化完毕，开始训练......')

        # batches = train_data.batch_iter(list(zip(train_data.doc_image, train_data.label_image)), config.batch_size, config.training.epoches)

        for i in range(config.training.epoches): #50000
            x_batch, y_batch = train_data.next_batch()

            train_step(x_batch, y_batch)
            current_step = tf.train.global_step(session, global_step)  # 将Session和global_step值传进来
            if current_step % config.training.evaluate_every == 0:  # 每evaluate_every次每100执行一次测试
                print('\nEvaluation')
                # x_dev_batch, y_dev_batch = dev_data.next_batch()
                dev_step(dev_data.doc_image, dev_data.label_image, writer=dev_summary_writer)
                print('')
            if current_step % config.training.checkpoint_every == 0:  # 每checkpoint_every次(100)执行一次保存模型
                path = saver.save(session, checkpoint_prefix, global_step=current_step)  # 定义模型保存路径
                print('Saved model checkpoint to {}\n'.format(path))

        # for batch in batches:
        #     x_batch, y_batch = zip(*batch)
        #     train_step(x_batch, y_batch)
        #     current_step = tf.train.global_step(session, global_step)
        #     if current_step % config.training.evaluate_every == 0:
        #         total_dev_loss = 0.0
        #         total_dev_accuracy = 0.0
        #
        #         print("\nEvaluation:")
        #         dev_batches = Dataset.batch_iter(list(zip(train_data.doc_image, train_data.label_image)), config.batch_size, 1)
        #         for dev_batch in dev_batches:
        #             x_dev_batch, y_dev_batch = zip(*dev_batch)
        #             dev_loss, dev_accuracy = dev_step(x_dev_batch, y_dev_batch)
        #             total_dev_loss += dev_loss
        #             total_dev_accuracy += dev_accuracy
        #         total_dev_accuracy = total_dev_accuracy / (len(dev_data.label_image) / config.batch_size)
        #         print("dev_loss {:g}, dev_acc {:g}, num_dev_batches {:g}".format(total_dev_loss, total_dev_accuracy,
        #                                                                          len(dev_data.label_image) / config.batch_size))


