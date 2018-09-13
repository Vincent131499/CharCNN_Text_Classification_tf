# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     data_helper
   Description :   数据处理文件
   Author :       Stephen
   date：          2018/9/11
-------------------------------------------------
   Change Activity:
                   2018/9/11:
-------------------------------------------------
"""
__author__ = 'Stephen'

import config
import numpy as np
import csv
import re

class Dataset(object):
    def __init__(self, data_source):
        self.data_source = data_source #数据集路径
        self.index_in_epoch = 0
        self.alphabet = config.config.alphabet #字母表
        self.alphabet_size = config.config.alphabet_size #字母表长度
        self.num_classes = config.config.nums_classes #类别数目
        self.l0 = config.config.l0
        self.epochs_completed = 0
        self.batch_size = config.config.batch_size
        self.example_nums = config.config.example_nums
        self.doc_image = [] #所有文章的向量集合
        self.label_image = [] #所有文章的标签集合

    """得到Dataset对象的batch"""
    def next_batch(self):
        start = self.index_in_epoch
        self.index_in_epoch += self.batch_size
        if self.index_in_epoch > self.example_nums:
            #一个epoch完成
            self.epochs_completed += 1
            #shuffle data重新洗牌
            perm = np.arange(self.example_nums)
            np.random.shuffle(perm)
            self.doc_image = self.doc_image[perm]
            self.label_image = self.label_image[perm]
            #开始下一个epoch
            start = 0
            self.index_in_epoch = self.batch_size
            assert self.batch_size <= self.example_nums
        end = self.index_in_epoch
        batch_x = np.array(self.doc_image[start:end], dtype='int64')
        batch_y = np.array(self.label_image[start:end], dtype='float32')

        return batch_x, batch_y

    """读取数据集"""
    def dataset_read(self):
        docs = []
        label = []
        doc_count = 0
        csvfile = open(self.data_source, 'r', encoding='utf-8')
        for line in csv.reader(csvfile, delimiter = ',', quotechar = '"'): #quotechar代表各字段的字符串引用字符
            content = line[1] + ". " + line[2]
            content = content.replace('\n', '')
            docs.append(content.lower())
            label.append(line[0])
            doc_count = doc_count + 1

        #引入embedding矩阵和字典
        print('引入嵌入词典和矩阵')
        embedding_w, embedding_dic = self.onehot_dic_build()

        print('开始进行文档处理')
        doc_image = []
        label_image = []
        for i in range(doc_count):
            #将每个句子中的所有字母，转换为embedding矩阵的索引
            doc_vec = self.doc_process(docs[i], embedding_dic)
            doc_image.append(doc_vec)
            #类别标签为1-4
            label_class = np.zeros(self.num_classes, dtype='float32')
            label_class[int(label[i]) - 1] = 1
            label_image.append(label_class)

        del embedding_w, embedding_dic

        print('求得训练集与测试集的tensor并赋值')
        self.doc_image = np.asarray(doc_image, dtype='int64')
        self.label_image = np.asarray(label_image, dtype='float32')
        print('样本维度：', self.doc_image.shape)
        print('标签维度：', self.label_image.shape)

    """one-hot编码"""
    def onehot_dic_build(self):
        alphabet = self.alphabet #字母表
        embedding_dic = {}
        embedding_w = []
        #对于字母表中不存在的或者空的字符用权0向量代替
        embedding_dic['UNK'] = 0
        embedding_w.append(np.zeros(len(alphabet), dtype='float32'))
        #遍历字母表
        for i, alpha in enumerate(alphabet):
            onehot = np.zeros(len(alphabet), dtype='float32')
            embedding_dic[alpha] = i + 1 #key:index
            onehot[i] = 1
            embedding_w.append(onehot)

        embedding_w = np.array(embedding_w, dtype='float32')
        return embedding_w, embedding_dic

    """文档处理，将其转换为索引集合"""
    def doc_process(self, doc, embedding_dic):
        #如果在embedding_dic中存在该词，则将该词的索引加入到doc的向量表示doc_vec中，不存在则用UNK代替
        #不到l0的文章，进行填充，填UNK的value值，即0
        min_len = min(self.l0, len(doc))
        doc_vec = np.zeros(self.l0, dtype='int64')
        for j in range(min_len):
            if doc[j] in embedding_dic:
                doc_vec[j] = embedding_dic[doc[j]]
            else:
                doc_vec[j] = embedding_dic['UNK']
        return doc_vec

    def batch_iter(data, batch_size, num_epochs, shuffle=True):
        data = np.array(data)
        data_size = len(data)
        num_batches_per_epoch = int((len(data) - 1) / batch_size) + 1
        for epoch in range(num_epochs):
            # Shuffle the data at each epoch
            if shuffle:
                shuffle_indices = np.random.permutation(np.arange(data_size))
                shuffled_data = data[shuffle_indices]
            else:
                shuffled_data = data
            for batch_num in range(num_batches_per_epoch):
                start_index = batch_num * batch_size
                end_index = min((batch_num + 1) * batch_size, data_size)
                yield shuffled_data[start_index:end_index]

"""运行此文件则执行此命令，否则略过"""
if __name__ == '__main__':
    dataset = Dataset(data_source='./data/ag_news_csv/train.csv')
    dataset.dataset_read()
