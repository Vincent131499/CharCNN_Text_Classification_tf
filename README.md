CharCNN_Text_Classification_tf:构建用于文本分类的基于字符的CNN网络
===============================================================
基于这篇[paper](https://arxiv.org/abs/1509.01626)的实现

Requirements:
==============
*Python3.6<br>
*Tensorflow-GPU1.8.0<br>
*numpy<br>

项目架构：
========
config.py:各种参数及超参数配置；<br>
data_helper.py:数据处理函数，包括读取数据集、生成batch；<br>
CharCNN_model.py:构建的模型文件；<br>
train.py:模型训练文件<br>
test.py:纯粹的测试代码，与项目无关，用于验证自己所想<br>

数据集：
=======
可以去[这里](https://github.com/Irvinglove/char-CNN-text-classification-tensorflow/tree/master/data/ag_news_csv)下载

运行：
=====
```python
python train.py<br>

Have a good time!
-----------------
