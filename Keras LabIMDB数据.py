#使用Keras分析IMDB电影数据
#import
import numpy as np
import keras
from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
import matplotlib.pyplot as plt
%matplotlib inline
np.random.seed(42)
#加载数据
#该数据集预先加载了keras,所以一个简单的命令就会帮助我们训练和测试数据。这里有一个我们想看多少个单词的参数，我么已经经它设置为1000，
#但是你可以随时尝设置为其他数字
#Loading the data(it is preloaded in Keras)
(x_train,y_train),(x_test,y_test)=imdb.load_data(num_word=1000)
print(x_train.shape)
print(x_test.shape)
#检查数据
#请注意，数据已经经过预处理，其中所有单词都包含数字，评论作为向量与评论中包含的单词一起出现，例如，如果单词the是我们词典中的第一个单词，
#并且评论包含单词the,那么相应的向量中有1
#输出结果是1和0的向量，其中1表示正面评论，0是负面评论
print(x_train[0])
print(y_train[0])
#3输出的one-hot编码
#在这里，我们将输入向量转换为（0,1）向量，例如，如果预处理的向量包含数字14，则在处理的问题中，第14个输入将是1
#ont-hot encoding the output into vector mode.each of length 1000
tokenizer=Tokenizer(num_words=1000)
x_train=tokenizer.sequences_to_matrix(x_train,mode="binary")
x_test=tokenizer.sequences_to_matrix(x_test,mode="binary")
print(x_train[0])
#同时我们将对输出进行one-hot编码
#one-hot encoding the output
num_classes=2
y_train=keras.utils.to_categorical(y_train,num_classes)
y_test=keras.utils.to_categoricak(y_test,num_classes)
print(y_train.shape)
print(y_test.shape)











