#序列模型
from keras.models import Sequential
#Create the Sequential model
model=Sequential()
#Keras.models.Sequential类是神经网络模型的封装容器。它会提供常见的函数，例如fit() evaluate()和compile().
#层
#Keras层就像神经网络层。有全连接层、最大池化层和激活层。可以使用模型的add()函数添加层。例如，简单的模型可以如下所示：
from keras.models import Sequential
from keras.layers.core import Dense,Acitvation,Flatten
#创建序列模型
model=Sequential()
#第一层，添加有128个节点的全连接层以及32个节点的输入层
model.add(Dense(128,input_dim=32))

#第二层，添加softmax激活层
model.add(Activation("softmax"))

#第三层，添加全连接层
model.add(Dense(10))

#第四层，添加sigmoid激活层
model.add(Activation("sigmoid"))

#Keras将根据第一层自动推断后续所有层的形状。这意味着，只需为第一层设置输入维度
#上面第一层model.add(Dense(input_dim=32))将维度设为32（表示数据来自32维空间）。第二层级获取第一层级的输出，并将输出维度设为128个节点。
#这种将输出传递给下一层及的链继续下去，直到最后一个层级（即模型的输出）。可以用看出输出维度是10
#构建好模型后，就可以用一下命令对其进行编译，将损失函数指定为一直处理的categorical_crossentropy。还可以指定优化程序，稍后将了解这一概念，
#暂时将使用adam.最后，可以指定评估模型用到的指标，将使用准确率
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
#可以使用一下命令来查看模型架构：
model.summary()
#使用以下命令对其进行拟合，指定epoch次数和我们希望在屏幕上显示的信息详细程度
#然后使用fit命令训练模型并通过epoch参数来指定训练轮数（周期），每epoch完成对整数据集的一次遍历。
#verbose参数可以指定显示训练过程信息类型，这里定义为0表示不显示信息
model.fit(X,y,nb_epoch=1000,verbose=0)
#注意：在Keras1中，nb_epoch会设置epoch次数，但是在Keras2中，变成了epochs
#最后，可以使用一下命令来评估模型：
model.evaluate()
#练习
#从最简单的示例开始。在此测验中，将构建一个简单的多层前向反馈神经网络以解决XOR问题
#1将第一层设为Dense()层，并将节点设为8，且input_dim设为2
#2将第二层之后使用softmax激活函数
#3将输出层节点设为2，因为输出只有2个类别
#4在输出层之后使用softmax激活函数
#对模型运行10个epoch
#准确度应该为50%。可以接受，当然肯定是不太雷翔，在4个点中，只有2个点分类正确？我们试着修改某些参数，以改变这一状况。
#例如，你可以增加epoch次数以及改变激活函数的类型。如果准确率达到75%，你将通过这道测验，能尝试达到100%
#首先，查看关于模型和层级的Keras文档，Keras多层感知器网络示例和你要构建的类似。请将该示例当做指南，但是注意有很多不同之处。

import numpy as np
from keras.utils import np_utils
import tensorflow as tf
tf.python.control_flow_ops=tf

#Set random seed
np.random.seed(42)

#Our data
X=np.array([[0,0],[0,1],[1,0],[1,1]]).astype("float32")
y=np.array([[0],[1],[1],[0]]).astype("float32")

#Initial Setup for Keras
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Flatten

#One-hot encoding the output
y=np_utils.to_categorical(y)

#Building the model
xor=Sequential()
xor.add(Dense(32,input_dim=2))
xor.add(Activation("tanh"))
xor.add(Dense(2))
xor.add(Activation("sigmoid"))

xor.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])

#Uncomment this line to print the model architecture
#xor.summary()
#Fitting the model
history=xor.fit(X,y,nb_epoch=1000,verbose=0)

#Scoring the model
score=xor.evalutae(X,y)
print("\nAccurayc:",score[-1])

#Checking the predicitons
print("\nPredictions:")
print(xor.predict_proba(X))








































