#在该notebook中，我们基于以下三条数据预测了加州大学洛杉矶分校的研究生录取情况：
#GRE分数（测试）即GRE Scores（Test）
#GPA分数（成绩）即GPA Scores(Grades)
#评级（1-4）即Class rank(1-4)
#数据集来源http://www.ats.ucla.edu/
#importing pandas and numpy
import pandas as pd
import numpy as np
#Reading the csv file into a pandas DataFrame
data=pd.read_csv("student_data.csv")
#printing out the first 10 rows of our data
data[:10]

#绘制数据
#首先让我们对数据进行绘制，看看它是什么样的，为了绘制二维图，让我们先忽略评级（rank）
#import matplotlib
import matplotlib.pyplot as plt
#function to help us plot
def plot_points(data):
    X=np.array(data[["gre","gpa"]])
    y=np.array(data["admit"])
    admitted=X[np.argwhere(y==1)]
    rejected=X[np.argwhere(y==0)]
    plt.scatter([s[0][0] for s in rejected],[s[0][1] for s in rejected],s=25,color="red",edgcolor="k")
    plt.scatter([s[0][0] for s in admitted],[s[0][1] for s in admitted],s=25,color="cyan",edgcolor="k")
    plt.xlabel("Test(GRE)")
    plt.ylabel("Grades(GPA)")
#Plotting the points
plot_points(data)
plt.show()
#粗略的说，它看起来像是，成绩和测试分数高的学生通过了，而得分低的学生却没有，但数据并没有如我们所希望的那样，很好的分离，也许将评级考虑
#进来会有帮助，接下来我们将绘制4个图，每个图代表一个级别
#Separating the ranks
data_rank1=data[data["rank"]==1]
data_rank2=data[data["rank"]==2]
data_rank3=data[data["rank"]==3]
data_rank4=data[data["rank"]==4]

#plotting the graphs
plot_points(data_rank1)
plt.title("Rank 1")
plt.show()
plot_points(data_rank2)
plt.title("Rank 2")
plt.show()
plot_points(data_rank3)
plt.title("Rank 3")
plt.show()
plot_points(data_rank4)
plt.title("Rank 4")
plt.show()
#看上去评级越低，录取率越高。让我们使用评级作为我们的输入之一，为了做到这一点，我们应该对它进行一次one-hot编码
#将评级进行one-hot编码
#我们将在pandas中使用get_dummies函数
#Make dummy variables for rank
one_hot_data=pd.concat([data,pd.get_dummies(data["rank"],prefix="rank")],axis=1)
#Drop the previous rank column
one_hot_data=one_hot_data.drop("rank",axis=1)
#print the first 10 rows of our data
one_hot_data[:10]

#缩放数据
#下一步是缩放数据，注意到成绩的方位是1.0-1.4而测试分数的范围大概是200-800.这个范围要大的多，这意味着我们的数据存在偏差，使得神经网络
#很难处理，让我们将两个特征放在0-1的范围内，将分数除以4.0，将测试分数除以800
#copying our data
processed_data=one_hot_data[:]
processed_data["gre"]=processed_data["gre"]/800
processed_data["gpa"]=processed_data["gpa"]/4.0
processed_data[:10]
#将数据分成训练集和测试集
#为了测试我们的算法，我们将数据分为训练集和测试集，测试集合的大小将占总数据的10%
sample=np.random.choice(prcessed_data.index,size=int(len(processed_data)*0.9),replace=False)
train_data,test_data=processed_data.iloc[sample],processed_data.drop(sample)
pirnt("Number of training samples is",len(train_data))
print("Number of testinng samples is ",len(test_data))
print(train_data[:10])
print(test_data[:10])
#将数据分成特征和目标
#现在，在培训前的最后一步，将数据分为特征和目标
#在Keras中，我们需要对输出进行one-hot编码，我们将使用to_categorical function 来做到这一点
import keras
#Separate data and ont-hot encode the output
#Building the model
model=Sequential()

#Note: we are also turning the data into numpy array,in order to train the model in Keras
features=np.array(train_data.drop("admit",axis=1))
targets=np.array(keras.utils.to_categorical(train_data["admit"],2))
features_test=np.array(test_data.drop("admit",axis=1))
targets_test=np.array(keras.utils.to_categorical(test_data["admit"],2))
print(features[:10])
print(targets[:10])
#定义模型架构
#我们将使用keras来构建神经网络
#imports
import numpy as np
from keras.models import Sequential 
from keras.layers.core import Dense,Dropout,Activation
from keras.optimizers import SGD
from keras.utils imoprt np_utils
#building the model 
model=Sequential()
model.add(Dense(128,acitvation="relu",input_shape=(6,)))
model.add(Dropout(.2))
modle.add(Dense(64,acitvation="relu"))
modle.add(Dropout(.1))
model.add(Dense(2,activation="softmax"))
#compiling the model
model.compile(loss="categorical_crossentropy",optimizer="adam",metrics=["accuracy"])
model.summary()
#训练模型
#Training the model
model.fit(features,targets,epochs=200,batch_size=100,verbose=0)
#模型评分
#evaluating the model on the training and testing set
score=model.evaluate(features,targets)
print("\n Training Accuracy:",score[1])
score=model.evaluate(features_test,targets_test)
print("\n Testing Accuracy:",score[1])
