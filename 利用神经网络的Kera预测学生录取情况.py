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
data_rank2=data[data[]]





