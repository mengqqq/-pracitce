import numpy as np
def softmax(L)：
   expL=np.exp(L)
   sumExpL=sum(expL)
   result=[]
   for i in expL:
       result.append(i*1.0/sumExpL)
   return result
   
   #Note:The function np.divide can also be used here,as followes:
   #def softmax(L):
