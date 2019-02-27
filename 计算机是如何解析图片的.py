import matplotlib.pyplot as plt
%matplotlib inline
import matplotlib.cm as cm
import numpy as np

#plot first six training images
fig=plt.figure(figsize=(20,20))
for i in range(6):
    ax=fig.add_subplot(1,6,i+1,xticks=[],yticks=[])
    ax.imshow(X_train[i],cmap="gray")
    ax.set_title(str(y_train[i]))
from keras.utils import np_utils
#print first ten(integer- alued)training labels
print("Integer- alued labels:")
print(y——train[:10])
#one-hot encode the labels
y_train=np_utils.to_categorical(y_train,10)
y_test=np_utils.to_categorical(y_test,10)
#print first ten(one-hot) training labels
print("One-hot labels:")
print(y_train[:10])
