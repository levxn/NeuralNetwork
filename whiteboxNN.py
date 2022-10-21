from turtle import update
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

data=pd.read_csv(r'train.csv')
# print(data.head())
data=np.array(data)
m, n = data.shape
np.random.shuffle(data)

data_dev=data[0:1000].T
Y_dev=data_dev[0]
X_dev=data_dev[1:n]

data_train=data[1000:m].T
Y_train=data_train[0]
X_train=data_train[1:n]

# X_train[:, 0].shape

def init_params():
    w1=np.random.randn(10,784)
    b1=np.random.randn(10,1)
    w2=np.random.randn(10,10)
    b2=np.random.randn(10,1)
    return w1,b1,w2,b2

def ReLu(z):
    return np.maximum(0,z)

def softmax(z):
    return np.exp(z)/np.sum(np.exp(z))

def forward_prop(w1,b1,w2,b2,x):
    z1=w1.dot(x)+b1
    a1=ReLu(z1)
    z2=w2.dot(a1)+b2
    a2=softmax(a1)

def one_hot(y):
    one_hot_Y=np.zeros((y.size,y.max()+1))
    one_hot_Y[np.arange(y.size),y]=1
    one_hot_Y=one_hot_Y.T
    return one_hot_Y

def deriv_ReLu(z):
    return z>0
    
def back_prop(z1,a1,z2,a2,w2,x,y):
    one_hot_Y=one_hot(y)
    dz2=a2-one_hot_Y
    dw2=1/ m* dz2.dot(a1.T)
    db2=1/ m* np.sum(dz2)
    dz1=w2.T.dot(dz2)*deriv_ReLu(z1)
    dw1=1/ m* dz1.dot(x.T)
    db1=1/ m* np.sum(dz1,2)
    return dw1,db1,dw2,db2

def update_params(w1,b1,w2,b2,dw1,db1,dw2,db2,alpha):
    w1=w1-alpha*dw1
    b1=b1-alpha*db1
    w2=w2-alpha*dw2
    b2=b2-alpha*db2
    return w1,b1,w2,b2

#---------------------------------------------------


def get_predictions(a2):
    return np.argmax(a2,0)

def get_accuracy(predictions, y):
    print(predictions,y)
    return np.sum(predictions==y)/y.size

def gradient_descent(x,y,iterations,alpha):
    w1,b1,w2,b2 = init_params()
    for i in range(int(iterations)):
        z1,a1,z2,a2 = forward_prop(w1,b1,w2,b2,x)
        dw1,db1,dw2,db2 =back_prop(z1,a1,z2,a2,w2,x,y)
        w1,b1,w2,b2 =update_params(w1,b1,w2,b2,dw1,db1,dw2,dw2,db2,alpha)
        if i%50==0:
            print("Iterataion: ",i)
            print("Accuracy ",get_accuracy(get_predictions(a2,y)))
    return w1,b1,w2,b2
   
   
w1,b1,w2,b2=gradient_descent(X_train,Y_train,0.10,500) #100,0.1

#------------------------------------------------------------------------

def make_predictions(x,w1,b1,w2,b2):
    _, _, _, a2=forward_prop(w1,b1,w2,b2,x)
    predictions=get_predictions(a2)
    return predictions

def test_prediction(index,w1,b1,w2,b2):
    current_image=X_train[:,index,None]
    prediction=make_predictions(X_train[:,index,None],w1,b1,w2,b2)
    label=Y_train[index]
    print("Predictions: ",prediction)
    print("Label: ",label)

    current_image=current_image.reshape((28,28))*255
    plt.gray()
    plt.imshow(current_image,interpolation='nearest')
    plt.show()

test_prediction(5,w1,b1,w2,b2)
