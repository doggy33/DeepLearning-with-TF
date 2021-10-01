import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"


def MNIST():
    # MNIST shape:(28, 28, 1) 
    # Total 70k:60k training and 10k testing
    # return Numpy type 
    (x, y) , (x_test, y_test) = keras.datasets.mnist.load_data()
    print("==train==")
    print('x :',x.shape, 'y:',y.shape)
    print('x min :',x.min(),'x max :',x.max())
    print('==test==')
    print('x test: ', x_test.shape, 'y_test: ', y_test.shape)
    print("Top4 y", y[:4])
    y = tf.one_hot(y, depth = 10)
    print(y[:4])


def CIFAR():
    # CIFAR 10/100 shape:(32, 32, 3)
    # Total 60k
    # keras.datasets.cifar100.load_data()
    (x, y) , (x_test, y_test) = keras.datasets.cifar10.load_data()
    print("==train==")
    print('x :',x.shape, 'y:',y.shape)
    print('x min :',x.min(),'x max :',x.max())
    print('==test==')
    print('x test: ', x_test.shape, 'y_test: ', y_test.shape)    
    print("Top4 y", y[:4])

    db = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    # shuffle  
    db = db.shuffle(1000)
    next(iter(db))


def preprocess(x , y):
    x = tf.cast(x, dtype= tf.float32) / 255.
    y = tf.cast(y, dtype= tf.int32)
    y = tf.one_hot(y, depth = 10)
    return x, y



if __name__=='__main__':
    #MNIST()
    CIFAR()