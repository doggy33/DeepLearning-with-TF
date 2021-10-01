import tensorflow as tf
from tensorflow import keras
from tensorflow.keras  import datasets
import os

#Let cpp don't print nonuse message
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
#basic matrix calculate

# Step 1 Load dataset
# x = [60k, 28 , 28]
# y = [60k]
(x,y), _ = datasets.mnist.load_data()

# Step 2 convert to Tensor
# x :[0~255.] -> [0~1.]
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.

y = tf.convert_to_tensor(y, dtype=tf.int32)

#check data type
#print(x.shape, y.shape,x.dtype,y.dtype)

#check data range

#We can see x range is between 255 and 0 
#generally  our input data set -1 to 1 or 1 to 0 is much better for tensorflow
#print(tf.reduce_max(x),tf.reduce_min(x))
#print(tf.reduce_max(y),tf.reduce_min(y))

#creat a batch (a dataset)

train_db = tf.data.Dataset.from_tensor_slices((x,y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)

# batch x:(128, 28, 28) y:(128,)
print('batch',sample[0].shape,sample[1].shape)

# [b,784] => [b,512] => [b,128] => [b,10]
# [dim_in , dim_out],[dim_out]
w1 = tf.Variable(tf.random.truncated_normal([784,512], stddev = 0.1))
b1 = tf.Variable(tf.zeros([512]))
w2 = tf.Variable(tf.random.truncated_normal([512,128], stddev = 0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128,10], stddev = 0.1))
b3 = tf.Variable(tf.zeros([10]))
#learning rate
lr = 1e-3

#Backpropagation
for epoch in range(10): # iteration db for 10
    for step , (x,y) in enumerate(train_db): # for every batch
        # x.shape:[128 ,28, 28]
        # y.shape:[128]
        # reshape [b, 28 ,28 ] => [b ,28 *28]
        x = tf.reshape(x,[-1, 28*28])

        with tf.GradientTape() as tape: #default trace tf.variable
            # x:[b, 28*28]
            # h1 = x@w1 + b1 
            # [b, 784]@[784 ,512] +[512] => [b, 512]+[512]        
            # h1 = x@w1 +tf.broadcast_to(b1, [x.shape[0],512])
            h1 = x@w1 + b1
            # activate functoin
            h1 = tf.nn.relu(h1)

            #[b ,512] => [b ,128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)

            #[b ,128] => [b, 10]
            out = h2@w3 + b3

            #compute loss
            # out : [b, 10] 
            # y :[b]
            y_onehot = tf.one_hot(y ,depth = 10)

            #mse = mean(sum(y-out)^2)
            loss = tf.square(y_onehot - out)
            #mean : scalar
            loss = tf.reduce_mean(loss)

        #compute gradients
        grads = tape.gradient(loss ,[w1, b1, w2, b2, w3, b3])
        # w1 = w1 -lr * w1_grad
        #print(grads)
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])    
        # w1 = w1 - lr  * grads[0]
        # b1 = b1 - lr  * grads[1]
        # w2 = w2 - lr  * grads[2]
        # b2 = b2 - lr  * grads[3]
        # w3 = w3 - lr  * grads[4]
        # b3 = b3 - lr  * grads[5]
    
        if step % 100 == 0:
            print(epoch ,step , 'loss',float(loss))