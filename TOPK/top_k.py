import tensorflow as tf
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
tf.random.set_seed(1234)


def accuracy(output, target, top_k = (1.)):
    maxk = max(top_k)
    batch_size = target.shape[0]

    pred = tf.math.top_k(output, maxk).indices
    pred = tf.transpose(pred , perm = [1,0])
    target_ = tf.broadcast_to(target , pred.shape)
    # [10 , b] 
    correct = tf.equal(pred , target_ )

    res = []
    for k in top_k:
        correct_k = tf.cast(tf.reshape(correct[:k], [-1]) ,dtype=tf.float32)
        correct_k = tf.reduce_sum(correct_k)
        acc = float(correct_k *(100.0 / batch_size))
        res.append(acc)

    return res



# normal distribution 
output = tf.random.normal([10,6])
# set total out equal 1
output = tf.math.softmax(output, axis = 1)
# Sample 10 times between 0 to 5
target = tf.random.uniform([10], maxval=6, dtype=tf.int32)

print('prob:' , output.numpy())
pred = tf.argmax(output.numpy() ,axis = 1)
print('pred:' , pred.numpy())
print('label:' , target.numpy())

acc = accuracy(output , target , top_k = (1,2,3,4,5,6))
print('Top 1-6 acc', acc)