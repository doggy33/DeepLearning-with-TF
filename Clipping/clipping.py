import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets ,layers , optimizers
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
print("======")
print(tf.__version__)
print(tf.test.is_gpu_available())

def main():
    (x, x_label) , _ = datasets.mnist.load_data()    
    x = tf.convert_to_tensor(x , dtype= tf.float32) / 50.
    x_label = tf.convert_to_tensor(x_label)
    x_label = tf.one_hot(x_label , depth = 10)
    print('x:', x.shape ,'label:' ,x_label.shape)
    train_db = tf.data.Dataset.from_tensor_slices((x,x_label)).batch(128).repeat(30)
    x , x_label = next(iter(train_db))
    print('sample:', x.shape ,x_label.shape)


    w1, b1 = tf.Variable(tf.random.truncated_normal([784,512] ,stddev=0.1)) , tf.Variable(tf.zeros([512]))
    w2, b2 = tf.Variable(tf.random.truncated_normal([512,256] ,stddev=0.1)) , tf.Variable(tf.zeros([256]))
    w3, b3 = tf.Variable(tf.random.truncated_normal([256,10] ,stddev=0.1)) , tf.Variable(tf.zeros([10]))

    optimizer = optimizers.SGD(lr = 0.01)

    for step, (x,x_label) in enumerate(train_db):
        #[b, 28, 28] => [b ,784]
        print()
        x = tf.reshape(x,[-1,784])

        with tf.GradientTape() as tape:
            #Layer1
            h1 = x @ w1 + b1
            print(h1.shape)
            h1 = tf.nn.relu(h1)

            #Layer2
            h2 = h1 @ w2 + b2
            h2 = tf.nn.relu(h2)

            #Output
            out = h2 @ w3 + b3
            #Loss
            # [b, 10] - [b, 10]
            loss = tf.square(x_label - out)
            # [b ,10] => [b] =>scalar
            loss = tf.reduce_mean(tf.reduce_mean(loss, axis =1))
            
        #compute gradient
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        #No Clipping
        print("==before==")   
        for g in grads:
            print(tf.norm(g))
        
        grads , _ = tf.clip_by_global_norm(grads, 15)
        print("==After==")   
        for g in grads:
            print(tf.norm(g))
        
        optimizer.apply_gradients(zip(grads, [w1, b1, w2, b2, w3, b3]))

        if step % 100 == 0:
            print("Loss :",float(loss))


if __name__ == "__main__":
    main()



