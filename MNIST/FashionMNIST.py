from matplotlib.pyplot import minorticks_off
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import datasets, layers, optimizers, Sequential, metrics

def preprocess(x,y):
    x = tf.cast(x, dtype =tf.float32) /255.
    y = tf.cast(y, dtype = tf.int32)
    return x,y 


batchsz = 128

(x, y), (x_test, y_test) = datasets.fashion_mnist.load_data()
db = tf.data.Dataset.from_tensor_slices((x,y))
db = db.map(preprocess).shuffle(10000).batch(batchsz) 

db_test = tf.data.Dataset.from_tensor_slices((x_test,y_test))
db_test = db_test.map(preprocess).batch(batchsz)

sample = next(iter(db))
print(sample[0].shape)

model = Sequential([
    layers.Dense(256, activation = tf.nn.relu), #[b, 784] => [b, 256]
    layers.Dense(128, activation = tf.nn.relu), #[b, 256] => [b, 128]
    layers.Dense(64, activation = tf.nn.relu), #[b, 128] => [b, 64]
    layers.Dense(32, activation = tf.nn.relu), #[b, 64] => [b, 32]
    layers.Dense(10) #[b, 32] => [b, 10] ,330 =32*10 + 10(b)
])

model.build(input_shape = [None, 28*28]) 
model.summary()
# w = w -lr* grad
optimizer = optimizers.Adam(lr = 1e-3)
def main():
    for epoch in range(30):
        for step, (x,y) in enumerate(db):
            #x: [b, 28, 28]
            #y: [b]
            x = tf.reshape(x, [-1, 28*28])
            with tf.GradientTape() as tape:
                #[b, 784] => [b, 10]
                logits = model(x)
                y_onehot = tf.one_hot(y, depth = 10)
                #[b]
                loss_mes = tf.reduce_mean(tf.losses.MSE(y_onehot, logits))
                loss2_ce = tf.reduce_mean(tf.losses.categorical_crossentropy(y_onehot, logits, from_logits = True))
            grads = tape.gradient(loss2_ce, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if step % 100 ==0:
                print(epoch, step, 'loss:',float(loss2_ce),float(loss_mes))

        #test
        total_correct = 0
        total_num = 0
        for x,y in db_test:
            #x: [b, 28, 28]
            #y: [b]
            x_test = tf.reshape(x, [-1, 28*28])
            #[b, 10]
            logits = model(x_test)
            #logits =>prob[b, 10]
            prob = tf.nn.softmax(logits, axis = 1)
            # [b, 10] => [b]
            pred = tf.argmax(prob ,axis = 1)
            pred = tf.cast(pred, dtype = tf.int32)
            #pred: [b]
            #y: [b]
            #correct [b], True: equal, Fasle: not equal
            correct = tf.equal(pred, y)
            correct = tf.reduce_sum(tf.cast(correct, dtype = tf.int32))
            total_correct += int(correct)
            total_num += x.shape[0]
            
        acc= total_correct / total_num
        print(epoch, 'test acc',acc)




if __name__ =='__main__':
    main()