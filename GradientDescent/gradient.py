import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape

w = tf.Variable(1.0)
b = tf.Variable(2.0)
x = tf.Variable(3.0)

with GradientTape() as tape1:
    with GradientTape() as tape2:
        y = x * w + b
    dy_dw, dy_db = tape2.gradient(y, [w, b])
d2y_dw2 = tape1.gradient(dy_dw, w)

print(dy_dw)  
print(dy_db)
print(d2y_dw2)   