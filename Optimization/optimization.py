import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.python.eager.backprop import GradientTape

def himmelblau(x):
    return (x[0]**2 + x[1] -11) ** 2 + (x[0] + x[1] ** 2 -7) **2

x = np.arange(-6, 6, 0.1)
y = np.arange(-6, 6, 0.1)
print('x, y range :', x.shape, y.shape)
X, Y = np.meshgrid(x,y)
print('x, y maps :', x.shape, y.shape)
Z = himmelblau([X,Y])


fig = plt.figure('himmelblau')
ax = fig.gca(projection ='3d')
ax.plot_surface(X,Y,Z)
ax.view_init(60,-30)
ax.set_xlabel('x')
ax.set_ylabel('y')
plt.show()

x = tf.constant([-4. , 0.])

for step in range(200):
    with GradientTape() as tape:
        tape.watch([x])
        y = himmelblau(x)
    grads = tape.gradient(y, [x])[0]
    x -= 0.01 * grads

    if step % 20 == 0:
        print("Step {}: x ={}, f(x) = {}".format(step, x.numpy(), y.numpy()))