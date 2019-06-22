# -*- coding: utf-8 -*-
# Solving a Sytem of ODEs
#----------------------------------
#
# In this script, we use Tensorflow to solve a sytem
#   of ODEs.
#
# The system of ODEs we will solve is the Lotka-Volterra
#   predator-prey system.

import matplotlib.pyplot as plt
import tensorflow as tf
# from sklearn import datasets
# from scipy.spatial import cKDTree
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import scale

# from tensorflow.python.framework import ops
# ops.reset_default_graph()

sess = tf.Session()

x_initial = tf.constant(1.0)
y_initial = tf.constant(1.0)
X_t1 = tf.Variable(x_initial)
Y_t1 = tf.Variable(y_initial)

t_delta = tf.placeholder(tf.float32, shape=())
a = tf.placeholder(tf.float32, shape=())
b = tf.placeholder(tf.float32, shape=())
c = tf.placeholder(tf.float32, shape=())
d = tf.placeholder(tf.float32, shape=())

X_t2 = X_t1 + (a * X_t1 + b * X_t1 * Y_t1) * t_delta
Y_t2 = Y_t1 + (c * Y_t1 + d * X_t1 * Y_t1) * t_delta

step = tf.group(X_t1.assign(X_t2), Y_t1.assign(Y_t2))

init = tf.global_variables_initializer()
sess.run(init)

prey_values = []
predator_values = []
for i in range(1000):
	# Step 
	step.run({a: (2./3.), b: (-4./3.), c: -1.0, d: 1.0, t_delta: 0.01}, session=sess)
	temp_prey, temp_pred = sess.run([X_t1, Y_t1])
	prey_values.append(temp_prey)
	predator_values.append(temp_pred)

plt.plot(prey_values, label='Prey')
plt.plot(predator_values, label='Predator')
plt.legend(loc='upper right')
plt.show()

