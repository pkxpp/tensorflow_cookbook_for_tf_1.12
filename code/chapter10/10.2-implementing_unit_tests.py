# -*- coding: utf-8 -*-
# Implementing Unit Tests
#----------------------------------
#
# Here, we will show how to implement different unit tests
#  on the MNIST example

import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
from tensorflow.python.framework import ops
ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
data_dir = '../dataset'
mnist = read_data_sets(data_dir)

# Convert images into 28x28 (they are downloaded as 1x784)
train_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28,28)) for x in mnist.test.images])

# Convert labels into one-hot encoded vectors
train_labels = mnist.train.labels
test_labels = mnist.test.labels

# Set model parameters
batch_size = 100
learning_rate = 0.005
evaluation_size = 100
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
num_channels = 1 # greyscale = 1 channel
generations = 100
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2 # NxN window for 1st max pool layer
max_pool_size2 = 2 # NxN window for 2nd max pool layer
fully_connected_size1 = 100
dropout_prob = 0.75

# Declare model placeholders
x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape=eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape=(evaluation_size))

# Dropout placeholder
dropout = tf.placeholder(tf.float32, shape=())

# Declare model parameters
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features],
                                               stddev=0.1, dtype=tf.float32))
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype=tf.float32))

conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features],
                                               stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype=tf.float32))

# fully connected variables
resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1],
                          stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))
full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size],
                                               stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))


# Initialize Model Operations
def my_conv_net(input_data):
	# First Conv-ReLU-MaxPool Layer
	conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
	max_pool1 = tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')

	# Second Conv-ReLU-MaxPool Layer
	conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1, 1, 1, 1], padding='SAME')
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
	max_pool2 = tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')

	# Transform Output into a 1xN layer for next fully connected layer
	final_conv_shape = max_pool2.get_shape().as_list()
	final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
	flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])

	# First Fully Connected Layer
	fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))

	# Second Fully Connected Layer
	final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)

	# Add dropout
	final_model_output = tf.nn.dropout(final_model_output, dropout)

	return(final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output, labels=y_target
))
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
	batch_predictions = np.argmax(logits, axis=1)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return (100.0 * num_correct/batch_predictions.shape[0])

my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
sess.run(init)

class drop_out_test(tf.test.TestCase):
	def dropout_greaterthan(self):
		with self.test_session():
			self.assertGreater(dropout.eval(), 0.25)

class accuracy_test(tf.test.TestCase):
	def accuracy_exact_test(self):
		with self.test_session():
			test_preds = [[0.9, 0.1], [0.01, 0.99]]
			test_targets = [0, 1]
			test_acc = get_accuracy(test_preds, test_targets)
			self.assertEqual(test_acc.eval(), 100.)

class shape_test(tf.test.TestCase):
	def output_shape_test(self):
		with self.test_session():
			numpy_array = np.ones([batch_size, target_size])
			self.assertShapeEqual(numpy_array, model_output)

tf.test.main()

train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
	rand_index = np.random.choice(len(train_xdata), size=batch_size)
	rand_x = train_xdata[rand_index]
	rand_x = np.expand_dims(rand_x, 3)
	rand_y = train_labels[rand_index]
	train_dict = {x_input: rand_x, y_target: rand_y, dropout: dropout_prob}
	
	sess.run(train_step, feed_dict = train_dict)
	temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict = train_dict)
	temp_train_acc = get_accuracy(temp_train_preds, rand_y)

	if (i+1)% eval_every == 0:
		eval_index = np.random.choice(len(test_xdata), size=evaluation_size)
		eval_x = test_xdata[eval_index]
		eval_x = np.expand_dims(eval_x, 3)
		eval_y = test_labels[eval_index]
		test_dict = {eval_input: eval_x, eval_target: eval_y, dropout: 1.0}
		test_preds = sess.run(test_prediction, feed_dict=test_dict)
		temp_test_acc = get_accuracy(test_preds, eval_y)
		
		train_loss.append(temp_train_loss)
		train_acc.append(temp_train_acc)
		test_acc.append(temp_test_acc)
		acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
		acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
		print(i, "111111")
		print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

# https://www.tensorflow.org/versions/r1.12/api_guides/python/test?hl=en
