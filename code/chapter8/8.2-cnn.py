# Introductory CNN Model: MNIST Digits
#---------------------------------------
#
# In this example, we will download the MNIST handwritten
# digits and create a simple CNN network to predict the
# digit category (0-9)

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.datasets.mnist import read_data_sets
import pylab
from PIL import Image
# from tensorflow.examples.tutorials.mnist import input_data
# from tensorflow.python.framework import ops
# ops.reset_default_graph()

# Start a graph session
sess = tf.Session()

# Load data
data_dir = 'temp'
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
mnist = read_data_sets(data_dir)
print(len(mnist.train.images))
print(len(mnist.train.images[0]))

train_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.train.images])
test_xdata = np.array([np.reshape(x, (28, 28)) for x in mnist.test.images])
train_labels = mnist.train.labels
test_labels = mnist.test.labels
# print(len(train_labels))
# print(train_labels[0])


batch_size = 100
learning_rate = 0.005
evaluation_size = 500
image_width = train_xdata[0].shape[0]
image_height = train_xdata[0].shape[1]
target_size = max(train_labels) + 1
# print(target_size)
num_channels = 1
generations = 500
eval_every = 5
conv1_features = 25
conv2_features = 50
max_pool_size1 = 2
max_pool_size2 = 2
fully_connected_size1 = 100

x_input_shape = (batch_size, image_width, image_height, num_channels)
x_input = tf.placeholder(tf.float32, shape=x_input_shape)
y_target = tf.placeholder(tf.int32, shape=(batch_size))
eval_input_shape = (evaluation_size, image_width, image_height, num_channels)
eval_input = tf.placeholder(tf.float32, shape = eval_input_shape)
eval_target = tf.placeholder(tf.int32, shape = (evaluation_size))

# conv1_features和conv2_features就是隐藏层的节点个数，可以理解为特征数量，看网上的文章之后可以说是特征图，即25个特征图
conv1_weight = tf.Variable(tf.truncated_normal([4, 4, num_channels, conv1_features], stddev=0.1, dtype=tf.float32))
print(conv1_weight)
conv1_bias = tf.Variable(tf.zeros([conv1_features], dtype = tf.float32))
conv2_weight = tf.Variable(tf.truncated_normal([4, 4, conv1_features, conv2_features], stddev=0.1, dtype=tf.float32))
conv2_bias = tf.Variable(tf.zeros([conv2_features], dtype = tf.float32))

resulting_width = image_width // (max_pool_size1 * max_pool_size2)
resulting_height = image_height // (max_pool_size1 * max_pool_size2)
print(resulting_width, resulting_height)

full1_input_size = resulting_width * resulting_height * conv2_features
full1_weight = tf.Variable(tf.truncated_normal([full1_input_size, fully_connected_size1], stddev=0.1, dtype=tf.float32))
full1_bias = tf.Variable(tf.truncated_normal([fully_connected_size1], stddev=0.1, dtype=tf.float32))


full2_weight = tf.Variable(tf.truncated_normal([fully_connected_size1, target_size], stddev=0.1, dtype=tf.float32))
full2_bias = tf.Variable(tf.truncated_normal([target_size], stddev=0.1, dtype=tf.float32))

def my_conv_net(input_data):
	# First Conv-ReLU-MaxPool Layer
	conv1 = tf.nn.conv2d(input_data, conv1_weight, strides=[1,1,1,1], padding='SAME')
	# 开始很纳闷为啥卷积4x4，28x28的输入还是28x28，原因是padding参数的缘故，'SAME'允许卷积核停留在图像边缘
	# print(conv1)
	relu1 = tf.nn.relu(tf.nn.bias_add(conv1, conv1_bias))
	# print(relu1)
	max_pool1 =tf.nn.max_pool(relu1, ksize=[1, max_pool_size1, max_pool_size1, 1], strides=[1, max_pool_size1, max_pool_size1, 1], padding='SAME')
	# print(max_pool1)
	
	# Second Conv-ReLU-MaxPool Layer
	conv2 = tf.nn.conv2d(max_pool1, conv2_weight, strides=[1,1,1,1], padding='SAME')
	relu2 = tf.nn.relu(tf.nn.bias_add(conv2, conv2_bias))
	max_pool2 =tf.nn.max_pool(relu2, ksize=[1, max_pool_size2, max_pool_size2, 1], strides=[1, max_pool_size2, max_pool_size2, 1], padding='SAME')
	
	# Transform Output into a 1xN layer for next fully connected layer
	final_conv_shape = max_pool2.get_shape().as_list()
	print(final_conv_shape)
	# 这里所谓的摊平就是因为最后一层卷积层的特征图是多个的，那么7x7的图像有50个，这样和输出层没办法做矩阵乘法，因为输出层就是一个(输入大小xtarget_size)的连接层，所以把50个特征图变成一个一维数组
	final_shape = final_conv_shape[1] * final_conv_shape[2] * final_conv_shape[3]
	flat_output = tf.reshape(max_pool2, [final_conv_shape[0], final_shape])
	print(flat_output)
	
	# First Fully Connected Layer
	fully_connected1 = tf.nn.relu(tf.add(tf.matmul(flat_output, full1_weight), full1_bias))
	
	# Second Fully Connected layer
	final_model_output = tf.add(tf.matmul(fully_connected1, full2_weight), full2_bias)
	return(final_model_output)

model_output = my_conv_net(x_input)
test_model_output = my_conv_net(eval_input)

loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=model_output,labels=y_target))
prediction = tf.nn.softmax(model_output)
test_prediction = tf.nn.softmax(test_model_output)

def get_accuracy(logits, targets):
	batch_predictions = np.argmax(logits, axis=1) #取得索引(即0-9，10个数值的索引)
	num_correct = np.sum(np.equal(batch_predictions, targets))
	return (100. * num_correct / batch_predictions.shape[0])

my_optimizer = tf.train.MomentumOptimizer(learning_rate, 0.9)
train_step = my_optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess.run(init)

train_loss = []
train_acc = []
test_acc = []
for i in range(generations):
	rand_index = np.random.choice(len(train_xdata), size = batch_size)
	rand_x = train_xdata[rand_index]
	rand_x = np.expand_dims(rand_x, 3)
	rand_y = train_labels[rand_index]
	train_dict = {x_input: rand_x, y_target: rand_y}
	sess.run(train_step, feed_dict = train_dict)
	temp_train_loss, temp_train_preds = sess.run([loss, prediction], feed_dict = train_dict)
	temp_train_acc = get_accuracy(temp_train_preds, rand_y)
	if (i + 1) % eval_every == 0:
		eval_index = np.random.choice(len(test_xdata), size = evaluation_size)
		eval_x = test_xdata[eval_index]
		eval_x = np.expand_dims(eval_x, 3)
		eval_y = test_labels[eval_index]
		test_dict = {eval_input: eval_x, eval_target: eval_y}
		test_preds = sess.run(test_prediction, feed_dict = test_dict)
		temp_test_acc = get_accuracy(test_preds, eval_y)
		
		train_loss.append(temp_train_loss)
		train_acc.append(temp_train_acc)
		test_acc.append(temp_test_acc)
		acc_and_loss = [(i+1), temp_train_loss, temp_train_acc, temp_test_acc]
		acc_and_loss = [np.round(x, 2) for x in acc_and_loss]
		print('Generation # {}. Train Loss: {:.2f}. Train Acc (Test Acc): {:.2f} ({:.2f})'.format(*acc_and_loss))

eval_indices = range(0, generations, eval_every)
plt.plot(eval_indices, train_loss, 'k-')
plt.title('Softmax Loss per Generation')
plt.xlabel('Generation')
plt.ylabel('Softmax Loss')
plt.show()

plt.plot(eval_indices, train_acc, 'k-', label='Train Set Accuracy')
plt.plot(eval_indices, test_acc, 'r--', label='Test Set Accuracy')
plt.title('Train adn Test Accuracy')
plt.xlabel('Generation')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

actuals = rand_y[0:6]
predictions = np.argmax(temp_train_preds, axis=1)[0:6]
images = np.squeeze(rand_x[0:6])
print(images)
Nrows = 2
Ncols = 3
for i in range(6):
	plt.subplot(Nrows, Ncols, i+1)
	plt.imshow(np.reshape(images[i], [28, 28]), cmap='Greys_r')
	plt.title('Actual: ' + str(actuals[i]) + ' Pred: ' + str(predictions[i]), fontsize=10)
	frame = plt.gca()
	frame.axes.get_xaxis().set_visible(False)
	frame.axes.get_yaxis().set_visible(False)

# http://scarlet.stanford.edu/teach/index.php/An_Introduction_to_Convolutional_Neural_Networks
# http://neuralnetworksanddeeplearning.com/chap6.html
# http://cs.nju.edu.cn/wujx/paper/CNN/pdf
# 上面这个网址打不开，搜索找到了下面这个
# https://cs.nju.edu.cn/wujx/teaching/15_CNN.pdf

# https://arxiv.org/pdf/1904.13353.pdf