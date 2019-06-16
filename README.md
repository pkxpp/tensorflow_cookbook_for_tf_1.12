# TensorFlow Cookbook for version 1.12
针对TellnsorFlow-1.12的《Tensorflow机器学习实践指南》代码

自己在学习《Tensorflow机器学习实践指南》这本书的时候，用的是Tensorflow-1.12的版本，虽然很多代码都是可以运行并且ok的，但是还是存在函数改名和代码运行不对的情况，所以把针对Tensorflow-1.12的代码上传到这里，希望对跟我一样的菜鸟有些许帮助。如果觉得有疑惑的话，参考作者的源码[1]，以及Tensorflow官方网站。也可以留言互相学习进步~ ^_^

一些零星的笔记也记录在下面
## Chapter3 基于TensorFlow的线性回归

### 3.2
把这两个例子总算看明白了，虽然写的是Ax=b，求得就是x。这个x才是真正得系数矩阵。
了解了cholesky矩阵分解，对于代码主要是知道函数tf.matrix_solve得意思

### 3.5
```
for i in range(iterations):
rand_index = np.random.choice(len(x_vals), size=batch_size)
rand_x = np.transpose([x_vals[rand_index]])
rand_y = np.transpose([y_vals[rand_index]]) 
```
知道为毛这里训练要转置了，因为这里的数据的格式是这样子的，所以需要转置。

### 3.9
用的是出生体重的数据，但是源链接打不开，后面是再github源码里面找到了数据，但是这里的数据是处理过的，结果折腾了好久。下面这个代码就是一开始没对上，结果训练的结果完全不对
```
# 这里的数据错误，导致最后结果和书上不一样，书上写的是x[0], x[2:9]，但是索引是从0开始的，坑爹
# 书上用的是源数据，而我这里用的是处理过的数据，书上也说了去掉了实际出生体特征和ID两列，估计是源数据里面的第一列和最后一列
y_vals = np.array([x[0] for x in birth_data])
x_vals = np.array([x[1:8] for x in birth_data])
```
## Chapter4 基于TensorFlow的支持向量机

支持向量机就是找到要给超平面，使得margin最大。那么怎么求呢？最后用到了拉格朗日乘子，并且最后对于每一个超平面只有超平面上的点（即支持向量）对于求极值是有效的，其他的在超平面外面的点都不参与计算。所以，对于任意一个超平面，如果对于训练数据，他的支持向量使得Margin最大，那么就是最优的超平面

并不能每次都能恰好分开数据，参数估计还是得调。另外，增加训练次数到1000次，几乎每次都能得到不错的数据，训练数据集和测试数据集的准确率都几乎是100%。这主要还是说明训练次数不够
### 4.4 TensorFlow上核函数的使用
* tf.reduce_sum
指定维度的和，执行完就降了一维
```
x = tf.constant([[1, 1, 1], [1, 1, 1]])
tf.reduce_sum(x) # 6
# 第0维相加[1, 1, 1] + [1, 1, 1]
tf.reduce_sum(x, 0) # [2, 2, 2]
# 第1维相加，[1+1+1], [1+1+1]
tf.reduce_sum(x, 1) # [3, 3]
tf.reduce_sum(x, 1, keep_dims=True) # [[3], [3]]
tf.reduce_sum(x, [0, 1]) # 6
```

* 高斯核函数
一开始很难理解这段代码和公式是对应上的$k(x_i, p_i) = e^{-y{||x_i - p_i||}^2}$
最后发现代码里面是把平方拆开之后的写法：${(a - b)}^2 = a^2 - 2ab + b^2$，这里的a=x_data[0], b = x_data[1]，因为dist是平方之后的值了
```
# Gaussian (RBF) kernel
gamma = tf.constant(-50.0)
dist = tf.reduce_sum(tf.square(x_data), 1)
dist = tf.reshape(dist, [-1,1])
sq_dists = tf.add(tf.sub(dist, tf.mul(2., tf.matmul(x_data, tf.transpose(x_data)))), tf.transpose(dist))
my_kernel = tf.exp(tf.mul(gamma, tf.abs(sq_dists)))
```

## Chapter5 最近邻域法

### 5.1 最近邻域法介绍
* numpy.ptp函数
计算范围，相当于(最大值-最小值)

### 5.3 如何度量文本距离
* tf.subtract的不同维度相减的意义：交叉相减
```
# 1.这里的x_data_train 减去 扩维的x_data_test，是为了让x_data_train的每一分数据都能减去x_data_test的每一分数据。这样才能计算，任意一个x_data_test的数据到x_data_train任意一个数据的距离
# 2.降维：把每一分数据相减后的和作为两份数据之前的距离，求的距离就是cols_used列的和
distance = tf.reduce_sum(tf.abs(tf.subtract(x_data_train, tf.expand_dims(x_data_test,1))), reduction_indices=2)
```
### 5.4 用TensorFlow实现混合距离计算
* numpy.std
求某个维度的标准差，对于x_vals.shape = (506, 10)，最后得出的是一个shape(10, )的一维数组
* tf.diag
生成对角矩阵，其他值用0填充
* tf.tile
*This operation creates a new tensor by replicating input multiples times. *
* tf.transpose的第二个参数perm
指定转换指定维度

* tf.argmax
tf.argmax是tensorflow用numpy的np.argmax实现的，它能给出某个tensor对象在某一维上的其数据最大值所在的索引值，常用于metric（如acc）的计算

## Chapter6 神经网络算法

### 6.4 用TensorFlow实现单层神经网络
* 门函数
前面的门函数不知道啥意思，其实他只是神经网络每个节点的表达方式

### 6.5 用TensorFlow实现神经网络常见层
* 层函数思维数据

说的是输入数据的思维[batch_size, width, height, channels]，batch_size是批量处理的图像个数，就像前一天学习的动图里面的批量数据为3，3幅图像数据。width和height是图像的宽度和高度，这个好理解。最后一个channels是颜色通道，黑白是1（比如minist的数据），RGB是3。

### 6.8 用TensorFlow基于神经网络实现井字棋

> To train our model, we will have a list of board positions followed by the best optimal response for a number of different boards. We can reduce the amount of boards to train on by considering only board positions that are different with respect to symmetries. The non-identity transformations of a Tic Tac Toe board are a rotation (either direction) by 90 degrees, 180 degrees, 270 degrees, a horizontal reflection, and a vertical reflection.Given this idea, we will use a shortlist of boards with the optimal move, apply two random transformations, and feed that into our neural network to learn.
意思就是说因为对称性，可以通过旋转和镜像生成许多不一样的数据，而无须都从文件里面读出来

![image](https://github.com/pkxpp/tensorflow_cookbook_for_tf_1.12/blob/master/image/1.png?raw=true)

## Chapter8 卷积神经网络

### 8.4 再训练已有的CNN模型
python data/build_image_data.py --train_directory="temp/train_dir/" --validation_directory="temp/validation_dir" --output_directory="temp/" --labels_file="temp/cifar10_labels.txt

python data/build_image_data.py --train_directory="../../../../../'TensorFlow Machine Learning Cookbook/dataset'/train_dir" --validation_directory="../../../../../'TensorFlow Machine Learning Cookbook'/dataset/validation_dir" --output_directory="../../../../../'TensorFlow Machine Learning Cookbook'/dataset/" --labels_file=""../../../../../'TensorFlow Machine Learning Cookbook'/dataset/cifar10_labels.txt"

报错：
```
(ENV) E:\study\machinelearning\code\models\research\inception\inception>python data/build_image_data.py --train_directory="temp/train_dir/" --validation_directory="temp/validation_dir" --output_directory="temp/" --labels_file="temp/cifar10_labels.txt
Saving results to temp/
Determining list of input files and labels from temp/validation_dir.
WARNING:tensorflow:From data/build_image_data.py:369: FastGFile.__init__ (from tensorflow.python.platform.gfile) is deprecated and will be removed in a future version.
Instructions for updating:
Use tf.gfile.GFile.
Traceback (most recent call last):
  File "data/build_image_data.py", line 437, in <module>
    tf.app.run()
  File "E:\study\machinelearning\ENV\lib\site-packages\tensorflow\python\platform\app.py", line 125, in run
    _sys.exit(main(argv))
  File "data/build_image_data.py", line 431, in main
    FLAGS.validation_shards, FLAGS.labels_file)
  File "data/build_image_data.py", line 417, in _process_dataset
    filenames, texts, labels = _find_image_files(directory, labels_file)
  File "data/build_image_data.py", line 369, in _find_image_files
    labels_file, 'r').readlines()]
  File "E:\study\machinelearning\ENV\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 188, in readlines
    self._preread_check()
  File "E:\study\machinelearning\ENV\lib\site-packages\tensorflow\python\lib\io\file_io.py", line 85, in _preread_check
    compat.as_bytes(self.__name), 1024 * 512, status)
  File "E:\study\machinelearning\ENV\lib\site-packages\tensorflow\python\framework\errors_impl.py", line 528, in __exit__
    c_api.TF_GetCode(self.status.status))
tensorflow.python.framework.errors_impl.NotFoundError: NewRandomAccessFile failed to Create/Open: temp/cifar10_labels.txt : ϵͳ\udcd5Ҳ\udcbb\udcb5\udcbdָ\udcb6\udca8\udcb5\udcc4·\udcbe\udcb6\udca1\udca3
; No such process
```
改成全路径解决：参考[1]
```
python data/build_image_data.py --train_directory="E:/temp/train_dir/" --validation_directory="E:/temp/validation_dir" --output_directory="E:/temp/" --labels_file="E:/temp/cifar10_labels.txt
```

### 8.5 用TensorFlow实现模仿大师绘画

github的代码有更新，真是大赞，参考[1]
```
def vgg_network(network_weights, init_image):
 network = {}
 image = init_image
 for i, layer in enumerate(vgg_layers):
  if layer[0] == 'c':
   weights, bias = network_weights[i][0][0][0][0]
   weights = np.transpose(weights, (1, 0, 2, 3))
   bias = bias.reshape(-1)
   conv_layer = tf.nn.conv2d(image, tf.constant(weights), (1,1,1,1), 'SAME')
   image = tf.nn.bias_add(conv_layer, bias)
  elif layer[0] == 'r':
   image = tf.nn.relu(image)
  else:
   image = tf.nn.max_pool(image, (1,2,2,1), (1,2,2,1), 'SAME')
  network[layer] = image
 return (network)
```
原来代码是layer[1]，改成layer[0]即可，这个问题也是从参考[1]中找到的

## Chapter9 递归神经网络
### 9.2 用TensorFlow实现RNN模型进行垃圾短信预测
* re正则表达式
```
Python 的 re 模块提供了re.sub用于替换字符串中的匹配项。

语法：

re.sub(pattern, repl, string, count=0, flags=0)
参数：

pattern : 正则中的模式字符串。
repl : 替换的字符串，也可为一个函数。
string : 要被查找替换的原始字符串。
count : 模式匹配后替换的最大次数，默认 0 表示替换所有的匹配。

来源： https://www.runoob.com/python/python-reg-expressions.htm
```
*text_string = re.sub(r'([^\s\w]|_|[0-9])+', '', text_string)*
这句是把指定的字符换成空''，即去掉
（0）匹配括号内的表达式，也表示一个组
（1）[...]用来表示一组字符，单独列出
（2）[^...]：表示不在[]中的字符，[^\s\w]即任意字母数字下划线，以及空白字符排除
（3）\s：匹配任意空白字符，等价于[\t\n\r\f]
（4）\w：匹配字母数字即下划线
（5）|：或者
（6）re+：匹配1个或多个的表达式
所以，这句话就是匹配非字母的字符，其他都替换掉，后面的*|_|[0-9]*，就是把下划线和数字也算上去了。但是，不知道为毛空白字符最后也算进去了！明明前面排除了

## Chapter11 TensorFlow的进阶应用
### 11.2 TensorFlow可视化：Tensorboard
Tensorboard的地址看log：http://localhost:6006

Chrome打不开，改成下面指令就可以了：
```
tensorboard --logdir=tensorboard/ --host localhost --port 8088
```

## 代码
放在github上了，参考[2]

## 参考
[1][Tensorflow cookbook](https://github.com/nfmcclure/tensorflow_cookbook)
[2][github](https://github.com/pkxpp/tensorflow_cookbook_for_tf_1.12)
