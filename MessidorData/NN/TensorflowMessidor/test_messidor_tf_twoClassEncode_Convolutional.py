import messidor_tf_twoClassEncode as tce
import tensorflow as tf
import numpy as np
from sklearn import cross_validation
import settings


class DataSets(object):
    pass

def weight_variable(shape):
    # initial = tf.truncated_normal(shape, stddev=0.1)
    initial = tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32,seed=1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

data_directory_path = settings.dataDirectoryPath
data_file_path = settings.dataFilePath
TRAIN_DATA_SIZE = settings.trainDataSize


sess = tf.InteractiveSession()

#image
IMAGE_D1 = settings.imageDimension1
IMAGE_D2 = settings.imageDimension2
IMAGE_D3 = settings.imageDimension3
IMAGE_SIZE = settings.imageSize
IMAGE_CONVOLUTIONAL_LAYER_OUTPUT = settings.firstConvolutionalLayerOutput
IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT = settings.denselyConnectedLayerOutput

RANGE = settings.range
BATCH = settings.batch

print('Reading dataset..')
# labels = tce.read_labels(data_file_path)
# file_names = tce.read_image_file_names(data_file_path)
file_names, labels = tce.read_labels_and_image_names(data_file_path)

images = tce.create_images_arrays(file_names, data_directory_path)

train_images = images[:TRAIN_DATA_SIZE]
test_images = images[TRAIN_DATA_SIZE:]
train_labels = labels[:TRAIN_DATA_SIZE]
test_labels = labels[TRAIN_DATA_SIZE:]


data_sets = DataSets()
data_sets.train = tce.DataSet(train_images, train_labels)
data_sets.test = tce.DataSet(test_images, test_labels)
print("Data set is loaded..")
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
y_ = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_variable([5,5,3,IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
b_conv1 = weight_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
x_image = tf.reshape(x, [-1, IMAGE_D1, IMAGE_D2, IMAGE_D3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
b_conv2 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)
#devrim lalalal
W_conv3 = weight_variable([5, 5, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
b_conv3 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])

h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
h_pool3 = max_pool_2x2(h_conv3)


#reklamlar bitti


W_fc1 = weight_variable([(IMAGE_D1/8)*(IMAGE_D2/8)*IMAGE_CONVOLUTIONAL_LAYER_OUTPUT, IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT])
b_fc1 = bias_variable([IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT])

h_pool_flat = tf.reshape(h_pool3, [-1,(IMAGE_D1/8)*(IMAGE_D2/8)*IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT,2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE")


train_step = tf.train.GradientDescentOptimizer(learning_rate=1e-4, use_locking=False, name='GradientOptimizer').minimize(cross_entropy)

correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print("initializing all variables")
sess.run(tf.initialize_all_variables())
for i in range(RANGE):
    print ("iteration : ", i)
    batch = data_sets.train.next_batch(BATCH)
    train_accuracy = accuracy.eval(feed_dict={
        x: batch[0], y_: batch[1], keep_prob:1.0})
    print "step %d, training accuracy %g" % (i, train_accuracy)
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob:0.5})


#print "test accuracy %g" % accuracy.eval(feed_dict={x: data_sets.test.images, y_: data_sets.test.labels})

feed_dict={x: data_sets.test.images, keep_prob:1.0}
prediction = tf.argmax(y_conv, 1)
print prediction.eval(feed_dict=feed_dict, session=sess)
probabilities = y_conv
print probabilities.eval(feed_dict=feed_dict, session=sess)