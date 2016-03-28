import messidor_tf_twoClassEncode as tce
import tensorflow as tf
import numpy as np


class DataSets(object):
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides= [1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 3, 3, 1], strides=[1, 2, 2, 1], padding='SAME')

#400 10MB data
DATA_DIRECTORY_PATH = '/Users/macbookair/Dropbox/image-eye/test'
data_file_path = '/Users/macbookair/Dropbox/image-eye/test/AnnotationBaseTest1.csv'
TRAIN_DATA_SIZE = 350

#1200 mix data
#DATA_DIRECTORY_PATH = '/Users/macbookair/Dropbox/image-eye/data'
#data_file_path = '/Users/macbookair/Dropbox/image-eye/data/data.csv'
#TRAIN_DATA_SIZE = 1000

sess = tf.InteractiveSession()

#image
IMAGE_D1 = 280
IMAGE_D2 = 186
IMAGE_D3 = 3
IMAGE_SIZE = IMAGE_D1*IMAGE_D2*IMAGE_D3

print('Reading dataset..')
labels = tce.read_labels(data_file_path)
file_names = tce.read_image_file_names(data_file_path)
images = tce.create_images_arrays(file_names, DATA_DIRECTORY_PATH)

train_images = images[:TRAIN_DATA_SIZE]
test_images = images[TRAIN_DATA_SIZE:]
train_labels = labels[:TRAIN_DATA_SIZE]
test_labels = labels[TRAIN_DATA_SIZE:]
print(test_labels)


data_sets = DataSets()
data_sets.train = tce.DataSet(train_images, train_labels)
data_sets.test = tce.DataSet(test_images, test_labels)

#x = tf.placeholder(tf.float32, [None, 9999360]) Image resized - it was for 10mb images
x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
y_ = tf.placeholder(tf.float32, [None, 2])

W_conv1 = weight_variable([5,5,3,32])
b_conv1 = weight_variable([32])
x_image = tf.reshape(x, [-1, IMAGE_D1, IMAGE_D2, IMAGE_D3])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)
W_fc1 = weight_variable([140*93*32, 512])
b_fc1 = bias_variable([512])

h_pool_flat = tf.reshape(h_pool1, [-1,140*93*32])
h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)


W_fc2 = weight_variable([512,2])
b_fc2 = bias_variable([2])
y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.initialize_all_variables())
for i in range(50):
    batch = data_sets.train.next_batch(50)
    train_step.run(feed_dict={x: batch[0], y_: batch[1]})

print "test accuracy %g" % accuracy.eval(feed_dict={x: data_sets.test.images, y_: data_sets.test.labels})
