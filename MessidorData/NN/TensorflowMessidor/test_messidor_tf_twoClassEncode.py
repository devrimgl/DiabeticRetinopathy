import messidor_tf_twoClassEncode as tce
import tensorflow as tf
import numpy as np
import settings


class DataSets(object):
    pass

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


data_directory_path = settings.dataDirectoryPath
data_file_path = settings.dataFilePath
TRAIN_DATA_SIZE = settings.trainDataSize
IMAGE_SIZE = settings.imageSize
RANGE = settings.range
BATCH = settings.batch

print('Reading dataset..')
labels = tce.read_labels(data_file_path)
file_names = tce.read_image_file_names(data_file_path)
images = tce.create_images_arrays(file_names, data_directory_path)

# 1000 - 200
train_images = images[:TRAIN_DATA_SIZE]
test_images = images[TRAIN_DATA_SIZE:]
train_labels = labels[:TRAIN_DATA_SIZE]
test_labels = labels[TRAIN_DATA_SIZE:]
print(test_labels)


data_sets = DataSets()
data_sets.train = tce.DataSet(train_images, train_labels)
data_sets.test = tce.DataSet(test_images, test_labels)

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
# variable for bias and weight
#W = tf.Variable(tf.zeros([156240, 2]))
#b = tf.Variable(tf.zeros([2]))

W = weight_variable([IMAGE_SIZE, 2])
b = bias_variable([2])
# Softmax Reggression
# y is our predicted probability distribution
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross Entropy
y_ = tf.placeholder(tf.float32, [None, 2])  # correct answers
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables we created:
init = tf.initialize_all_variables()

# launch a model in a Session, and run the operation that initializes the variables
sess = tf.InteractiveSession()
sess.run(init)

# Model Evalution
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Kaci dogru kaci yanlis karsilastirmasi
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# train
for i in range(RANGE):
    print ('Iteration', i)
    batch = data_sets.train.next_batch(BATCH)
    train_accuracy = accuracy.eval(feed_dict= {
        x: batch[0], y_: batch[1]})
    # print "step %d, training accuracy %g" %(i, train_accuracy)
    train_step.run(feed_dict={x: data_sets.train.images, y_: data_sets.train.labels})

    #sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})


# Accuracy on test data
# print(sess.run(accuracy, feed_dict={x: data_sets.test.images, y_: data_sets.test.labels}))
print "test accuracy %g" % accuracy.eval(feed_dict={x: data_sets.test.images, y_: data_sets.test.labels})