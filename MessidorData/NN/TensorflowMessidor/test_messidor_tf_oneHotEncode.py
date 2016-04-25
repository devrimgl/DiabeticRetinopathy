import messidor_data as ms
import tensorflow as tf
import settings


data_directory_path = settings.dataDirectoryPath
data_file_path = settings.dataFilePath
TRAIN_DATA_SIZE = settings.trainDataSize
IMAGE_SIZE = settings.imageSize
RANGE = settings.range
BATCH = settings.batch

print('Reading dataset..')
labels = ms.read_labels(data_file_path)
file_names = ms.read_image_file_names(data_file_path)
images = ms.create_images_arrays(file_names, data_directory_path)

train_images = images[:TRAIN_DATA_SIZE]
test_images = images[TRAIN_DATA_SIZE:]
train_labels = labels[:TRAIN_DATA_SIZE]
test_labels = labels[TRAIN_DATA_SIZE:]


class DataSets(object):
    pass


data_sets = DataSets()
data_sets.train = ms.DataSet(train_images, train_labels)
data_sets.test = ms.DataSet(test_images, test_labels)

x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
# variable for bias and weight
W = tf.Variable(tf.zeros([IMAGE_SIZE, 4]))
b = tf.Variable(tf.zeros([4]))

# Softmax Reggression
# y is our predicted probability distribution
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross Entropy
y_ = tf.placeholder(tf.float32, [None, 4])  # correct answers
cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables we created:
init = tf.initialize_all_variables()

# launch a model in a Session, and run the operation that initializes the variables
sess = tf.Session()
sess.run(init)

# train
for i in range(RANGE):
    print ('Iteration', i)
    batch_xs, batch_ys = data_sets.train.next_batch(BATCH)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Model Evalution
correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))

# Kaci dogru kaci yanlis karsilastirmasi
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Accuracy on test data
print(sess.run(accuracy, feed_dict={x: data_sets.test.images, y_: data_sets.test.labels}))
