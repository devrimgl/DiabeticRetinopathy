import Minst
import tensorflow as tf

#datayi okudu
mnist = Minst.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, [None, 784])

# variable for bias and weight
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

# Softmax Reggression
# y is our predicted probability distribution
y = tf.nn.softmax(tf.matmul(x, W) + b)

# Cross Entropy
y_ = tf.placeholder(tf.float32, [None, 10]) #correct answers
cross_entropy = -tf.reduce_sum(y_*tf.log(y))

train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

# initialize the variables we created:
init = tf.initialize_all_variables()

# launch a model in a Session, and run the operation that initializes the variables
sess = tf.Session()
sess.run(init)

#train
for i in range(1000):
  batch_xs, batch_ys = mnist.train.next_batch(100)
  sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

# Model Evalution
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# Kaci dogru kaci yanlis karsilastirmasi
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Accuracy on test data
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))