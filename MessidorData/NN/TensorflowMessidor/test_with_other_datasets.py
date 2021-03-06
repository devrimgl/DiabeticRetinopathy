import tensorflow as tf
from sklearn.metrics import roc_curve, auc, roc_auc_score, log_loss

import settings
import os
import messidor_tf_twoClassEncode as tce
import numpy as np
import matplotlib.pyplot as plt


def weight_variable(shape):
    """

    :param shape:
    :return:
    """
    initial = tf.truncated_normal(shape, stddev=0.1, seed=23)
    # initial = tf.random_uniform(shape, minval=0, maxval=None, dtype=tf.float32,seed=1)
    return tf.Variable(initial)


def bias_variable(shape):
    """

    :param shape:
    :return:
    """
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv2d(x, W):
    """

    :param x:
    :param W:
    :return:
    """
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')


def max_pool_2x2(x):
    """

    :param x:
    :return:
    """
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')  # strides changed to 1


data_directory_path = settings.dataDirectoryPath
data_file_path = settings.dataFilePath
TRAIN_DATA_SIZE = settings.trainDataSize
KERNEL_SIZE = settings.kernelSize
LAYER_CONSTANT = settings.layerPoolConstant

# image
IMAGE_D1 = settings.imageDimension1
IMAGE_D2 = settings.imageDimension2
IMAGE_D3 = settings.imageDimension3
IMAGE_SIZE = settings.imageSize
IMAGE_CONVOLUTIONAL_LAYER_OUTPUT = settings.firstConvolutionalLayerOutput
IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT = settings.denselyConnectedLayerOutput

RANGE = settings.range
BATCH = settings.batch


with tf.Session() as sess:
    x = tf.placeholder(tf.float32, [None, IMAGE_SIZE])
    y_ = tf.placeholder(tf.float32, [None, 2])

    W_conv1 = weight_variable([KERNEL_SIZE, KERNEL_SIZE, IMAGE_D3, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
    b_conv1 = weight_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
    x_image = tf.reshape(x, [-1, IMAGE_D1, IMAGE_D2, IMAGE_D3])
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)

    W_conv2 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])
    b_conv2 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT])

    h_conv2 = tf.nn.relu(conv2d(h_conv1, W_conv2) + b_conv2)
    h_pool1 = max_pool_2x2(h_conv2)

    W_conv3 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2])
    b_conv3 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2])

    h_conv3 = tf.nn.relu(conv2d(h_pool1, W_conv3) + b_conv3)

    W_conv4 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2])
    b_conv4 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2])

    h_conv4 = tf.nn.relu(conv2d(h_conv3, W_conv4) + b_conv4)
    h_pool2 = max_pool_2x2(h_conv4)

    W_conv5 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 2, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4])
    b_conv5 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4])

    h_conv5 = tf.nn.relu(conv2d(h_pool2, W_conv5) + b_conv5)

    W_conv6 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4])
    b_conv6 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4])

    h_conv6 = tf.nn.relu(conv2d(h_conv5, W_conv6) + b_conv6)
    h_pool3 = max_pool_2x2(h_conv6)

    W_conv7 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 4, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8])
    b_conv7 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8])

    h_conv7 = tf.nn.relu(conv2d(h_pool3, W_conv7) + b_conv7)
    h_pool4 = max_pool_2x2(h_conv7)

    W_conv8 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8])
    b_conv8 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8])

    h_conv8 = tf.nn.relu(conv2d(h_pool4, W_conv8) + b_conv8)
    h_pool5 = max_pool_2x2(h_conv8)

    W_conv9 = weight_variable(
        [KERNEL_SIZE, KERNEL_SIZE, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 8, IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 16])
    b_conv9 = bias_variable([IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 16])

    h_conv9 = tf.nn.relu(conv2d(h_pool5, W_conv9) + b_conv9)
    h_pool6 = max_pool_2x2(h_conv9)

    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_pool6, keep_prob)

    W_fc1 = weight_variable(
        [(IMAGE_D1 / LAYER_CONSTANT) * (IMAGE_D2 / LAYER_CONSTANT) * IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 16,
         IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT])
    b_fc1 = bias_variable([IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT])

    h_pool_flat = tf.reshape(h_pool6, [-1, (IMAGE_D1 / LAYER_CONSTANT) * (
        IMAGE_D2 / LAYER_CONSTANT) * IMAGE_CONVOLUTIONAL_LAYER_OUTPUT * 16])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool_flat, W_fc1) + b_fc1)

    W_fc2 = weight_variable([IMAGE_DENSELY_CONNECTED_LAYER_OUTPUT, 2])
    b_fc2 = bias_variable([2])
    y_conv = tf.nn.softmax(tf.matmul(h_fc1, W_fc2) + b_fc2)

    cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
    cross_entropy = tf.Print(cross_entropy, [cross_entropy], "CrossE")
    train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

    correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    saver = tf.train.Saver()
    saver.restore(sess, settings.cnnModelPath)
    print('Data model trained with messidor is loaded from the disk')
    # Load other dataset to test with messidor image.
    # PATH = '/home/devrim/Downloads/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images'

    #PATH = "/home/devrim/Downloads/ROCtraining"
    NORMAL_PATH = "/home/devrim/Downloads/e-optha/e_optha_MA/healthy_eoptha"
    ABNORMAL_PATH = "/home/devrim/Downloads/e-optha/e_optha_MA/MA_eoptha"
    normal_file_names = []
    abnormal_file_names = []
    for (dirpath, dirnames, filenames) in os.walk(NORMAL_PATH):
        normal_file_names.extend(filenames)
        break
    for (dirpath, dirnames, filenames) in os.walk(ABNORMAL_PATH):
        abnormal_file_names.extend(filenames)
        break
    normal_test_images = tce.create_images_arrays(normal_file_names, NORMAL_PATH)
    normal_test_labels = np.asarray([tce.two_class_encode('0') for i in range(len(normal_test_images))])
    abnormal_test_images = tce.create_images_arrays(abnormal_file_names, ABNORMAL_PATH)
    abnormal_test_labels = np.asarray([tce.two_class_encode('1') for i in range(len(abnormal_test_images))])
    test_images = np.concatenate([normal_test_images, abnormal_test_images])
    test_labels = np.concatenate([normal_test_labels, abnormal_test_labels])
    test_data = tce.DataSet(test_images, test_labels)

    test_acc = accuracy.eval(
        feed_dict={x: test_data.images, y_: test_data.labels, keep_prob: 0.5})
    print('Overall Test Accuracy:', test_acc)

    label_true = [tce.convert_one_hot_encode(item) for item in test_data.labels]
    probabilities = y_conv
    label_probabilities = probabilities.eval(
        feed_dict={x: test_data.images, y_: test_data.labels, keep_prob: 0.5})
    print(label_probabilities)
    label_scores = [item[0] for item in label_probabilities]
    label_true = np.array(label_true)
    label_scores = np.array(label_scores)

    normal = len(normal_test_labels)
    abnormal = len(abnormal_test_labels)
    sensitivty = 0
    specificity = 0
    for i in range(normal):
        if label_scores[i] >= 0.5:
            specificity += 1
    for i in range(normal, normal + abnormal):
        if label_scores[i] < 0.5:
            sensitivty += 1
    sensitivty /= float(abnormal)
    specificity /= float(normal)
    print('Sensitivity', sensitivty)
    print('specificity', specificity)
    fpr, tpr, thresholds = roc_curve(label_true, label_scores, pos_label=0)
    test_auc_result = auc(fpr, tpr)
    test_log_loss_result = log_loss(label_true, label_probabilities)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area= %0.2f)' % test_auc_result)
    plt.plot([0,1], [0,1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Sensitivity')
    plt.ylabel('Specitivity')
    plt.title('ROC curve')
    plt.legend(loc='lower right')
    plt.show()



