import csv
import os
import sys
from PIL import Image
import numpy as np

# (DR1, DR2, DR3) and DR0
labels_file_path = '/Users/macbookair/Dropbox/image-eye/test/AnnotationBaseTest1.csv'


def two_class_encode(label, number_of_classes=2):
    result = np.zeros(number_of_classes)
    if label == "0":
        result[1] = 1
    else:
        result[0] = 1
    return result

# dictionary of file name and one_hot_encoded labels
def read_labels(labels_file_path):
    labelData = open(labels_file_path, 'r')
    labels = []
    try:
        reader = csv.reader(labelData)
        for row in reader:
            label = row[2]
            label = two_class_encode(label)
            # labels[row[0]] = one_hot_encode(label)
            labels.append(label)
    finally:
        labelData.close()
    return np.asarray(labels)

def read_image_file_names(image_file_path):
    image_list = []
    image_data = open(image_file_path, 'r')
    try:
        reader = csv.reader(image_data)
        for row in reader:
            image = row[0]
            image_list.append(image)
    finally:
        image_data.close()
    return image_list

def create_images_arrays(image_list, DATA_DIRECTORY_PATH):
    """
    It reads all image files and created a list of image arrays,
    :param image_list:
    :param DATA_DIRECTORY_PATH:
    :return:
    """
    images = []
    for image in image_list:
        image_path = os.path.join(DATA_DIRECTORY_PATH, image)
        im = Image.open(image_path)
        imarray = np.array(im)
        images.append(imarray)
    return np.asarray(images)


class DataSet(object):
  def __init__(self, images, labels, fake_data=False):
    if fake_data:
      self._num_examples = 10000
    else:
      assert images.shape[0] == labels.shape[0], (
          "images.shape: %s labels.shape: %s" % (images.shape,
                                                 labels.shape))
      self._num_examples = images.shape[0]
      # Convert shape from [num examples, rows, columns, depth]
      # to [num examples, rows*columns] (assuming depth == 1)
      # assert images.shape[3] == 1
      images = images.reshape(images.shape[0],
                              images.shape[1] * images.shape[2]*images.shape[3])
      # Convert from [0, 255] -> [0.0, 1.0].
      images = images.astype(np.float32)
      images = np.multiply(images, 1.0 / 255.0)
    self._images = images
    self._labels = labels
    self._epochs_completed = 0
    self._index_in_epoch = 0
  @property
  def images(self):
    return self._images
  @property
  def labels(self):
    return self._labels
  @property
  def num_examples(self):
    return self._num_examples
  @property
  def epochs_completed(self):
    return self._epochs_completed
  def next_batch(self, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    if fake_data:
      fake_image = [1.0 for _ in xrange(784)]
      fake_label = 0
      return [fake_image for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = self._index_in_epoch
    self._index_in_epoch += batch_size
    if self._index_in_epoch > self._num_examples:
      # Finished epoch
      self._epochs_completed += 1
      # Shuffle the data
      perm = np.arange(self._num_examples)
      np.random.shuffle(perm)
      self._images = self._images[perm]
      self._labels = self._labels[perm]
      # Start next epoch
      start = 0
      self._index_in_epoch = batch_size
      assert batch_size <= self._num_examples
    end = self._index_in_epoch
    return self._images[start:end], self._labels[start:end]

# Test..
read_labels(labels_file_path)
