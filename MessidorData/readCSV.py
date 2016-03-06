import csv
import sys
import numpy as np

labels_file_path = '/Users/macbookair/Dropbox/image-eye/Base11/AnnotationBase11.csv'


def one_hot_encode(label, number_of_classes=4):
    result = np.zeros(number_of_classes)
    result[label] = 1
    return result

def read_labels(labels_file_path):
    labelData = open(labels_file_path, 'r')
    labels = []
    try:
        reader = csv.reader(labelData)
        for row in reader:
            label = row[2]
            labels.append(one_hot_encode(label))
            print(label)
    finally:
        labelData.close()
    return labels

