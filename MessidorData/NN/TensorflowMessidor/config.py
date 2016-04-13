import cv2, glob, numpy

# Data path

data_directory_path = '/Users/macbookair/Dropbox/image-eye/data'
data_file_path = '/Users/macbookair/Dropbox/image-eye/data/data.csv'
train_data_size = 900

# Image Dimensions

image_d1 = 280
image_d2 = 186
image_d3 = 3
image_size = image_d1*image_d2*image_d3


# Range

range = 50
batch = 50