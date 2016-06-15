import csv
import os
import settings
from PIL import Image
import numpy as np

DATA_DIRECTORY_PATH = settings.dataDirectoryPath
image_file_path = settings.dataFilePath
path = settings.path



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

image_list = read_image_file_names(image_file_path)

images = []

for image in image_list:
    image_path = os.path.join(DATA_DIRECTORY_PATH, image)
    im = Image.open(image_path)
    imarray = np.array(im)
    images.append(imarray)

images = np.asarray(images)
print(images.shape)
im = Image.open(path)
imarray = np.array(im)
print(imarray.shape)

imarray = imarray.reshape(imarray.shape[0]*imarray.shape[1]*imarray.shape[2])
print(imarray.shape)
