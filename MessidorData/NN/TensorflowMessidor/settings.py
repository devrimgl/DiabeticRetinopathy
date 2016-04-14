import csv
import os
import sys
from PIL import Image
import numpy as np

dataDirectoryPath = '/home/devrim/DR_data/data'
dataFilePath = '/home/devrim/DR_data/data/data.csv'
trainDataSize = 1000

imageDimension1 = 280
imageDimension2 = 186
imageDimension3 = 3

imageSize = imageDimension1*imageDimension2*imageDimension3

range = 50
batch = 50



