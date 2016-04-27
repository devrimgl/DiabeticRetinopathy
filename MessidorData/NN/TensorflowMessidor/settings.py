import math
import time

dataDirectoryPath = '/home/devrim/DR_data/data'
convertDataDirectoryPath = '/home/devrim/DR_data/convertData'
dataFilePath = '/home/devrim/DR_data/data/data.csv'
path = '/home/devrim/DR_data/data/20051019_38557_0100_PP.tif'
trainDataSize = 1000

imageDimension1 = 128
imageDimension2 = 128
imageDimension3 = 3
RGBlayer = 3

imageSize = imageDimension1 * imageDimension2 * imageDimension3
firstConvolutionalLayerOutput = 16
denselyConnectedLayerOutput = 96

range = 5001
batch = 50

kernelSize = 3
layer = 6
layerPoolConstant = int(math.pow(2, layer))
currentTime = time.strftime("%Y %m %d - %H:%M")
