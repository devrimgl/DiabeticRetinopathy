import math
import time

dataDirectoryPath = '/home/devrim/DR_data/data'
diaretdb1_path= '/home/devrim/Downloads/diaretdb1_v_1_1/resources/images/ddb1_fundusimages/'
diaretdb1_converted_path= '/home/devrim/Downloads/diaretdb1_v_1_1/resources/images/converted/'
diaretdb0_path= '/home/devrim/Downloads/diaretdb0_v_1_1/resources/images/diaretdb0_fundus_images'
diaretdb0_converted_path= '//home/devrim/Downloads/diaretdb0_v_1_1/resources/images/converted/'
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

range = 2001
batch = 50

kernelSize = 3
layer = 6
layerPoolConstant = int(math.pow(2, layer))
currentTime = time.strftime("%Y %m %d - %H:%M")

#cnnModelPath = '/home/devrim/DR_data/models/cnnModel/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnn_noneq/model.ckpt'
cnnModelPath = '/home/devrim/DR_data/models/cnnRotateNoEqualizeModelMA/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnnRotateNoEqualizeModel/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnnRotateEqualizeModel/model.ckpt'
# cnnModelPath = '/home/devrim/DR_data/models/cnnGrayScaleRotateNoEqualizeModel/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnnRotateContrast15NoEq/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnnRotateContrast25NoEq/model.ckpt'
#cnnModelPath = '/home/devrim/DR_data/models/cnnRotateContrast15NoEqSharpenEdgeEnhance/model.ckpt'
