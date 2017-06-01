import os
import configparser
import numpy
import skimage.transform
import matplotlib.pyplot as plt

def CheckFileExist(filePath,
                   throwError = True):
    if not os.path.exists(filePath):
        print ('%s does not exist ! !' % (filePath))
        if throwError is True:
            assert ('%s does not exist ! !' % (filePath))
        return False
    return True

def CheckPathExist(path,
                   throwError = True):
    if not os.path.isdir(path):
        print ('%s does not exist ! Make sure you choose right location !' % (path))
        if throwError is True:
            assert ('%s does not exist ! Make sure you choose right location !' % (path))
        return False
    return True

def CheckNotNone(something,
                 name,
                 throwError = True):
    if something is None:
        print ('%s can not be none !' % (name))
        if throwError is True:
            assert ('%s can not be none !' % (name))
        return False
    return True

def GetAllSubfoldersPath(path):
    # Check parameter
    CheckNotNone(path, 'path'); CheckPathExist(path)

    allSubFoldersPath = sorted([os.path.join(os.path.abspath(path), name + '/') for name in os.listdir(path)
                                if CheckPathExist(os.path.join(path, name))])
    return allSubFoldersPath

def GetAllFiles(path):
    # Check parameters
    CheckNotNone(path, 'path'); CheckPathExist(path)

    allFilesPath = sorted([os.path.join(os.path.abspath(path), filename) for filename in os.listdir(path)
                           if CheckFileExist(os.path.join(path, filename))])
    return allFilesPath

def ReadFile(filePath):
    # Check file exist
    CheckFileExist(filePath)

    file = open(filePath)
    allData = tuple(file)
    file.close()

    return allData

def ReadImages(imsPath,
               batchSize):
    ims = []
    numHasData = 0
    for imPath in imsPath:
        if imPath != '':
            extension = imPath.split('.')[-1]
            im = plt.imread(imPath, extension)
            im = skimage.transform.resize(im, (300, 300), preserve_range=True)
            im = im[:, :, [2, 1, 0]]
            im = numpy.transpose(im, (2, 0, 1))
            ims.append(im)
            numHasData += 1
    ims = numpy.asarray(ims, dtype = 'float32')

    VGG_MEAN = numpy.asarray([103.939, 116.779, 123.68], dtype = 'float32')
    VGG_MEAN = numpy.reshape(VGG_MEAN, (1, 3, 1, 1))

    ims = ims - VGG_MEAN

    if numHasData == 0:
        numHasData = batchSize
        ims        = numpy.zeros((batchSize, 3, 300, 300), dtype = 'float32')
    ims     = numpy.pad(ims, ((0, batchSize - numHasData), (0, 0), (0, 0), (0, 0)), mode = 'constant', constant_values = 0)
    return ims

def ReadFileIni(filePath):
    # Check file exist
    CheckFileExist(filePath)

    config = configparser.ConfigParser()
    config.sections()
    config.read(filePath)
    return config