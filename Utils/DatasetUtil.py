import scipy.io
from random import shuffle
from Utils.FileHelper import *


class DatasetUtil():
    TRAIN_RATIO = 0.9
    VALID_RATIO = 0.05
    TEST_RATIO  = 0.05

    def __init__(self,
                 datasetPath = None):
        # Check parameters
        CheckNotNone(datasetPath, 'datasetPath'); CheckPathExist(datasetPath)

        # Set parameters
        self.DatasetPath = datasetPath

        # Config train dataset
        self.TrainImgPath   = self.DatasetPath + 'Img/'
        self.TrainTrjPath   = self.DatasetPath + 'Trj/'

        # Constant
        self.Character = '01234567890ABCCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

        # Load data
        self.loadTrainFiles()

    def readOneFolder(self,
                      trjFolderIdx,
                      trainTrjFolder):
        trainTrjFiles = GetAllFiles(trainTrjFolder)
        for trainTrjFile in trainTrjFiles:
            data = self.readMFile(trainTrjFile)

            oneData = dict()
            oneData['Trj']  = data
            oneData['Char'] = numpy.asarray(trjFolderIdx, dtype = 'int32')
            self.AllData.append(oneData)

    def loadTrainFiles(self):
        self.AllData = []

        trainTrjFolders = sorted(GetAllSubfoldersPath(self.TrainTrjPath))
        for trjFolderIdx, trainTrjFolder in enumerate(trainTrjFolders):
            self.readOneFolder(trjFolderIdx, trainTrjFolder)

        # Split dataset
        shuffle(self.AllData)
        numSamples = self.AllData.__len__()
        self.TrainData = self.AllData[0 : int(numSamples * self.TRAIN_RATIO)]
        self.ValidData = self.AllData[int(numSamples * self.TRAIN_RATIO) : int(numSamples * (self.TRAIN_RATIO + self.VALID_RATIO))]
        self.TestData  = self.AllData[int(numSamples * (self.TRAIN_RATIO + self.VALID_RATIO)) : ]

    def readMFile(self,
                  pathFile):
        file = open(pathFile)

        content = file.read()
        rows    = content.split('\r\n\r\n')[0]
        cols    = content.split('\r\n\r\n')[1]

        rows    = map(float, rows[rows.find('[') + 1 : rows.find(']')].split('\r\n')[:-1])
        cols    = map(float, cols[cols.find('[') + 1 : cols.find(']')].split('\r\n')[:-1])

        data = dict()
        data['rows'] = numpy.asarray(rows, dtype = 'float32')
        data['cols'] = numpy.asarray(cols, dtype = 'float32')

        file.close()

        return data