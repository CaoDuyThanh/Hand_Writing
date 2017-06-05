import random
from Tkinter import *
from Models.LSTMModel import *
from Models.RNNModel import *
from Utils.DatasetUtil import *

# TRAINING CONFIG
NUM_EPOCH          = 10
MAX_ITERATION      = 1000000
LEARNING_RATE      = 0.0001
DISPLAY_FREQUENCY  = 100;            DISPLAY_CONTENT = 'LearningRate = %f, Epoch = %d, iteration = %d, cost = %f. Best valid cost = %f'
VALIDATE_FREQUENCY = 5000
SAVE_FREQUENCY     = 2500

# EARLY STOPPING
PATIENCE              = 50000
PATIENCE_INCREASE     = 2
IMPROVEMENT_THRESHOLD = 0.9999995

# AUGEMENT SETTING
ROT_ANGLE_LIMIT = 15. / 180 * 3.14
TRANS_LIMIT     = 0.2
SHEAR_LIMIT     = 0.5
SCALE_LIMIT     = 0.2

# NETWORK CONFIG
NETWORK_TYPE = 'RNN'
INPUT_SIZE   = 2
HIDDEN_SIZE  = 256
OUTPUT_SIZE  = 62

# PATH SETTINGS
DATASET_PATH  = '../Dataset/English/'
SAVE_PATH     = '../Pretrained/' + NETWORK_TYPE + '/Epoch=%d_Iter=%d.pkl'
BEST_PATH     = '../Pretrained/' + NETWORK_TYPE + '/Best.pkl'
RECORD_PATH   = '../Pretrained/' + NETWORK_TYPE + '/Record.pkl'
STATE_PATH    = '../Pretrained/' + NETWORK_TYPE + '/CurrentState.pkl'

# GLOBAL VARIABLES
Dataset = None
Model   = None

######################################
#      READ TIME SERIES DATASET      #
######################################
def ReadDataset():
    global Dataset
    Dataset = DatasetUtil(datasetPath = DATASET_PATH)

################################
#   CREATE TIME-SERIES MODEL   #
################################
def CreateModel():
    global Model
    if NETWORK_TYPE == 'LSTM':
        Model = LSTMModel(
                    inputSize  = INPUT_SIZE,
                    numHidden  = HIDDEN_SIZE,
                    outputSize = OUTPUT_SIZE
                )
    else:
        Model = RNNModel(
                    inputSize = INPUT_SIZE,
                    numHidden=HIDDEN_SIZE,
                    outputSize=OUTPUT_SIZE
                )

###########################
#      VALID MODEL        #
###########################
def ValidModel(Model, validData):
    print ('----------------------------------- VALIDATION -----------------------------------------------------------')

    AllCosts = []
    iter  = 0
    epoch = 0
    for validDataIdx, validSample in enumerate(validData):
        trj  = validSample['Trj']
        rows = trj['rows']
        cols = trj['cols']

        rows = rows.reshape((rows.shape[0], 1))
        cols = cols.reshape((cols.shape[0], 1))

        input = numpy.concatenate((rows, cols), axis=1)
        input = input.reshape(input.shape[0], 1, input.shape[1])

        char = validSample['Char']
        char = char.reshape((1,))

        iter += 1
        cost = Model.ValidFunc(input, char)
        AllCosts.append(cost)

        if iter % DISPLAY_FREQUENCY == 0:
            print (DISPLAY_CONTENT % (LEARNING_RATE, epoch, iter, numpy.mean(AllCosts), epoch))

    print ('Valid cost = %f' % (numpy.mean(AllCosts)))
    print ('----------------------------------- VALIDATION (DONE) ----------------------------------------------------')

    return numpy.mean(AllCosts)



###########################
#      TRAIN MODEL        #
###########################
def TrainModel():
    global Dataset, Model, \
           PATIENCE, PATIENCE_INCREASE, \
           IMPROVEMENT_THRESHOLD

    # Load record
    AllCosts = []
    BestCost = 1000
    if CheckFileExist(RECORD_PATH, throwError = False):
        file = open(RECORD_PATH)
        AllCosts = pickle.load(file)
        BestCost = pickle.load(file)
        file.close()

    # Load model
    if CheckFileExist(STATE_PATH, throwError = False):
        file = open(STATE_PATH)
        Model.LoadState(file)
        file.close()
        print ('Load state !')

    trainData = Dataset.TrainData
    validData = Dataset.ValidData
    epoch = 0
    iter  = 0
    costs = []
    while iter < MAX_ITERATION:
        for trainDataIdx, trainSample in enumerate(trainData):
            trj  = trainSample['Trj']
            rows = trj['rows']
            cols = trj['cols']

            rows = rows.reshape((rows.shape[0], 1))
            cols = cols.reshape((cols.shape[0], 1))

            input = numpy.concatenate((rows, cols), axis = 1)
            input = input.reshape(input.shape[0], 1, input.shape[1])

            char  = trainSample['Char']
            char  = char.reshape((1,))

            auInput = AugmentData(input)

            iter += 1
            cost = Model.TrainFunc(auInput, char)
            costs.append(cost)

            if iter % DISPLAY_FREQUENCY == 0:
                print (DISPLAY_CONTENT % (LEARNING_RATE, epoch, iter, numpy.mean(costs), BestCost))
                costs = []

            if iter % VALIDATE_FREQUENCY == 0:
                validCost = ValidModel(Model, validData)
                if validCost < BestCost:
                    if validCost < BestCost * IMPROVEMENT_THRESHOLD:
                        PATIENCE = max(PATIENCE, iter * PATIENCE_INCREASE)

                    BestCost = validCost
                    # Save best model
                    file = open(BEST_PATH, 'wb')
                    Model.SaveModel(file)
                    file.close()
                    print ('Save best model !')

            if iter % SAVE_FREQUENCY == 0:
                # Save model
                file = open(SAVE_PATH % (epoch, iter), 'wb')
                Model.SaveModel(file)
                file.close()
                print ('Save model !')

                # Save record
                file = open(RECORD_PATH, 'wb')
                pickle.dump(AllCosts, file, 0)
                pickle.dump(BestCost, file, 0)
                file.close()
                print ('Save record !')

                # Save state
                file = open(STATE_PATH, 'wb')
                Model.SaveState(file)
                file.close()
                print ('Save state !')
        epoch += 1

# - Affine matrix --------------------------------------------------------------------------------------------------
def RotateMatrix(alpha):
    return numpy.asarray([[math.cos(alpha), -math.sin(alpha), 0],
                          [math.sin(alpha),  math.cos(alpha), 0],
                          [              0,                0, 1]], dtype = 'float32')

def TranslateMatrix(dx, dy):
    return numpy.asarray([[1, 0, dx],
                          [0, 1, dy],
                          [0, 0,  1]], dtype = 'float32')

def ScaleMatrix(scaleX, scaleY):
    return numpy.asarray([[scaleX,      0, 0],
                          [     0, scaleY, 0],
                          [     0,      0, 1]], dtype = 'float32')

def ShearMatrix(dx, dy):
    return numpy.asarray([[ 1, dy, 0],
                          [ 0,  1, 0],
                          [ 0,  0, 1]], dtype = 'float32').dot(
           numpy.asarray([[ 1, 0, 0],
                          [dx, 1, 0],
                          [ 0, 0, 1]], dtype='float32')
           )

def AugmentData(input):
    alpha = random.uniform(-ROT_ANGLE_LIMIT, ROT_ANGLE_LIMIT)
    dx    = random.uniform(-TRANS_LIMIT, TRANS_LIMIT)
    dy    = random.uniform(-TRANS_LIMIT, TRANS_LIMIT)
    scaleX = random.uniform(1 - SCALE_LIMIT, 1 + SCALE_LIMIT)
    scaleY = random.uniform(1 - SCALE_LIMIT, 1 + SCALE_LIMIT)
    shearX = random.uniform(-SHEAR_LIMIT, SHEAR_LIMIT)
    shearY = random.uniform(-SHEAR_LIMIT, SHEAR_LIMIT)

    rotMax   = RotateMatrix(alpha)
    transMax = TranslateMatrix(dx, dy)
    scaleMax = ScaleMatrix(scaleX, scaleY)
    shearMax = ShearMatrix(shearX, shearY)

    input = input.reshape((input.shape[0], input.shape[2]))

    ones  = numpy.ones((input.shape[0], 1), dtype = 'float32')
    input = numpy.concatenate((input, ones), axis=1)

    input = rotMax.dot(shearMax).dot(scaleMax).dot(transMax).dot(input.T)
    # input = transMax.dot(scaleMax).dot(shearMax).dot(rotMax).dot(input.T)
    # input = shearMax.dot(input.T)
    input = input.T
    input = input[:, :2]
    input = input.reshape((input.shape[0], 1, input.shape[1]))
    return input

###########################
#      TEST MODEL         #
###########################
def TestModel():
    global Model, \
           Dataset

    # Load best model
    if CheckFileExist(BEST_PATH, throwError=False):
        file = open(BEST_PATH)
        Model.LoadModel(file)
        file.close()
        print ('Load best model !')

    print ('----------------------------------- TEST -----------------------------------------------------------------')
    testData = Dataset.TestData
    AllPrec = 0
    iter = 0
    for validDataIdx, validSample in enumerate(testData):
        trj = validSample['Trj']
        rows = trj['rows']
        cols = trj['cols']

        rows = rows.reshape((rows.shape[0], 1))
        cols = cols.reshape((cols.shape[0], 1))

        input = numpy.concatenate((rows, cols), axis=1)
        input = input.reshape(input.shape[0], 1, input.shape[1])

        char = validSample['Char']
        char = char.reshape((1,))

        iter += 1
        pred = Model.PredFunc(input)
        prec = 0
        for idx in range(len(pred)):
            if pred[idx] == char[idx]:
                prec += 1
        AllPrec += prec

    precision = AllPrec * 1.0 / len(testData)
    print ('Precision = %f' % (precision))
    print ('----------------------------------- TEST (DONE) ----------------------------------------------------------')

if __name__ == '__main__':
    ReadDataset()
    CreateModel()
    TrainModel()
    TestModel()


# print ('%s' % Dataset.Character[char])
#     plt.pause(2)
#     plt.close()
