from Models.LSTMModel import *
from Utils.DatasetUtil import *

# TRAINING CONFIG
NUM_EPOCH          = 10
MAX_ITERATION      = 200000
LEARNING_RATE      = 0.0001
DISPLAY_FREQUENCY  = 100;            DISPLAY_CONTENT = 'LearningRate = %f, Epoch = %d, iteration = %d, cost = %f'
VALIDATE_FREQUENCY = 10000
SAVE_FREQUENCY     = 1000

# EARLY STOPPING
PATIENCE = 50000
PATIENCE_INCREASE = 2
IMPROVEMENT_THRESHOLD = 0.995


# PATH SETTINGS
DATASET_PATH  = '../Dataset/English/'
SAVE_PATH     = '../Pretrained/Epoch=%d_Iter=%d.pkl'
BEST_PATH     = '../Pretrained/Best.pkl'
RECORD_PATH   = '../Pretrained/Record.pkl'
STATE_PATH    = '../Pretrained/CurrentState.pkl'

# NETWORK CONFIG
INPUT_SIZE  = 2
HIDDEN_SIZE = 256
OUTPUT_SIZE = 64

# GLOBAL VARIABLES
Dataset   = None
Model = None

######################################
#      READ TIME SERIES DATASET      #
######################################
def ReadDataset():
    global Dataset
    Dataset = DatasetUtil(datasetPath = DATASET_PATH)


###############################
#      CREATE LSTM MODEL      #
###############################
def CreateModel():
    global Model
    Model = LSTMModel(
                inputSize  = INPUT_SIZE,
                numHidden  = HIDDEN_SIZE,
                outputSize = OUTPUT_SIZE
            )

###########################
#      VALID MODEL        #
###########################
def ValidModel(LSTMModel, validData):
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
        cost = Model.PredFunc(input, char)
        AllCosts.append(cost)

        if iter % DISPLAY_FREQUENCY == 0:
            print (DISPLAY_CONTENT % (LEARNING_RATE, epoch, iter, numpy.mean(AllCosts)))

    return numpy.mean(AllCosts)

###########################
#      TRAIN MODEL        #
###########################
def TrainModel():
    global Dataset, Model

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
        LSTMModel.LoadState(file)
        file.close()
        print ('Load state !')

    trainData = Dataset.TrainData
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

            iter += 1
            cost = Model.TrainFunc(input, char)
            costs.append(cost)

            if iter % DISPLAY_FREQUENCY == 0:
                print (DISPLAY_CONTENT % (LEARNING_RATE, epoch, iter, numpy.mean(costs)))
                costs = []

            if iter % VALIDATE_FREQUENCY == 0:
                validCost = ValidModel()
                if validCost < BestCost:
                    if validCost < BestCost * IMPROVEMENT_THRESHOLD:
                        PATIENCE = max(PATIENCE, iter * PATIENCE_INCREASE)

                    BestCost = validCost
                    # Save best model
                    file = open(BEST_PATH, 'wb')
                    LSTMModel.SaveModel(file)
                    file.close()
                    print ('Save best model !')

            if iter % SAVE_FREQUENCY == 0:
                # Save model
                file = open(SAVE_PATH, 'wb')
                LSTMModel.SaveModel(file)
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
                LSTMModel.SaveState(file)
                file.close()
                print ('Save state !')
        epoch += 1

#############################################################################################
#                                                                                           #
#      GENERATE SEQUENCE                                                                    #
#                                                                                           #
#############################################################################################
def GenerateData():
    return 0


if __name__ == '__main__':
    ReadDataset()
    CreateModel()
    TrainModel()
    GenerateData()
