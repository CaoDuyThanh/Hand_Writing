import theano
import theano.tensor as T
import numpy
from Layers.LayerHelper import *

def numpyFloatX(data):
    return numpy.asarray(data, dtype = 'float32')

class LSTMModel():
    def __init__(self,
                 inputSize,
                 numHidden,
                 outputSize):
        ####################################
        #       Create model               #
        ####################################
        # Create tensor variables to store input / output data
        self.Input  = T.tensor3('Input')
        self.Target = T.ivector('Target')

        # Parse data
        numSteps   = self.Input.shape[0]
        numSamples = self.Input.shape[1]

        # Setting LSTM architecture
        self.Net         = LSTMNet()
        self.Net.NetName = 'LSTM_Encoder'
        self.Net.LayerOpts['lstm_num_hidden']   = numHidden
        self.Net.LayerOpts['lstm_inputs_size']  = inputSize
        self.Net.LayerOpts['lstm_outputs_size'] = outputSize

        # Create LSTM layer
        self.Net.Layer['lstm_truncid'] = LSTMLayer(self.Net)

        # Truncate model
        yVals, modelUpdates = theano.scan(self.Net.Layer['lstm_truncid'].Step,
                                     sequences    = [self.Input],
                                     outputs_info = [T.alloc(numpyFloatX(0.),
                                                             numSamples,
                                                             numHidden),
                                                     T.alloc(numpyFloatX(0.),
                                                             numSamples,
                                                             numHidden),
                                                     T.alloc(numpyFloatX(0.),
                                                             numSamples,
                                                             outputSize)],
                                     n_steps      = numSteps)

        output = yVals[2][-1]
        self.Net.LayerOpts['softmax_axis'] = 1
        predProb = SoftmaxLayer(self.Net, output).Output
        pred     = predProb.argmax(axis = 1)


        # Calculate cost
        cost   = -T.log(predProb[T.arange(numSamples), self.Target]).mean()
        params = self.Net.Layer['lstm_truncid'].Params
        grads  = T.grad(cost, params)
        self.Optimizer = AdamGDUpdate(net    = self.Net,
                                      params = params,
                                      grads  = grads)
        gdUpdates = self.Optimizer.Updates

        updates = modelUpdates + \
                  gdUpdates

        self.TrainFunc = theano.function(inputs  = [self.Input, self.Target],
                                         updates = updates,
                                         outputs = [cost])

        self.ValidFunc = theano.function(inputs  = [self.Input, self.Target],
                                         outputs = [cost])

        self.PredFunc  = theano.function(inputs  = [self.Input],
                                         outputs = [pred])

    def SaveModel(self,
                  file):
        self.Net.Layer['lstm_truncid'].SaveModel(file)

    def SaveState(self,
                  file):
        self.Net.Layer['lstm_truncid'].SaveModel(file)
        self.Optimizer.SaveModel(file)

    def LoadModel(self,
                  file):
        self.Net.Layer['lstm_truncid'].LoadModel(file)

    def LoadState(self,
                  file):
        self.Net.Layer['lstm_truncid'].LoadModel(file)
        self.Optimizer.LoadModel(file)