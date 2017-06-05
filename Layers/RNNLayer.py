import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class RNNLayer(BasicLayer):
    def __init__(self,
                 net):
        # Save all information to its layer
        self.NetName     = net.NetName
        self.NumHidden   = net.LayerOpts['rnn_num_hidden']
        self.InputSize   = net.LayerOpts['rnn_inputs_size']
        self.OutputSize  = net.LayerOpts['rnn_outputs_size']
        self.Params      = net.LayerOpts['rnn_params']

        if self.Params is None:
            # Parameters for input
            # Init Wi
            Wi = CreateSharedParameter(net.NetOpts['rng'], (self.InputSize, self.NumHidden), 1, '%s_Wi' % (self.NetName))

            # Init Ui | bi
            Ui = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, '%s_Ui' % (self.NetName))
            bi = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, '%s_bi' % (self.NetName))

            # Parameters for output
            Wy = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.OutputSize), 1, '%s_Wy' % (self.NetName))
            by = CreateSharedParameter(net.NetOpts['rng'], (self.OutputSize,), 0, '%s_by' % (self.NetName))

            self.Params = [Wi] + \
                          [Wy, by] + \
                          [Ui] + \
                          [bi]

    def Step(self,
             inputs,
             hkm1,
             output):
        # Get all weight from param
        Wi = self.Params[0]
        Wy = self.Params[4]
        by = self.Params[5]
        Ui = self.Params[6]
        bi = self.Params[10]

        H      = T.dot(inputs, Wi) + T.dot(hkm1, Ui) + bi
        Output = T.dot(H, Wy) + by

        return H, Output