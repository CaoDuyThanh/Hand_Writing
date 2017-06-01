import theano.tensor as T
import pickle
import cPickle
from UtilLayer import *
from BasicLayer import *

class LSTMLayer(BasicLayer):
    def __init__(self,
                 net):
        # Save all information to its layer
        self.NetName     = net.NetName
        self.NumHidden   = net.LayerOpts['lstm_num_hidden']
        self.InputSize   = net.LayerOpts['lstm_inputs_size']
        self.OutputSize  = net.LayerOpts['lstm_outputs_size']
        self.Params      = net.LayerOpts['lstm_params']

        if self.Params is None:
            # Parameters for list of input
            # Init Wi | Wf | Wc | Wo
            Wi = CreateSharedParameter(net.NetOpts['rng'], (self.InputSize, self.NumHidden), 1, '%s_Wi' % (self.NetName))
            Wf = CreateSharedParameter(net.NetOpts['rng'], (self.InputSize, self.NumHidden), 1, '%s_Wf' % (self.NetName))
            Wc = CreateSharedParameter(net.NetOpts['rng'], (self.InputSize, self.NumHidden), 1, '%s_Wc' % (self.NetName))
            Wo = CreateSharedParameter(net.NetOpts['rng'], (self.InputSize, self.NumHidden), 1, '%s_Wo' % (self.NetName))

            # Init Ui | bi
            Ui = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, '%s_Ui' % (self.NetName))
            bi = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, '%s_bi' % (self.NetName))

            # Init Uf | bf
            Uf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, '%s_Uf' % (self.NetName))
            bf = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, '%s_bf' % (self.NetName))

            # Init Uc | bc
            Uc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, '%s_Uc' % (self.NetName))
            bc = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, '%s_bc' % (self.NetName))

            # Init Uo | bo
            Uo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.NumHidden), 1, '%s_Uo' % (self.NetName))
            bo = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, )              , 0, '%s_bo' % (self.NetName))

            # Parameters for list of output
            Wy = CreateSharedParameter(net.NetOpts['rng'], (self.NumHidden, self.OutputSize), 1, '%s_Wy' % (self.NetName))
            by = CreateSharedParameter(net.NetOpts['rng'], (self.OutputSize,), 0, '%s_by' % (self.NetName))

            self.Params = [Wi, Wf, Wc, Wo] + \
                          [Wy, by] + \
                          [Ui, Uf, Uc, Uo] + \
                          [bi, bf, bc, bo]

    def Step(self,
             inputs,
             ckm1,
             hkm1,
             output):
        # Get all weight from param
        Wi = self.Params[0]
        Wf = self.Params[1]
        Wc = self.Params[2]
        Wo = self.Params[3]
        Wy = self.Params[4]
        by = self.Params[5]
        Ui = self.Params[6]
        Uf = self.Params[7]
        Uc = self.Params[8]
        Uo = self.Params[9]
        bi = self.Params[10]
        bf = self.Params[11]
        bc = self.Params[12]
        bo = self.Params[13]

        inputI = T.dot(inputs, Wi)
        inputF = T.dot(inputs, Wf)
        inputO = T.dot(inputs, Wo)
        inputG = T.dot(inputs, Wc)

        # Calculate to next layer
        i = T.nnet.sigmoid(inputI + T.dot(hkm1, Ui) + bi)
        f = T.nnet.sigmoid(inputF + T.dot(hkm1, Uf) + bf)
        o = T.nnet.sigmoid(inputO + T.dot(hkm1, Uo) + bo)
        g = T.tanh(inputG + T.dot(hkm1, Uc) + bc)

        C = ckm1 * f + g * i
        H = T.tanh(C) * o
        Output = T.dot(H, Wy) + by

        return C, H, Output