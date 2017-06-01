import pickle
import cPickle
import theano.tensor as T
from UtilLayer import *
from theano.tensor.nnet import conv2d

class ConvLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.Input          = input
        self.InputShape     = net.LayerOpts['conv2D_input_shape']
        self.FilterShape    = net.LayerOpts['conv2D_filter_shape']
        self.FilterFlip     = net.LayerOpts['conv2D_filter_flip']
        self.BorderMode     = net.LayerOpts['conv2D_border_mode']
        self.Subsample      = net.LayerOpts['conv2D_stride']
        self.FilterDilation = net.LayerOpts['conv2D_filter_dilation']
        self.W              = net.LayerOpts['conv2D_W']
        self.WName          = net.LayerOpts['conv2D_WName']
        self.b              = net.LayerOpts['conv2D_b']
        self.bName          = net.LayerOpts['conv2D_bName']

        if self.W is None:
            self.W = CreateSharedParameter(
                        rng     = net.NetOpts['rng'],
                        shape   = self.FilterShape,
                        nameVar = self.WName
                    )

        if self.b is None:
            bShape = (self.FilterShape[0],)
            self.b = CreateSharedParameter(
                        rng     = net.NetOpts['rng'],
                        shape   = bShape,
                        nameVar = self.bName
                    )
        self.Params = [self.W, self.b]

        convOutput = conv2d(
                        input           = self.Input,
                        input_shape     = self.InputShape,
                        filters         = self.W,
                        filter_shape    = self.FilterShape,
                        border_mode     = self.BorderMode,
                        subsample       = self.Subsample,
                        filter_flip     = self.FilterFlip,
                        filter_dilation = self.FilterDilation
                    )

        output_shape = T.shape(convOutput)
        rp_biases = self.b.reshape((1, self.FilterShape[0], 1, 1))
        rp_biases = T.extra_ops.repeat(
            rp_biases,
            output_shape[0],
            0)
        rp_biases = T.extra_ops.repeat(
            rp_biases,
            output_shape[2],
            2)
        rp_biases = T.extra_ops.repeat(
            rp_biases,
            output_shape[3],
            3)
        self.Output = convOutput + rp_biases

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow = True) for param in self.Params]

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, -1) for param in self.Params]