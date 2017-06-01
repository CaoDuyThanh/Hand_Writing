import theano.tensor as T
import numpy

class SoftmaxLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.Axis    = net.LayerOpts['softmax_axis']
        self.Input   = input

        e_x = T.exp(input - input.max(axis = self.Axis, keepdims = True))
        out = e_x / e_x.sum(axis = self.Axis, keepdims = True)

        self.Output = out