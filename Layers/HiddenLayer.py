import theano.tensor as T
from UtilLayer import *

class HiddenLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.Input = input
        self.InputSize   = net.LayerOpts['hidden_input_size']
        self.OutputSize  = net.LayerOpts['hidden_output_size']
        self.W           = net.LayerOpts['hidden_W']
        self.WName       = net.LayerOpts['hidden_WName']
        self.b           = net.LayerOpts['hidden_b']
        self.bName       = net.LayerOpts['hidden_bName']

        if self.W is None:
            self.W = CreateSharedParameter(
                        rng     = net.NetOpts['rng'],
                        shape   = (self.InputSize, self.OutputSize),
                        nameVar = self.WName
                    )

        if self.b is None:
            self.b = CreateSharedParameter(
                        rng     = net.NetOpts['rng'],
                        shape   = (self.OutputSize,),
                        nameVar = self.bName
                    )

        self.Params = [self.W, self.b]

        self.Output = T.dot(input, self.W) + self.b
