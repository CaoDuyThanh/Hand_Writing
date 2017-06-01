import theano.tensor as T

class FlattenLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.NDim    = net.LayerOpts['flatten_ndim']

        self.Output = input.flatten(ndim = self.NDim)