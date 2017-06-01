import theano.tensor as T

class ConcatLayer():
    def __init__(self,
                 net,
                 inputs):
        # Save all information to its layer
        self.Axis = net.LayerOpts['concatenate_axis']

        self.Output = T.concatenate(tuple(inputs), axis = self.Axis)