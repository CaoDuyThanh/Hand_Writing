import theano

class PermuteLayer():
    def __init__(self,
                 net,
                 input):
        # Save information to its layer
        self.PermuteDimension = net.LayerOpts['permute_dimension']

        self.Output = input.dimshuffle(self.PermuteDimension)