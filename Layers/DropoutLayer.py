import theano

class DropoutLayer():
    def __init__(self,
                 net,
                 input):
        # Save information config to its layer
        self.DropRate = net.LayerOpts['drop_rate']

        theanoRng = net.NetOpts['theano_rng']
        self.Output = theanoRng.binomial(size  = input.shape,
                                         n     = 1,
                                         p     = 1 - self.DropRate,
                                         dtype = theano.config.floatX) * input