import theano.tensor as T

class SigmoidLayer():
    def __init__(self,
                 input):
        # Save information to its layer
        self.Input = input

        self.Output = T.nnet.sigmoid(input)