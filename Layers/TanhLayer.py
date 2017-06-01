import theano.tensor as T

class TanhLayer():
    def __init__(self,
                 input):
        # Save information to its layer
        self.Input = input

        self.Output = T.tanh(input)