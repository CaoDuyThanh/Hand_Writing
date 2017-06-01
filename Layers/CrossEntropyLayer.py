import theano.tensor as T

class CrossEntropyLayer():
    def __init__(self,
                 pred,
                 target):
        self.Output = - target * T.log(pred) - (1 - target) * T.log(1 - pred)