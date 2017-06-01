import theano.tensor as T

class L2CostLayer():
    def __init__(self,
                 net,
                 pred,
                 target):
        # Save all information to its layer
        self.Axis = net.LayerOpts['l2cost_axis']

        out = T.mean(T.sqr(target - pred), axis=self.Axis, keepdims=True)

        self.Output = T.mean(out)