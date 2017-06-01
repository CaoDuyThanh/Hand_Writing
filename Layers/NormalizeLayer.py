import theano.tensor as T
from UtilLayer import *

class NormalizeLayer():
    def __init__(self,
                 net,
                 input):
        # Save all information to its layer
        self.NormalizeScale = net.LayerOpts['normalize_scale']
        self.FilterShape    = net.LayerOpts['normalize_filter_shape']
        self.ScaleName      = net.LayerOpts['normalize_scale_name']

        self.Scale = CreateSharedParameter(
                        rng     = net.NetOpts['rng'],
                        shape   = self.FilterShape,
                        nameVar = self.ScaleName
                    )

        input2    = T.sqr(input)
        inputSum  = input2.sum(axis = 1, keepdims = True)
        inputSqrt = T.sqrt(inputSum)

        output_shape = T.shape(input)
        scaleReshape = self.Scale.reshape((1, self.FilterShape[0], 1, 1))
        scaleReshape = T.extra_ops.repeat(
            scaleReshape,
            output_shape[0],
            0)
        scaleReshape = T.extra_ops.repeat(
            scaleReshape,
            output_shape[2],
            2)
        scaleReshape = T.extra_ops.repeat(
            scaleReshape,
            output_shape[3],
            3)

        self.Output = input / inputSqrt * scaleReshape