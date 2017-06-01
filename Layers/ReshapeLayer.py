import theano.tensor as T

class ReshapeLayer():
    def __init__(self,
                 net,
                 input):
        # Save all information
        self.NewShape = net.LayerOpts['reshape_new_shape']

        shapeInput = input.shape

        newShape = []
        pos = -1
        for idx, shape in enumerate(self.NewShape):
            if shape == -1:
                newShape += 1
                pos = idx
            else:
                if shape == 0:
                    newShape.append(shapeInput[idx])
                else:
                    newShape.append(shape)
        if pos >= 0:
            newShape[pos] = T.prod(shapeInput) / T.prod(newShape)

        self.Output = input.reshape(tuple(newShape))