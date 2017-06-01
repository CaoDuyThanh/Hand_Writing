from theano.tensor.signal.pool import pool_2d

class Pool2DLayer():
    def __init__(self,
                 net,
                 input):
        # Save config information to its layer
        self.Ws           = net.LayerOpts['pool_filter_size']
        self.IgnoreBorder = net.LayerOpts['pool_ignore_border']
        self.Stride       = net.LayerOpts['pool_stride']
        self.Padding      = net.LayerOpts['pool_padding']
        self.Mode         = net.LayerOpts['pool_mode']

        self.Output = pool_2d(input         = input,
                              ws            = self.Ws,
                              ignore_border = self.IgnoreBorder,
                              stride        = self.Stride,
                              pad           = self.Padding,
                              mode          = self.Mode)