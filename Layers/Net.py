import numpy
import theano
import caffe
from theano.tensor.shared_randomstreams import RandomStreams

class NeuralNet():
    def __init__(self):
        # Setting default options for NetOpts
        self.NetOpts = {}
        self.NetOpts['rng_seed']         = 1610
        self.NetOpts['rng']              = numpy.random.RandomState(self.NetOpts['rng_seed'])
        self.NetOpts['theano_rng']       = RandomStreams(self.NetOpts['rng'].randint(2 ** 30))
        self.NetOpts['learning_rate']    = numpy.asarray(0.0005, theano.config.floatX)
        self.NetOpts['batch_size']       = 1

        self.LayerOpts = {}
        # Default options for softmax layers
        self.LayerOpts['softmax_axis'] = 1

        # Default options for relu layers
        self.LayerOpts['relu_alpha'] = 0.01

        # Deafult options for elu layers
        self.LayerOpts['elu_alpha'] = 1

        # Default l2 term
        self.LayerOpts['l2_term'] = 0.0005

        # Default l2 cost layer
        self.LayerOpts['l2cost_axis'] = 1

        # Default dropping rate for dropout
        self.LayerOpts['drop_rate']       = 0.5

        # Default options for hidden layer
        self.LayerOpts['hidden_input_size']  = 100
        self.LayerOpts['hidden_output_size'] = 100
        self.LayerOpts['hidden_W']           = None
        self.LayerOpts['hidden_WName']       = ''
        self.LayerOpts['hidden_b']           = None
        self.LayerOpts['hidden_bName']       = ''

        # Default options for reshape layer
        self.LayerOpts['reshape_new_shape']  = None

        # Permute layer
        self.LayerOpts['permute_dimension']  = None

        # Flatten layer
        self.LayerOpts['flatten_ndim'] = 2

        # Normalize layer
        self.LayerOpts['normalize_scale']        = 1
        self.LayerOpts['normalize_filter_shape'] = 1
        self.LayerOpts['normalize_'] = 1

        # Concatenate layer
        self.LayerOpts['concatenate_axis'] = 1

        self.UpdateOpts = {}
        # Adam update
        self.UpdateOpts['adam_beta1'] = 0.9
        self.UpdateOpts['adam_beta2'] = 0.999
        self.UpdateOpts['adam_delta'] = 0.000001


        # Network name for saving
        self.NetName = 'SimpleNet'

        # The content dictionary will store actual layers (LayerHelper)
        self.Layer  = {}
        self.Params = []

    def LoadCaffeModel(self,
                       caffePrototxtPath,
                       caffeModelPath):
        netCaffe = caffe.Net(caffePrototxtPath, caffeModelPath, caffe.TEST)
        layersCaffe = dict(zip(list(netCaffe._layer_names), netCaffe.layers))

        for name, layer in self.Layer.items():
            try:
                if name not in layersCaffe:
                    continue
                if name == 'conv4_3_norm':
                    layer.Scale.set_value(layersCaffe[name].blobs[0].data)
                layer.W.set_value(layersCaffe[name].blobs[0].data)
                layer.b.set_value(layersCaffe[name].blobs[1].data)
            except AttributeError:
                continue

class ConvNeuralNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for conv layers
        self.LayerOpts['conv2D_input_shape']     = None
        self.LayerOpts['conv2D_filter_shape']    = (32, 3, 3, 3)
        self.LayerOpts['conv2D_W']               = None
        self.LayerOpts['conv2D_WName']           = ''
        self.LayerOpts['conv2D_b']               = None
        self.LayerOpts['conv2D_bName']           = ''
        self.LayerOpts['conv2D_border_mode']     = 'valid'
        self.LayerOpts['conv2D_stride']          = (1, 1)
        self.LayerOpts['conv2D_filter_flip']     = False
        self.LayerOpts['conv2D_filter_dilation'] = (1, 1)

        # Default options for pooling layers
        self.LayerOpts['pool_stride']        = (2,2)
        self.LayerOpts['pool_padding']       = (0,0)
        self.LayerOpts['pool_mode']          = 'max'
        self.LayerOpts['pool_filter_size']   = (2,2)
        self.LayerOpts['pool_ignore_border'] = False

        # Network name for saving
        self.NetName = 'ConvNet'

class LSTMNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for lstm layers
        self.LayerOpts['lstm_num_hidden']   = 500
        self.LayerOpts['lstm_inputs_size']  = None
        self.LayerOpts['lstm_num_truncate'] = 20
        self.LayerOpts['lstm_params']       = None

        # Network name for saving
        self.NetName = 'LSTMNet'

class RNNNet(NeuralNet):
    def __init__(self):
        NeuralNet.__init__(self)

        # Setting default options for layer_opts
        # Default options for rnn layers
        self.LayerOpts['rnn_num_hidden']   = 500
        self.LayerOpts['rnn_inputs_size']  = None
        self.LayerOpts['rnn_num_truncate'] = 20
        self.LayerOpts['rnn_params']       = None

        # Network name for saving
        self.NetName = 'RNNNet'