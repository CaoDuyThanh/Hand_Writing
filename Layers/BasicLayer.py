import pickle
import cPickle

class BasicLayer():
    def __init__(self):
        # Do nothing here
        return

    def SaveModel(self, file):
        [pickle.dump(param.get_value(borrow = True), file, 0) for param in self.Params]

    def LoadModel(self, file):
        [param.set_value(cPickle.load(file), borrow=True) for param in self.Params]