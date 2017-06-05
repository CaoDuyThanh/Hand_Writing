import random
from Tkinter import *
from Models.LSTMModel import *
from Models.RNNModel import *
from Utils.DatasetUtil import *

# NETWORK CONFIG
NETWORK_TYPE = 'LSTM'
INPUT_SIZE   = 2
HIDDEN_SIZE  = 256
OUTPUT_SIZE  = 62

# PATH SETTINGS
BEST_PATH     = '../Pretrained/' + NETWORK_TYPE + '/Best.pkl'

# GLOBAL VARIABLES
Model = None

################################
#   CREATE TIME-SERIES MODEL   #
################################
def CreateModel():
    global Model
    if NETWORK_TYPE == 'LSTM':
        Model = LSTMModel(
                    inputSize  = INPUT_SIZE,
                    numHidden  = HIDDEN_SIZE,
                    outputSize = OUTPUT_SIZE
                )
    else:
        Model = RNNModel(
                    inputSize = INPUT_SIZE,
                    numHidden=HIDDEN_SIZE,
                    outputSize=OUTPUT_SIZE
                )

######################################
#              DEMO                  #
######################################
inputRaw = None
w        = None
def DrawCharacter():
    global Model, \
           Dataset, \
           inputRaw, \
           w

    Character = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'

    # Load model
    if CheckFileExist(BEST_PATH, throwError=False):
        file = open(BEST_PATH)
        Model.LoadModel(file)
        file.close()
        print ('Load best model !')

    canvas_width  = 900
    canvas_height = 600

    inputRaw = []
    def paint(event):
        global inputRaw

        python_green = "#476042"
        x1, y1 = (event.x - 1), (event.y - 1)
        x2, y2 = (event.x + 1), (event.y + 1)
        w.create_oval(x1, y1, x2, y2, fill=python_green)

        col = event.x / 900. - 0.5
        row = event.y / 600. - 0.5
        inputRaw.append([row, col])

    def KeyPress(event):
        global inputRaw, \
               Dataset,  \
               w

        if event.char == 'r':
            inputRaw = []
            w.delete('all')
        if event.char == 'e':
            input = numpy.asarray(inputRaw, dtype = 'float32')
            input = input.reshape((input.shape[0], 1, input.shape[1]))

            pred = Model.PredFunc(input)
            print Character[pred[0]]

    master = Tk()
    master.title("Painting using Ovals")
    w = Canvas(master,
               width  = canvas_width,
               height = canvas_height)
    w.pack(expand = NO, fill = BOTH)
    w.focus_set()
    w.bind('<Key>', KeyPress)
    w.bind('<B1-Motion>', paint)


    message = Label(master, text = "Press and Drag the mouse to draw")
    message.pack(side = BOTTOM)

    mainloop()

if __name__ == '__main__':
    CreateModel()
    DrawCharacter()