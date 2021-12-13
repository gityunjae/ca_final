import os
import json
os.environ["CUDA_VISIBLE_DEVICES"]="0"

# This is the path for mel-spectrograms
TMP_PATH = '/HDD/storage/ca_final/dataset/arena_mel'
# This is the path to save the model checkpoints
MODELS_PATH = '/home/yun/yj/컴청/icassp2021/runs'

# SET GPUs to use:
 #"0,1,2,3"

# General Imports
import argparse
import numpy as np
import random

# Deep Learning
import keras
import tensorflow
from tensorflow.keras import optimizers
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Convolution2D, MaxPooling2D, Dense, Dropout, Activation, Flatten, merge
from keras.layers import BatchNormalization
from keras.layers.advanced_activations import ELU, ReLU

# Machine Learning preprocessing and evaluation
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, auc, roc_auc_score, average_precision_score
from keras.callbacks import ModelCheckpoint
from keras.regularizers import l2
from keras.models import Model
from keras.layers import Input, Conv2D, MaxPooling2D, Concatenate
import math
from keras.layers.core import Layer, Dense, Dropout, Activation, Flatten, Reshape, Lambda
from keras.layers import BatchNormalization
import json

from cf_train import load_feats, save


def add_channel(data, n_channels=1):
    # n_channels: 1 for grey-scale, 3 for RGB, but usually already present in the data

    N, ydim, xdim = data.shape

    if keras.backend.image_data_format() == 'channels_last':  # TENSORFLOW
        # Tensorflow ordering (~/.keras/keras.json: "image_dim_ordering": "tf")
        data = data.reshape(N, ydim, xdim, n_channels)
    else: # THEANO
        # Theano ordering (~/.keras/keras.json: "image_dim_ordering": "th")
        data = data.reshape(N, n_channels, ydim, xdim)

    return data

def load_spectrograms(item_ids, enc=True):
    list_spectrograms = []
    ret_ids = []
    for p, kid in enumerate(item_ids):
        if enc:
            kid = kid.decode()
        filename = '{}/{}.npy'.format(kid[:-3], kid)
        if len(kid)<=3:
            filename = '{}/{}.npy'.format('0', kid)
        npz_spec_file = os.path.join(TMP_PATH, filename)
        if os.path.exists(npz_spec_file):
            melspec = np.load(npz_spec_file)
            if melspec.shape[1]< 1876:
                print (melspec.shape)
            else:
                list_spectrograms.append(melspec[:40,:1876])
                ret_ids.append(p)
        else:
            print ("File not exists", filename)
    item_list = np.array(list_spectrograms, dtype=K.floatx())
    item_list[np.isinf(item_list)] = 0
    item_list = add_channel(item_list)
    return item_list, ret_ids

def SubSpec(input_shape, n_mels, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout):
    ##melSize = Input(shape=input_shape)
    #melSize = x_train.shape[1]
    # Sub-Spectrogram Size
    splitSize = 20
    # Mel-bins overlap
    overlap = 10
    # Time Indices
    timeInd = 500
    # Channels used
    channels = 2
    #melSize = 48
    ####### Generate the model ###########
    #inputLayer = Input((melSize, timeInd, channels))
    inputLayer = Input(shape=input_shape)
    melSize = inputLayer.shape[1]
    subSize = int(splitSize / 10)
    i = 0
    outputs = []
    toconcat = []
    while (overlap * i <= melSize - splitSize):    # 48 - 20 = 28, i=3
        # Create Sub-Spectrogram
        INPUT = Lambda(lambda inputLayer: inputLayer[:, i * overlap:i * overlap + splitSize, :, :],
                       output_shape=(splitSize, timeInd, channels))(inputLayer)

        # First conv-layer -- 32 kernels
        CONV = Conv2D(32, kernel_size=(7, 7), padding='same', kernel_initializer="he_normal")(INPUT)
        CONV = BatchNormalization(axis=1)(CONV)
        CONV = Activation('relu')(CONV)

        # Max pool by SubSpectrogram <mel-bin>/10 size. For example for sub-spec of 30x500, max pool by 3 vertically.
        CONV = MaxPooling2D((subSize, 5))(CONV)
        CONV = Dropout(0.3)(CONV)

        # Second conv-layer -- 64 kernels
        CONV = Conv2D(64, kernel_size=(7, 7), padding='same',
                      kernel_initializer="he_normal")(CONV)
        CONV = BatchNormalization(axis=1)(CONV)
        CONV = Activation('relu')(CONV)

        # Max pool
        CONV = MaxPooling2D((4, 100))(CONV)
        CONV = Dropout(0.30)(CONV)

        # Flatten
        FLATTEN = Flatten()(CONV)

        OUTLAYER = Dense(32, activation='relu')(FLATTEN)
        DROPOUT = Dropout(0.30)(OUTLAYER)

        # Sub-Classifier Layer
        FINALOUTPUT = Dense(output_shape, activation='softmax')(DROPOUT)
        outputs.append(FINALOUTPUT)
        toconcat.append(OUTLAYER)

        i = i + 1

    x = Concatenate()(toconcat)

    # Automatically chooses appropriate number of hidden layers -- in a factor of 2.
    # For example if  the number of sub-spectrograms is 9, we have 9*32 neurons by
    # concatenating. So number of hidden layers would be 5 -- 512, 256, 128, 64
    numFCs = int(math.log(i * 32, 2))
    print(i * 32)
    print(numFCs)
    print(math.pow(2, numFCs))
    neurons = math.pow(2, numFCs)

    # Last layer before Softmax is a 64-neuron hidden layer
    while (neurons >= 64):
        x = Dense(int(neurons), activation='relu')(x)
        x = Dropout(0.30)(x)
        neurons = neurons / 2

    # softmax -- GLOBAL CLASSIFIER
    x = Dense(output_shape, activation=activation)(x)
    # Create model
    model = Model(inputLayer, x)

    return model

'''
def HCNN(input_shape, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout):
    # CNN using 1xm Horizontal filter
    # use two filters of different size and do depth-wise concatenation, and go through another conv
    # input: (48, 1876)
    melgram_input = Input(shape=input_shape)

    # Determine input axis
    if keras.backend.image_data_format() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=channel_axis, name='bn_0_freq')(melgram_input)

    if normalize == 'batch':
        pass
    elif normalize in ('data_sample', 'time', 'freq', 'channel'):
        x = Normalization2D(normalize, name='nomalization')(melgram_input)
    elif normalize in ('no', 'False'):
        x = melgram_input

    # Conv 10 times..?
    x1 = Convolution2D(1, 16)(x)
    x1 = BatchNormalization(axis=channel_axis, name='bn1')(x1)
    x1 = ReLU()(x1)
    x2 = Convolution2D(1, 26)(x)
    x2 = BatchNormalization(axis=channel_axis, name='bn2')(x2)
    x2 = ReLU()(x2)
    x3 = Convolution2D(1, 50)(x)
    x3 = BatchNormalization(axis=channel_axis, name='bn3')(x3)
    x3 = ReLU()(x3)
    x4 = Convolution2D(1, 76)(x)
    x4 = BatchNormalization(axis=channel_axis, name='bn4')(x4)
    x4 = ReLU()(x4)
    x5 = Convolution2D(1, 96)(x)
    x5 = BatchNormalization(axis=channel_axis, name='bn5')(x5)
    x5 = ReLU()(x5)
    x6 = Convolution2D(1, 120)(x)
    x6 = BatchNormalization(axis=channel_axis, name='bn6')(x6)
    x6 = ReLU()(x6)
    x7 = Convolution2D(1, 146)(x)
    x7 = BatchNormalization(axis=channel_axis, name='bn7')(x7)
    x7 = ReLU()(x7)
    x8 = Convolution2D(1, 170)(x)
    x8 = BatchNormalization(axis=channel_axis, name='bn8')(x8)
    x8 = ReLU()(x8)
    x9 = Convolution2D(1, 196)(x)
    x9 = BatchNormalization(axis=channel_axis, name='bn9')(x9)
    x9 = ReLU()(x9)
    x10 = Convolution2D(1, 220)(x)
    x10 = BatchNormalization(axis=channel_axis, name='bn10')(x10)
    x10 = ReLU()(x10)

    # concat 10 results..?
    x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim=2)

    x = MaxPooling2D(pool_size=(10, 4), name='pool1')(x)

    # 2nd Conv
    x = Convolution2D(7, 7)(x)
    x = BatchNormalization(axis=channel_axis, name='bn22')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(2, 2), name='pool2')(x)

    return x

def PCNN(input_shape, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout):
    # CNN using nx1 Vertical filter
    # use two filters of different size and do depth-wise concatenation, and go through another conv
    melgram_input = Input(shape=input_shape)

    # Determine input axis
    if keras.backend.image_data_format() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=channel_axis, name='bn_0_freq')(melgram_input)

    if normalize == 'batch':
        pass
    elif normalize in ('data_sample', 'time', 'freq', 'channel'):
        x = Normalization2D(normalize, name='nomalization')(melgram_input)
    elif normalize in ('no', 'False'):
        x = melgram_input

    # Conv 10 times..?
    x1 = Convolution2D(2, 1)(x)
    x1 = BatchNormalization(axis=channel_axis, name='bn1')(x1)
    x1 = ReLU()(x1)
    x2 = Convolution2D(4, 1)(x)
    x2 = BatchNormalization(axis=channel_axis, name='bn2')(x2)
    x2 = ReLU()(x2)
    x3 = Convolution2D(6, 1)(x)
    x3 = BatchNormalization(axis=channel_axis, name='bn3')(x3)
    x3 = ReLU()(x3)
    x4 = Convolution2D(8, 1)(x)
    x4 = BatchNormalization(axis=channel_axis, name='bn4')(x4)
    x4 = ReLU()(x4)
    x5 = Convolution2D(10, 1)(x)
    x5 = BatchNormalization(axis=channel_axis, name='bn5')(x5)
    x5 = ReLU()(x5)
    x6 = Convolution2D(12, 1)(x)
    x6 = BatchNormalization(axis=channel_axis, name='bn6')(x6)
    x6 = ReLU()(x6)
    x7 = Convolution2D(14, 1)(x)
    x7 = BatchNormalization(axis=channel_axis, name='bn7')(x7)
    x7 = ReLU()(x7)
    x8 = Convolution2D(16, 1)(x)
    x8 = BatchNormalization(axis=channel_axis, name='bn8')(x8)
    x8 = ReLU()(x8)
    x9 = Convolution2D(18, 1)(x)
    x9 = BatchNormalization(axis=channel_axis, name='bn9')(x9)
    x9 = ReLU()(x9)
    x10 = Convolution2D(24, 1)(x)
    x10 = BatchNormalization(axis=channel_axis, name='bn10')(x10)
    x10 = ReLU()(x10)

    # concat 10 results..?
    x = torch.cat([x1, x2, x3, x4, x5, x6, x7, x8, x9, x10], dim=2)

    x = MaxPooling2D(pool_size=(10, 4), name='pool1')(x)

    # 2nd Conv
    x = Convolution2D(5, 5)(x)
    x = BatchNormalization(axis=channel_axis, name='bn22')(x)
    x = ReLU()(x)
    x = MaxPooling2D(pool_size=(10, 2), name='pool2')(x)

    return x

def biCNN():
    # implemented based on paper, 'Acoustic Scene Classification Using Bilinear Pooling on Time-liked and
    # Frequency-liked Convolution Neural Network
    # HCNN과 PCNN을 각각 받아서 Bilinear Pooling을 한 후 softmax 적용
    return
'''
def CompactCNN(input_shape, nb_conv, nb_filters, n_mels, normalize, nb_hidden, dense_units,
               output_shape, activation, dropout, multiple_segments=False, graph_model=False, input_tensor=None):

    melgram_input = Input(shape=input_shape)

    if n_mels >= 256:
        poolings = [(2, 4), (4, 4), (4, 5), (2, 4), (4, 4)]
    elif n_mels >= 128:
        poolings = [(2, 4), (4, 5), (4, 8), (4, 7), (4, 4)]
    elif n_mels >= 96:
        poolings = [(2, 4), (4, 5), (3, 8), (4, 7), (4, 3)] #(2, 8), (4, 3)]
    elif n_mels >= 72:
        poolings = [(2, 4), (3, 4), (2, 5), (2, 4), (3, 4)]
    elif n_mels >= 64:
        poolings = [(2, 4), (2, 4), (2, 5), (2, 4), (4, 4)]
    elif n_mels >= 48: # 우리의 케이스로 보임)
        poolings = [(2, 4), (4, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 32:
        poolings = [(2, 4), (2, 5), (3, 8), (2, 7), (4, 4)]
    elif n_mels >= 24:
        poolings = [(2, 4), (2, 4), (3, 8), (2, 8), (4, 4)]
    elif n_mels >= 16:
        poolings = [(2, 4), (2, 5), (2, 8), (2, 7), (4, 4)]
    elif n_mels >= 8:
        poolings = [(2, 4), (2, 4), (2, 8), (1, 8), (4, 4)]

    # Determine input axis
    if keras.backend.image_data_format() == 'th':
        channel_axis = 1
        freq_axis = 2
        time_axis = 3
    else:
        channel_axis = 3
        freq_axis = 1
        time_axis = 2

    # Input block
    x = BatchNormalization(axis=channel_axis, name='bn_0_freq')(melgram_input)

    if normalize == 'batch':
        pass
    elif normalize in ('data_sample', 'time', 'freq', 'channel'):
        x = Normalization2D(normalize, name='nomalization')(melgram_input)
    elif normalize in ('no', 'False'):
        x = melgram_input

    # Conv block 1
    x = Convolution2D(nb_filters[0], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn1')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[0], name='pool1')(x)

    # Conv block 2
    x = Convolution2D(nb_filters[1], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn2')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[1], name='pool2')(x)

    # Conv block 3
    x = Convolution2D(nb_filters[2], (3, 3), padding='same')(x)
    x = BatchNormalization(axis=channel_axis, name='bn3')(x)
    x = ELU()(x)
    x = MaxPooling2D(pool_size=poolings[2], name='pool3')(x)

    # Conv block 4
    if nb_conv > 3:
        x = Convolution2D(nb_filters[3], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn4')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=poolings[3], name='pool4')(x)

    # Conv block 5
    if nb_conv == 5:
        x = Convolution2D(nb_filters[4], (3, 3), padding='same')(x)
        x = BatchNormalization(axis=channel_axis, name='bn5')(x)
        x = ELU()(x)
        x = MaxPooling2D(pool_size=poolings[4], name='pool5')(x)

    # Flatten the outout of the last Conv Layer
    x = Flatten()(x)

    if nb_hidden == 1:
        x = Dropout(dropout)(x)
        x = Dense(dense_units, activation='relu')(x)
    elif nb_hidden == 2:
        x = Dropout(dropout)(x)
        x = Dense(dense_units[0], activation='relu')(x)
        x = Dropout(dropout)(x)
        x = Dense(dense_units[1], activation='relu')(x)
    else:
        pass

    # Output Layer
    x = Dense(output_shape, activation=activation, name = 'output')(x)

    # Create model
    model = Model(melgram_input, x)

    return model

class TestCallback(keras.callbacks.Callback):
    def __init__(self, test_data, test_classes, val_set, val_classes):
        self.test_data = test_data
        self.test_classes = test_classes
        self.val_data = val_set
        self.val_classes = val_classes

    def on_epoch_end(self, epoch, logs={}):
        #if (epoch+1) % 5 ==0:
        test_pred_prob = self.model.predict(self.test_data)
        roc_auc = 0
        pr_auc = 0
        for i in range(50):
            roc_auc += roc_auc_score(self.test_classes[:, i], test_pred_prob[:, i])
            pr_auc += average_precision_score(self.test_classes[:, i], test_pred_prob[:, i])
        print('Test:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))

        val_pred_prob = self.model.predict(self.val_data)
        roc_auc = 0
        pr_auc = 0
        for i in range(50):
            roc_auc += roc_auc_score(self.val_classes[:, i], val_pred_prob[:, i])
            pr_auc += average_precision_score(self.val_classes[:, i], val_pred_prob[:, i])
        print('Validation:')
        print('Epoch: '+str(epoch)+' ROC-AUC '+str(roc_auc/50)+' PR-AUC '+str(pr_auc/50))


def batch_block_generator(train_set, item_vecs_reg, batch_size=32, dimms="200"):
    block_step = 50000
    n_train = len(train_set)
    randomize = True
    while 1:
        for i in range(0, n_train, block_step):
            npy_train_mtrx_x = os.path.join(MODELS_PATH, 'repr_x_{}.npy'.format(i))
            npy_train_mtrx_y = os.path.join(MODELS_PATH, 'repr_y_{}_{}_fm.npy'.format(dimms,i))
            msdid_block = train_set[i:min(n_train, i+block_step)]
            if os.path.exists(npy_train_mtrx_y):
                x_block = np.load(npy_train_mtrx_x)
                x_block = x_block[:, :40, :, :]  #####
                y_block = np.load(npy_train_mtrx_y)
            else:
                x_block, loaded_positions = load_spectrograms(msdid_block)
                x_block = x_block[:,:40,:,:] #####
                y_block = item_vecs_reg[[i+p for p in loaded_positions]]
                np.save(npy_train_mtrx_x, x_block)
                np.save(npy_train_mtrx_y, y_block)
            items_list = list(range(x_block.shape[0]))
            if randomize:
                random.shuffle(items_list)
            for j in range(0, len(items_list), batch_size):
                if j+batch_size <= x_block.shape[0]:
                    items_in_batch = items_list[j:j+batch_size]
                    x_batch = x_block[items_in_batch,:,:,:]
                    y_batch = y_block[items_in_batch]
                    yield (x_batch, y_batch)


def step_decay(epoch):
    initial_lrate = 0.001
    drop = 0.1
    epochs_drop = 100.0
    lrate = initial_lrate * math.pow(drop,
        math.floor((1+epoch)/epochs_drop))
    return lrate

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(
        description='Train the model',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-d',
                        '--dim',
                        dest="dimms",
                        help='Dimension of the leatures',
                        type=str,
                        default="300")


    args = parser.parse_args()
    model_folder = "models_split"
    # train에서 10%를 test로 설정해두고 그것과 독립적으로 10%를 val set으로 설정한듯..? 왜 따로따로 설정 안했지
    item_features_file = os.path.join(model_folder, 'cf_item_{}_{}.feats'.format(args.dimms, 'train'))
    item_ids, item_vecs_reg =  load_feats(item_features_file)
    train_item_vecs, test_item_vecs, train_item_ids, test_item_ids = train_test_split(item_vecs_reg, item_ids, test_size=0.10, random_state=42)
    train_len = len(train_item_vecs)
    print("train data: "+str(train_len))
    test_len = len(test_item_vecs)
    print("test data: " + str(test_len))
    test_data, test_positions = load_spectrograms(test_item_ids)
    save([test_item_ids[i].decode() for i in test_positions], test_item_vecs[test_positions], os.path.join(model_folder, 'valid_orig_cf.npy'))
    print ("Finished loading CF features")

    input_shape = test_data[0,:,:,:].shape
    print (test_data.shape)
    print ("Input shape: ", input_shape)

    # the loss in this case MSE
    loss = 'mean_squared_error'

    # number of Convolutional Layers
    nb_conv_layers = 4

    # number of Filters in each layer
    nb_filters = [128, 384, 768, 2048]
    # nb_filters = [32, 96, 192, 512]

    # number of hidden layers at the end of the model
    nb_hidden = 0
    dense_units = 200
    # dense_units = 50

    # which activation function to use for OUTPUT layer
    # IN A MULTI-LABEL TASK with N classes we use SIGMOID activation same as with a BINARY task
    # as EACH of the classes can be 0 or 1
    output_activation = 'linear'

    # which type of normalization
    normalization = 'batch'

    # droupout
    dropout = 0


    # how many output units
    # IN A SINGLE-LABEL MULTI-CLASS or MULTI-LABEL TASK with N classes, we need N output units
    #output_shape = 64
    output_shape = item_vecs_reg.shape[1]

    # Optimizers

    # simple case:
    # Stochastic Gradient Descent
    #optimizer = 'sgd'

    # advanced:
    sgd = optimizers.SGD(momentum=0.9, nesterov=True)

    # We use mostly ADAM
    adam = optimizers.Adam(lr=0.001) #0.001

    metrics = ['accuracy']

    # Optimizer
    optimizer = adam

    batch_size = 32 # batch for train
    tb = 32 # batch for test

    # epochs = 20

    random_seed = 0

    import tensorflow as tf
    # 이 부분을 Bilinear CNN으로 수정..!
    # model = CompactCNN
    model = SubSpec(input_shape, n_mels = input_shape[0],
                               normalize=normalization,
                               nb_hidden = nb_hidden, dense_units = dense_units,
                               output_shape = output_shape, activation = output_activation,
                               dropout = dropout)
    model.summary()

    # COMPILE MODEL
    model.compile(loss=loss, metrics=metrics, optimizer=optimizer)

    # past_epochs is only for the case that we execute the next code box multiple times (so that Tensorboard is displaying properly)
    past_epochs = 0

    model_checkpoint = keras.callbacks.ModelCheckpoint(os.path.join(MODELS_PATH, "model_vgg_{}_fm_200.h5".format(args.dimms)), monitor='val_loss', save_best_only=True, mode='max')
    callbacks = [model_checkpoint]
    # START TRAINING
    # epochs = 200
    # epochs = 50
    # epochs = 3
    epochs = 100

    history = model.fit(batch_block_generator(item_ids, item_vecs_reg, batch_size, dimms=args.dimms),
                                                               steps_per_epoch = int(len(item_ids)/batch_size),
                                                               validation_data = (test_data, test_item_vecs[test_positions]),
                                                               epochs=epochs,
                                                               verbose=2,
                                                               initial_epoch=past_epochs,
                                                               callbacks=callbacks
                                                               )

    """
    model.load_weights(os.path.join(MODELS_PATH, "model_vgg_{}_fm_200.h5".format(args.dimms)))
    """

    test_pred = model.predict(test_data, batch_size=tb)
    save([test_item_ids[i].decode() for i in test_positions], test_pred, os.path.join(model_folder, 'test_pred.npy'))
    for split in ['train', '8', '5', '1']:
        tr_ids = json.load(open(os.path.join(model_folder, 'track_ids_{}.json'.format(split)), 'r'))
        #test_ids_1 = tr_ids[81219:]
        test_ids_1 = tr_ids[train_len:]
        for i in range(0, len(test_ids_1), tb):
            test_data_1, test_positions_1 = load_spectrograms([str(x) for x in test_ids_1[i:i+tb]], enc=False)
            test_pred_1 = model.predict(test_data_1, tb)
            save([str(test_ids_1[i+j]) for j in test_positions_1], test_pred_1, os.path.join(model_folder, 'test_pred_{}_{}.npy'.format(split, i)))


