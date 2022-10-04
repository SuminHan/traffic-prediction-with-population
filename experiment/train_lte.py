import os, datetime, argparse, tqdm, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utils
import myltemodels as mymodels
import pickle
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


from os.path import join as pjoin

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = '../dataprocess/prepdata-1718') #'../dataprocess/prepdata2_1803-1807')
parser.add_argument('--region', type = str, default = 'gangnam')
parser.add_argument('--mark', type = str, default = '')
parser.add_argument('--cell_size', type=str, default='500', help='LTE cell_size (m).')
# parser.add_argument('--POIMODE', type=str2bool, default=True)

parser.add_argument('--model_name', type = str, default = 'MyConvLSTM')
parser.add_argument('--memo', type = str, default = '')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--P', type = int, default = 6)
parser.add_argument('--Q', type = int, default = 3)

parser.add_argument('--time_slot', type = int, default = 60, 
                    help = 'a time step is 60 mins')
parser.add_argument('--cnn_size', type = int, default = 3)
parser.add_argument('--L', type = int, default = 3,
                    help = 'number of STAtt Blocks')
parser.add_argument('--LZ', type = int, default = 2,
                    help = 'number of Regional STAtt Blocks')
parser.add_argument('--K', type = int, default = 8,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 8,
                    help = 'dims of each head attention outputs')
parser.add_argument('--D', type = int, default = 64)
parser.add_argument('--batch_size', type = int, default = 32,
                    help = 'batch size')
parser.add_argument('--max_epoch', type = int, default = 200,
                    help = 'epoch to run')
parser.add_argument('--patience', type = int, default = 10,
                    help = 'patience for early stop')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--decay_epoch', type=int, default = 5,
                    help = 'decay epoch')


args = parser.parse_args()

    
# print(os.listdir(args.data_dir))
args.dataset_name = args.data_dir.split('/')[-1]
args.test_name = args.memo + args.model_name + args.mark
args.model_checkpoint_dir = f'checkpoint/{args.dataset_name}/{args.region}-LTE/'
args.model_checkpoint = os.path.join(args.model_checkpoint_dir, args.test_name)
args.test_dir = f'test/{args.dataset_name}/{args.region}-LTE'

if not os.path.isdir(args.model_checkpoint_dir):
    os.makedirs(args.model_checkpoint_dir)
if not os.path.isdir(args.test_dir):
    os.makedirs(args.test_dir)


(trainX, trainZ, trainTE, trainY, trainZY, 
            valX, valZ, valTE, valY, valZY, 
            testX, testZ, testTE, testY, testZY,
            SE, SEZ, DIST_GG, DIST_GR, DIST_RR, mean, std, height, width, maxLTE) = utils.loadDataCell(args)
# mean = 0
# std = 1
print(trainX.shape, trainX.dtype, trainZ.shape, trainZ.dtype, trainTE.shape, trainTE.dtype)
print(valX.shape, valX.dtype, valZ.shape, valZ.dtype, valTE.shape, valTE.dtype)
print(testX.shape, testX.dtype, testZ.shape, testZ.dtype, testTE.shape, testTE.dtype)

pred = np.tile(np.expand_dims(testZ.mean(axis=1) * maxLTE, 1), [1, args.Q, 1])
label = testZY
print('Closeness Mean', utils.metric(pred, label), sep='\t')

pred = np.tile(testZ[:, -1:, :] * maxLTE, [1, args.Q, 1])
label = testZY
print('Closeness Last', utils.metric(pred, label), sep='\t')


pred = trainZY.mean(0)[np.newaxis, :, :]
label = testZY
print('Historical Avg', utils.metric(pred, label), sep='\t')


extdata = dict()
extdata['DIST_GG'] = DIST_GG
extdata['DIST_GR'] = None#DIST_GR
extdata['DIST_RR'] = DIST_RR
extdata['ADJ_GG'] = 1*(DIST_GG < 1) #utils.convert_to_adj_mx(DIST_GG)
extdata['ADJ_GR'] = None#utils.convert_to_adj_mx(DIST_GR)
extdata['ADJ_RR'] = utils.convert_to_adj_mx(DIST_RR)
extdata['mean'] = mean
extdata['std'] = std
extdata['SE'] = SE
extdata['SEZ'] = SEZ
extdata['num_nodes'] = DIST_GG.shape[0]
extdata['height'] = height
extdata['width'] = width


adj_gg = utils.row_normalize(extdata['ADJ_GG'])
adj_ggt = utils.row_normalize(extdata['ADJ_GG'].T)

def rwr_result(adj_gg):
    myeye = np.eye(extdata['num_nodes'])
    r = 0.15
    res = myeye
    for _ in range(100):
        res = (1-r) * (adj_gg @ res) + r * myeye
    return res

extdata['RWR1'] = utils.row_normalize(rwr_result(adj_gg).T)
extdata['RWR2'] = utils.row_normalize(rwr_result(adj_ggt).T)
# extdata['RWR1'] = utils.row_normalize(1*(rwr_result(adj_gg) > 0))
# extdata['RWR2'] = utils.row_normalize(1*(rwr_result(adj_ggt) > 0))


def model_define():
    ### model train ###
    Z = layers.Input(shape=trainZ.shape[1:], dtype=tf.float32)
    TE = layers.Input(shape=trainTE.shape[1:], dtype=tf.int32)

    Y = mymodels.ModelSet(model_name=args.model_name, extdata=extdata, args=args, Z=Z, TE=TE)
    model = keras.models.Model((Z, TE), Y)
    return model

model = model_define()
        
model.summary()
optimizer = keras.optimizers.Adam(lr=args.learning_rate)
model.compile(loss=tf.keras.metrics.mean_absolute_error, optimizer=optimizer)

# Define some callbacks to improve training.
early_stopping = keras.callbacks.EarlyStopping(monitor="val_loss", patience=21)
reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.2, patience=5, min_lr=0.0001)
model_ckpt = tf.keras.callbacks.ModelCheckpoint(args.model_checkpoint, save_weights_only=True, save_best_only=True, monitor='val_loss', mode='min', verbose=0)



model.fit(
    (trainZ, trainTE), trainZY,
    batch_size=args.batch_size,
    epochs=args.max_epoch,
    validation_data=((valZ, valTE), valZY),
    callbacks=[early_stopping, reduce_lr, model_ckpt],
)

model = model_define()
model.load_weights(args.model_checkpoint)

pred = model.predict((testZ, testTE))
label = testZY
print(f'{args.model_name}', utils.metric(pred, label), sep='\t')


np.save(f'{args.test_dir}/label.npy', label)
np.save(f'{args.test_dir}/pred_{args.test_name}.npy', pred)