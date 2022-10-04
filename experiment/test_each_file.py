import os, datetime, argparse, tqdm, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utils
import mymodels

from os.path import join as pjoin


def str2bool(v): 
    if isinstance(v, bool): 
        return v 
    if v.lower() in ('yes', 'true', 't', 'y', '1'): 
        return True 
    elif v.lower() in ('no', 'false', 'f', 'n', '0'): 
        return False 
    else: raise argparse.ArgumentTypeError('Boolean value expected.')

parser = argparse.ArgumentParser()
parser.add_argument('--data_dir', type = str, default = '../dataprocess/prepdata-1803-1808') # prepdata2_1803-1807
parser.add_argument('--mark', type = str, default = 'aaai')
parser.add_argument('--region', type = str, default = 'gangnam')
parser.add_argument('--memo', type = str, default = '')
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--P', type = int, default = 12)
parser.add_argument('--Q', type = int, default = 3)
parser.add_argument('--time_slot', type = int, default = 60, 
                    help = 'a time step is 60 mins')
parser.add_argument('--cell_size', type = int, default = 500)
parser.add_argument('--save_dir', default = 'data',
                    help = 'save_dir')
parser.add_argument('--NOBIM', default = False, type=str2bool,
                    help = 'NOBIM')
parser.add_argument('--NODYC', default = False, type=str2bool,
                    help = 'NODYC')
parser.add_argument('--NOLTE', default = False, type=str2bool,
                    help = 'NOLTE')
parser.add_argument('--NOPOI', default = False, type=str2bool,
                    help = 'NOPOI')
parser.add_argument('--NOSAT', default = False, type=str2bool,
                    help = 'NOSAT')
parser.add_argument('--lambda_value', default = 0.6, type=float,
                    help = 'lambda_value')

args = parser.parse_args()

# (trainX, trainZ, trainTE, trainY, trainZY, 
#             valX, valZ, valTE, valY, valZY, 
#             testX, testZ, testTE, testY, testZY,
#             SE, SEZ, DIST_GR, ADJ_DY, DIST_GG, mean, std, height, width) = utils.loadData(args)


# print(os.listdir(args.data_dir))

print(args.region)

skip_models = []# ['DNN', 'MyDCGRU_GST', 'MyGRU', 'MyLSTM']

args.dataset_name = args.data_dir.split('/')[-1]
args.test_dir = f'test_exp/'

label = np.load(os.path.join(args.test_dir, 'label.npy'))

for fname in sorted(os.listdir(args.test_dir)):
    skip = True
    for w in ['MyGMSTARK']:
        # if w in fname:
        #     skip = True
        #     break
        skip = False
    if 'pred' in fname and not skip:
        pred_tmp = np.load(os.path.join(args.test_dir, fname))
        print((fname[5:-4] + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in utils.metric(pred_tmp, label)), sep='\t')