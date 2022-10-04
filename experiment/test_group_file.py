import os, datetime, argparse, tqdm, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utils
import mymodels

from os.path import join as pjoin

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
parser.add_argument('--cell_size', type = int, default = 1000)

args = parser.parse_args()

# (trainX, trainZ, trainTE, trainY, trainZY, 
#             valX, valZ, valTE, valY, valZY, 
#             testX, testZ, testTE, testY, testZY,
#             SE, SEZ, DIST_GG, DIST_GR, DIST_RR, mean, std) = utils.loadData(args)


# print(os.listdir(args.data_dir))

print(args.region)

skip_models = []# ['DNN', 'MyDCGRU_GST', 'MyGRU', 'MyLSTM']

args.dataset_name = args.data_dir.split('/')[-1]
args.test_dir = f'test_exp/'

label = np.load(os.path.join(args.test_dir, 'label.npy'))

group_results = dict()
for fname in sorted(os.listdir(args.test_dir)):
    skip = False
    for w in skip_models:
        if w in fname:
            skip = True
            break

    if 'pred' in fname and not skip:
        pred_tmp = np.load(os.path.join(args.test_dir, fname))
        # print((fname[5:-4] + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in utils.metric(pred_tmp, label)), sep='\t')

        group_results.setdefault(fname[7:-4], {})
        # try:
        for q in range(args.Q):
            group_results[fname[7:-4]].setdefault(q, [])
            group_results[fname[7:-4]][q].append(utils.metric(pred_tmp[:, q, :], label[:, q, :]))
        group_results[fname[7:-4]].setdefault('all', [])
        group_results[fname[7:-4]]['all'].append(utils.metric(pred_tmp, label))
        # except:
        #     pass

for k in sorted(group_results):
    for q in range(args.Q):
        ol = len(group_results[k][q])
        print((k + '_'+str(ol) + f'-T{q}' + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in np.array(group_results[k][q]).mean(0)), sep='\t')
    print((k + '_'+str(ol) + f'-TA' + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in np.array(group_results[k]['all']).mean(0)), sep='\t')
         