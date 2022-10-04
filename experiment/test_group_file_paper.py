import os, datetime, argparse, tqdm, pickle
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import utils
import mymodels

from os.path import join as pjoin

parser = argparse.ArgumentParser()
parser.add_argument('--train_ratio', type = float, default = 0.7,
                    help = 'training set [default : 0.7]')
parser.add_argument('--val_ratio', type = float, default = 0.1,
                    help = 'validation set [default : 0.1]')
parser.add_argument('--test_ratio', type = float, default = 0.2,
                    help = 'testing set [default : 0.2]')
parser.add_argument('--P', type = int, default = 6)
parser.add_argument('--Q', type = int, default = 1)
parser.add_argument('--time_slot', type = int, default = 60, 
                    help = 'a time step is 60 mins')

args = parser.parse_args()

# (trainX, trainZ, trainTE, trainY, trainZY, 
#             valX, valZ, valTE, valY, valZY, 
#             testX, testZ, testTE, testY, testZY,
#             SE, SEZ, DIST_GG, DIST_GR, DIST_RR, mean, std) = utils.loadData(args)


# print(os.listdir(args.data_dir))


skip_models = []# ['DNN', 'MyDCGRU_GST', 'MyGRU', 'MyLSTM']

args.test_dir = f'test_exp/'

label = np.load(os.path.join(args.test_dir, 'label.npy'))

traf_df = pd.read_hdf('../prepdata/traffic-volume-A-20180101-20190101.df')

warns = np.isnan(traf_df.iloc[24*59:24*151]).sum(0) > 900

skip_models = []# ['DNN', 'MyDCGRU_GST', 'MyGRU', 'MyLSTM']

args.test_dir = f'test_exp/'
label = np.load(os.path.join(args.test_dir, 'label.npy'))[..., ~warns]

alive_list = dict()
group_results = dict()
group_preds = dict()
for fname in sorted(os.listdir(args.test_dir)):
    skip = False
    for w in skip_models:
        if w in fname:
            skip = True
            break

    if 'pred' in fname and not skip:
        pred_tmp = np.load(os.path.join(args.test_dir, fname))[..., ~warns]
        group_preds.setdefault(fname[7:-4], [])
        group_preds[fname[7:-4]].append(pred_tmp)
        # print((fname[5:-4] + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in utils.metric(pred_tmp, label)), sep='\t')

        group_results.setdefault(fname[7:-4], {})
        # try:
        for q in range(args.Q):
            # print(pred_tmp.shape, label.shape)
            group_results[fname[7:-4]].setdefault(q, [])
            group_results[fname[7:-4]][q].append(utils.metric(pred_tmp[:, q, :], label[:, q, :]))

        group_results[fname[7:-4]].setdefault('all', [])
        group_results[fname[7:-4]]['all'].append(utils.metric(pred_tmp, label))
        # except:
        #     pass


def Sorting(lst):
    lst2 = sorted(lst, key=len)
    return lst2
      
# Driver code
lst = list(group_results.keys())
# print(Sorting(lst))
lst = Sorting(lst)


for k in lst: #sorted(group_results):
    res_list = []
    for q in range(args.Q):
        ol = len(group_results[k][q])
        tmae = np.array(group_results[k][q]).mean(0)[0]
        if args.Q > 1:
            res_list.append(tmae)

    # print((k + '_'+str(ol) + f'-TA' + ' '*20)[:20], '\t'.join('%.4f'%_ for _ in np.array(group_results[k]['all']).mean(0)), sep='\t')
    res_list.extend(list(np.array(group_results[k]['all']).mean(0)))
    print((k + '_'+str(ol) + ' '*30)[:20], '\t'.join('%.3f'%_ for _ in res_list), sep='\t')

    
for key in group_preds:
    mae_preds = []
    for pred in group_preds[key]:
        mae_pred = utils.metric(pred_tmp[:, q, :], label[:, q, :])[0]
        mae_preds.append(mae_pred)
    
    mae_preds    
        
    
    group_preds[key] = np.mean(np.array(group_preds[key]), 0)