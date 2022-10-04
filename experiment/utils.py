import numpy as np
import pandas as pd
import geopandas as gpd
import os, h5py
import scipy.sparse as sp
import pickle

def load_h5(filename, keywords):
	f = h5py.File(filename, 'r')
	data = []
	for name in keywords:
		data.append(np.array(f[name]))
	f.close()
	if len(data) == 1:
		return data[0]
	return data


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mask = np.not_equal(label, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)
        rmse = np.square(mae)
        mape = np.divide(mae, label)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        mape = np.nan_to_num(mape * mask)
        mape = np.mean(mape)
    return mae, rmse, mape

def seq2instance(data, P, Q):
    num_step = data.shape[0]
    data_type = data.dtype
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(num_sample):
        x[i] = data[i : i + P].astype(data_type)
        y[i] = data[i + P : i + P + Q].astype(data_type)
    return x, y

def seq2instance_fill(data, data_fill, P, Q):
    num_step = data.shape[0]
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, *data.shape[1:]))
    y = np.zeros(shape = (num_sample, Q, *data.shape[1:]))
    for i in range(num_sample):
        x[i] = data_fill[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def seq2instance2(data, P, Q):
    print(data.shape)
    num_step, nodes, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, nodes, dims))
    y = np.zeros(shape = (num_sample, Q, nodes, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def fill_missing(data):
	data = data.copy()
	data[data < 1e-5] = float('nan')
	data = data.fillna(method='pad')
	data = data.fillna(method='bfill')
	return data

from scipy.special import softmax
from scipy import stats
def pearson_corr(X, Y):
    return stats.pearsonr(X, Y)[0]

def row_normalize(a):
    row_sums = a.sum(axis=1)
    new_matrix = a / row_sums[:, np.newaxis]
    return new_matrix

    
def load_graph_data(pkl_filename):
    sensor_ids, sensor_id_to_ind, adj_mx = load_pickle(pkl_filename)
    return sensor_ids, sensor_id_to_ind, adj_mx

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data

# def convert_to_adj_mx(dist_mx, threshold=3000):
#     if threshold > 0:
#         dist_mx[dist_mx > threshold] = np.inf
#     distances = dist_mx[~np.isinf(dist_mx)].flatten()
#     std = distances.std()
#     adj_mx = np.exp(-np.square(dist_mx / std))
#     return row_normalize(adj_mx)

def convert_to_adj_mx(dist_mx, threshold=3000):
    adj_mx = np.zeros(dist_mx.shape)
    if threshold > 0:
        adj_mx[dist_mx > threshold] = 0
        adj_mx[dist_mx <= threshold] = 1
    
    return row_normalize(adj_mx)


def loadVolumeData(args):
    #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
    traf_df = pd.read_hdf(args.file_traf)

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    #Traffic_fill = fill_missing(df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train)
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    
    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)

    extdata = dict()
    extdata['maxval'] = maxval 
    extdata['num_nodes'] = num_sensors 

    return (trainX, trainTE, trainY, 
            valX, valTE, valY, 
            testX, testTE, testY, extdata)


def loadVolumeData2(args):
    #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
    traf_df = pd.read_hdf(args.file_traf); traf_df = traf_df.iloc[24*59:24*151]
    coarse_df = pd.read_hdf(args.file_coarse); coarse_df = coarse_df.iloc[24*59:24*151]
    fine_df = pd.read_hdf(args.file_fine); fine_df = fine_df.iloc[24*59:24*151]
    
    extdata = dict()

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    #Traffic_fill = fill_missing(df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = train_traf = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train); extdata['maxval'] = maxval 
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    
    
    # print(train_traf.shape)
    # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    # # cr = int(args.cnn_size//2)
    # for i in range(num_sensors):
    #     adj_prcc[i, i] = 1
    #     for j in range(i+1, num_sensors):
    #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
    # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

    # coarse train/val/test
    CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    # CoarseLTE = np.concatenate((np.zeros((0, CoarseLTE.shape[1])), np.diff(CoarseLTE, axis=0)), 0)
    CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    extdata['CH'] = CH; extdata['CW'] = CW
    CoarseLTE = CoarseLTE.reshape(-1, CH, CW)
    
    train = CoarseLTE[: train_steps]
    val = CoarseLTE[train_steps : train_steps + val_steps]
    test = CoarseLTE[-test_steps :]

    # X, Y 
    trainZC, _ = seq2instance(train, args.P, args.Q)
    valZC, _ = seq2instance(val, args.P, args.Q)
    testZC, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
    trainZC = trainZC / maxvalZC
    valZC = valZC / maxvalZC
    testZC = testZC / maxvalZC

    
    # fine train/val/test
    FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))
    # FineLTE = np.concatenate((np.zeros((0, FineLTE.shape[1])), np.diff(FineLTE, axis=0)), 0)
    FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
    extdata['FH'] = FH; extdata['FW'] = FW
    FineLTE = FineLTE.reshape(-1, FH, FW)
    
    train = FineLTE[: train_steps]
    val = FineLTE[train_steps : train_steps + val_steps]
    test = FineLTE[-test_steps :]

    # X, Y 
    trainZF, _ = seq2instance(train, args.P, args.Q)
    valZF, _ = seq2instance(val, args.P, args.Q)
    testZF, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF
    trainZF = trainZF / maxvalZF
    valZF = valZF / maxvalZF
    testZF = testZF / maxvalZF

    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2)
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


    return (trainX, trainZC, trainZF, trainTE, trainY, 
            valX, valZC, valZF, valTE, valY, 
            testX, testZC, testZF, testTE, testY, extdata)



def loadVolumeData3(args):
    #traf_df = pd.read_csv(args.filepath, index_col=0, parse_dates=True)
    traf_df = pd.read_hdf(args.file_traf); traf_df = traf_df.iloc[24*59:24*151]
    coarse_df = pd.read_hdf(args.file_coarse); coarse_df = coarse_df.iloc[24*59:24*151]
    fine_df = pd.read_hdf(args.file_fine); fine_df = fine_df.iloc[24*59:24*151]
    
    extdata = dict()

    Traffic = np.nan_to_num(traf_df.values.astype(np.float32))
    #Traffic_fill = fill_missing(df).values.astype(np.float32)
    num_step, num_sensors = traf_df.shape; extdata['num_nodes'] = num_sensors
    
    # train/val/test 
    train_steps = round(args.train_ratio * num_step)
    test_steps = round(args.test_ratio * num_step)
    val_steps = num_step - train_steps - test_steps

    train = train_traf = Traffic[: train_steps]
    val = Traffic[train_steps : train_steps + val_steps]
    test = Traffic[-test_steps :]

    # X, Y 
    trainX, trainY = seq2instance(train, args.P, args.Q)
    valX, valY = seq2instance(val, args.P, args.Q)
    testX, testY = seq2instance(test, args.P, args.Q)
    # normalization
    maxval = np.max(train); extdata['maxval'] = maxval 
    trainX = trainX / maxval
    valX = valX / maxval
    testX = testX / maxval
    
    
    # print(train_traf.shape)
    # adj_prcc = np.zeros((num_sensors, num_sensors), dtype=np.float32)
    # # cr = int(args.cnn_size//2)
    # for i in range(num_sensors):
    #     adj_prcc[i, i] = 1
    #     for j in range(i+1, num_sensors):
    #         adj_prcc[j, i] = adj_prcc[i, j] = pearson_corr(train_traf[:, i], train_traf[:, j])
    # extdata['adj_prcc_X'] = row_normalize(adj_prcc)

    # coarse train/val/test
    CoarseLTE = np.nan_to_num(coarse_df.values.astype(np.float32))
    prev_CoarseLTE = np.concatenate((np.zeros((1, CoarseLTE.shape[1])), CoarseLTE[:-1, ...]), 0)
    print(CoarseLTE.shape, prev_CoarseLTE.shape)
    CoarseLTE = np.concatenate((prev_CoarseLTE, CoarseLTE), -1)


    CH, CW = coarse_df.columns[-1].split(','); CH = int(CH)+1; CW = int(CW)+1; 
    extdata['CH'] = CH; extdata['CW'] = CW
    CoarseLTE = CoarseLTE.reshape(-1, CH, CW, 2)
    
    train = CoarseLTE[: train_steps]
    val = CoarseLTE[train_steps : train_steps + val_steps]
    test = CoarseLTE[train_steps + val_steps :]

    # X, Y 
    trainZC, _ = seq2instance(train, args.P, args.Q)
    valZC, _ = seq2instance(val, args.P, args.Q)
    testZC, _ = seq2instance(test, args.P, args.Q)
    # normalization
    maxvalZC = np.max(train); extdata['maxvalZC'] = maxvalZC
    trainZC = trainZC / maxvalZC
    valZC = valZC / maxvalZC
    testZC = testZC / maxvalZC

    
    # fine train/val/test
    FineLTE = np.nan_to_num(fine_df.values.astype(np.float32))
    prev_FineLTE = np.concatenate((np.zeros((1, FineLTE.shape[1])), FineLTE[:-1, ...]), 0)
    FineLTE = np.concatenate((prev_FineLTE, FineLTE), -1)

    FH, FW = fine_df.columns[-1].split(','); FH = int(FH)+1; FW = int(FW)+1
    extdata['FH'] = FH; extdata['FW'] = FW
    FineLTE = FineLTE.reshape(-1, FH, FW, 2)
    
    train = FineLTE[: train_steps]
    val = FineLTE[train_steps : train_steps + val_steps]
    test = FineLTE[train_steps + val_steps :]

    # X, Y 
    trainZF, _ = seq2instance(train, args.P, args.Q)
    valZF, _ = seq2instance(val, args.P, args.Q)
    testZF, _ = seq2instance(test, args.P, args.Q)
    # normalization 
    maxvalZF = np.max(train); extdata['maxvalZF'] = maxvalZF
    trainZF = trainZF / maxvalZF
    valZF = valZF / maxvalZF
    testZF = testZF / maxvalZF

    
    # temporal embedding 
    Time = traf_df.index
    dayofweek =  np.reshape(Time.weekday, newshape = (-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // (args.time_slot*60) #// Time.freq.delta.total_seconds()
    timeofday = np.reshape(timeofday, newshape = (-1, 1))    
    Time = np.concatenate((dayofweek, timeofday), axis = -1)
    # train/val/test 
    train = Time[: train_steps]
    val = Time[train_steps : train_steps + val_steps]
    test = Time[-test_steps :]
    # shape = (num_sample, P + Q, 2) 
    trainTE = seq2instance(train, args.P, args.Q)
    trainTE = np.concatenate(trainTE, axis = 1).astype(np.int32)
    valTE = seq2instance(val, args.P, args.Q)
    valTE = np.concatenate(valTE, axis = 1).astype(np.int32)
    testTE = seq2instance(test, args.P, args.Q)
    testTE = np.concatenate(testTE, axis = 1).astype(np.int32)


    return (trainX, trainZC, trainZF, trainTE, trainY, 
            valX, valZC, valZF, valTE, valY, 
            testX, testZC, testZF, testTE, testY, extdata)