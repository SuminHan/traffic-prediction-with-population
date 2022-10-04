import math
import argparse
import utils, model
import time, datetime
import numpy as np
import tensorflow as tf

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"  # specify which GPU(s) to be used

class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


args = dotdict()
args.time_slot = 60
args.P = 12
args.Q = 3
args.L = 5
args.K = 8
args.d = 8
args.train_ratio = 0.7
args.val_ratio = 0.1
args.test_ratio = 0.2
args.batch_size = 16
args.cell_size = 150
args.cnn_size = 5
args.lambda_value = 0.6
args.NODYC=True # This option allows not to calculate ADJ_DYC again if it's already calculated
args.data_dir = '../../dataprocess/prepdata'


for region in ['gangnam', 'hongik', 'jamsil']:
    args.region = region

    for fname in os.listdir('data'):
        if 'OURS' in fname and 'meta' in fname and '3-2_DY0.6_2021' in fname and region in fname:
            print(fname)
            break
    model_file = 'data/' + fname[:-5]

    start = time.time()

    (trainX, trainZ, trainTE, trainY, _, 
        valX, valZ, valTE, valY, _, 
        testX, testZ, testTE, testY, _,
        SE, SEZ, ADJ_GR, ADJ_DY, mean, std, LH, LW) = utils.loadData(args)

    num_train, num_val, num_test = trainX.shape[0], valX.shape[0], testX.shape[0]
    print('trainX: %s\ttrainY: %s' % (trainX.shape, trainY.shape))
    print('valX:   %s\t\tvalY:   %s' % (valX.shape, valY.shape))
    print('testX:  %s\t\ttestY:  %s' % (testX.shape, testY.shape))
    print('data loaded!')

    # test model
    # utils.log_string(log, '**** testing model ****')
    # utils.log_string(log, 'loading model from %s' % model_file)
    graph = tf.Graph()
    with graph.as_default():
        saver = tf.compat.v1.train.import_meta_graph(model_file + '.meta')
    print('model loaded!')


    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    with tf.Session(graph = graph, config = config) as sess:
        saver.restore(sess, model_file)
        parameters = 0
        for variable in tf.compat.v1.trainable_variables():
            parameters += np.product([x.value for x in variable.get_shape()])
        pred = graph.get_collection(name = 'pred')[0]
        testPred = []
        num_batch = math.ceil(num_test / args.batch_size)
        start_test = time.time()
        for batch_idx in range(num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)
            feed_dict = {
                'X:0': testX[start_idx : end_idx],
                'Z:0': testZ[start_idx : end_idx],
                'TE:0': testTE[start_idx : end_idx],
                'is_training:0': False}
            pred_batch = sess.run(pred, feed_dict = feed_dict)
            testPred.append(pred_batch)
        end_test = time.time()
        testPred = np.concatenate(testPred, axis = 0)
    # train_mae, train_rmse, train_mape = utils.metric(trainPred, trainY)
    # val_mae, val_rmse, val_mape = utils.metric(valPred, valY)
    test_mae, test_rmse, test_mape = utils.metric(testPred, testY)
    print('testing time: %.1fs' % (end_test - start_test))
    print('                MAE\t\tRMSE\t\tMAPE')
    # print('train            %.3f\t\t%.3f\t\t%.3f%%' %
    #                  (train_mae, train_rmse, train_mape * 100))
    # print('val              %.3f\t\t%.3f\t\t%.3f%%' %
    #                  (val_mae, val_rmse, val_mape * 100))
    print('test             %.3f\t\t%.3f\t\t%.3f%%' %
                     (test_mae, test_rmse, test_mape * 100))
    print('performance in each prediction step')
    MAE, RMSE, MAPE = [], [], []
    for q in range(args.Q):
        mae, rmse, mape = utils.metric(testPred[:, q], testY[:, q])
        MAE.append(mae)
        RMSE.append(rmse)
        MAPE.append(mape)
        print('step: %02d         %.3f\t\t%.3f\t\t%.3f%%' %
                         (q + 1, mae, rmse, mape * 100))
    average_mae = np.mean(MAE)
    average_rmse = np.mean(RMSE)
    average_mape = np.mean(MAPE)
    print('average:         %.3f\t\t%.3f\t\t%.3f%%' %
        (average_mae, average_rmse, average_mape * 100))
    end = time.time()
    print('total time: %.1fmin' % ((end - start) / 60))
    # log.close()

    np.save(f'test/OURS_{args.region}_pred', testPred)
    np.save(f'test/OURS_{args.region}_label', testY)