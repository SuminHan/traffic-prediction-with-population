
import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dcgru_cell_tf2 import *
from submodules import *
from MATURE import *
from mymodels_DCGRU import * # MyDCGRUSTE0ZCF, MyDCGRUSTE0ZC, MyDCGRUSTE0ZF
from mymodels_GMAN import * # MyGM0ZCF, MyGM0ZC, MyGM0ZF
from mymodels_ASTGCN import *

def row_normalize(an_array):
    sum_of_rows = an_array.sum(axis=1)
    normalized_array = an_array / sum_of_rows[:, np.newaxis]
    return normalized_array

def str_to_class(classname):
    return getattr(sys.modules[__name__], classname)

def ModelSet(model_name, extdata, args, **kwargs):
    model = str_to_class(model_name)(extdata, args)
    return (model(kwargs) ) * extdata['maxval']


###########################################################################

class LastRepeat(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(LastRepeat, self).__init__()
        pass
        
    def build(self, input_shape):
        pass

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        return X[:, -1:, :]



class MyGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGRU, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FCs_1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FCs_2 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])
        self.gru = layers.GRU(self.D)


    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        X = self.FCs_1(X)
        X = self.gru(X)
        Y = self.FCs_2(X)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y


class MyLSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyLSTM, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FCs_1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        self.FCs_2 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])
        self.lstm = layers.LSTM(self.D)


    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        X = self.FCs_1(X)
        X = self.lstm(X)
        Y = self.FCs_2(X)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y

##############


class MyDCGRUSTE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        X = X + STEX
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y
        


class MyGMAN0(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN0, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        # self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAP_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.P_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAP_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.GSTAQ_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.Q_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAQ_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)
        return Y


        
class MyGMDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMDCGRU, self).__init__()
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        self.mydcgru = MyDCGRUSTE(self.extdata, self.args)
        self.mygman = MyGMAN0(self.extdata, self.args)
    
    def call(self, kwargs):
        return self.mydcgru(kwargs) + self.mygman(kwargs)



########################
