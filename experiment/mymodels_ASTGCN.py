import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dcgru_cell_tf2 import *
from submodules import *


class MyGM0ZCFC(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGM0ZCFC, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        self.num_cells_C = self.CH*self.CW
        self.num_cells_F = self.FH*self.FW
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(1)])

        

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.concat_fusion = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])


    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)
        STEX_P, STEX_Q = STEX[:, :self.P, :], STEX[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX_P
        for i in range(self.L):
            X = self.GSTA_enc[i](X, STEX_P)


        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = tf.concat((X, ZF, ZC), -1)
        X = self.concat_fusion(X)
        
        X = self.C_trans_layer(X, STEX_P, STEX_Q)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, STEX_Q)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)

        return Y