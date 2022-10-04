import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dcgru_cell_tf2 import *
from submodules import *



class MyDCGRUSTE0ZCFC(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFC, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

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
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
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
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyDCGRUSTE0ZCFW(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFW, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

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

        self.weight_fusion = WeightFusion(D)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
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

        X = tf.stack((X, ZF, ZC), -1)
        X = self.weight_fusion(X)
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyDCGRUSTE0ZCFBPB(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBPB, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.num_nodes = extdata['num_nodes']
        self.adj_mat = np.eye(self.num_nodes)
        self.patch_size = args.patch_size

        self.CH = extdata['CH']
        self.CW = extdata['CW']
        self.FH = extdata['FH']
        self.FW = extdata['FW']
        # self.num_cells_C = self.CH*self.CW
        # self.num_cells_F = self.FH*self.FW
        import math
        self.num_cells_C = math.ceil(self.CH / self.patch_size)*math.ceil(self.CW / self.patch_size)
        self.num_cells_F = math.ceil(self.FH / self.patch_size)*math.ceil(self.FW / self.patch_size)
        
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
        
        

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.ZC_Pooling = layers.AveragePooling2D(pool_size=(self.patch_size, self.patch_size), padding="same")
        self.ZC_trans_layer = BipartiteAttention(self.K, self.d)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.ZF_Pooling = layers.AveragePooling2D(pool_size=(self.patch_size, self.patch_size), padding="same")
        self.ZF_trans_layer = BipartiteAttention(self.K, self.d)
        # self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # STEZC_T = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF_T = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        #ZC = ZC + STEZC_T
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, ZC.shape[2], ZC.shape[3], self.D))
        ZC = self.ZC_Pooling(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[1]*ZC.shape[2], self.D))
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))

        print(ZC.shape, STEZC.shape, STEX.shape)

        ZC = self.ZC_trans_layer(ZC, STEZC, STEX)
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        #ZF = ZF + STEZF_T
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, ZF.shape[2], ZF.shape[3], self.D))
        ZF = self.ZC_Pooling(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[1]*ZF.shape[2], self.D))
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.ZF_trans_layer(ZF, STEZF, STEX)
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyDCGRUSTE0ZCFBB2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBB2, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.ZC_trans_layer = BipartiteAttention(self.K, self.d)
        self.ZC_trans_layer2 = BipartiteAttention(self.K, self.d)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.ZF_trans_layer = BipartiteAttention(self.K, self.d)
        self.ZF_trans_layer2 = BipartiteAttention(self.K, self.d)
        # self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC_T = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF_T = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC_T
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.ZC_trans_layer(ZC, STEZC, STEX) - self.ZC_trans_layer2(ZC, STEZC, STEX)
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF_T
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.ZF_trans_layer(ZF, STEZF, STEX) - self.ZF_trans_layer2(ZF, STEZF, STEX)
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyDCGRUSTE0ZCFB2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFB2, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 5, strides=3, padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 5, strides=3, padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 5, strides=3, padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 5, strides=3, padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y




class MyDCGRUSTE0ZCFBV(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBV, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        # self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        # self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZC', dtype=tf.float32)
        # self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        # self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        # STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        # STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        # ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        # ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y

class MyDCGRUSTE0ZCFBU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBU, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyDCGRUSTE0ZCFB1(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFB1, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y

class MyDCGRUSTE0ZCFBTA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBTA, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.d = args.d
        self.L = args.L
        self.K = args.K
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.ZC_TA_layer = [TemporalAttention(self.K, self.d) for _ in range(self.L)]

        # self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        # self.FC_ZC_Bi = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        # self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        
        self.ZF_TA_layer = [TemporalAttention(self.K, self.d) for _ in range(self.L)]
        
        # self.FC_ZF_Bi = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC_P = STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF_P = STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        # ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)

        ZC = tf.reshape(ZC, (-1, self.P, self.CH * self.CW, self.D))
        for i in range(self.L):
            ZC = self.ZC_TA_layer[i](ZC, STEZC_P)

        # ZC = self.FC_ZC_Bi(ZC)
        # ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        # ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)

        ZF = tf.reshape(ZF, (-1, self.P, self.FH * self.FW, self.D))
        for i in range(self.L):
            ZF = self.ZF_TA_layer[i](ZF, STEZF_P)

        # ZF = self.FC_ZF_Bi(ZF)
        # ZF = tf.reshape(ZF, (-1, self.P, self.FH * self.FW, self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y





class MyDCGRUSTE0ZCFBD0(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBD0, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.STDE_layer = STDEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.STDEZC_layer = STDEmbedding(self.num_cells_C, D)
        self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
                                        initializer='glorot_uniform',
                                        name='SEZC', dtype=tf.float32)
        self.STDEZF_layer = STDEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        from deconvolutional_recurrent import DeConvLSTM2D

        self.FC_ZC_ConvLSTM = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STDEX = self.STDE_layer(self.SE, TE)
        

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STDEX[:, :self.P, :, 0, :]
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STDEZC = self.STDEZC_layer(self.SEZC, TE)[:, :self.P, :, 0, :]
        STDEZC = tf.reshape(STDEZC, (-1, self.P, self.CH, self.CW, self.D))

        STDEZF = self.STDEZF_layer(self.SEZF, TE)[:, :self.P, :, 0, :]
        STDEZF = tf.reshape(STDEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1) 
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STDEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STDEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



from deconvolutional_recurrent import DeConvLSTM2D
class MyDCGRUSTE0ZCFBDE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBDE, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_ConvLSTM3 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_ConvLSTM3 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM1(ZC), self.FC_ZC_ConvLSTM2(ZC), self.FC_ZC_ConvLSTM3(ZC), self.FC_ZC_ConvLSTM4(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM1(ZF), self.FC_ZF_ConvLSTM2(ZF), self.FC_ZF_ConvLSTM3(ZF), self.FC_ZF_ConvLSTM4(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y




class MyDCGRUSTE0ZCFBDER(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBDER, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM1(ZC), self.FC_ZC_ConvLSTM4(ZC)[:, ::-1, ...]), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM1(ZF), self.FC_ZF_ConvLSTM4(ZF)[:, ::-1, ...]), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyDCGRUSTE0ZCFBDER2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBDER2, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in1 = keras.Sequential([layers.Dense(D, activation="relu"),layers.Dense(D)])
        self.FC_ZC_in2 = keras.Sequential([layers.Dense(D, activation="relu"),layers.Dense(D)])
        self.FC_ZC_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in1 = keras.Sequential([layers.Dense(D, activation="relu"),layers.Dense(D)])
        self.FC_ZF_in2 = keras.Sequential([layers.Dense(D, activation="relu"),layers.Dense(D)])
        self.FC_ZF_ConvLSTM1 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM4 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        ZC1 = self.FC_ZC_in1(ZC[..., :1]) + STEZC
        ZC2 = self.FC_ZC_in2(ZC[..., 1:]) + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM1(ZC1), self.FC_ZC_ConvLSTM4(ZC2)[:, ::-1, ...]), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        ZF1 = self.FC_ZF_in1(ZF[..., :1]) + STEZF
        ZF2 = self.FC_ZF_in2(ZF[..., 1:]) + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM1(ZF1), self.FC_ZF_ConvLSTM4(ZF2)[:, ::-1, ...]), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyDCGRUSTE0ZCFBDEV(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFBDEV, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_ConvLSTM3 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_ConvLSTM3 = DeConvLSTM2D(D, 3, strides=(1, 1), padding='same', use_bias=False, return_sequences=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM2(ZC)[:, ::-1, ...], self.FC_ZC_ConvLSTM3(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM2(ZF)[:, ::-1, ...], self.FC_ZF_ConvLSTM3(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y





class MyDCGRUSTE0ZCFSAB(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFSAB, self).__init__()
        self.D = args.D
        self.d = args.d
        self.K = args.K
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.SA_ZC = TransformAttention2(self.K, self.d)
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.SA_ZF = TransformAttention2(self.K, self.d)
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 1, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))




        # ZC part
        ZC = self.FC_ZC_in(ZC)

        # ZC = ZC + STEZC
        # ZC: (batch, P, CH, CW, D)
        ZCV = []
        STEZCK = []
        for i in range(3):
            for j in range(3):
                ZCv = ZC[:, :, i:self.CH-3+i, j:self.CW-3+j, :]; ZCV.append(ZCv)
                STEZCk = STEZC[:, :, i:self.CH-3+i, j:self.CW-3+j, :]; STEZCK.append(STEZCk)
        STEZCQ = STEZC[:, :, 1:self.CH-2, 1:self.CW-2, :]

        ZCV = tf.stack(ZCV, -2)
        STEZCK = tf.stack(STEZCK, -2)
        STEZCQ = tf.expand_dims(STEZCQ, -2)
        ZC = self.SA_ZC(ZCV, STEZCK, STEZCQ)
        ZC = tf.squeeze(ZC, -2)


        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF part
        ZF = self.FC_ZF_in(ZF)

        # ZF = ZF + STEZF
        # ZF: (batch, P, CH, CW, D)
        ZFV = []
        STEZFK = []
        for i in range(3):
            for j in range(3):
                ZFv = ZF[:, :, i:self.FH-3+i, j:self.FW-3+j, :]; ZFV.append(ZFv)
                STEZFk = STEZC[:, :, i:self.FH-3+i, j:self.FW-3+j, :]; STEZFK.append(STEZFk)
        STEZFQ = STEZF[:, :, 1:self.FH-2, 1:self.FW-2, :]

        ZFV = tf.stack(ZFV, -2)
        STEZFK = tf.stack(STEZFK, -2)
        STEZFQ = tf.expand_dims(STEZFQ, -2)
        ZF = self.SA_ZF(ZFV, STEZFK, STEZFQ)
        ZF = tf.squeeze(ZF, -2)


        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y











class MyDCGRUSTE0ZCFB(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFB, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y




class MyDCGRUSTE0ZCFB_s(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCFB_s, self).__init__()
        self.D = 1 #args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZC_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_ConvLSTM2 = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True, go_backwards=True)
        self.FC_ZF_Bi = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.concat((self.FC_ZC_ConvLSTM(ZC), self.FC_ZC_ConvLSTM2(ZC)), -1)
        ZC = self.FC_ZC_Bi(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = tf.concat((self.FC_ZF_ConvLSTM(ZF), self.FC_ZF_ConvLSTM2(ZF)), -1)
        ZF = self.FC_ZF_Bi(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyDCGRUSTE0ZCF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZCF, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

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



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))

        
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

        X = X + ZF + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyDCGRUSTE0ZC(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZC, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        # self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        # self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        # self.FC_ZF_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)
        # self.FC_ZF_trans2 = layers.Dense(self.D)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        # TE = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)
        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))



        
        STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        # STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        # STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        ZC = tf.expand_dims(ZC, -1)
        ZC = self.FC_ZC_in(ZC)
        ZC = ZC + STEZC
        ZC = self.FC_ZC_ConvLSTM(ZC)
        ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        ZC = tf.transpose(ZC, (0, 1, 3, 2))
        ZC = self.FC_ZC_trans(ZC)
        ZC = tf.transpose(ZC, (0, 1, 3, 2))


        # ZF = tf.expand_dims(ZF, -1)
        # ZF = self.FC_ZF_in(ZF)
        # ZF = ZF + STEZF
        # ZF = self.FC_ZF_ConvLSTM(ZF)
        # ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))
        # ZF = self.FC_ZF_trans(ZF)
        # ZF = tf.transpose(ZF, (0, 1, 3, 2))
        # ZF = self.FC_ZF_trans2(ZF)

        X = X + ZC
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y


class MyDCGRUSTE0ZF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE0ZF, self).__init__()
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
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
        # self.STEZC_layer = STEmbedding(self.num_cells_C, D)
        # self.SEZC = self.add_weight(shape=(self.num_cells_C, D),
        #                                 initializer='glorot_uniform',
        #                                 name='SEZC', dtype=tf.float32)
        self.STEZF_layer = STEmbedding(self.num_cells_F, D)
        self.SEZF = self.add_weight(shape=(self.num_cells_F, D),
                                        initializer='glorot_uniform',
                                        name='SEZF', dtype=tf.float32)
                                        
        self.FC_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_X_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False, return_sequences=True)
        self.FC_X_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 1, self.num_nodes, 'random_walk'), return_state=False)
        self.FC_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        # self.FC_ZC_in = keras.Sequential([
        #                     layers.Dense(D, activation="relu"),
        #                     layers.Dense(D)])
        # self.FC_ZC_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.FC_ZC_trans = layers.Dense(self.num_nodes, use_bias=False)

        self.FC_ZF_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZF_ConvLSTM = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        self.FC_ZF_trans = layers.Dense(self.num_nodes, use_bias=False)



    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STEX = self.STE_layer(self.SE, TE)[:, :self.P, :]

        # TE = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)
        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))

        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X = self.FC_X_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))



        
        # STEZC = self.STEZC_layer(self.SEZC, TE)[:, :self.P, :]
        # STEZC = tf.reshape(STEZC, (-1, self.P, self.CH, self.CW, self.D))

        STEZF = self.STEZF_layer(self.SEZF, TE)[:, :self.P, :]
        STEZF = tf.reshape(STEZF, (-1, self.P, self.FH, self.FW, self.D))


        # ZC = tf.expand_dims(ZC, -1)
        # ZC = self.FC_ZC_in(ZC)
        # ZC = ZC + STEZC
        # ZC = self.FC_ZC_ConvLSTM(ZC)
        # ZC = tf.reshape(ZC, (-1, self.P, ZC.shape[2]*ZC.shape[3], self.D))
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))
        # ZC = self.FC_ZC_trans(ZC)
        # ZC = tf.transpose(ZC, (0, 1, 3, 2))


        ZF = tf.expand_dims(ZF, -1)
        ZF = self.FC_ZF_in(ZF)
        ZF = ZF + STEZF
        ZF = self.FC_ZF_ConvLSTM(ZF)
        ZF = tf.reshape(ZF, (-1, self.P, ZF.shape[2]*ZF.shape[3], self.D))
        ZF = tf.transpose(ZF, (0, 1, 3, 2))
        ZF = self.FC_ZF_trans(ZF)
        ZF = tf.transpose(ZF, (0, 1, 3, 2))

        X = X + ZF
        
        X = self.FC_X_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y
