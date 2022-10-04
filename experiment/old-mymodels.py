

import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dcgru_cell_tf2 import *
from submodules import *
from MATURE import *

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


##############


class MyDCGRUSTE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        # TE = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)
        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        X = X + STEX
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y

        




class MyGMBISTGRUPOIv3(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMBISTGRUPOIv3, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.adj_mat = extdata['ADJ_GG']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.SEZs = extdata['SEZ']
        print('len(SEZ)', len(self.SEZs))
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        D = self.D
        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)

        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_sequences=True, return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out1 = keras.Sequential([
                            layers.Dense(D*self.Q)])
        

        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]

        
        if not self.args.NOMOD:
            self.FCQ_feat = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.FCK_feat = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.num_mx = 1 + 1 + len(self.SEZs)
            self.FC_gcn = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
            self.FCQ_feat2 = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.FCK_feat2 = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.FC_gcn2 = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        

        
        self.FC_cat = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.RSTA_enc = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
    
    def call(self, kwargs):        
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X0 = X

        
        X0 = self.FC_XC_DCGRU1(X0)
        X0 = tf.reshape(X0, (-1, self.P, self.num_nodes, self.D))
        X0 = self.FC_XC_DCGRU2(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.D))
        X0 = self.FC_XC_out1(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.Q, self.D))
        Y1 = tf.transpose(X0, (0, 2, 1, 3))




        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)

        if not self.args.NOMOD:
            z0 = Z
            z_support = self.ADJ_RR@z0

            z = z0
            z += z_support
            # z = tf.concat([z0, z_support], -1)
            for i, feat in enumerate(self.SEZs):
                # premultiply the concatened inputs and state with support matrices
                FEQ = self.FCQ_feat[i](feat)
                FEK = self.FCK_feat[i](feat)
                
                support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5 # @ tf.transpose(FE)
                support = tf.nn.softmax(support) 

                z_support = support@z0
                # x_support = tf.expand_dims(x_support, 0) 
                # concatenate convolved signal
                # z = tf.concat([z, z_support], axis=-1)
                z += z_support
            Z = self.FC_gcn(z)

        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
            
        if not self.args.NOMOD:
            z0 = Z
            z_support = self.ADJ_RR@z0

            z = z0
            z += z_support
            # z = tf.concat([z0, z_support], -1)
            for i, feat in enumerate(self.SEZs):
                # premultiply the concatened inputs and state with support matrices
                FEQ = self.FCQ_feat2[i](feat)
                FEK = self.FCK_feat2[i](feat)
                
                support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5 # @ tf.transpose(FE)
                support = tf.nn.softmax(support) 

                z_support = support@z0
                # x_support = tf.expand_dims(x_support, 0) 
                # concatenate convolved signal
                # z = tf.concat([z, z_support], axis=-1)
                z += z_support
            Z = self.FC_gcn2(z)
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)

        X = tf.concat((X, Y1), -1)
        X = self.FC_cat(X)
        
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        Y2 = X

        print(Y1.shape, Y2.shape)
        # Y = tf.concat((Y1, Y2), -1)
        Y = self.FC_X_out(Y2)
        Y = tf.squeeze(Y, -1)

        return Y

        

class MyGMBISTGRUPOIv2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMBISTGRUPOIv2, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.adj_mat = extdata['ADJ_GG']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.SEZs = extdata['SEZ']
        print('len(SEZ)', len(self.SEZs))
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        D = self.D
        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)

        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_sequences=True, return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out1 = keras.Sequential([
                            layers.Dense(D*self.Q)])
        

        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]

        
        if not self.args.NOMOD:
            self.FCQ_feat = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.FCK_feat = [keras.Sequential([
                                tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
            self.num_mx = 1 + 1 + len(self.SEZs)
            self.FC_gcn = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        

        
        self.FC_cat = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.RSTA_enc = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
    
    def call(self, kwargs):        
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X0 = X

        
        X0 = self.FC_XC_DCGRU1(X0)
        X0 = tf.reshape(X0, (-1, self.P, self.num_nodes, self.D))
        X0 = self.FC_XC_DCGRU2(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.D))
        X0 = self.FC_XC_out1(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.Q, self.D))
        Y1 = tf.transpose(X0, (0, 2, 1, 3))


        if not self.args.NOMOD:
            z0 = Z
            z_support = self.ADJ_RR@z0

            z = z0
            z += z_support
            # z = tf.concat([z0, z_support], -1)
            for i, feat in enumerate(self.SEZs):
                # premultiply the concatened inputs and state with support matrices
                FEQ = self.FCQ_feat[i](feat)
                FEK = self.FCK_feat[i](feat)
                
                support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5 # @ tf.transpose(FE)
                support = tf.nn.softmax(support) 

                z_support = support@z0
                # x_support = tf.expand_dims(x_support, 0) 
                # concatenate convolved signal
                # z = tf.concat([z, z_support], axis=-1)
                z += z_support
                
            Z = self.FC_gcn(z)


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)

        X = tf.concat((X, Y1), -1)
        X = self.FC_cat(X)
        
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        Y2 = X

        print(Y1.shape, Y2.shape)
        # Y = tf.concat((Y1, Y2), -1)
        Y = self.FC_X_out(Y2)
        Y = tf.squeeze(Y, -1)

        return Y



class MyGMBISTGRUPOI(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMBISTGRUPOI, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.adj_mat = extdata['ADJ_GG']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.SEZs = extdata['SEZ']
        print('len(SEZ)', len(self.SEZs))
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
    
    def build(self, input_shape):
        D = self.D
        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)

        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_sequences=True, return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out1 = keras.Sequential([
                            layers.Dense(D*self.Q)])
        

        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]

        
        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(D, activation='relu', use_bias=False)]) for i in range(len(self.SEZs))]
        self.num_mx = 1 + 1 + len(self.SEZs)
        self.FC_gcn = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        

        
        
        self.RSTA_enc = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
    
    def call(self, kwargs):        
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X0 = X

        
        X0 = self.FC_XC_DCGRU1(X0)
        X0 = tf.reshape(X0, (-1, self.P, self.num_nodes, self.D))
        X0 = self.FC_XC_DCGRU2(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.D))
        X0 = self.FC_XC_out1(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.Q, self.D))
        Y1 = tf.transpose(X0, (0, 2, 1, 3))


        z0 = Z
        z_support = self.ADJ_RR@z0

        z = z0
        z += z_support
        # z = tf.concat([z0, z_support], -1)
        for i, feat in enumerate(self.SEZs):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5 # @ tf.transpose(FE)
            support = tf.nn.softmax(support) 

            z_support = support@z0
            # x_support = tf.expand_dims(x_support, 0) 
            # concatenate convolved signal
            # z = tf.concat([z, z_support], axis=-1)
            z += z_support
            
        Z = self.FC_gcn(z)


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        Y2 = X

        print(Y1.shape, Y2.shape)
        Y = tf.concat((Y1, Y2), -1)
        Y = self.FC_X_out(Y)
        Y = tf.squeeze(Y, -1)

        return Y





class MyGMBISTDCGRU2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMBISTDCGRU2, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.adj_mat = extdata['ADJ_GG']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
    
    def build(self, input_shape):
        D = self.D
        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)




        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_sequences=True, return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out1 = keras.Sequential([
                            layers.Dense(D*self.Q)])



        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.RSTA_enc = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
    
    def call(self, kwargs):        
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX
        X0 = X

        
        X0 = self.FC_XC_DCGRU1(X0)
        X0 = tf.reshape(X0, (-1, self.P, self.num_nodes, self.D))
        X0 = self.FC_XC_DCGRU2(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.D))
        X0 = self.FC_XC_out1(X0)
        X0 = tf.reshape(X0, (-1, self.num_nodes, self.Q, self.D))
        Y1 = tf.transpose(X0, (0, 2, 1, 3))


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        Y2 = X

        print(Y1.shape, Y2.shape)
        Y = tf.concat((Y1, Y2), -1)
        Y = self.FC_X_out(Y)
        Y = tf.squeeze(Y, -1)

        return Y

class MyGMBISTDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMBISTDCGRU, self).__init__()
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        self.mydcgru = MyDCGRU(self.extdata, self.args)
        self.mygmstark = MyGMSTARKBICNN(self.extdata, self.args)
    
    def call(self, kwargs):
        return self.mydcgru(kwargs) + self.mygmstark(kwargs)


class MyGMSTDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTDCGRU, self).__init__()
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        self.mydcgru = MyDCGRU(self.extdata, self.args)
        self.mygmstark = MyGMSTARKS(self.extdata, self.args)
    
    def call(self, kwargs):
        return self.mydcgru(kwargs) + self.mygmstark(kwargs)

    
class MyGMDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMDCGRU, self).__init__()
        self.extdata = extdata
        self.args = args
    
    def build(self, input_shape):
        self.mydcgru = MyDCGRU(self.extdata, self.args)
        self.mygman = MyGMAN(self.extdata, self.args)
    
    def call(self, kwargs):
        return self.mydcgru(kwargs) + self.mygman(kwargs)


class MyGMSTARK_SUB3(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTARK_SUB3, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        # self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        # self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.num_nodes = extdata['num_nodes']
        self.num_cells = extdata['num_cells']
        self.extdata = extdata
        self.args = args
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        # self.RSTA_enc = [RSTAttBlock(self.K, self.d, self.ADJ_RR) for _ in range(self.L)]
        # self.R_trans_layer = TransformAttention(self.K, self.d)
        # self.RSTA_dec = [RSTAttBlock(self.K, self.d, self.ADJ_RR) for _ in range(self.L)]
        self.RSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)

        self.S_trans_layer1 = TransformAttention2(self.K, self.d)
        self.S_trans_layer2 = TransformAttention2(self.K, self.d)
        self.RSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_Z_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        self.mydcgru = MyDCGRUSTE(self.extdata, self.args)
        
    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        # Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        # ZX = self.ADJ_GR @ Z
        ZX = self.S_trans_layer1(Z, STEZ, STEX)
        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        # ZY = self.ADJ_GR @ Z
        ZY = self.S_trans_layer2(Z, STEZY, STEXY)

        
        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)

        print(X.shape, ZX.shape, )

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1) + self.mydcgru(kwargs)


        
        return Y


class MyKALSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyKALSTM, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.height = args.height
        self.width = args.width
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.movement_GR = self.add_weight(shape=(self.num_nodes, self.num_cells),
                                                    initializer='glorot_uniform',
                                                    name='ZtoN', dtype=tf.float32)
                            
        self.ConvLSTM_Z = layers.ConvLSTM2D(D, 3, strides=(1, 1), padding='valid', use_bias=False, return_sequences=True)
        # self.LSTM_Z = layers.LSTM(D, return_sequences=True)
        self.DCGRU_Z = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'), return_sequences=True)
        # self.FCs_Z_decoder = layers.Dense(self.P*D, activation="relu")
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ

        # (batch_size, time_seq, num_cells, D)
        Z = tf.reshape(Z, (-1, self.P, self.height, self.width, self.D))
        Z = self.ConvLSTM_Z(Z)
        Z = tf.reshape(Z, (-1, self.P, self.height*self.width, self.D))
        
        adj_GR = gumbel_softmax(self.movement_GR)
        Z = adj_GR @ Z
        
        Z = self.DCGRU_Z(Z)
        ZX = tf.reshape(Z, (-1, self.P, self.num_nodes, self.D))

        # Z = self.FCs_Z_decoder(Z)
        # Z = tf.reshape(Z, (-1, self.height*self.width, self.P, self.D))
        # Z = tf.transpose(Z, [0, 2, 1, 3])
        # ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX

        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyKAConvLSTM2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyKAConvLSTM2, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.height = args.height
        self.width = args.width
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
                            
        self.ConvLSTM_Z = layers.ConvLSTM2D(D, 3, padding='same', use_bias=False, return_sequences=False)
        self.FCs_Z_decoder = layers.Dense(self.P*D, activation="relu")
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ

        # (batch_size, time_seq, num_cells, D)
        Z = tf.reshape(Z, (-1, self.P, self.height, self.width, self.D))
        Z = self.ConvLSTM_Z(Z)
        Z = self.FCs_Z_decoder(Z)
        Z = tf.reshape(Z, (-1, self.height*self.width, self.P, self.D))
        Z = tf.transpose(Z, [0, 2, 1, 3])
        ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX

        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y

class MyKAMovement(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyKAMovement, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.height = args.height
        self.width = args.width
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        self.movement_Z = self.add_weight(shape=(self.num_cells, self.num_cells),
                                                    initializer='glorot_uniform',
                                                    name='movement_Z', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.movement_FC = layers.Dense(D, activation="relu")
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
                            
        self.ConvLSTM_Z = layers.ConvLSTM2D(D, 3, padding='same', use_bias=False, return_sequences=True)
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ



        # (batch_size, time_seq, num_cells, D)
        Z = tf.reshape(Z, (-1, self.P, self.height, self.width, self.D))
        Z = self.ConvLSTM_Z(Z)
        Z = tf.reshape(Z, (-1, self.P, self.height*self.width, self.D))
        
        movement_Z = gumbel_softmax(self.movement_Z)
        Z = movement_Z @ Z 
        Z = self.movement_FC(Z)
        ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX

        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y


class MyKAConvLSTM(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyKAConvLSTM, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.height = args.height
        self.width = args.width
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])
                            
        self.ConvLSTM_Z = layers.ConvLSTM2D(D, 3, padding='same', use_bias=False, return_sequences=True)
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ

        # (batch_size, time_seq, num_cells, D)
        Z = tf.reshape(Z, (-1, self.P, self.height, self.width, self.D))
        Z = self.ConvLSTM_Z(Z)
        Z = tf.reshape(Z, (-1, self.P, self.height*self.width, self.D))
        ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX

        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y

class MyKAmodel(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyKAmodel, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        self.height = args.height
        self.width = args.width
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        self.CNN_Z = layers.Conv2D(D, 3, padding='same', use_bias=False)
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ

        # (batch_size, time_seq, num_cells, D)
        Z = tf.reshape(Z, (-1, self.height, self.width, self.D))
        Z = self.CNN_Z(Z)
        Z = tf.reshape(Z, (-1, self.P, self.height*self.width, self.D))
        ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX

        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyMultiGraph(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyMultiGraph, self).__init__()
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        self.adj_gg = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.D = args.D
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.FCs_X_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FCs_Z_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])

        self.FCs_X_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

        # self.DCGRU_Z = layers.RNN(DCGRUCell(D, self.adj_rr, 1, self.num_cells, 'laplacian'), return_sequences=True)
        # self.FCN_Z = layers.Dense(self.num_nodes, use_bias=False)
        self.FCN_Z = keras.Sequential([
                            layers.Dense(D, use_bias=False),
                            layers.Dense(self.num_nodes, use_bias=False)])
        self.DCGRU_X = layers.RNN(DCGRUCell(D, self.adj_gg, 3, self.num_nodes, 'laplacian'))


    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        print('TE', TE.dtype)

        STEZ = self.STEZ_layer(self.SE_Z, TE[:, :self.P, :])
        # Z = tf.expand_dims(Z, -1)
        Z = self.FCs_Z_in(Z)
        Z = Z + STEZ

        Z = tf.transpose(Z, (0, 1, 3, 2))
        Z = self.FCN_Z(Z)
        ZX = tf.transpose(Z, (0, 1, 3, 2))
        
        
        # Z = self.DCGRU_Z(Z)
        # Z = tf.reshape(Z, (-1, self.P, self.num_cells, self.D))
        # ZX = self.adj_gr @ Z

        STEX = self.STEX_layer(self.SE_X, TE[:, :self.P, :])
        X = tf.expand_dims(X, -1)
        X = self.FCs_X_in(X)
        X = X + STEX + ZX


        X = self.DCGRU_X(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FCs_X_out(X)
        Y = tf.transpose(Y, (0, 2, 1))

        return Y



class MyMATURE_STE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyMATURE_STE, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.num_K = 15
        self.num_S = 60
        self.gamma = 0.4
        
    def build(self, input_shape):
        self.marn_cell_r = MARN(self.D, self.num_K, self.num_S)
        self.marn_cell_s = MARN(self.D, self.num_K, self.num_S)
        self.KAM_cell = KAM(self.D, self.gamma, self.num_K, self.num_S)
        self.FCs_final = keras.Sequential([
                        layers.Dense(self.D, activation="relu"),
                        layers.Dense(self.D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])


    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        
        TEX = tf.cast(TE[:, :-1, :], tf.int32)
        dayofweek = tf.one_hot(TEX[..., 0], depth = 7)
        timeofday = tf.one_hot(TEX[..., 1], depth = 24)
        TEX = tf.concat((dayofweek, timeofday), axis = -1)
        X = tf.concat((X, TEX), -1)
        Z = tf.concat((Z, TEX), -1)

        batch_size = tf.shape(X)[0]
        h_zero = tf.zeros((batch_size, self.D))
        c_zero = tf.zeros((batch_size, self.D))
        M_r_zero = tf.zeros((batch_size, self.num_K, self.num_S))
        M_s_zero = tf.zeros((batch_size, self.num_K, self.num_S))
        k_zero = tf.zeros((batch_size, self.num_S))
        l_zero = tf.zeros((batch_size, self.num_S))
        b_zero = tf.zeros((batch_size, self.num_S))

        h_r_prev, h_s_prev = h_zero, h_zero
        c_r_prev, c_s_prev = c_zero, c_zero
        M_r_prev, M_s_prev = M_r_zero, M_s_zero
        k_r_prev, k_s_prev = k_zero, k_zero
        l_prev = l_zero
        b_prev = b_zero


        # build model (memory adaption frmo r -> s)
        for i in range(self.P):
            x_r_inp = Z[:, i, :]
            x_s_inp = X[:, i, :]
            
            # MARN cell, mode R
            output_h_r, (h_r_curr, c_r_curr, M_r_curr, k_r_curr) = self.marn_cell_r(x_r_inp, (h_r_prev, c_r_prev, M_r_prev, k_r_prev))
            # Knowledge Adaption module
            M_s_new, (l_curr, b_curr) = self.KAM_cell((M_r_prev, M_s_prev, M_r_curr), (l_prev, b_prev))
            (l_prev, b_prev) = (l_curr, b_curr)
            # MARN cell, mode S
            output_h_s, (h_s_curr, c_s_curr, M_s_curr, k_s_curr) = self.marn_cell_s(x_s_inp, (h_s_prev, c_s_prev, M_s_new, k_s_prev))
            
            # update states
            (h_r_prev, c_r_prev, M_r_prev, k_r_prev) = (h_r_curr, c_r_curr, M_r_curr, k_r_curr)
            (h_s_prev, c_s_prev, M_s_prev, k_s_prev) = (h_s_curr, c_s_curr, M_s_curr, k_s_curr)
            
        final_h_r = output_h_r
        final_h_s = output_h_s
        
        final_h = tf.concat((final_h_r, final_h_s), -1)
        Y = self.FCs_final(final_h)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y



class MyMATURE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyMATURE, self).__init__()
        self.D = args.D
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.num_K = 15
        self.num_S = 60
        self.gamma = 0.4
        
    def build(self, input_shape):
        self.marn_cell_r = MARN(self.D, self.num_K, self.num_S)
        self.marn_cell_s = MARN(self.D, self.num_K, self.num_S)
        self.KAM_cell = KAM(self.D, self.gamma, self.num_K, self.num_S)
        self.FCs_final = keras.Sequential([
                        layers.Dense(self.D, activation="relu"),
                        layers.Dense(self.D, activation="relu"),
                        layers.Dense(self.Q*self.num_nodes)])


    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']

        batch_size = tf.shape(X)[0]
        h_zero = tf.zeros((batch_size, self.D))
        c_zero = tf.zeros((batch_size, self.D))
        M_r_zero = tf.zeros((batch_size, self.num_K, self.num_S))
        M_s_zero = tf.zeros((batch_size, self.num_K, self.num_S))
        k_zero = tf.zeros((batch_size, self.num_S))
        l_zero = tf.zeros((batch_size, self.num_S))
        b_zero = tf.zeros((batch_size, self.num_S))

        h_r_prev, h_s_prev = h_zero, h_zero
        c_r_prev, c_s_prev = c_zero, c_zero
        M_r_prev, M_s_prev = M_r_zero, M_s_zero
        k_r_prev, k_s_prev = k_zero, k_zero
        l_prev = l_zero
        b_prev = b_zero


        # build model (memory adaption frmo r -> s)
        for i in range(self.P):
            x_r_inp = Z[:, i, :]
            x_s_inp = X[:, i, :]
            
            # MARN cell, mode R
            output_h_r, (h_r_curr, c_r_curr, M_r_curr, k_r_curr) = self.marn_cell_r(x_r_inp, (h_r_prev, c_r_prev, M_r_prev, k_r_prev))
            # Knowledge Adaption module
            M_s_new, (l_curr, b_curr) = self.KAM_cell((M_r_prev, M_s_prev, M_r_curr), (l_prev, b_prev))
            (l_prev, b_prev) = (l_curr, b_curr)
            # MARN cell, mode S
            output_h_s, (h_s_curr, c_s_curr, M_s_curr, k_s_curr) = self.marn_cell_s(x_s_inp, (h_s_prev, c_s_prev, M_s_new, k_s_prev))
            
            # update states
            (h_r_prev, c_r_prev, M_r_prev, k_r_prev) = (h_r_curr, c_r_curr, M_r_curr, k_r_curr)
            (h_s_prev, c_s_prev, M_s_prev, k_s_prev) = (h_s_curr, c_s_curr, M_s_curr, k_s_curr)
            
        final_h_r = output_h_r
        final_h_s = output_h_s
        
        final_h = tf.concat((final_h_r, final_h_s), -1)
        Y = self.FCs_final(final_h)
        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))
        return Y


class DNN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(DNN, self).__init__()
        self.D = args.D
        self.NX = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FCs = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D, activation="relu"),
                        layers.Dense(self.NX)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        X = tf.reshape(X, (-1, self.P*self.NX))
        X = self.FCs(X)
        Y = tf.reshape(X, (-1, self.Q, self.NX))
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
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        
        # TEX = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TEX[..., 0], depth = 7)
        # timeofday = tf.one_hot(TEX[..., 1], depth = 24)
        # TEX = tf.concat((dayofweek, timeofday), axis = -1)
        # X = tf.concat((X, TEX), -1)

        X = self.FCs_1(X)
        X = self.lstm(X)
        Y = self.FCs_2(X)

        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))

        return Y


class MyLSTMKA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyLSTMKA, self).__init__()
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

        self.ZFCs_1 = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        # self.ZFCs_2 = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(self.Q*self.num_nodes)])
        self.Zlstm = layers.LSTM(self.D)

    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        
        # TEX = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TEX[..., 0], depth = 7)
        # timeofday = tf.one_hot(TEX[..., 1], depth = 24)
        # TEX = tf.concat((dayofweek, timeofday), axis = -1)
        # X = tf.concat((X, TEX), -1)

        X = self.FCs_1(X)
        X = self.lstm(X)
        # X = self.FCs_2(X)

        Z = self.ZFCs_1(Z)
        Z = self.Zlstm(Z)
        # Y = self.FCs_2(X)
        X = tf.concat((X, Z), -1)
        Y = self.FCs_2(X)

        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))

        return Y


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
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        
        # TEX = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TEX[..., 0], depth = 7)
        # timeofday = tf.one_hot(TEX[..., 1], depth = 24)
        # TEX = tf.concat((dayofweek, timeofday), axis = -1)
        # X = tf.concat((X, TEX), -1)

        X = self.FCs_1(X)
        X = self.gru(X)
        Y = self.FCs_2(X)

        Y = tf.reshape(Y, (-1, self.Q, self.num_nodes))

        return Y

class MyDCGRU_GST(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU_GST, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRU_GST_Cell(D, 1, self.num_nodes), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']

        TE = tf.cast(TE[:, :self.P :], tf.int32)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        TE = tf.concat((dayofweek, timeofday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)

        TE = tf.tile(TE, (1, 1, self.num_nodes, 1))
        X = tf.expand_dims(X, -1)
        X = tf.concat((X, TE), -1)

        X = self.FC_XC_in(X)
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y

class MyDCGRUSTE(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTE, self).__init__()
        self.num_nodes = extdata['ADJ_GG'].shape[0]
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        # TE = tf.cast(TE[:, :-1, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)
        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        X = X + STEX
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y

class MyDCGRUSTEKA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTEKA, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.SEZ = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SEZ', dtype=tf.float32)
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

                            
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_ZC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        STE = self.STE_layer(self.SE, TE)
        STEZ = self.STE_layer(self.SEZ, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        # TE = tf.cast(TE[:, :self.P, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)

        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))
        X = tf.expand_dims(X, -1) #+ STEX
        Z = tf.expand_dims(Z, -1) #+ STEZX
        # X = tf.concat((X, TE), -1)

        X = self.FC_XC_in(X) + STEX
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        X = self.FC_XC_out(X)

        
        Z = self.FC_ZC_in(Z) + STEZX
        Z = self.FC_ZC_DCGRU(Z)
        Z = tf.reshape(Z, (-1, self.num_nodes, self.D))
        Z = self.FC_ZC_out(Z)

        Y = Z + X


        Y = tf.transpose(Y, (0, 2, 1))
        return Y

from MFGCGRU_cell import * 

class MyDCGRUSTEKAP(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUSTEKAP, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.num_cells = extdata['num_cells']
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.P = args.P
        self.Q = args.Q
        self.POISEZ = extdata['SEZ']
        # print('SEZ',len(self.adj_rr), len(self.SEZ))
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.SEZ = self.add_weight(shape=(self.num_cells, self.D),
                                        initializer='glorot_uniform',
                                        name='SEZ', dtype=tf.float32)
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

                            
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        # self.FC_ZC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_ZC_DCGRU = tf.keras.layers.RNN(MFGCGRU(D, self.num_cells, adj_mats=[self.adj_rr], ext_feats = self.POISEZ))
        self.FC_ZC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        STE = self.STE_layer(self.SE, TE)
        STEZ = self.STE_layer(self.SEZ, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        # TE = tf.cast(TE[:, :self.P, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)

        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))
        # X = tf.expand_dims(X, -1) + STEX
        # Z = tf.expand_dims(Z, -1) + STEZX
        # X = tf.concat((X, TE), -1)

        # print(X.shape, STEX.shape)

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X) + STEX
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        X = self.FC_XC_out(X)

        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z) + STEZX
        Z = self.FC_ZC_DCGRU(Z)
        Z = tf.reshape(Z, (-1, self.num_cells, self.D))
        Z = self.FC_ZC_out(Z)

        print('Z, X', Z.shape, X.shape, self.adj_gr.shape)
        Z = self.adj_gr @ Z
        Y = Z + X
        # Y = self.FC_XC_out(Y)


        Y = tf.transpose(Y, (0, 2, 1))
        return Y

class MyDCGRUKA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRUKA, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

                            
        self.FC_ZC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_ZC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_ZC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']

        # TE = tf.cast(TE[:, :self.P, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)

        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))
        X = tf.expand_dims(X, -1)
        Z = tf.expand_dims(Z, -1)
        # X = tf.concat((X, TE), -1)

        X = self.FC_XC_in(X)
        X = self.FC_XC_DCGRU(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        X = self.FC_XC_out(X)

        
        Z = self.FC_ZC_in(Z)
        Z = self.FC_ZC_DCGRU(Z)
        Z = tf.reshape(Z, (-1, self.num_nodes, self.D))
        Z = self.FC_ZC_out(Z)

        Y = Z + X


        Y = tf.transpose(Y, (0, 2, 1))
        return Y


class MyDCGRU(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyDCGRU, self).__init__()
        self.num_nodes = extdata['num_nodes']
        self.D = args.D
        self.adj_mat = extdata['ADJ_GG']
        self.P = args.P
        self.Q = args.Q
        
    def build(self, input_shape):
        D = self.D
        self.FC_XC_in = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D)])
        self.FC_XC_DCGRU1 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_sequences=True, return_state=False)
        self.FC_XC_DCGRU2 = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 2, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_XC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(self.Q)])

    def call(self, kwargs):
        X, TE = kwargs['X'], kwargs['TE']

        # TE = tf.cast(TE[:, :self.P, :], tf.int32)
        # dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        # timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # TE = tf.concat((dayofweek, timeofday), axis = -1)
        # TE = tf.expand_dims(TE, axis = 2)

        # TE = tf.tile(TE, (1, 1, self.num_nodes, 1))
        X = tf.expand_dims(X, -1)
        # X = tf.concat((X, TE), -1)

        X = self.FC_XC_in(X)
        X = self.FC_XC_DCGRU1(X)
        X = tf.reshape(X, (-1, self.P, self.num_nodes, self.D))
        X = self.FC_XC_DCGRU2(X)
        X = tf.reshape(X, (-1, self.num_nodes, self.D))
        Y = self.FC_XC_out(X)
        Y = tf.transpose(Y, (0, 2, 1))
        return Y



class MyGMANKA_EF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANKA_EF, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        
        
        self.WF = WeightFusion(D)
        self.FC_ZC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_ZC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZ = self.STEZ_layer(self.SE, TE)
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X) + STEX

        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z) + STEZX

        X = tf.stack((X, Z), -1)
        X = self.WF(X)
        
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)

        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)
        return Y


class MyGMANKA_MF(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANKA_MF, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        self.ZGSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.WF = WeightFusion(D)
        self.FC_ZC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_ZC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZ = self.STEZ_layer(self.SE, TE)
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)

        
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z)
        for i in range(self.L):
            Z = self.ZGSTAC_enc[i](Z, STEZX)

        X = tf.stack((X, Z), -1)
        X = self.WF(X)
        X = self.C_trans_layer(X, STEX, STEY)

        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)

        return Y



class MyGMANKAOD(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANKAOD, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.adjrwr1 = extdata['RWR1'] 
        self.adjrwr2 = extdata['RWR2'] 
        
    def build(self, input_shape):
        D = self.D

        # self.Zout = self.add_weight(shape=(self.num_nodes, D),
        #                                 initializer='glorot_uniform',
        #                                 name='Zout', dtype=tf.float32)
        # self.FC_Zout = keras.Sequential([
        #                 layers.Dense(self.Q, activation="relu")])

        # self.Ain = self.add_weight(shape=(self.num_nodes_in, self.num_nodes, D),
        #                                 initializer='glorot_uniform',
        #                                 name='Ain', dtype=tf.float32)
        # self.Aout = self.add_weight(shape=(self.num_nodes_out, self.num_nodes, D),
        #                                 initializer='glorot_uniform',
        #                                 name='Aout', dtype=tf.float32)
        # self.FC_Ain = keras.Sequential([
        #                 layers.Dense(1, activation="relu")])
        # self.FC_Aout = keras.Sequential([
        #                 layers.Dense(1, activation="relu")])

        



        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        self.ZGSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.ZC_trans_layer = TransformAttention(self.K, self.d)
        self.ZGSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.WF = WeightFusion(D)
        # self.WF0 = WeightFusion(D)
        self.FC_ZC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_ZC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)



        # netflow = self.FC_Ain(self.Ain) - self.FC_Aout(self.Aout)

        # Zout = self.FC_Zout(self.Zout)



        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZ = self.STEZ_layer(self.SE, TE)
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X0 = X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        #X = self.FC_XC_out(X)
        #Y = tf.squeeze(X, -1)

        
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z)
        for i in range(self.L):
            Z = self.ZGSTAC_enc[i](Z, STEZX)
        Z = self.ZC_trans_layer(Z, STEZX, STEZY)
        # Z = self.WF0(tf.stack((Z, X0), -1))
        for i in range(self.L):
            Z = self.ZGSTAC_dec[i](Z, STEZY)
        # Z = self.FC_ZC_out(Z)
        # Z = tf.squeeze(Z, -1)

        Z1 = self.adjrwr1 @ Z
        Z2 = self.adjrwr2 @ Z

        
        X = tf.stack((X, Z, Z1, Z2), -1)
        X = self.WF(X)
        
        #X = tf.concat((X, Z), -1)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)

        # ZY = self.FC_ZC_out(Z)
        # ZY = tf.squeeze(ZY, -1) 
        return Y



class MyGMANKA(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANKA, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_nodes, D)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        self.ZGSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.ZC_trans_layer = TransformAttention(self.K, self.d)
        self.ZGSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.WF = WeightFusion(D)
        self.FC_ZC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_ZC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZ = self.STEZ_layer(self.SE, TE)
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        #X = self.FC_XC_out(X)
        #Y = tf.squeeze(X, -1)

        
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z)
        for i in range(self.L):
            Z = self.ZGSTAC_enc[i](Z, STEZX)
        Z = self.ZC_trans_layer(Z, STEZX, STEZY)
        for i in range(self.L):
            Z = self.ZGSTAC_dec[i](Z, STEZY)
        #Z = self.FC_ZC_out(Z)
        #ZY = tf.squeeze(Z, -1)

        

        
        X = tf.stack((X, Z), -1)
        print(X.shape)
        X = self.WF(X)
        
        #X = tf.concat((X, Z), -1)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)

        return Y





class MyGMANKAP(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANKAP, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.num_cells = extdata['num_cells']
        self.adj_mat = extdata['ADJ_GG']
        self.adj_gr = extdata['ADJ_GR']
        self.adj_rr = extdata['ADJ_RR']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        self.SE = self.add_weight(shape=(self.num_nodes, self.D),
                                        initializer='glorot_uniform',
                                        name='SE', dtype=tf.float32)
        self.SEZ = self.add_weight(shape=(self.num_cells, self.D),
                                        initializer='glorot_uniform',
                                        name='SEZ', dtype=tf.float32)
        
        self.GSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.C_trans_layer = TransformAttention(self.K, self.d)
        self.GSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.FC_XC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_XC_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

                        
        # self.ZGSTAC_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        # self.ZC_trans_layer = TransformAttention(self.K, self.d)
        # self.ZGSTAC_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        
        self.WF = WeightFusion(D)
        self.FC_ZC_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])

        # self.FC_ZC_DCGRU = tf.keras.layers.RNN(DCGRUCell(D, self.adj_mat, 3, self.num_nodes, 'laplacian'), return_state=False)
        self.FC_ZC_DCGRU = tf.keras.layers.RNN(MFGCGRU(D, self.num_cells, adj_mats=[self.adj_rr], ext_feats = self.POISEZ))
        self.FC_ZC_out = keras.Sequential([
                            layers.Dense(D, activation="relu"),
                            layers.Dense(D*self.Q)])
        
        # self.FC_ZC_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]
        STEZ = self.STEZ_layer(self.SEZ, TE)
        STEZX, STEZY = STEZ[:, :self.P, :], STEZ[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        #X = self.FC_XC_out(X)
        #Y = tf.squeeze(X, -1)

        
        # Z = tf.expand_dims(Z, -1)
        # Z = self.FC_ZC_in(Z)
        # for i in range(self.L):
        #     Z = self.ZGSTAC_enc[i](Z, STEZX)
        # Z = self.ZC_trans_layer(Z, STEZX, STEZY)
        # for i in range(self.L):
        #     Z = self.ZGSTAC_dec[i](Z, STEZY)
        # #Z = self.FC_ZC_out(Z)
        # #ZY = tf.squeeze(Z, -1)

        Z = tf.expand_dims(Z, -1)
        Z = self.FC_ZC_in(Z) + STEZX
        Z = self.FC_ZC_DCGRU(Z)
        Z = tf.reshape(Z, (-1, self.num_cells, self.D))
        Z = self.FC_ZC_out(Z)
        Z = tf.reshape(Z, (-1, self.num_cells, self.Q, self.D))
        Z = tf.transpose(Z, (0, 2, 1, 3))
        Z = self.adj_gr @ Z

        X = tf.stack((X, Z), -1)
        print(X.shape)
        X = self.WF(X)
        
        #X = tf.concat((X, Z), -1)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)

        return Y




class MyGMANB(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMANB, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        
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
        X, TE, Z = kwargs['X'], kwargs['TE'], kwargs['Z']
        TE = tf.cast(TE, tf.int32)
        
        STE = self.STE_layer(self.SE, TE)
        STEX, STEY = STE[:, :self.P, :], STE[:, self.P:, :]

        X = tf.expand_dims(X, -1)
        Z = tf.expand_dims(Z, -1)
        X = tf.concat((X, Z), -1)
        X = self.FC_XC_in(X)
        for i in range(self.L):
            X = self.GSTAC_enc[i](X, STEX)
        X = self.C_trans_layer(X, STEX, STEY)
        for i in range(self.L):
            X = self.GSTAC_dec[i](X, STEY)
        X = self.FC_XC_out(X)
        Y = tf.squeeze(X, -1)
        return Y

class MyGMAN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMAN, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.num_nodes = extdata['num_nodes']
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        
    def build(self, input_shape):
        D = self.D
        self.STE_layer = STEmbedding(self.num_nodes, D)
        
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


class MyGMSTARKS(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTARKS, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.RSTA_enc = [RSTAttBlock2(self.K, self.d, self.ADJ_RR, self.POISEZ) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlock2(self.K, self.d, self.ADJ_RR, self.POISEZ) for _ in range(self.L)]
        
        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_Z_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        ZX = self.ADJ_GR @ Z
        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        ZY = self.ADJ_GR @ Z

        
        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)

        print(X.shape, ZX.shape, )

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)
        return Y


class MyGMSTARKBICNN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTARKBICNN, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.RSTA_enc = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNN(self.K, self.d, self.cnn_size) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_Z_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        # ZX = self.ADJ_GR @ Z
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        # ZY = self.ADJ_GR @ Z
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        
        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)

        print(X.shape, ZX.shape, )

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)
        return Y


class MyGMSTARKBICNNPOI(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTARKBICNNPOI, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.cnn_size = args.cnn_size
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)
        # self.FC_POISEZ = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(D)])

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.RSTA_enc = [RSTAttBlockCNNPOI(self.K, self.d, self.cnn_size, self.POISEZ) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlockCNNPOI(self.K, self.d, self.cnn_size, self.POISEZ) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_Z_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        # SEZ = self.FC_POISEZ(self.POISEZ)
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        # ZX = self.ADJ_GR @ Z
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        # ZY = self.ADJ_GR @ Z
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        
        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)

        print(X.shape, ZX.shape, )

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)
        return Y



class MyGMSTARKBI2(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyGMSTARKBI2, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        
    def build(self, input_shape):
        D = self.D

        self.SE_X = self.add_weight(shape=(self.num_nodes, D),
                                        initializer='glorot_uniform',
                                        name='SE1', dtype=tf.float32)
        self.SE_Z = self.add_weight(shape=(self.num_cells, D),
                                        initializer='glorot_uniform',
                                        name='SE2', dtype=tf.float32)

        self.STEX_layer = STEmbedding(self.num_nodes, D)
        self.STEZ_layer = STEmbedding(self.num_cells, D)
        
        self.GSTA_enc = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        self.G_trans_layer = TransformAttention(self.K, self.d)
        self.GSTA_dec = [GSTAttBlock(self.K, self.d) for _ in range(self.L)]
        
        self.RSTA_enc = [RSTAttBlock2(self.K, self.d, self.ADJ_RR, self.POISEZ) for _ in range(self.L)]
        self.R_trans_layer = TransformAttention(self.K, self.d)
        self.RSTA_dec = [RSTAttBlock2(self.K, self.d, self.ADJ_RR, self.POISEZ) for _ in range(self.L)]
    
        self.SX_trans_layer = BipartiteAttention(self.K, self.d)
        self.SY_trans_layer = BipartiteAttention(self.K, self.d)

        self.FC_X_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        self.FC_X_out = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(1)])

        self.FC_Z_in = keras.Sequential([
                        layers.Dense(D, activation="relu"),
                        layers.Dense(D)])
        
        # self.FC_Z_out = keras.Sequential([
        #                 layers.Dense(D, activation="relu"),
        #                 layers.Dense(1)])
        
    def call(self, kwargs):
        X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)
        
        STEZ0 = self.STEZ_layer(self.SE_Z, TE)
        STEZ, STEZY = STEZ0[:, :self.P, :], STEZ0[:, self.P:, :]
        Z = tf.expand_dims(Z, -1)
        Z = self.FC_Z_in(Z)
        Z = Z + STEZ

        STEX0 = self.STEX_layer(self.SE_X, TE)
        STEX, STEXY = STEX0[:, :self.P, :], STEX0[:, self.P:, :]
        X = tf.expand_dims(X, -1)
        X = self.FC_X_in(X)
        X = X + STEX


        for i in range(self.L):
            Z = self.RSTA_enc[i](Z, STEZ)
        # ZX = self.ADJ_GR @ Z
        ZX = self.SX_trans_layer(Z, STEZ, STEX)

        Z = self.R_trans_layer(Z, STEZ, STEZY)
        for i in range(self.L):
            Z = self.RSTA_dec[i](Z, STEZY)
        # ZY = self.ADJ_GR @ Z
        ZY = self.SY_trans_layer(Z, STEZY, STEXY)

        
        # X = tf.expand_dims(X, -1)
        # X = self.FC_X_in(X)

        print(X.shape, ZX.shape, )

        for i in range(self.L):
            X = self.GSTA_enc[i](X, ZX)
        X = self.G_trans_layer(X, ZX, ZY)
        for i in range(self.L):
            X = self.GSTA_dec[i](X, ZY)
        X = self.FC_X_out(X)
        Y = tf.squeeze(X, -1)
        return Y



from tgcn import tgcnCell

class MyTGCN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyTGCN, self).__init__()
        self.D = args.D
        self.K = args.K
        self.d = args.d
        self.L = args.L
        self.P = args.P
        self.Q = args.Q
        self.SE = extdata['SE']
        self.POISEZ = extdata['SEZ']
        self.ADJ_RR = extdata['ADJ_RR']
        self.ADJ_GR = extdata['ADJ_GR']
        self.ADJ_GG = extdata['ADJ_GG']
        self.num_nodes = int(extdata['ADJ_GG'].shape[0])
        self.num_cells = int(extdata['ADJ_RR'].shape[0])
        
    def build(self, input_shape):
        self.cell_1 = tgcnCell(self.D, self.ADJ_GG, num_nodes=self.num_nodes)
        self.cell = tf.nn.rnn_cell.MultiRNNCell([self.cell_1], state_is_tuple=True)
        # self.FC_out = keras.Sequential([
        #                     layers.Dense(self.D, activation="relu"),
        #                     layers.Dense(self.Q)])
        self.m_weights = self.add_weight(shape=(self.D, self.Q),
                                        initializer='glorot_uniform',
                                        name='weight_o', dtype=tf.float32)
        self.m_biases = self.add_weight(shape=(self.Q,),
                                        initializer='glorot_uniform',
                                        name='bias_o', dtype=tf.float32)
        # self.weights = {
        #     'out': tf.Variable(tf.random_normal([self.D, self.Q], mean=1.0), name='weight_o')}
        # self.biases = {
        #     'out': tf.Variable(tf.random_normal([self.Q]),name='bias_o')}
        
    def call(self, kwargs):
        _X, Z, TE = kwargs['X'], kwargs['Z'], kwargs['TE']
        TE = tf.cast(TE, tf.int32)

        _X = tf.unstack(_X, axis=1)
        outputs, states = tf.nn.static_rnn(self.cell, _X, dtype=tf.float32)
        m = []
        for i in outputs:
            o = tf.reshape(i,shape=[-1,self.num_nodes,self.D])
            o = tf.reshape(o,shape=[-1,self.D])
            m.append(o)
        last_output = m[-1]
        output = tf.matmul(last_output, self.m_weights) + self.m_biases
        output = tf.reshape(output,shape=[-1,self.num_nodes,self.Q])
        output = tf.transpose(output, perm=[0,2,1])
        output = tf.reshape(output, shape=[-1,self.num_nodes])
        return output



# def ModelSet(model_name, extdata, args, **kwargs):
#     model = str_to_class(model_name)(extdata, args)
#     model_output = model(kwargs)
#     return (model_output[0]) * extdata['std'] + extdata['mean'], model_output[1]