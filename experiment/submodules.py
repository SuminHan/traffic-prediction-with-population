import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class STDEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STDEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.D = D
        self.Kd = 4

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        self.DE = self.add_weight(shape=(1+self.Kd, self.D),
                                        initializer='glorot_uniform',
                                        name='DE', dtype=tf.float32)
        
    def call(self, SE, TE):
        print('SE', SE.shape, 'TE', TE.shape)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # monthofweek = tf.one_hot(TE[..., 2], depth = 12)
        TE = tf.concat((dayofweek, timeofday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        STE = SE + TE


        STE = tf.expand_dims(STE, axis = -2)
        STDE = STE + self.DE

        return STDE

class STEmbedding(tf.keras.layers.Layer):
    def __init__(self, num_nodes, D):
        super(STEmbedding, self).__init__()
        self.num_nodes = num_nodes
        self.D = D

    def build(self, input_shape):
        self.FC_TE = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, SE, TE):
        print('SE', SE.shape, 'TE', TE.shape)
        dayofweek = tf.one_hot(TE[..., 0], depth = 7)
        timeofday = tf.one_hot(TE[..., 1], depth = 24)
        # monthofweek = tf.one_hot(TE[..., 2], depth = 12)
        TE = tf.concat((dayofweek, timeofday), axis = -1)
        TE = tf.expand_dims(TE, axis = 2)
        TE = self.FC_TE(TE)
        
        STE = SE + TE
        return STE
    


# class DeconvLSTM(tf.keras.layers.Layer):
#     def __init__(self, num_nodes, D):
#         super(DeconvLSTM, self).__init__()
#         self.num_nodes = num_nodes
#         self.D = D

#     def build(self, input_shape):
#         self.FC_TE = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
        
#     def call(self, SE, TE):
#         print('SE', SE.shape, 'TE', TE.shape)
#         dayofweek = tf.one_hot(TE[..., 0], depth = 7)
#         timeofday = tf.one_hot(TE[..., 1], depth = 24)
#         # monthofweek = tf.one_hot(TE[..., 2], depth = 12)
#         TE = tf.concat((dayofweek, timeofday), axis = -1)
#         TE = tf.expand_dims(TE, axis = 2)
#         TE = self.FC_TE(TE)
        
#         STE = SE + TE
#         return STE




import tensorflow as tf
class DeConvLSTM():
    def __init__(self, num_classes, num_lstm_cells=128, num_lstm_layers=1,
                 kernel_size=(10), filter_size=[128, 256, 128], pool_size=(2),
                 num_cnn_layers=3, dropout_rate=0.8):
        self.num_classes = num_classes
        self.num_lstm_cells = num_lstm_cells
        self.num_lstm_layers = num_cnn_layers
        self.pool_size = pool_size
        self.num_cnn_layers = num_cnn_layers
        self.num_classes = num_classes
        self.dropout_rate = dropout_rate
        self.kernel_size = kernel_size
        self.filter_size = filter_size
        self.model = None

    def create_cnn_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Conv1D(self.filter_size[0], self.kernel_size, input_shape=input_shape))
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation('relu'))
        for layer in range(1, self.num_cnn_layers):
            model.add(tf.keras.layers.Conv1D(self.filter_size[layer], self.kernel_size))
            model.add(tf.keras.layers.BatchNormalization())
            model.add(tf.keras.layers.Activation('relu'))
        model.add(tf.keras.layers.AveragePooling1D(self.pool_size))
        model.add(tf.keras.layers.Flatten())
        return model

    def create_lstm_model(self, input_shape):
        model = tf.keras.models.Sequential()
        model.add(tf.keras.layers.Permute((2,1), input_shape=input_shape))
        model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells,
                  return_sequences=True))
        model.add(tf.keras.layers.Dropout(self.dropout_rate))
        for layer in range(1, self.num_lstm_layers):
            model.add(tf.keras.layers.CuDNNLSTM(self.num_lstm_cells, return_sequences=True))
            model.add(tf.keras.layers.Dropout(self.dropout_rate))
        model.add(tf.keras.layers.Flatten())
        return model

    def create_network(self, input_data, time_steps, num_features):
        cnn_input = tf.reshape(input_data, (-1, time_steps, num_features))
        lstm_input = tf.reshape(input_data, (-1, time_steps, num_features))
        shape_cnn = cnn_input.shape[1:]
        shape_lstm = lstm_input.shape[1:]
        lstm_input = tf.keras.layers.Input(shape=shape_lstm, name='lstm_input')
        cnn_input = tf.keras.layers.Input(shape=shape_cnn, name='cnn_input')
        cnn_out = self.create_cnn_model(shape_cnn)(cnn_input)
        lstm_out = self.create_lstm_model(shape_lstm)(lstm_input)
        network_output = tf.keras.layers.concatenate([cnn_out, lstm_out])
        network_output = tf.keras.layers.Dense(self.num_classes,
                                               activation=tf.nn.softmax,
                                               name='network_output'
                                               )(network_output)
        model = tf.keras.models.Model(inputs=[lstm_input, cnn_input],
                                      outputs=[network_output])
        model.compile(optimizer='adam', loss='categorical_crossentropy',
                      metrics=['accuracy'])
        self.model = model

    def fit(self, input_data, labels, num_epochs, time_steps, num_features,
            batch_size, learn_rate=0.01):
            cnn_input = np.reshape(input_data, (-1, time_steps, num_features))
            lstm_input = np.reshape(input_data, (-1, time_steps, num_features))
            self.model.fit({'lstm_input': lstm_input, 'cnn_input': cnn_input},
                       {'network_output': labels},
                       epochs=num_epochs, batch_size=batch_size,
                       validation_split=0.2)

    def evaluate(self, test_data, test_labels, time_steps, num_features):
        cnn_data = np.reshape(test_data, (-1, time_steps, num_features))
        lstm_data = np.reshape(test_data, (-1, time_steps, num_features))
        loss, accuracy = self.model.evaluate(x=[lstm_data, cnn_data], y=test_labels, steps=2)
        print("Model loss:", loss, ", Accuracy:", accuracy)
        return loss, 


# class STEmbedding_Y(tf.keras.layers.Layer):
#     def __init__(self, num_nodes, D):
#         super(STEmbedding_Y, self).__init__()
#         self.num_nodes = num_nodes
#         self.D = D

#     def build(self, input_shape):
#         self.FC_TE = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D),])
        
#     def call(self, SE, TEC, TEP, TEQ, TEY):
#         TEY = tf.expand_dims(TEY, axis=1)
        
#         num_c = TEC.shape[-2]
#         num_p = TEP.shape[-2]
#         num_q = TEQ.shape[-2]
#         num_y = TEY.shape[-2]
        
#         TE = tf.concat((TEC, TEP, TEQ, TEY), -2)
        
#         dayofweek = tf.one_hot(TE[..., 0], depth = 7)
#         timeofday = tf.one_hot(TE[..., 1], depth = 24)
#         minuteofday = tf.one_hot(TE[..., 2], depth = 4)
#         holiday = tf.one_hot(TE[..., 3], depth = 1)
#         TE = tf.concat((dayofweek, timeofday, minuteofday, holiday), axis = -1)
#         TE = tf.expand_dims(TE, axis = 2)
#         TE = self.FC_TE(TE)
        
#         STE = SE + TE
#         STE_C = STE[:, : num_c, ...]
#         STE_P = STE[:, num_c : num_c+num_p, ...]
#         STE_Q = STE[:, num_c+num_p : num_c+num_p+num_q, ...]
#         STE_Y = STE[:, num_c+num_p+num_q : , ...]
        
#         return STE_C, STE_P, STE_Q, STE_Y
    
class WeightFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(WeightFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FCs = []
        for i in range(input_shape[-1]):
            self.FCs.append(keras.Sequential([layers.Dense(1, use_bias=True),]))

        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),])
        
    def call(self, Xs):
        Zs = []
        for i in range(Xs.shape[-1]):
            Zs.append(self.FCs[i](Xs[..., i]))
        Z = tf.stack(Zs, -1)
        Z = tf.nn.softmax(Z, -1)
        #Z = tf.expand_dims(Z, -2)
        Z = tf.reduce_sum(Z * Xs, -1)
        return self.FC_H(Z)


class CPTFusion(tf.keras.layers.Layer):
    def __init__(self, D, out_dim):
        super(CPTFusion, self).__init__()
        self.D = D
        self.out_dim = out_dim

    def build(self, input_shape):
        self.FC_C = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_P = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_Q = keras.Sequential([
            layers.Dense(1, use_bias=False),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.out_dim),])
        
    def call(self, XC, XP, XQ):
        ZC = self.FC_C(XC)
        ZP = self.FC_P(XP)
        ZQ = self.FC_Q(XQ)

        Z = tf.concat((ZC, ZP, ZQ), -1)
        Z = tf.nn.softmax(Z)
        return self.FC_H(Z[..., 0:1] * XC + Z[..., 1:2] * XP + Z[..., 2:] * XQ)

class SpatialConvolution(tf.keras.layers.Layer):
    def __init__(self, D, adj_mat):
        super(SpatialConvolution, self).__init__()
        self.D = D
        self.adj_mat = adj_mat

    def build(self, input_shape):
        self.FC = keras.Sequential([
            layers.Dense(self.D, activation="relu", use_bias=False),])
        
    def call(self, X):
        X = self.FC(self.adj_mat @ X)
        return X

    
    
class SpatialAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(SpatialAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class TemporalAttention(tf.keras.layers.Layer):
    def __init__(self, K, d, use_mask=False):
        super(TemporalAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.use_mask = use_mask

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        D = self.D
        
        X = tf.concat((X, STE), axis = -1)
        query = self.FC_Q(X)
        key = self.FC_K(X)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        # if self.use_mask:
        #     batch_size = tf.shape(X)[0]
        #     num_step = X.get_shape()[1]#.value
        #     N = X.get_shape()[2]#.value
        #     mask = tf.ones(shape = (num_step, num_step))
        #     mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        #     mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
        #     mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
        #     mask = tf.cast(mask, dtype = tf.bool)
        #     attention = tf.compat.v2.where(
        #         condition = mask, x = attention, y = -2 ** 15 + 1)
            
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
class BipartiteAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(BipartiteAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        
        
        # query = tf.transpose(query, perm = (0, 2, 1, 3))
        # key = tf.transpose(key, perm = (0, 2, 3, 1))
        # value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        # X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    
    
class GatedFusion(tf.keras.layers.Layer):
    def __init__(self, D):
        super(GatedFusion, self).__init__()
        self.D = D

    def build(self, input_shape):
        self.FC_S = keras.Sequential([
            layers.Dense(self.D, use_bias=False),])
        self.FC_T = keras.Sequential([
            layers.Dense(self.D),])
        self.FC_H = keras.Sequential([
            layers.Dense(self.D, activation='relu'),
            layers.Dense(self.D),])
        
    def call(self, HS, HT):
        XS = self.FC_S(HS)
        XT = self.FC_T(HT)
        
        z = tf.nn.sigmoid(tf.add(XS, XT))
        H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
        H = self.FC_H(H)
        return H
    
class GSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(GSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.SA_layer = SpatialAttention(self.K, self.d)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SA_layer(X, STE)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H
    
class RSTAttBlock2(tf.keras.layers.Layer):
    def __init__(self, K, d, adj_mat, SEZ):
        super(RSTAttBlock2, self).__init__()
        self.K = K
        self.d = d
        self.units = self.D = K*d
        self.adj_mat = adj_mat
        self.adj_mats = [adj_mat]
        self.ext_feats = SEZ
        self.num_nodes = adj_mat.shape[0]

    def build(self, input_shape):
        # self.SA_layer = SpatialAttention(self.K, self.d)
        # self.SC_layer = SpatialConvolution(self.D, self.adj_mat)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]

        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = self.D
        self.rows_kernel = (self.D) * self.num_mx
        self.FC_gcn = tf.keras.layers.Dense(self.units, activation='relu')
        
        # self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
        #                                 initializer='glorot_uniform',
        #                                 name='c_kernel')
        # self.c_bias = self.add_weight(shape=(self.units,),
        #                               initializer='zeros',
        #                               name='c_bias')
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        support = self.adj_mat
        x = x0 = X
        x_support = support@x0
        # x_support = tf.expand_dims(x_support, 0)
        # concatenate convolved signal
        x = tf.concat([x, x_support], axis=-1)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5 # @ tf.transpose(FE)
            support = tf.nn.softmax(support) 

            x_support = support@x0
            # x_support = tf.expand_dims(x_support, 0) 
            # concatenate convolved signal
            x = tf.concat([x, x_support], axis=-1)


        # x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        # x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        # x = tf.reshape(x, shape=[-1, input_size * self.num_mx])
        x = self.FC_gcn(x)
        # x = tf.matmul(x, self.c_kernel)
        # x = tf.nn.bias_add(x, self.c_bias)

        HS = x
        # HS = self.SC_layer(X)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H




class RSTAttBlock3(tf.keras.layers.Layer):
    def __init__(self, K, d, adj_mat, SEZ, SE):
        super(RSTAttBlock3, self).__init__()
        self.K = K
        self.d = d
        self.units = self.D = K*d
        self.adj_mat = adj_mat
        self.adj_mats = [adj_mat]
        self.ext_feats = SEZ
        self.num_nodes = adj_mat.shape[0]
        self.SE = SE

    def build(self, input_shape):
        # self.SA_layer = SpatialAttention(self.K, self.d)
        # self.SC_layer = SpatialConvolution(self.D, self.adj_mat)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
        self.FCQ_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
                            tf.keras.layers.Dense(self.units, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        # self.FC_sent_feat = [keras.Sequential([
        #                         tf.keras.layers.Dense(self.units, activation='relu'),
        #                         tf.keras.layers.Dense(1, activation='relu')]) for i in range(len(self.ext_feats))]

        self.num_mx = 1 + len(self.adj_mats) + len(self.ext_feats)
        self.input_dim = self.D
        self.rows_kernel = (self.D) * self.num_mx
        self.FC_gcn = [tf.keras.layers.Dense(self.units, activation='relu') for _ in range(self.num_mx)]
        
        # self.c_kernel = self.add_weight(shape=(self.rows_kernel, self.units),
        #                                 initializer='glorot_uniform',
        #                                 name='c_kernel')
        # self.c_bias = self.add_weight(shape=(self.units,),
        #                               initializer='zeros',
        #                               name='c_bias')
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        support = self.adj_mat
        x_list = []
        x = x0 = X
        x_list.append(x0)
        x_support = support@x0
        # x_support = tf.expand_dims(x_support, 0)
        # concatenate convolved signal
        # x = tf.concat([x, x_support], axis=-1)
        x_list.append(x_support)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)
            # FES = tf.squeeze(self.FC_sent_feat[i](tf.concat((feat, self.SE), -1)), -1)
            # FES = tf.squeeze(self.FC_sent_feat[i](feat), -1)
            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5
            support = tf.nn.softmax(support) 

            # support_exp = tf.exp(support)
            # support_exp_sum = tf.reduce_sum(support_exp, -1)
            # support = support_exp / tf.expand_dims((FES + support_exp_sum), -1)

            x_support = support@x0 
                        
            # support = tf.matmul(FEQ, FEK, transpose_b=True) / self.units**0.5 # @ tf.transpose(FE)
            # support = tf.nn.softmax(support) 
            # x_support = support@x0
            # x_support = tf.expand_dims(x_support, 0) 
            # concatenate convolved signal
            # x = tf.concat([x, x_support], axis=-1)
            x_list.append(x_support)
        
        xs = []
        for i in range(self.num_mx):
            x = x_list[i]
            x = self.FC_gcn[i](x)
            xs.append(x)
        
        xs = tf.stack(xs, -1)
        x = tf.reduce_mean(xs, -1)

        # x = tf.reshape(x, shape=[self.num_mx, self.num_nodes, input_size, -1])
        # x = tf.transpose(x, perm=[3, 1, 2, 0])  # (batch_size, num_nodes, input_dim + units, order)
        # x = tf.reshape(x, shape=[-1, input_size * self.num_mx])
        # x = self.FC_gcn(x)
        # x = tf.matmul(x, self.c_kernel)
        # x = tf.nn.bias_add(x, self.c_bias)

        HS = x
        # HS = self.SC_layer(X)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H


class RSTAttBlock(tf.keras.layers.Layer):
    def __init__(self, K, d, adj_mat):
        super(RSTAttBlock, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.adj_mat = adj_mat

    def build(self, input_shape):
        # self.SA_layer = SpatialAttention(self.K, self.d)
        self.SC_layer = SpatialConvolution(self.D, self.adj_mat)
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.SC_layer(X)
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H


class RSTAttBlockCNN(tf.keras.layers.Layer):
    def __init__(self, K, d, H, W, cnn_size):
        super(RSTAttBlockCNN, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.H = H
        self.W = W
        self.cnn_size = cnn_size

    def build(self, input_shape):
        # self.SA_layer = SpatialAttention(self.K, self.d)
        # self.SC_layer = SpatialConvolution(self.D, self.adj_mat)
        self.CNN_layer = layers.Conv2D(self.D, self.cnn_size, padding='same')
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
    def call(self, X, STE):
        K = self.K
        d = self.d

        num_seq = X.shape[1]
        
        HT = self.TA_layer(X, STE)

        XS = tf.reshape(X, (-1, self.H, self.W, self.D))
        HS = self.CNN_layer(XS)
        HS = tf.reshape(HS, (-1, num_seq, self.H*self.W, self.D))
        
        H = self.GF(HS, HT)
        return X + H


class POIConvolution(tf.keras.layers.Layer):
    def __init__(self, D, ext_feats):
        super(POIConvolution, self).__init__()
        self.D = D
        self.ext_feats = ext_feats
        self.num_mx = 1 + len(self.ext_feats)

    def build(self, input_shape):
        self.FCQ_feat = [keras.Sequential([
            tf.keras.layers.Dense(self.D, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FCK_feat = [keras.Sequential([
            tf.keras.layers.Dense(self.D, activation='relu', use_bias=False)]) for i in range(len(self.ext_feats))]
        self.FC_gcn = [tf.keras.layers.Dense(self.D, activation='relu') for _ in range(self.num_mx)]
        
        
    def call(self, X):
        x_list = []
        x0 = X
        x_list.append(X)

        for i, feat in enumerate(self.ext_feats):
            # premultiply the concatened inputs and state with support matrices
            FEQ = self.FCQ_feat[i](feat)
            FEK = self.FCK_feat[i](feat)

            support = tf.matmul(FEQ, FEK, transpose_b=True) / self.D**0.5
            support = tf.nn.softmax(support) 
            x_support = support@x0 
            x_list.append(x_support)
        
        xs = []
        for i in range(self.num_mx):
            xs.append(self.FC_gcn[i](x_list[i]))
        
        xs = tf.stack(xs, -1)
        x_output = tf.reduce_mean(xs, -1)

        return x_output

class RSTAttBlockCNNPOI(tf.keras.layers.Layer):
    def __init__(self, K, d, cnn_size, ext_feats):
        super(RSTAttBlockCNNPOI, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d
        self.cnn_size = cnn_size
        self.ext_feats = ext_feats

    def build(self, input_shape):
        # self.SA_layer = SpatialAttention(self.K, self.d)
        # self.SC_layer = SpatialConvolution(self.D, self.adj_mat)
        self.CNN_layer = layers.Conv2D(self.D, self.cnn_size, padding='same')
        self.POI_layer = POIConvolution(self.D, self.ext_feats)
        self.FC_Slayer = keras.Sequential([
                            layers.Dense(self.D, activation="relu"),
                            layers.Dense(self.D)])
        self.TA_layer = TemporalAttention(self.K, self.d)
        self.GF = GatedFusion(self.D)
        
        
    def call(self, X, STE):
        K = self.K
        d = self.d
        
        HS = self.FC_Slayer(tf.concat((self.CNN_layer(X), self.POI_layer(X)), -1))
        HT = self.TA_layer(X, STE)
        H = self.GF(HS, HT)
        return X + H

class TransformAttention(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu")])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D)])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)
    
        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        query = tf.transpose(query, perm = (0, 2, 1, 3))
        key = tf.transpose(key, perm = (0, 2, 3, 1))
        value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
        attention = tf.matmul(query, key)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.transpose(X, perm = (0, 2, 1, 3))
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X
    

# class TransformAttention2(tf.keras.layers.Layer):
#     def __init__(self, K, d):
#         super(TransformAttention2, self).__init__()
#         self.K = K
#         self.d = d
#         self.D = K*d

#     def build(self, input_shape):
#         self.FC_Q = keras.Sequential([
#             layers.Dense(self.D, activation="relu")])
#         self.FC_K = keras.Sequential([
#             layers.Dense(self.D, activation="relu")])
#         self.FC_V = keras.Sequential([
#             layers.Dense(self.D, activation="relu")])
#         self.FC_X = keras.Sequential([
#             layers.Dense(self.D, activation="relu"),
#             layers.Dense(self.D)])
        
#     def call(self, X, STE_P, STE_Q):
#         X = tf.transpose(X, (0, 2, 1, 3))
#         STE_P = tf.transpose(STE_P, (0, 2, 1, 3))
#         STE_Q = tf.transpose(STE_Q, (0, 2, 1, 3))
#         K = self.K
#         d = self.d
#         D = self.D
        
#         query = self.FC_Q(STE_Q)
#         key = self.FC_K(STE_P)
#         value = self.FC_V(X)
    
#         query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
#         key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
#         value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
#         query = tf.transpose(query, perm = (0, 2, 1, 3))
#         key = tf.transpose(key, perm = (0, 2, 3, 1))
#         value = tf.transpose(value, perm = (0, 2, 1, 3))   
    
#         attention = tf.matmul(query, key)
#         attention /= (d ** 0.5)
#         attention = tf.nn.softmax(attention, axis = -1)
        
#         # [batch_size, num_step, N, D]
#         X = tf.matmul(attention, value)
#         X = tf.transpose(X, perm = (0, 2, 1, 3))
#         X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
#         X = self.FC_X(X)
#         return X
    
    
class TransformAttention2(tf.keras.layers.Layer):
    def __init__(self, K, d):
        super(TransformAttention2, self).__init__()
        self.K = K
        self.d = d
        self.D = K*d

    def build(self, input_shape):
        self.FC_Q = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_K = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_V = keras.Sequential([
            layers.Dense(self.D, activation="relu"),])
        self.FC_X = keras.Sequential([
            layers.Dense(self.D, activation="relu"),
            layers.Dense(self.D),])
        
    def call(self, X, STE_P, STE_Q):
        K = self.K
        d = self.d
        D = self.D
        
        query = self.FC_Q(STE_Q)
        key = self.FC_K(STE_P)
        value = self.FC_V(X)

        query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
        key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
        value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
        
        attention = tf.matmul(query, key, transpose_b = True)
        attention /= (d ** 0.5)
        attention = tf.nn.softmax(attention, axis = -1)
        
        # [batch_size, num_step, N, D]
        X = tf.matmul(attention, value)
        X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
        X = self.FC_X(X)
        return X