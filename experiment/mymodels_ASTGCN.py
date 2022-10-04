import sys
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from scipy.sparse.linalg import eigs
# from dcgru_cell_tf2 import *
# from submodules import *



class Spatial_Attention_layer(tf.keras.layers.Layer):
    '''
    compute spatial attention scores
    '''
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Spatial_Attention_layer, self).__init__()
        self.in_channels = in_channels
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps
        

    def build(self, input_shape):
        self.W1 =  self.add_weight(shape=(self.num_of_timesteps, 1), initializer='glorot_uniform', dtype=tf.float32, name='W1')
        self.W2 =  self.add_weight(shape=(self.in_channels, self.num_of_timesteps), initializer='glorot_uniform', dtype=tf.float32, name='W2')
        self.W3 =  self.add_weight(shape=(1, self.in_channels), initializer='glorot_uniform', dtype=tf.float32, name='W3')
        self.bs =  self.add_weight(shape=(1, self.num_of_vertices, self.num_of_vertices), initializer='glorot_uniform', dtype=tf.float32, name='bs')
        self.Vs =  self.add_weight(shape=(self.num_of_vertices, self.num_of_vertices), initializer='glorot_uniform', dtype=tf.float32, name='Vs')
        

    def call(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B,N,N)
        '''

        # lhs = torch.matmul(torch.matmul(x, self.W1), self.W2)  # (b,N,F,T)(T)->(b,N,F)(F,T)->(b,N,T)
        lhs = tf.squeeze((x @ self.W1) , -1)
        lhs = lhs @ self.W2
        
        # rhs = torch.matmul(self.W3, x).transpose(-1, -2)  # (F)(b,N,F,T)->(b,N,T)->(b,T,N)
        print('rhs', self.W3.shape, x.shape)
        rhs = (self.W3 @ x)
        rhs = tf.transpose(tf.squeeze(rhs, 2), (0, 2, 1))
        
        # product = torch.matmul(lhs, rhs)  # (b,N,T)(b,T,N) -> (B, N, N)
        print(lhs.shape, rhs.shape)
        product = lhs @ rhs

        # S = torch.matmul(self.Vs, torch.sigmoid(product + self.bs))  # (N,N)(B, N, N)->(B,N,N)
        S = self.Vs @ tf.sigmoid(product + self.bs)

        # S_normalized = F.softmax(S, dim=1)
        S_normalized = tf.nn.softmax(S, 1)

        return S_normalized



class cheb_conv_withSAt(tf.keras.layers.Layer):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv_withSAt, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
    
    def build(self, input_shape):
        self.Theta = [self.add_weight(shape=(self.in_channels, self.out_channels), initializer='glorot_uniform', name=f'Theta_{_}') for _ in range(self.K)]


    def call(self, x, spatial_attention):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = x.shape
        batch_size = tf.shape(x)[0]
        print(x.shape)

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = tf.zeros((batch_size, num_of_vertices, self.out_channels), dtype=tf.float32)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                T_k_with_at = T_k * (spatial_attention)   # (N,N)*(N,N) = (N,N) 多行和为1, 按着列进行归一化

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = tf.transpose(T_k_with_at, (0, 2, 1)) @ graph_signal  # (N, N)(b, N, F_in) = (b, N, F_in) 因为是左乘，所以多行和为1变为多列和为1，即一行之和为1，进行左乘

                output = output + rhs @ theta_k # (b, N, F_in)(F_in, F_out) = (b, N, F_out)

            outputs.append(tf.expand_dims(output, -1))  # (b, N, F_out, 1)

        return tf.nn.relu(tf.concat(outputs, -1))  # (b, N, F_out, T)



class Temporal_Attention_layer(tf.keras.layers.Layer):
    def __init__(self, in_channels, num_of_vertices, num_of_timesteps):
        super(Temporal_Attention_layer, self).__init__()
        self.in_channels = in_channels
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps

        # self.U1 = nn.Parameter(torch.FloatTensor(num_of_vertices).to(DEVICE))
        # self.U2 = nn.Parameter(torch.FloatTensor(in_channels, num_of_vertices).to(DEVICE))
        # self.U3 = nn.Parameter(torch.FloatTensor(in_channels).to(DEVICE))
        # self.be = nn.Parameter(torch.FloatTensor(1, num_of_timesteps, num_of_timesteps).to(DEVICE))
        # self.Ve = nn.Parameter(torch.FloatTensor(num_of_timesteps, num_of_timesteps).to(DEVICE))

    def build(self, input_shape):
        self.U1 =  self.add_weight(shape=(self.num_of_vertices, 1), initializer='glorot_uniform', dtype=tf.float32, name='U1')
        self.U2 =  self.add_weight(shape=(self.in_channels, self.num_of_vertices), initializer='glorot_uniform', dtype=tf.float32, name='U2')
        self.U3 =  self.add_weight(shape=(1, self.in_channels), initializer='glorot_uniform', dtype=tf.float32, name='U3')
        self.be =  self.add_weight(shape=(1, self.num_of_timesteps, self.num_of_timesteps), initializer='glorot_uniform', dtype=tf.float32, name='be')
        self.Ve =  self.add_weight(shape=(self.num_of_timesteps, self.num_of_timesteps), initializer='glorot_uniform', dtype=tf.float32, name='Ve')
        

    def call(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (B, T, T)
        '''
        _, num_of_vertices, num_of_features, num_of_timesteps = x.shape

        # lhs = torch.matmul(torch.matmul(x.permute(0, 3, 2, 1), self.U1), self.U2)
        print(tf.transpose(x, (0, 3, 2, 1)).shape, self.U1.shape, self.U2.shape)
        lhs = tf.squeeze((tf.transpose(x, (0, 3, 2, 1)) @ self.U1) , -1) @ self.U2
        
        # x:(B, N, F_in, T) -> (B, T, F_in, N)
        # (B, T, F_in, N)(N) -> (B,T,F_in)
        # (B,T,F_in)(F_in,N)->(B,T,N)

        # rhs = torch.matmul(self.U3, x)  # (F)(B,N,F,T)->(B, N, T)
        print('rhs = ', self.U3.shape, x.shape)
        rhs = self.U3 @ x  # (F)(B,N,F,T)->(B, N, T)

        rhs = tf.squeeze(rhs, -2)
        print('lhs, rhs', lhs.shape, rhs.shape)

        # product = torch.matmul(lhs, rhs)  # (B,T,N)(B,N,T)->(B,T,T)
        product = lhs @ rhs  # (B,T,N)(B,N,T)->(B,T,T)

        # E = torch.matmul(self.Ve, torch.sigmoid(product + self.be))  # (B, T, T)
        E = self.Ve @ tf.sigmoid(product + self.be)  # (B, T, T)

        E_normalized = tf.nn.softmax(E, 1)

        return E_normalized



class cheb_conv(tf.keras.layers.Layer):
    '''
    K-order chebyshev graph convolution
    '''

    def __init__(self, K, cheb_polynomials, in_channels, out_channels):
        '''
        :param K: int
        :param in_channles: int, num of channels in the input sequence
        :param out_channels: int, num of channels in the output sequence
        '''
        super(cheb_conv, self).__init__()
        self.K = K
        self.cheb_polynomials = cheb_polynomials
        self.in_channels = in_channels
        self.out_channels = out_channels
        

    def build(self, input_shape):
        # self.Theta = nn.ParameterList([nn.Parameter(torch.FloatTensor(in_channels, out_channels).to(self.DEVICE)) for _ in range(K)])
        self.Theta = [self.add_weight(shape=(self.in_channels, self.out_channels), initializer='glorot_uniform', name=f'Theta_{_}') for _ in range(self.K)]


    def call(self, x):
        '''
        Chebyshev graph convolution operation
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, F_out, T)
        '''

        batch_size, num_of_vertices, in_channels, num_of_timesteps = tf.shape(x)

        outputs = []

        for time_step in range(num_of_timesteps):

            graph_signal = x[:, :, :, time_step]  # (b, N, F_in)

            output = tf.zeros((batch_size, num_of_vertices, self.out_channels))  # (b, N, F_out)

            for k in range(self.K):

                T_k = self.cheb_polynomials[k]  # (N,N)

                theta_k = self.Theta[k]  # (in_channel, out_channel)

                rhs = tf.transpose(graph_signal, (0, 2, 1)) @ tf.transpose(T_k, (0, 2, 1))

                output = output + rhs @ theta_k

            outputs.append(tf.unsqueeze(output, -1))

        return tf.nn.relu(tf.concat(outputs, -1))



class ASTGCN_block(tf.keras.layers.Layer):

    def __init__(self, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, num_of_timesteps):
        super(ASTGCN_block, self).__init__()
        self.K = K
        self.in_channels = in_channels
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.cheb_polynomials = cheb_polynomials
        self.num_of_vertices = num_of_vertices
        self.num_of_timesteps = num_of_timesteps

        
    
    def build(self, input_shape):
        self.TAt = Temporal_Attention_layer(self.in_channels, self.num_of_vertices, self.num_of_timesteps)
        self.SAt = Spatial_Attention_layer(self.in_channels, self.num_of_vertices, self.num_of_timesteps)
        self.cheb_conv_SAt = cheb_conv_withSAt(self.K, self.cheb_polynomials, self.in_channels, self.nb_chev_filter)
        # self.time_conv = nn.Conv2d(nb_chev_filter, nb_time_filter, kernel_size=(1, 3), stride=(1, time_strides), padding=(0, 1))
        # self.residual_conv = nn.Conv2d(in_channels, nb_time_filter, kernel_size=(1, 1), stride=(1, time_strides))
        # self.ln = nn.LayerNorm(nb_time_filter)  #需要将channel放到最后一个维度上
        self.time_conv = layers.Conv2D(self.nb_time_filter, kernel_size=(1, 3), strides=(1, self.time_strides), padding='same', data_format="channels_first")
        self.residual_conv = layers.Conv2D(self.nb_time_filter, kernel_size=(1, 1), strides=(1, self.time_strides), padding='same', data_format="channels_first")
        # self.ln = layers.LayerNormalization(axis=-1)  #需要将channel放到最后一个维度上
    

    def call(self, x):
        '''
        :param x: (batch_size, N, F_in, T)
        :return: (batch_size, N, nb_time_filter, T)
        '''
        batch_size, num_of_vertices, num_of_features, num_of_timesteps = x.shape
        batch_size = tf.shape(x)[0]
        print('ASTGCN:', x.shape)
        # TAt
        temporal_At = self.TAt(x)  # (b, T, T)

        # x_TAt = torch.matmul(x.reshape(batch_size, -1, num_of_timesteps), temporal_At).reshape(batch_size, num_of_vertices, num_of_features, num_of_timesteps)
        x_TAt = tf.reshape((tf.reshape(x, (batch_size, -1, num_of_timesteps)) @ temporal_At), 
                            (batch_size, num_of_vertices, num_of_features, num_of_timesteps))

        # SAt
        spatial_At = self.SAt(x_TAt)

        # cheb gcn
        spatial_gcn = self.cheb_conv_SAt(x, spatial_At)  # (b,N,F,T)
        # spatial_gcn = self.cheb_conv(x)

        # convolution along the time axis
        time_conv_output = self.time_conv(tf.transpose(spatial_gcn, (0, 2, 1, 3)))  # (b,N,F,T)->(b,F,N,T) 用(1,3)的卷积核去做->(b,F,N,T)

        # residual shortcut
        x_residual = self.residual_conv(tf.transpose(x, (0, 2, 1, 3)))  # (b,N,F,T)->(b,F,N,T) 用(1,1)的卷积核去做->(b,F,N,T)

        # x_residual = tf.transpose(self.ln(tf.transpose(tf.nn.relu(x_residual + time_conv_output), (0, 3, 2, 1))), (0, 2, 3, 1))
        x_residual = tf.transpose((tf.transpose(tf.nn.relu(x_residual + time_conv_output), (0, 3, 2, 1))), (0, 2, 3, 1))
        # (b,F,N,T)->(b,T,N,F) -ln-> (b,T,N,F)->(b,N,F,T)

        return x_residual



class ASTGCN_submodule(tf.keras.layers.Layer):

    def __init__(self, nb_block, in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_for_predict, len_input, num_of_vertices):
        '''
        :param nb_block:
        :param in_channels:
        :param K:
        :param nb_chev_filter:
        :param nb_time_filter:
        :param time_strides:
        :param cheb_polynomials:
        :param nb_predict_step:
        '''

        super(ASTGCN_submodule, self).__init__()
        self.nb_block = nb_block
        self.in_channels = in_channels
        self.K = K
        self.nb_chev_filter = nb_chev_filter
        self.nb_time_filter = nb_time_filter
        self.time_strides = time_strides
        self.cheb_polynomials = cheb_polynomials
        self.num_for_predict = num_for_predict
        self.len_input = len_input
        self.num_of_vertices = num_of_vertices
        
    def build(self, input_shape):
        nb_block = self.nb_block
        in_channels = self.in_channels
        K = self.K
        nb_chev_filter = self.nb_chev_filter
        nb_time_filter = self.nb_time_filter
        time_strides = self.time_strides
        cheb_polynomials = self.cheb_polynomials
        num_for_predict = self.num_for_predict
        len_input = self.len_input
        num_of_vertices = self.num_of_vertices

        self.BlockList = [ASTGCN_block(in_channels, K, nb_chev_filter, nb_time_filter, time_strides, cheb_polynomials, num_of_vertices, len_input)]

        self.BlockList.extend([ASTGCN_block(nb_time_filter, K, nb_chev_filter, nb_time_filter, 1, cheb_polynomials, num_of_vertices, len_input//time_strides) for _ in range(nb_block-1)])

        # self.final_conv = nn.Conv2d(int(len_input/time_strides), num_for_predict, kernel_size=(1, nb_time_filter))
        self.final_conv = layers.Conv2D(num_for_predict, kernel_size=(1, nb_time_filter), data_format="channels_first")

    def call(self, x):
        '''
        :param x: (B, N_nodes, F_in, T_in)
        :return: (B, N_nodes, T_out)
        '''
        x = tf.expand_dims(x, -2)
        x = tf.transpose(x, (0, 3, 2, 1))
        print('ASTGCN_submodule', x.shape)
        for block in self.BlockList:
            x = block(x)
            print('ASTGCN_submodule', x.shape)

            
        x = tf.transpose(x, (0, 3, 1, 2))
        output = self.final_conv(x)
        print('output', output.shape)
        output = output[:, :, :, -1]
        output = tf.transpose(output, (0, 2, 1))
        # (b,N,F,T)->(b,T,N,F)-conv<1,F>->(b,c_out*T,N,1)->(b,c_out*T,N)->(b,N,T)

        return output



class MyASTGCN(tf.keras.layers.Layer):
    def __init__(self, extdata, args):
        super(MyASTGCN, self).__init__()
        # self.nb_block = args.nb_block
        # self.in_channels = args.in_channels
        # self.K = args.K
        # self.nb_chev_filter = args.nb_chev_filter
        # self.nb_time_filter = args.nb_time_filter
        # self.time_strides = args.time_strides
        # self.adj_mx = extdata['adj_mat']
        # self.num_for_predict = args.num_for_predict
        # self.len_input = args.len_input
        # self.num_of_vertices = args.num_of_vertices
        
        self.nb_block = 2
        self.in_channels = 1
        self.K = 3
        self.nb_chev_filter = 64
        self.nb_time_filter = 64
        self.time_strides = 1
        self.adj_mx = extdata['adj_mat']
        self.num_for_predict = 1
        self.len_input = 6
        self.num_of_vertices = extdata['num_nodes']
        
        
        self.L_tilde = scaled_Laplacian(self.adj_mx)
        self.cheb_polynomials = [tf.constant(i) for i in cheb_polynomial(self.L_tilde, self.K)]
        
        
    def build(self, input_shape):
        self.model = ASTGCN_submodule(self.nb_block, 
                                    self.in_channels, 
                                    self.K, self.nb_chev_filter, 
                                    self.nb_time_filter, self.time_strides, 
                                    self.cheb_polynomials, self.num_for_predict, 
                                    self.len_input, self.num_of_vertices)


    def call(self, kwargs):
        X, ZC, ZF, TE = kwargs['X'], kwargs['ZC'], kwargs['ZF'], kwargs['TE']

        Y = self.model(X)

        return Y



### lib codes

def scaled_Laplacian(W):
    '''
    compute \tilde{L}
    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices
    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)
    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}
    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)
    K: the maximum order of chebyshev polynomials
    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}
    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    cheb_polynomials = [cheb.astype(np.float32) for cheb in cheb_polynomials]

    return cheb_polynomials