import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.python.keras import activations
import matplotlib.pyplot as plt

class MARN(keras.layers.Layer):
    def __init__(self, units, num_K, num_S, **kwargs):
        self.units = units
        self.state_size = [units, units]
        self.num_K = num_K
        self.num_S = num_S
        super(MARN, self).__init__(**kwargs)
        self.activation = activations.get('tanh')
        self.recurrent_activation = activations.get('sigmoid')

    def build(self, input_shape):
        """        
        input_shape_X (X): (None, input_dim)
        input_shape_M (M): (None, K, S)
        """
        
        self.input_dim = input_shape[-1]
        self.rows_kernel = (input_shape[-1] + self.units)
        self.kernels = {}
        self.bias = {}
        w_list = ['f', 't', 'i', 'o']
        b_list = ['f', 't', 'i', 'o']
        for w in w_list:
            self.kernels[w] = self.add_weight(shape=(self.rows_kernel, self.units),
                                               initializer='glorot_uniform',
                                               name=w+'_kernel')
        for b in b_list:
            self.bias[b] = self.add_weight(shape=(self.units,),
                                            initializer='zeros',
                                            name=b+'_bias')
        
        self.kernels['k'] = self.add_weight(shape=(self.units, self.num_S),
                                               initializer='glorot_uniform',
                                               name='k_kernel')
        self.bias['k'] = self.add_weight(shape=(self.num_S,),
                                               initializer='zeros',
                                               name='k_bias')
        self.kernels['r'] = self.add_weight(shape=(self.num_S, self.units),
                                           initializer='glorot_uniform',
                                           name='r_kernel')
        self.kernels['c'] = self.add_weight(shape=(self.units, self.units),
                                           initializer = 'glorot_uniform',
                                           name='c_kernel')
        self.kernels['h'] = self.add_weight(shape=(self.num_S, self.units),
                                           initializer='glorot_uniform',
                                           name='h_kernel')
        self.kernels['e'] = self.add_weight(shape=(self.units, self.num_S),
                                           initializer='glorot_uniform',
                                           name='e_kernel')
        self.bias['e'] = self.add_weight(shape=(self.num_S,),
                                               initializer='zeros',
                                               name='e_bias')
        self.kernels['a'] = self.add_weight(shape=(self.units, self.num_S),
                                           initializer='glorot_uniform',
                                           name='a_kernel')
        self.bias['a'] = self.add_weight(shape=(self.num_S,),
                                               initializer='zeros',
                                               name='a_bias')
        self.built = True

    def call(self, inputs, states):
        """
        Modified GRU cell, to account for graph convolution operations.
        
        inputs: (batch_size, input_dim)
        h states: (batch_size, units)
        c states: (batch_size, units)
        M states: (batch_size, K, S)
        k states: (batch_size, S)
        """
        X = inputs
        h_prev = states[0]
        c_prev = states[1]
        M_prev = states[2]
        k_prev = states[3]
        
        i = self.recurrent_activation(self.activation_dense(X, h_prev, 'input'))
        f = self.recurrent_activation(self.activation_dense(X, h_prev, 'forget'))
        o = self.recurrent_activation(self.activation_dense(X, h_prev, 'output'))
        t = self.recurrent_activation(self.activation_dense(X, h_prev, 'theta'))
        
        c = f * c_prev + i * t
        
        alphas = []
        for ki in range(self.num_K):
            alpha_ki = tf.keras.losses.cosine_similarity(M_prev[:,ki,:], k_prev)
            alphas.append(alpha_ki)
        
        alpha = tf.stack(alphas, axis=1)
        alpha = tf.nn.softmax(alpha, axis=-1)
        
        alpha_tmp = tf.expand_dims(alpha, -1)
        alpha_tmp = tf.tile(alpha_tmp, (1, 1, M_prev.shape[2]))
        
        r_tmp = M_prev * alpha_tmp
        r = tf.reduce_sum(r_tmp, 1)
        
        r_1 = r@self.kernels['r']
        c_1 = c@self.kernels['c']
        rc_1 = tf.nn.sigmoid(r_1+c_1)
        r_2 = r@self.kernels['h']
        
        rc_2 = rc_1*r_2
        rc_3 = tf.nn.tanh(c+rc_2)
        h_curr = o*rc_3
        
        k_curr = tf.nn.tanh(h_curr@self.kernels['k']+self.bias['k'])
        
        e = tf.nn.sigmoid(h_curr@self.kernels['e']+self.bias['e'])
        a = tf.nn.tanh(h_curr@self.kernels['a']+self.bias['a'])
        
        alpha_tmp2 = tf.expand_dims(alpha, -1)
        e_tmp2 = tf.expand_dims(e, 1)
        a_tmp2 = tf.expand_dims(a, 1)
        
        alpha_e = alpha_tmp2*e_tmp2
        alpha_a = alpha_tmp2*a_tmp2
        M_curr = M_prev*(1-alpha_e)+alpha_a
        
        return h_curr, [h_curr, c, M_curr, k_curr]

    def activation_dense(self, inputs, state, gate):
        """
        Graph convolution operation, based on the chosen support matrices
        
        inputs: (batch_size, input_dim)
        state: (batch_size, units)
        gate: "input", "forget", "output", "theta"
        """
                
        x = tf.concat([inputs, state], axis=-1)
        if gate in ['input', 'forget', 'output', 'theta']:
            gate_i = gate[0]
            x = tf.matmul(x, self.kernels[gate_i])
            x = tf.nn.bias_add(x, self.bias[gate_i])
        else:
            print('Error: Unknown gate')
        return x


class KAM(keras.layers.Layer):
    def __init__(self, units, gamma, num_K, num_S, **kwargs):
        self.units = units
        self.gamma = gamma
        self.num_K = num_K
        self.num_S = num_S
        super(KAM, self).__init__(**kwargs)

    def build(self, input_shape):
        """
        Defines kernel and biases of the KAM cell
        
        input_shape_M (M): (None, K, S)
        """
        
        self.g_kernel = self.add_weight(shape=(self.num_S*2, self.units),
                                               initializer='glorot_uniform',
                                               name='g_kernel')
        self.g_vector = self.add_weight(shape=(self.units,1),
                                               initializer='zeros',
                                               name='g_bias')
        self.b_kernel = self.add_weight(shape=(self.num_S, self.num_S),
                                               initializer='glorot_uniform',
                                               name='b_kernel')
        self.b_bias = self.add_weight(shape=(self.num_S,),
                                               initializer='zeros',
                                               name='b_bias')
        
        self.l_kernel = self.add_weight(shape=(self.num_S, self.num_S),
                                               initializer='glorot_uniform',
                                               name='l_kernel')
        self.l_bias = self.add_weight(shape=(self.num_S,),
                                               initializer='zeros',
                                               name='l_bias')
        
        self.built = True

    def call(self, inputs, states):
        """
        Knowledge Adaption Model, derive M_s_curr from M_s_prev, M_r_prev, M_r_curr
        
        inputs: M_r_prev, M_s_prev, M_r_curr (batch_size, K, S)
        l states: (batch_size, S)
        b states: (batch_size, S)
        """
        M_r_prev, M_s_prev, M_r_curr = inputs
        l_prev = states[0]
        b_prev = states[1]
        
        beta = self.g(M_r_prev, M_s_prev)
        
        b_curr = tf.nn.tanh(b_prev@self.b_kernel+self.b_bias)
        l_curr = tf.nn.sigmoid(l_prev@self.l_kernel+self.l_bias)
                
        beta_tmp = tf.expand_dims(beta, -1)
        beta_tmp = tf.tile(beta_tmp, (1, 1, M_r_prev.shape[2]))
        
        l_curr_tmp = tf.expand_dims(l_curr, 1)
        b_curr_tmp = tf.expand_dims(b_curr, 1)
        beta_l = beta_tmp*l_curr_tmp
        beta_b = beta_tmp*b_curr_tmp
        
        M_new = M_r_prev*(1-beta_l)+beta_b
        M_s_curr = self.gamma * M_s_prev + (1 - self.gamma) * M_new
        
        return M_s_curr, [l_curr, b_curr]
    
    def g(self, M_r, M_s):
        g_results = []
        for ki in range(M_r.shape[1]):
            g_ki = tf.nn.tanh(tf.concat((M_r[:,ki,:], M_s[:,ki,:]), axis=-1))@self.g_kernel
            g_ki = g_ki@self.g_vector
            g_ki = tf.squeeze(g_ki, -1)
            g_results.append(g_ki)
        beta = tf.stack(g_results, axis=1)
        beta = tf.nn.softmax(beta, axis=-1)
        return beta
    