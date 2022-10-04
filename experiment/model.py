import tf_utils
import tensorflow as tf

def placeholder(P, Q, NX, NZ):
    X = tf.compat.v1.placeholder(
        shape = (None, P, NX), dtype = tf.float32, name = 'X')
    Z = tf.compat.v1.placeholder(
        shape = (None, P, NZ), dtype = tf.float32, name = 'Z')
    TE = tf.compat.v1.placeholder(
        shape = (None, P + Q, 2), dtype = tf.int32, name = 'TE')
    label = tf.compat.v1.placeholder(
        shape = (None, Q, NX), dtype = tf.float32, name = 'label')
    is_training = tf.compat.v1.placeholder(
        shape = (), dtype = tf.bool, name = 'is_training')
    return X, Z, TE, label, is_training

def FC(x, units, activations, bn, bn_decay, is_training, use_bias = True, drop = None):
    if isinstance(units, int):
        units = [units]
        activations = [activations]
    elif isinstance(units, tuple):
        units = list(units)
        activations = list(activations)
    assert type(units) == list
    for num_unit, activation in zip(units, activations):
        if drop is not None:
            x = tf_utils.dropout(x, drop = drop, is_training = is_training)
        x = tf_utils.conv2d(
            x, output_dims = num_unit, kernel_size = [1, 1], stride = [1, 1],
            padding = 'VALID', use_bias = use_bias, activation = activation,
            bn = bn, bn_decay = bn_decay, is_training = is_training)
    return x

def STEmbedding(SE, SEZ, TE, T, D, bn, bn_decay, is_training):
    '''
    spatio-temporal embedding
    SE:     [N, D]
    TE:     [batch_size, P + Q, 2] (dayofweek, timeofday)
    T:      num of time steps in one day
    D:      output dims
    retrun: [batch_size, P + Q, N, D]
    '''
    # spatial embedding
    SE = tf.Variable(
        tf.glorot_uniform_initializer()(shape = SE.shape),
        dtype = tf.float32, trainable = True)
    SE = tf.expand_dims(tf.expand_dims(SE, axis = 0), axis = 0)
    SE = FC(
        SE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    SEZ = tf.expand_dims(tf.expand_dims(SEZ, axis = 0), axis = 0)
    SEZ = FC(
        SEZ, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # temporal embedding
    dayofweek = tf.one_hot(TE[..., 0], depth = 7)
    timeofday = tf.one_hot(TE[..., 1], depth = T)
    TE = tf.concat((dayofweek, timeofday), axis = -1)
    TE = tf.expand_dims(TE, axis = 2)
    TE = FC(
        TE, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)    

    return tf.add(SE, TE), tf.add(SEZ, TE)

def spatialAttention(X, STE, K, d, bn, bn_decay, is_training):
    '''
    spatial attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis = -1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
        
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X

def temporalAttention(X, STE, K, d, bn, bn_decay, is_training, mask=True):
    '''
    temporal attention mechanism
    X:      [batch_size, num_step, N, D]
    STE:    [batch_size, num_step, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, num_step, N, D]
    '''
    D = K * d
    X = tf.concat((X, STE), axis = -1)
    # [batch_size, num_step, N, K * d]
    query = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # query: [K * batch_size, N, num_step, d]
    # key:   [K * batch_size, N, d, num_step]
    # value: [K * batch_size, N, num_step, d]
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))
    # [K * batch_size, N, num_step, num_step]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    # mask attention score
    if mask:
        batch_size = tf.shape(X)[0]
        num_step = X.get_shape()[1].value
        N = X.get_shape()[2].value
        mask = tf.ones(shape = (num_step, num_step))
        mask = tf.linalg.LinearOperatorLowerTriangular(mask).to_dense()
        mask = tf.expand_dims(tf.expand_dims(mask, axis = 0), axis = 0)
        mask = tf.tile(mask, multiples = (K * batch_size, N, 1, 1))
        mask = tf.cast(mask, dtype = tf.bool)
        attention = tf.compat.v2.where(
            condition = mask, x = attention, y = -2 ** 15 + 1)
    # softmax   
    attention = tf.nn.softmax(attention, axis = -1)
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X

def gatedFusion(HS, HT, D, bn, bn_decay, is_training):
    '''
    gated fusion
    HS:     [batch_size, num_step, N, D]
    HT:     [batch_size, num_step, N, D]
    D:      output dims
    return: [batch_size, num_step, N, D]
    '''
    XS = FC(
        HS, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = False)
    XT = FC(
        HT, units = D, activations = None,
        bn = bn, bn_decay = bn_decay,
        is_training = is_training, use_bias = True)
    z = tf.nn.sigmoid(tf.add(XS, XT))
    H = tf.add(tf.multiply(z, HS), tf.multiply(1 - z, HT))
    H = FC(
        H, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return H

def GSTAttBlock(X, STE, K, d, bn, bn_decay, is_training):
    HS = spatialAttention(X, STE, K, d, bn, bn_decay, is_training)
    HT = temporalAttention(X, STE, K, d, bn, bn_decay, is_training)
    H = gatedFusion(HS, HT, K * d, bn, bn_decay, is_training)
    return tf.add(X, H)
    
def RSTAttBlock(Z, STEZ, ADJ_DY, LH, LW, K, d, bn, bn_decay, is_training, args):
    D = K*d
    num_step = tf.shape(Z)[1]

    if not args.NODYC:
        assert ADJ_DY.shape[0] == LH*LW and ADJ_DY.shape[1] == LH*LW
        ZS = DyConv(ADJ_DY, Z, num_step, D)
        ZS = tf_utils.batch_norm(ZS, is_training = is_training, bn_decay = bn_decay)
    else:
        ZS = tf.reshape(Z, (-1, LH, LW, D))
        ZS = tf_utils.conv2d(
            ZS, D, [args.cnn_size, args.cnn_size], activation = tf.nn.relu, bn=bn,  bn_decay=bn_decay, is_training = is_training)
        ZS = tf.reshape(ZS, (-1, num_step, LH*LW, D))

    ZT = temporalAttention(Z, STEZ, K, d, bn, bn_decay, is_training, mask=True)
    H = gatedFusion(ZS, ZT, K * d, bn, bn_decay, is_training)

    return tf.add(Z, H)
    

def transformAttention(X, STE_P, STE_Q, K, d, bn, bn_decay, is_training):
    '''
    transform attention mechanism
    X:      [batch_size, P, N, D]
    STE_P:  [batch_size, P, N, D]
    STE_Q:  [batch_size, Q, N, D]
    K:      number of attention heads
    d:      dimension of each attention outputs
    return: [batch_size, Q, N, D]
    '''
    D = K * d
    # query: [batch_size, Q, N, K * d]
    # key:   [batch_size, P, N, K * d]
    # value: [batch_size, P, N, K * d]
    query = FC(
        STE_Q, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    key = FC(
        STE_P, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        X, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # query: [K * batch_size, Q, N, d]
    # key:   [K * batch_size, P, N, d]
    # value: [K * batch_size, P, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # query: [K * batch_size, N, Q, d]
    # key:   [K * batch_size, N, d, P]
    # value: [K * batch_size, N, P, d]
    query = tf.transpose(query, perm = (0, 2, 1, 3))
    key = tf.transpose(key, perm = (0, 2, 3, 1))
    value = tf.transpose(value, perm = (0, 2, 1, 3))    
    # [K * batch_size, N, Q, P]
    attention = tf.matmul(query, key)
    attention /= (d ** 0.5)
    attention = tf.nn.softmax(attention, axis = -1)
    # [batch_size, Q, N, D]
    X = tf.matmul(attention, value)
    X = tf.transpose(X, perm = (0, 2, 1, 3))
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    return X


def DyConv(DyAdj, Z, num_seq, D):
    # Z = tf.transpose(Z, perm=[2, 1, 0, 3])
    W = tf.Variable(
        tf.glorot_uniform_initializer()(shape = (2*D, D)),
        dtype = tf.float32, trainable = True, name = 'kernel')
    # Z = Z @ W
    # Z = tf.reshape(Z, (Z.shape[0], -1))
    Z = tf.concat((DyAdj @ Z, Z), -1) @ W
    # Z = tf.reshape(Z, (Z.shape[0], num_seq, -1, D))
    # Z = tf.transpose(Z, perm=[2, 1, 0, 3])
    return tf.nn.relu(Z)



def sample_gumbel(shape, eps=1e-20):
    U = tf.random_uniform(shape, minval=0, maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax(logits, temperature=0.5, hard=False):
    gumbel_softmax_sample = logits + sample_gumbel(tf.shape(logits))
    y = tf.nn.softmax(gumbel_softmax_sample / temperature)

    if hard:
        k = tf.shape(logits)[-1]
        y_hard = tf.cast(tf.equal(y, tf.reduce_max(y, 1, keep_dims=True)),
                         y.dtype)
        y = tf.stop_gradient(y_hard - y) + y

    return y


def BipartiteGlob(Z, DIST_GR, K, d, bn, bn_decay, is_training):
    '''
    TE_P:   [batch_size, num_step, 2] (dayofweek, timeofday)
    STE:    [batch_size, num_step, NX, D]
    Z:      [batch_size, num_step, NZ, D]
    ADJ:    [N, NZ]
    NZ:     [LH*LW]
    '''

    D = K * d
    
    GR = tf.Variable(
        tf.glorot_uniform_initializer()(shape = (1, DIST_GR.shape[1])),
        dtype = tf.float32, trainable = True, name = 'kernel')
    Adj = gumbel_softmax(GR) 
    Z = tf.matmul(Adj, Z)
    Z = FC(
        Z, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training, use_bias = False)
    Z = tf.tile(Z, (1, 1, DIST_GR.shape[0], 1))
    return Z


def BipartiteGlobHead(Z, DIST_GR, K, d, bn, bn_decay, is_training):
    '''
    TE_P:   [batch_size, num_step, 2] (dayofweek, timeofday)
    STE:    [batch_size, num_step, NX, D]
    Z:      [batch_size, num_step, NZ, D]
    ADJ:    [N, NZ]
    NZ:     [LH*LW]
    '''

    D = K * d
    
    Zs = []
    for _ in range(K):
        Zt = BipartiteGlob(Z, DIST_GR, 1, d, bn, bn_decay, is_training)
        Zs.append(Zt)
    Z = tf.concat(Zs, -1)
    return Z


def BipartiteAttn(STE, Z, DIST_GR, num_step, K, d, bn, bn_decay, is_training, NOBIM):
    '''
    STE:    [batch_size, num_step, NX, D]
    Z:      [batch_size, num_step, NZ, D]
    ADJ:    [N, NZ]
    NZ:     [LH*LW]
    '''

    D = K * d
    # [batch_size, num_step, NX, K * d]
    query = FC(
        STE, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [batch_size, num_step, NZ, K * d]
    key = FC(
        Z, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    value = FC(
        Z, units = D, activations = tf.nn.relu,
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # [K * batch_size, num_step, N, d]
    query = tf.concat(tf.split(query, K, axis = -1), axis = 0)
    key = tf.concat(tf.split(key, K, axis = -1), axis = 0)
    value = tf.concat(tf.split(value, K, axis = -1), axis = 0)
    # [K * batch_size, num_step, N, N]
    attention = tf.matmul(query, key, transpose_b = True)
    attention /= (d ** 0.5)

    # Masking
    batch_size = tf.shape(Z)[0]
    num_step = tf.shape(Z)[1]

    if not NOBIM:
        MASK_GR = - ((DIST_GR / DIST_GR.std()) ** 2) / 2 # normalize before applying trainable std
        NX = DIST_GR.shape[0]
        NZ = DIST_GR.shape[1]
        std_weight = tf.Variable(
            tf.glorot_uniform_initializer()(shape = (K,NX,1)),
            dtype = tf.float32, trainable = True, name = 'bimstd')
        std_weight = tf.tile(std_weight, multiples = (1, 1, NZ))

        mask = []
        for i in range(K):
            mask.append(MASK_GR / (std_weight[i] ** 2))
        mask = tf.stack(mask, axis=0)
        mask = tf.expand_dims(mask, axis=1)
        mask = tf.tile(mask, multiples = (batch_size, num_step, 1, 1))
        attention = attention + mask

    attention = tf.nn.softmax(attention, axis = -1)
    
    # [batch_size, num_step, N, D]
    X = tf.matmul(attention, value)
    X = tf.concat(tf.split(X, K, axis = 0), axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
        
    return X
    

def GMSTARK(X, Z, DIST_GR, ADJ_DY, TE, SE, SEZ, P, Q, LH, LW, T, L, LZ, K, d, bn, bn_decay, is_training, args):
    '''
    GMAN
    X：       [batch_size, P, N]
    TE：      [batch_size, P + Q, 2] (time-of-day, day-of-week)
    SE：      [N, K * d]
    P：       number of history steps
    Q：       number of prediction steps
    T：       one day is divided into T steps
    L：       number of STAtt blocks in the encoder/decoder
    K：       number of attention heads
    d：       dimension of each attention head outputs
    return：  [batch_size, Q, N]
    '''
    D = K * d
    # input
    N = X.shape[-1]

    if args.NOLTE:
        SEZ = tf.reshape(SEZ, (1, LH, LW, -1))
        for _ in range(LZ): # conv number
            SEZ = tf_utils.conv2d(
                SEZ, D, [args.cnn_size, args.cnn_size], activation = tf.nn.relu, bn=bn,  bn_decay=bn_decay, is_training = is_training)
        SEZ = tf.reshape(SEZ, (LH*LW, D))
  

    STE, STEZ = STEmbedding(SE, SEZ, TE, T, D, bn, bn_decay, is_training)
    STE_P = STE[:, : P]
    STE_Q = STE[:, P :]
    STEZ_P = STEZ[:, : P]
    STEZ_Q = STEZ[:, P :]


    if args.GMAN:
        ZP = STE_P
        ZQ = STE_Q
    elif args.NOLTE:
        ZP = BipartiteAttn(STE_P, STEZ_P, DIST_GR, P, K, d, bn, bn_decay, is_training, args.NOBIM)
        ZQ = BipartiteAttn(STE_Q, STEZ_Q, DIST_GR, Q, K, d, bn, bn_decay, is_training, args.NOBIM)
    else: # default
        Z = tf.reshape(Z, (-1, P, LH*LW, 1))
        Z = FC(
            Z, units = [D, D], activations = [tf.nn.relu],
            bn = bn, bn_decay = bn_decay, is_training = is_training)

        for _ in range(args.LZ):
            Z = RSTAttBlock(Z, STEZ_P, ADJ_DY, LH, LW, K, d, bn, bn_decay, is_training, args)
            
        ZP_L = BipartiteAttn(STE_P, Z, DIST_GR, P, K, d, bn, bn_decay, is_training, args.NOBIM)
        ZP_G = BipartiteGlobHead(Z, DIST_GR, K, d, bn, bn_decay, is_training)
        ZP = tf.concat((ZP_L, ZP_G), -1)
    

        Z = transformAttention(
            Z, STEZ_P, STEZ_Q, K, d, bn, bn_decay, is_training)

        for _ in range(args.LZ):
            Z = RSTAttBlock(Z, STEZ_Q, ADJ_DY, LH, LW, K, d, bn, bn_decay, is_training, args)

        ZQ_L = BipartiteAttn(STE_Q, Z, DIST_GR, Q, K, d, bn, bn_decay, is_training, args.NOBIM)
        ZQ_G = BipartiteGlobHead(Z, DIST_GR, K, d, bn, bn_decay, is_training)
        ZQ = tf.concat((ZQ_L, ZQ_G), -1)

    X = tf.expand_dims(X, axis = -1)
    X = FC(
        X, units = [D, D], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training)
    # encoder
    for _ in range(L):
        X = GSTAttBlock(X, STE_P, K, d, bn, bn_decay, is_training)
    # transAtt
    X = transformAttention(
        X, ZP, ZQ, K, d, bn, bn_decay, is_training)
    # decoder
    for _ in range(L):
        X = GSTAttBlock(X, STE_Q, K, d, bn, bn_decay, is_training)
    # output
    X = FC(
        X, units = [D, 1], activations = [tf.nn.relu, None],
        bn = bn, bn_decay = bn_decay, is_training = is_training,
        use_bias = True, drop = 0.1)

    return tf.squeeze(X, axis = 3)

def mae_loss(pred, label):
    mask = tf.not_equal(label, 0)
    mask = tf.cast(mask, tf.float32)
    mask /= tf.reduce_mean(mask)
    mask = tf.compat.v2.where(
        condition = tf.math.is_nan(mask), x = 0., y = mask)
    loss = tf.abs(tf.subtract(pred, label))
    loss *= mask
    loss = tf.compat.v2.where(
        condition = tf.math.is_nan(loss), x = 0., y = loss)
    loss = tf.reduce_mean(loss)
    return loss
