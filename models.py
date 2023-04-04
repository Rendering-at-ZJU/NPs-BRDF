import numpy as np
import random
from tensorflow.python.keras.layers import *
from tensorflow.python.keras.utils import Sequence
from tensorflow.python.keras.callbacks import TensorBoard
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import Callback
from tensorflow.python.keras import backend as K

import tensorflow as tf

from config import Config

def sampling(args):
    z_mean, z_log_var = args
    epsilon = K.random_normal(shape=(K.shape(z_mean)[0], Config.z_dim))
    if Config.test_mode:
        z_sample = z_mean
    else:
        z_sample = z_mean + K.exp(z_log_var / 2) * epsilon
    return z_sample

def expand_z_dim(args):
    z_sample, x_all = args
    z = K.expand_dims(z_sample, 1)
    x_all = K.expand_dims(K.mean(x_all*0, -1), -1)
    z = z + x_all
    return z

def dot_mask(args):
    NoL, NoV = args
    s_L = K.relu(K.sign(NoL))
    s_V = K.relu(K.sign(NoV))
    return s_L * s_V

def get_decoder():
    x_all = Input(shape=(None, Config.in_dim), name='x')
    NoL = Input(shape=(None, 1))
    NoV = Input(shape=(None, 1))
    z_mean = Input(shape=(Config.z_dim,), name='z_mean')
    z_log_var = Input(shape=(Config.z_dim,), name='z_log_var')

    input_tensor = [z_mean, z_log_var, x_all]

    z_sample = Lambda(sampling)([z_mean, z_log_var])
    z_sample = Lambda(expand_z_dim)([z_sample, x_all])

    y = Concatenate(axis=-1)([x_all, z_sample])
    for i in range(Config.dc_layers):
        y = Dense(Config.dc_dims[i], activation='relu')(y)
    
    if Config.use_dot_mask:
        input_tensor += [NoL, NoV]
        mask = Lambda(dot_mask)([NoL, NoV])
        y_hat = Multiply()([y, mask])
    else:
        y_hat = y

    decoder = Model(input_tensor, y_hat, name='decoder')
    return decoder

def encoder_h(args):
    x, y = args
    h = Concatenate(axis=-1)([x, y])

    for i in range(Config.h_layers):
        h = Dense(Config.h_dims[i], activation='relu')(h)

    return h

def aggregation(args):
    return K.mean(args, 1)

def encoder_a(args):
    s = args
    for i in range(Config.a_layers - 1):
        s = Dense(Config.a_dims[i], activation='relu')(s)

    z_mean = Dense(Config.z_dim)(s)
    z_log_var = Dense(Config.z_dim)(s)

    return z_mean, z_log_var

def get_encoder():
    x = Input(shape=(None, Config.in_dim,))
    y = Input(shape=(None, Config.out_dim,))

    s_k = encoder_h([x, y])
    s = Lambda(aggregation)(s_k)
    # z_mean, z_log_var = encoder_a(s)
    pre_encoder = Model([x, y], s)

    s_context = Input(shape=(Config.h_dim, ))
    z_context = encoder_a(s_context)
    post_encoder = Model(s_context, z_context)

    s = pre_encoder([x, y])
    z_c = post_encoder(s)

    encoder =  Model([x, y], z_c, name='encoder')
    # encoder = Model([x, y], [z_mean, z_log_var], name='encoder')

    return encoder

def get_nps():
    encoder = get_encoder()
    decoder = get_decoder()

    x_context = Input(shape=(None, Config.in_dim,), name='x_context')
    y_context = Input(shape=(None, Config.out_dim, ), name='y_context')
    x_all = Input(shape=(None, Config.in_dim,), name='x_all')
    y_all = Input(shape=(None, Config.out_dim, ), name='y_all')
    NoL_map = Input(shape=(None, 1,), name='NdotL_map')
    NoV_map = Input(shape=(None, 1,), name='NdotV_map')

    z_context = encoder([x_context, y_context])
    z_all = encoder([x_all, y_all])

    decoder_input_tensor = [z_all[0], z_all[1], x_all]

    input_tensor = [x_context, y_context, x_all, y_all]

    if Config.use_dot_mask:
        decoder_input_tensor += [NoL_map, NoV_map]
        input_tensor += [NoL_map, NoV_map]

    y_hat = decoder(decoder_input_tensor)
    out_tensor = [z_all[0], z_context[0], z_all[1], z_context[1], y_hat]

    nps = Model(input_tensor, out_tensor)
    return nps, encoder, decoder