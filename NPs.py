import numpy as np
import tensorflow as tf
import random
import os

from tensorflow.python.keras.layers import *
from tensorflow.python.keras.callbacks import TensorBoard, ModelCheckpoint
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import backend as K
import os.path as osp

from merlFunctions import readMERLBRDF
import models
import util
from config import Config

from argparse import ArgumentParser

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--mode', type=str, default='decompress', choices=['compress', 'decompress', 'edit', 'interpolation'], help='')
    parser.add_argument('--input', type=str, required=True, help='input file')
    parser.add_argument('--trait', type=str, help='trait file')
    parser.add_argument('--trait_length', type=float, help='trait length')
    parser.add_argument('--input1', type=str, help='BRDF for interpolation')
    parser.add_argument('--int_weight', type=float, help='interpolation weight')
    parser.add_argument('--output', type=str, required=True, help='output file')
    parser.add_argument('--backbone', type=str, default='NPs', choices=['Nps', 'hypernet'], help='Backbone model')
    args = parser.parse_args()
    
    x_grid = util.generate_coordinate()
    x_grid[:, :, 0:1] /= (np.pi)
    x_grid[:, :, 1:2] /= (np.pi/2)
    x_grid[:, :, 2:3] /= (np.pi/2)

    if Config.use_ringmap:
        x_grid = util.ringmap(x_grid)

    NoL = np.load('data/NdotL.npy')
    NoV = np.load('data/NdotV.npy')

    nps, encoder, decoder = models.get_nps()
    nps.load_weights('models/LOG4_Mean_Dim7/weight/Epoch40000_400000_weights_nps.h5')
    if args.backbone == 'hypernet':
        hyper_decoder = models.get_hypernet()
        trainer = models.get_hypernet_trainer(hyper_decoder, decoder)
        trainer.load_weights('models/LOG4_Dim7_Post_Hyper7_5/weight/Epoch60000_600000_weights_nps.h5')
        decoder = hyper_decoder

    os.makedirs(osp.dirname(osp.abspath(args.output)), exist_ok=True)

    if args.mode == 'compress':
        assert args.input[-6:] == 'binary'
        z = util.encode_material(args.input, x_grid, NoL, NoV, encoder)
        print(z)
        np.save(args.output, z)
    elif args.mode == 'decompress':
        assert args.input[-3:] == 'npy'
        z = np.load(args.input)
        brdf = util.decode_material(x_grid, NoL, NoV, z, decoder, args.backbone == 'hypernet')
        util.save_BRDF(args.output, brdf)
    elif args.mode == 'edit':
        assert args.input[-3:] == 'npy'
        trait_vec = np.load(args.trait)
        z = np.load(args.input)
        z[0] += trait_vec * args.trait_length
        brdf = util.decode_material(x_grid, NoL, NoV, z, decoder, args.backbone == 'hypernet')
        util.save_BRDF(args.output, brdf)
    elif args.mode == 'interpolation':
        z = np.load(args.input)
        z1 = np.load(args.input1)
        z[0] = z[0] * args.int_weight + (1 - args.int_weight) * z1[0]
        brdf = util.decode_material(x_grid, NoL, NoV, z, decoder, args.backbone == 'hypernet')
        util.save_BRDF(args.output, brdf)