import os.path as osp
import glob

from coordinateFunctions import *
from merlFunctions import *
from config import Config


def generate_coordinate():
    shp = BRDFSHAPE
    id = np.arange(shp[0]*shp[1]*shp[2])
    merlID = IDToMERL(id)
    grid = MERLToRusink(merlID).reshape([shp[0], shp[1], shp[2], 3])
    return grid.reshape(1, -1, 3)

def ringmap(x_grid):
    x_grid_add = x_grid[:, :, 0:1] * 2 * np.pi
    x_grid[:, :, 0:1] = np.sin(x_grid_add) * 0.5 + 0.5
    x_grid_add = np.cos(x_grid_add) * 0.5 + 0.5
    x_grid = np.concatenate([x_grid_add, x_grid], axis=-1)
    return x_grid

def encode_material(mat_fn, x_grid, NoL, NoV, encoder):
    shp = BRDFSHAPE

    mat = readMERLBRDF(mat_fn)
    mat = mat.reshape(-1, 3)
    if Config.use_dot_mask:
        mat[NoL<0.0] = (0.0, 0.0, 0.0)
        mat[NoV<0.0] = (0.0, 0.0, 0.0)

    mat = mat.reshape(1, -1, 3)

    bs = 1
    data = mat

    for _ in range(Config.log_times):
        data = np.log(data + 1.0)

    data = data / Config.global_data_std

    datasize = data.shape[0]

    x_input = np.repeat(x_grid, bs, 0)
    z = encoder.predict([x_input, data])
    return z

def decode_material(x_grid, NoL, NoV, z, decoder, is_hypernet=False):
    shp = BRDFSHAPE
    bs = z[0].shape[0]

    # reconstruction the brdf slice into MERL format
    x_input = np.repeat(x_grid, bs, 0)
    NoL_batch = np.repeat(NoL.reshape(1, -1, 1), bs, 0)
    NoV_batch = np.repeat(NoV.reshape(1, -1, 1), bs, 0)
    target_input_tensor = [z[0], z[1], x_input] if not is_hypernet else [z[0], x_input]
    if Config.use_dot_mask:
        target_input_tensor += [NoL_batch, NoV_batch]

    mat = decoder.predict(target_input_tensor).reshape(bs, -1, Config.out_dim)

    mat = mat * Config.global_data_std
    for _ in range(Config.log_times):
        mat = np.exp(mat) - 1.0

    mat = mat.reshape(-1, 3)

    if Config.use_dot_mask:
        mat[(NoL<0.0) | (NoV<0.0)] = 0.0
        mat[(NoL<0.0) | (NoV<0.0)] = 0.0
        mat[(NoL<0.0) | (NoV<0.0)] = 0.0

    mat = mat.reshape(shp[0], shp[1], shp[2], 3)

    return mat

def get_latent_vector_list(paths):
    file_list = []
    for path in paths:
        file_list += glob.glob(osp.join(path, '*_latentVector.npy')) 

    file_list.sort(key=lambda x: osp.basename(x))
    return file_list

def save_BRDF(filename, brdf):
    saveMERLBRDF(filename, brdf)