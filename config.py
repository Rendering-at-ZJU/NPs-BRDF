from datetime import datetime
TIMESTAMP = "{0:%Y-%m-%dT%H-%M-%S}".format(datetime.now())

class Config:
    test_mode = True 
    mask_samples = False 

    batch_size = 16
    epochs = 40000

    MERL_path = './MERL_dataset/'
    EPFL_path = './EPFL_dataset/'

    output_path = 'output/{}_tmp/'.format(TIMESTAMP)
    model_path = 'models/{}_tmp/'.format(TIMESTAMP)
    log_path = 'runs/{}_tmp/'.format(TIMESTAMP)

    use_ringmap = True
    log_times = 4

    z_dim = 7
    in_dim = 3 if not use_ringmap else 4
    out_dim = 3

    h_dim = 64
    h_layers = 3
    h_dims = [400] * (h_layers - 1) + [h_dim]

    a_layers = 4
    a_dims = [h_dim] * (a_layers - 1)

    dc_layers = 7 #decoder layer
    dc_dims = [400] * (dc_layers - 1) + [out_dim]

    use_dot_mask = True

    global_data_std = 0.15

    saving_interval = 50

    max_samples = 180*90 #16200