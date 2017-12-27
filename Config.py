class Config(object):
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 3
    num_steps = 100  # number of steps to unroll the RNN for
    hidden_size = 1024  # size of hidden layer of neurons
    iteration = 30
    keep_prob = 0.75
    batch_size = 64
    ckpt_steps = 2000
    model_dir = '/home/liuduo/ckpt'
    model_path = model_dir + '/model.ckpt'  # the path of model that need to save or load
