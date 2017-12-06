class Config(object):
    init_scale = 0.04
    learning_rate = 0.001
    max_grad_norm = 15
    num_layers = 3
    num_steps = 30  # number of steps to unroll the RNN for
    hidden_size = 800  # size of hidden layer of neurons
    iteration = 30
    keep_prob = 0.5
    batch_size = 128
    ckpt_steps = 2000
    model_dir = '/home/liuduo/ckpt'
    model_path = model_dir + '/model.ckpt'  # the path of model that need to save or load
