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
    ckpt_steps = 200 
    model_path = '/home/liuduo/ckpt'  # the path of model that need to save or load

    # parameters for generation
    is_sample = True  # true means using sample, if not using max
    is_beams = True  # whether or not using beam search
    beam_size = 25  # size of beam search
    len_of_generation = 100  # The number of characters by generated
    start_sentence = 'Our business is not unknown to the senate'  # the seed sentence to generate text
