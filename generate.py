import tensorflow as tf
import numpy as np
import _pickle
import Config
import Model
import time

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()

char_to_idx, idx_to_char = _pickle.load(open(config.model_dir + '/vocab.bin', 'rb'))

config.vocab_size = len(char_to_idx)
flags = tf.flags
flags.DEFINE_boolean('is_sample', False, '')
flags.DEFINE_boolean('is_beams', True, '')
flags.DEFINE_integer('beam_size', 5, '')
flags.DEFINE_integer('len_of_generation', 50, '')
flags.DEFINE_string('input_path', '/home/liuduo/ckpt/test_input', '')
flags.DEFINE_string('output_path', '/home/liuduo/ckpt/test_output', '')
FLAGS = flags.FLAGS
is_sample = FLAGS.is_sample
is_beams = FLAGS.is_beams
beam_size = FLAGS.beam_size
len_of_generation = FLAGS.len_of_generation
start_sentence_list = [x.strip().replace('\n', '').lower() for x in open(FLAGS.input_path).readlines()]
output_fd = open(FLAGS.output_path, 'w')


def run_epoch(session, m, data, eval_op, state=None):
    """Runs the model on the given data."""
    x = data.reshape((1, 1))
    prob, _state, _ = session.run([m._prob, m.final_state, eval_op],
                                  {m.input_data: x,
                                   m.initial_state: state})
    return prob, _state


def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        config.batch_size = 1
        config.num_steps = 1

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model.Model(is_training=False, config=config)

        # tf.global_variables_initializer().run()

        model_saver = tf.train.Saver()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'model loading ...')
        model_saver.restore(session, tf.train.get_checkpoint_state(config.model_dir).model_checkpoint_path)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'Done!')

        if not is_beams:
            for start_sentence in start_sentence_list:
                # sentence state
                char_list = list(start_sentence)
                start_idx = char_to_idx[char_list[0]]
                _state = mtest.initial_state.eval()
                test_data = np.int32([start_idx])
                prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                gen_res = [char_list[0]]
                for i in range(1, len(char_list)):
                    char = char_list[i]
                    try:
                        char_index = char_to_idx[char]
                    except KeyError:
                        char_index = np.argmax(prob.reshape(-1))
                    prob, _state = run_epoch(session, mtest, np.int32([char_index]), tf.no_op(), _state)
                    gen_res.append(char)
                # gen text
                if is_sample:
                    gen = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                    gen = gen[0]
                else:
                    gen = np.argmax(prob.reshape(-1))
                test_data = np.int32(gen)
                gen_res.append(idx_to_char[gen])
                for i in range(len_of_generation - 1):
                    prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                    if is_sample:
                        gen = np.random.choice(config.vocab_size, 1, p=prob.reshape(-1))
                        gen = gen[0]
                    else:
                        gen = np.argmax(prob.reshape(-1))
                    test_data = np.int32(gen)
                    gen_res.append(idx_to_char[gen])
                output_lines = list()
                output_lines.append(start_sentence)
                output_lines.append('1: ' + ''.join(gen_res)[len(start_sentence):])
                output_fd.writelines([x + '\n' for x in output_lines])
                output_fd.flush()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'Generated Result: ',
                      ''.join(gen_res))
        else:
            for start_sentence in start_sentence_list:
                # sentence state
                char_list = list(start_sentence)
                start_idx = char_to_idx[char_list[0]]
                _state = mtest.initial_state.eval()
                beams = [(0.0, [idx_to_char[start_idx]], idx_to_char[start_idx])]
                test_data = np.int32([start_idx])
                prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
                y1 = np.log(1e-20 + prob.reshape(-1))
                beams = [(beams[0][0], beams[0][1], beams[0][2], _state)]
                for i in range(1, len(char_list)):
                    char = char_list[i]
                    try:
                        char_index = char_to_idx[char]
                    except KeyError:
                        top_indices = np.argsort(-y1)
                        char_index = top_indices[0]
                    prob, _state = run_epoch(session, mtest, np.int32([char_index]), tf.no_op(), beams[0][3])
                    y1 = np.log(1e-20 + prob.reshape(-1))
                    beams = [(beams[0][0], beams[0][1] + [char], char_index, _state)]
                # gen text
                if is_sample:
                    top_indices = np.random.choice(config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                else:
                    top_indices = np.argsort(-y1)
                b = beams[0]
                beam_candidates = []
                for i in range(beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
                    print('*****: ' + str(prob.reshape(-1)[wordix]) + '\t' + '$' + idx_to_char[wordix] + '$')
                beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
                beams = beam_candidates[:beam_size]  # truncate to get new beams
                for xy in range(len_of_generation - 1):
                    beam_candidates = []
                    for b in beams:
                        test_data = np.int32(b[2])
                        prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), b[3])
                        y1 = np.log(1e-20 + prob.reshape(-1))
                        if is_sample:
                            top_indices = np.random.choice(config.vocab_size, beam_size, replace=False,
                                                           p=prob.reshape(-1))
                        else:
                            top_indices = np.argsort(-y1)
                        for i in range(beam_size):
                            wordix = top_indices[i]
                            beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_char[wordix]], wordix, _state))
                            print('*****: ' + str(prob.reshape(-1)[wordix]) + '\t' + '$' + idx_to_char[wordix] + '$')
                    beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
                    beams = beam_candidates[:beam_size]  # truncate to get new beams
                output_lines = list()
                output_lines.append(start_sentence)
                for i in range(beam_size):
                    output_lines.append(str(i + 1) + ': ' + ''.join(beams[i][1])[len(start_sentence):])
                output_fd.writelines([x + '\n' for x in output_lines])
                output_fd.flush()
                print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'Generated Result: ',
                      ''.join(beams[0][1]))


if __name__ == "__main__":
    tf.app.run()
