import tensorflow as tf
import numpy as np
import _pickle
import Config
import Model
import time
import math
from collections import OrderedDict

config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True
config_tf.inter_op_parallelism_threads = 1
config_tf.intra_op_parallelism_threads = 1

config = Config.Config()

char_to_idx, idx_to_char = _pickle.load(open(config.model_dir + '/vocab.bin', 'rb'))
frequency = _pickle.load(open('/home/liuduo/frequency.bin', 'rb'))

config.vocab_size = len(char_to_idx)


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

        model_saver = tf.train.Saver()
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'model loading ...')
        model_saver.restore(session, tf.train.get_checkpoint_state(config.model_dir).model_checkpoint_path)
        print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()) + ' ' + 'Done!')

        # dictation、correction、replacement、nonsense
        test_pairs = [('How about we have a dinner tomorrow night? ', 'and then see a movie'),
                      ('How about we have a dinner tomorrow night? ', 'lunch'),
                      ('How about we have a dinner tomorrow night? ', 'tomorrow afternoon'),
                      ('How about we have a dinner tomorrow night? ', 'What is your name'),
                      ('I look forward to working with you in the following months. ', 'It is so great'),
                      ('I look forward to working with you in the following months. ', 'weeks'),
                      ('I look forward to working with you in the following months. ', 'your team'),
                      ('I look forward to working with you in the following months. ', 'I am hungry'),
                      ("What's the weather tomorrow? ", 'what about go to the park'),
                      ("What's the weather tomorrow? ", 'the day after tomorrow'),
                      ("What's the weather tomorrow? ", 'today'),
                      ("What's the weather tomorrow? ", 'What is your name')]
        output_lines = list()
        for s1, s2 in test_pairs:
            p_s2_s1 = get_prob(s1, s2, mtest, session)
            p_s2 = get_prob2(s2, mtest, session)
            # result = p_s2_s1 - p_s2
            print(s1 + '    ' + s2 + '    ' + str(p_s2_s1))
            output_lines.append(s1 + '    ' + s2 + '    ' + str(p_s2_s1) + '\n')
        open('/Users/liuduo/Desktop/result.txt', 'w').writelines(output_lines)


def get_prob2(s1, mtest, session):
    s1_chars = list(s1.lower())

    start_idx = char_to_idx[s1_chars[0]]
    _state = mtest.initial_state.eval()
    prob, _state = run_epoch(session, mtest, np.int32([start_idx]), tf.no_op(), _state)
    s1_probs = list()
    s1_probs.append(frequency[s1_chars[0]])
    for i in range(1, len(s1_chars)):
        idx = char_to_idx[s1_chars[i]]
        s1_probs.append(prob.reshape(-1)[idx])
        prob, _state = run_epoch(session, mtest, np.int32([idx]), tf.no_op(), _state)
    return sum([np.log(1e-20 + x) for x in s1_probs])


def get_prob(s1, s2, mtest, session):
    s1_chars = list(s1.lower())
    s2_chars = list(s2.lower())

    # 依次将s1的所有字符顺序输入到RNN模型中，得到最终的state
    start_idx = char_to_idx[s1_chars[0]]
    _state = mtest.initial_state.eval()
    test_data = np.int32([start_idx])
    prob, _state = run_epoch(session, mtest, test_data, tf.no_op(), _state)
    beams = [(0.0, [idx_to_char[start_idx]], idx_to_char[start_idx], _state)]
    for i in range(1, len(s1_chars)):
        char = s1_chars[i]
        char_index = char_to_idx[char]
        prob, _state = run_epoch(session, mtest, np.int32([char_index]), tf.no_op(), beams[0][3])
        beams = [(beams[0][0], beams[0][1] + [char], char_index, _state)]
    # 依次将s2的所有字符顺序输入到RNN模型中，计算每个字符产生的概率，从而得到最终整个s2字符序列的产生概率
    s2_probs = list()
    for i, char in enumerate(s2_chars):
        if i == 0:
            top_indices = np.argsort(prob.reshape(-1) * -1)[:5]
            first_letter_probs = OrderedDict([(idx_to_char[x], prob.reshape(-1)[x]) for x in top_indices])
        idx = char_to_idx[char]
        s2_probs.append(prob.reshape(-1)[idx])
        prob, _state = run_epoch(session, mtest, np.int32([idx]), tf.no_op(), _state)

    # 1. 相乘然后开s2.length次方
    result = 1
    for x in s2_probs:
        result *= x
    return math.pow(result, 1 / len(s2_chars)), first_letter_probs, s2_probs

    # 2. 相乘之后得到P(s2|s1)，然后再除以get_prob2()方法得到的P(s2)
    # return sum([np.log(1e-20 + x) for x in s2_probs])

    # 3. 直接相加
    # return sum(s2_probs)

    # 4. 套用OpenNMT的Length normalization
    # p_s2s1 = 1
    # for x in s2_probs:
    #     p_s2s1 *= x
    # numerator = np.log(p_s2s1)
    # alpha = 1.0
    # denominator = np.power(5 + len(s2_chars), alpha) / np.power(6, alpha)
    # return numerator / denominator


if __name__ == "__main__":
    tf.app.run()
