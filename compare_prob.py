import tensorflow as tf
import numpy as np
import _pickle
import Config
import Model
import time
import math
import csv

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
        test_pairs = [("google play store gift card,", "so i can buy movies to put on tablet"),
                      (
                          "my mom gave me twenty dollars for christmas yesterday,",
                          "so i can buy movies to put on tablet"),
                      ("how about we have a dinner tomorrow night?", "so i can buy movies to put on tablet"),
                      ("i look forward to working with you in the following months,",
                       "so i can buy movies to put on tablet"),
                      ("what's the weather tomorrow?", "so i can buy movies to put on tablet"),
                      ("can you resend last text,", "i deleted it by accident"),
                      ("oh i've made a big mistake,", "i deleted it by accident"),
                      ("what's the weather tomorrow?", "i deleted it by accident"),
                      ("how about we have a dinner tomorrow night?", "i deleted it by accident"),
                      ("i look forward to working with you in the following months,", "i deleted it by accident"),
                      ("it always gets harder before it gets better.", "don't worry things will get better"),
                      ("hi that isn't a big problem,", "don't worry things will get better"),
                      ("i look forward to working with you in the following months,",
                       "don't worry things will get better"),
                      ("i think you'd be surprised how much zola would like a puppy.",
                       "don't worry things will get better"),
                      ("i've got to get taylor another bag for her makeup brushes.",
                       "don't worry things will get better"),
                      ("i just wanna spend some time with you.", "i feel we are not connecting on any level"),
                      ("i want to go see a movie with you.", "i feel we are not connecting on any level"),
                      ("i look forward to working with you in the following months,",
                       "i feel we are not connecting on any level"),
                      ("there is no help button.", "i feel we are not connecting on any level"),
                      ("some of his comments throughout the post just rub me wrong.",
                       "i feel we are not connecting on any level"),
                      ("and the same for you.", "you can still talk to me as friends"),
                      ("let's forget those unpleasant things.", "you can still talk to me as friends"),
                      ("i look forward to working with you in the following months,",
                       "you can still talk to me as friends"),
                      ("what's the weather tomorrow?", "you can still talk to me as friends"),
                      ("google play store gift card,", "you can still talk to me as friends"),
                      ]
        rows = list()
        rows2 = list()
        for s1, s2 in test_pairs:
            length_penalty_prob, first_letter_probs, every_letter_prob = get_prob(s1, s2, mtest, session)
            rows.append([s1] + [length_penalty_prob] + every_letter_prob)
            keys = []
            values = []
            for k in sorted(first_letter_probs.keys()):
                keys.append(k)
                values.append(first_letter_probs[k])
            rows2.append(keys)
            rows2.append(values)
        with open('/Users/liuduo/Desktop/result.csv', 'w') as fd:
            fd_csv = csv.writer(fd)
            fd_csv.writerows(rows)
        with open('/Users/liuduo/Desktop/result2.csv', 'w') as fd:
            fd_csv = csv.writer(fd)
            fd_csv.writerows(rows2)


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
            top_indices = np.argsort(prob.reshape(-1) * -1)
            first_letter_probs = dict([(idx_to_char[x], prob.reshape(-1)[x]) for x in top_indices])
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
