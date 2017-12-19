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
        test_pairs = [("this wasn't because of anything you did.", 'but it is being canceled and i need you to delete it off of your pay pal'),
                      ('How about we have a dinner tomorrow night?', 'hello world'),
                      ('How about we have a dinner tomorrow night?', 'if you want'),
                      ('How about we have a dinner tomorrow night?', 'What is your name'),
                      ('How about we have a dinner tomorrow night?',
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ('I look forward to working with you in the following months.', 'lunch'),
                      ('I look forward to working with you in the following months.', 'hello world'),
                      ('I look forward to working with you in the following months.', 'if you want'),
                      ('I look forward to working with you in the following months.', 'What is your name'),
                      ('I look forward to working with you in the following months.',
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("What's the weather tomorrow?", 'lunch'),
                      ("What's the weather tomorrow?", 'hello world'),
                      ("What's the weather tomorrow?", 'if you want'),
                      ("What's the weather tomorrow?", 'What is your name'),
                      ("What's the weather tomorrow?",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("i can't even talk correctly.", 'lunch'),
                      ("i can't even talk correctly.", 'hello world'),
                      ("i can't even talk correctly.", 'if you want'),
                      ("i can't even talk correctly.", 'What is your name'),
                      ("i can't even talk correctly.",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("don't worry about the car seat tomorrow.", 'lunch'),
                      ("don't worry about the car seat tomorrow.", 'hello world'),
                      ("don't worry about the car seat tomorrow.", 'if you want'),
                      ("don't worry about the car seat tomorrow.", 'What is your name'),
                      ("don't worry about the car seat tomorrow.",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("as long as everything stays the same,", 'lunch'),
                      ("as long as everything stays the same,", 'hello world'),
                      ("as long as everything stays the same,", 'if you want'),
                      ("as long as everything stays the same,", 'What is your name'),
                      ("as long as everything stays the same,",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("i'll let you know when i close.", 'lunch'),
                      ("i'll let you know when i close.", 'hello world'),
                      ("i'll let you know when i close.", 'if you want'),
                      ("i'll let you know when i close.", 'What is your name'),
                      ("i'll let you know when i close.",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("i want to hang out with my friends,", 'lunch'),
                      ("i want to hang out with my friends,", 'hello world'),
                      ("i want to hang out with my friends,", 'if you want'),
                      ("i want to hang out with my friends,", 'What is your name'),
                      ("i want to hang out with my friends,",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("i'm really not feeling good today dad.", 'lunch'),
                      ("i'm really not feeling good today dad.", 'hello world'),
                      ("i'm really not feeling good today dad.", 'if you want'),
                      ("i'm really not feeling good today dad.", 'What is your name'),
                      ("i'm really not feeling good today dad.",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ("people rent out their house for travelers.", 'lunch'),
                      ("people rent out their house for travelers.", 'hello world'),
                      ("people rent out their house for travelers.", 'if you want'),
                      ("people rent out their house for travelers.", 'What is your name'),
                      ("people rent out their house for travelers.",
                       'but it is being canceled and i need you to delete it off of your pay pal'),
                      ]
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
