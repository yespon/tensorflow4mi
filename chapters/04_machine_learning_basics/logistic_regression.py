#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 21:06
# @Author  : Yespon
# @Email   : yespon@qq.com
# @File    : softmax_regression.py
# @Software: PyCharm Community Edition

import os
import tensorflow as tf

# class LogisticRegression():
# ##################################################
# 初始化变量和模型参数，定义训练闭环中的运算
# ##################################################

W = tf.Variable(tf.zeros([5, 1]), name="weights")
b = tf.Variable(0., name="bias")
# 创建一个 Saver 对象
# saver = tf.train.Saver()


# 之前的推断现在用于值的合并
# def inference(X):
def combine_inputs(X):
    # 计算推断模型在数据 X 上的输出，并将结果返回
    return tf.matmul(X, W) + b


#
def inference(X):
    return tf.sigmoid(combine_inputs(X))


def loss(X, Y):
    # 依据训练数据 X 及期望输出 Y 计算损失
    # Y_predicted = inference(X)
    # return tf.reduce_sum(tf.squared_difference(Y, Y_predicted))
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=combine_inputs(X), labels=Y))


def read_csv(batch_size, file_name, record_defaults):
    filename_queue = tf.train.string_input_producer([os.path.dirname(__file__) + "/" + file_name])
    reader = tf.TextLineReader(skip_header_lines=1)
    key, value = reader.read(filename_queue)

    # decode_csv 会将字符串（文本行）转换到具有指定默认值的张量列构成的元组中
    # 它还会为每一列设置数据类型
    decoded = tf.decode_csv(value, record_defaults=record_defaults)

    # 实际上会读取一个文件，并加载一个张量中的 batch_size 行
    return tf.train.shuffle_batch(decoded, batch_size=batch_size, capacity=batch_size * 50, min_after_dequeue=batch_size)


def inputs():
    # 读取或生成训练数据 X 及其期望输出 Y
    passenger_id, survived, pclass, name, sex, age, sibsp, parch, ticket, fare, cabin, embarked\
        = read_csv(100, "train.csv", [[0.0], [0.0], [0], [""],
                                       [""], [0.0], [0.0], [0.0],
                                       [""], [0.0], [""], [""]])
    # 转换属性数据
    is_first_class = tf.to_float(tf.equal(pclass, [1]))
    is_second_class = tf.to_float(tf.equal(pclass, [2]))
    is_third_class = tf.to_float(tf.equal(pclass, [3]))

    gender = tf.to_float(tf.equal(sex, ["female"]))

    # 最终将所有特征排列在一个矩阵中，然后对该矩阵装置，使其每行对应一个样本，每列对应一种特征
    features = tf.transpose(tf.stack([is_first_class, is_second_class, is_third_class, gender, age]))
    survived = tf.reshape(survived, [100, 1])
    return features, survived


def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    learning_rate = 0.01
    return tf.train.GradientDescentOptimizer(learning_rate).minimize(total_loss)


def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估
    predicted = tf.cast(inference(X) > 0.5, tf.float32)
    print(sess.run(tf.reduce_mean(tf.cast(tf.equal(predicted, Y), tf.float32))))

# #####################################################
# 在一个会话对象中启动数据流图，搭建流程
# #####################################################
with tf.Session() as sess:
    if 'sess' in locals() and sess is not None:
        print("Close interactive session.")
        sess.close

    tf.global_variables_initializer().run()

    X, Y = inputs()

    total_loss = loss(X, Y)
    train_op = train(total_loss)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    # 实际的训练迭代次数
    training_steps = 1000

    initial_step = 0

    # 验证之前是否已经保存了检查点文件
    # ckpt = tf.train.get_checkpoint_state(os.path.dirname(__file__))
    # if ckpt and ckpt.model_checkpoint_path:
    # 从检查点恢复模型参数
    # saver.restore(sess, ckpt.model_checkpoint_path)
    # initial_step = int(ckpt.model_checkpoint_path.rsplit('-', 1)[1])

    for step in range(initial_step, training_steps):
        sess.run([train_op])
        # 处于调试和学习的目的，查看损失在训练过程中递减的情况
        if step % 10 == 0:
            print("Epoch: ", step, "\nloss: ", sess.run([total_loss]))
            # saver.save(sess, 'my_model', global_step=step)

    print("Final model W=", sess.run(W), "b=", sess.run(b))
    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    # saver.save(sess, 'my_model', global_step=step)
    sess.close()
