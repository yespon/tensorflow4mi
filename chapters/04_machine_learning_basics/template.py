#!/usr/bin/python3
# -*- coding: utf-8 -*-
# @Time    : 2017/11/29 21:03
# @Author  : Yespon
# @Email   : yespon@qq.com
# @File    : linear_regression.py
# @Software: PyCharm Community Edition
import tensorflow as tf

# ##################################################
# 初始化变量和模型参数，定义训练闭环中的运算
# ##################################################


def inference(X):
    # 计算推断模型在数据 X 上的输出，并将结果返回
    pass


def loss(X, Y):
    # 依据训练数据 X 及期望输出 Y 计算损失
    pass


def inputs():
    # 读取或生成训练数据 X 及其期望输出 Y
    pass


def train(total_loss):
    # 依据计算的总损失训练或调整模型参数
    pass


def evaluate(sess, X, Y):
    # 对训练得到的模型进行评估
    pass

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
        if step % 50 == 0:
            print("Epoch: ", step, "\nloss: ", sess.run([total_loss]))
            # saver.save(sess, 'my_model', global_step=step)

    evaluate(sess, X, Y)

    coord.request_stop()
    coord.join(threads)
    # saver.save(sess, 'my_model', global_step=step)
    sess.close()
