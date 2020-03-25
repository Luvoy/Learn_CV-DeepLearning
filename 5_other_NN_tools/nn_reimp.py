# -*- coding: utf-8 -*-
# TODO 为什么要sqrt

import numpy as np
import matplotlib.pyplot as plt
import sklearn.datasets

np.random.seed(666)
N = 200

nn_input_dim = 2
nn_output_dim = 2
learn_rate = 0.001
reg_lambda = 0.001

x, y = sklearn.datasets.make_moons(n_samples=N, noise=0.1)
# print(x)
# print(type(y))
# print(y)
# print(y.shape)

# 初始层: N * input_dim    初始层可看成input_dim个列向量堆叠 每个列向量N个数
# 初始层·w1+b1 = 隐藏层
# w1: input_dim * hidden_dim
# b1: 标量
# 隐藏层: N * hidden_dim    可看成hidden_dim个列向量堆叠 每个列向量N个数
# 隐藏层·w2+b2 = 输出层
# w2: hidden_dim * output_dim
# b2: 标量
# 输出层: N * output_dim    可看成output_dim个列向量堆叠 每个列向量N个数

# 这里理解为对于N个样本中的每一个数,都要去看看它是哪个分类的,也就是output_dim中哪一个
# 所以是行求和


def active_func(input_np, type='sigmoid', paras=None):
    if type == 'sigmoid':
        return 1 / (1 + np.exp(-input_np))
    elif type == 'tanh':
        return np.tanh(input_np)
    elif type == 'relu':
        input_np[input_np <= 0] = 0
        input_np[input_np > 0] *= paras[0]
        return input_np
    else:
        return np.tanh(input_np)


def soft_max_np(input_np, axis=1):
    return np.exp(input_np) / np.sum(
        np.exp(input_np), axis=axis, keepdims=True)


def loss(model, reg_flag=True):
    w1, b1, w2, b2 = model["w1"], model['b1'], model["w2"], model["b2"]
    z1 = x @ w1 + b1
    z1_activ = active_func(z1, type='tanh', paras=None)
    z2 = z1_activ @ w2 + b2
    probs = soft_max_np(z2, axis=1)
    cross_entropy = -np.log(probs[range(N), y])
    sum_loss = np.sum(cross_entropy)
    if reg_flag is True:
        total_loss = sum_loss + reg_lambda * 1 / 2 * (np.sum(np.square(w1)) +
                                                      np.sum(np.square(w2)))
    else:
        total_loss = sum_loss
    return total_loss / N


def predict(model, x_new):
    w1, b1, w2, b2 = model["w1"], model['b1'], model["w2"], model["b2"]
    z1 = x_new @ w1 + b1
    z1_activ = active_func(z1, type='tanh', paras=None)
    z2 = z1_activ @ w2 + b2
    probs = soft_max_np(z2, axis=1)
    return np.argmax(probs, axis=1)


def build_model(nn_hidden_dim, max_iter=20000, print_loss=True, reg_flag=True):
    w1 = np.random.randn(nn_input_dim, nn_hidden_dim) / np.sqrt(nn_input_dim)
    b1 = np.zeros((1, nn_hidden_dim))
    w2 = np.random.randn(nn_hidden_dim, nn_output_dim) / np.sqrt(nn_hidden_dim)
    b2 = np.zeros((1, nn_output_dim))

    for i in range(max_iter):
        z1 = x @ w1 + b1
        z1_activ = active_func(z1, type='tanh', paras=None)
        z2 = z1_activ @ w2 + b2
        probs = soft_max_np(z2, axis=1)

        delta3 = probs

        # 反向传播

        delta3[range(N), y] -= 1

        dw2 = z1_activ.T @ delta3
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = delta3 @ w2.T * (1 - np.power(z1_activ, 2))
        dw1 = x.T @ delta2
        db1 = np.sum(delta2, axis=0, keepdims=True)
        if reg_flag is True:
            w1 = w1 - learn_rate * (dw1 + reg_lambda * w1)
            w2 = w2 - learn_rate * (dw2 + reg_lambda * w2)
            b1 = b1 - learn_rate * (db1 + reg_lambda * b1)
            b2 = b2 - learn_rate * (db2 + reg_lambda * b2)
        else:
            w1 -= learn_rate * dw1
            w2 -= learn_rate * dw2
            b1 -= learn_rate * db1
            b2 -= learn_rate * db2

        model = {"w1": w1, "b1": b1, "w2": w2, "b2": b2}

        if print_loss and i % 1000 == 0:
            print(f"loss after iter{i}: {loss(model,reg_flag=reg_flag)}")
    return model


model_1 = build_model(nn_hidden_dim=20,
                      max_iter=30000,
                      print_loss=True,
                      reg_flag=True)

print(predict(model_1, x_new=np.array([1, 1])))
plt.scatter(1, 1, color="green")

print(predict(model_1, x_new=np.array([2, 1.5])))
plt.scatter(2, 1.5, color="green")

plt.scatter(x[:, 0], x[:, 1], c=y)
plt.show()
