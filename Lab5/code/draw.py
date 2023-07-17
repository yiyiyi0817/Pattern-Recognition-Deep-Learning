from matplotlib import pyplot as plt
import mat4py
import numpy as np
import torch

# 用于绘制判别器D的决策边界, 对于每个背景点，用D判断其属于真实数据的概率，如果小于0.5，则用白色表示，否则用黑色表示。
def draw_background(D, x_min, x_max, y_min, y_max, device):
    i = x_min
    background = []
    color = []
    while i <= x_max - 0.01:
        j = y_min
        while j <= y_max - 0.01:
            background.append([i, j])
            j += 0.01
        background.append([i, y_max])
        i += 0.01
    j = y_min
    while j <= y_max - 0.01:
        background.append([i, j])
        j += 0.01
        background.append([i, y_max])
    background.append([x_max, y_max])
    result = D(torch.Tensor(background).to(device))
    for i in range(len(result)):
        if result[i] < 0.5:
            color.append('w')
        else:
            color.append('k')
    background = np.array(background)
    plt.scatter(background[:, 0], background[:, 1], c=color)

# 用于绘制真实数据和生成数据的散点图，首先调用draw_background绘制判别器的决策边界，然后用plt.scatter绘制真实数据data和生成数据xy，并用不同的颜色和大小区分
def draw_scatter(args, D, xy, epoch):
    data = mat4py.loadmat(args.data_dir)['xx']
    data = np.array(data)
    x = xy[:, 0]
    y = xy[:, 1]
    draw_background(D, -0.5, 2.2, -0.2, 1, args.device)
    plt.xlim(-0.5, 2.2)
    plt.ylim(-0.2, 1)
    plt.scatter(data[:, 0], data[:, 1], c='b', s=10)
    plt.scatter(x, y, c='r', s=10)
    plt.savefig(args.plot_result_dir + '/' + args.model + '/epoch-' + '{:04d}'.format(epoch) + '.jpg')