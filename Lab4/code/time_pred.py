import os
import argparse
import time
import numpy as np
import pandas as pd
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import random_split
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.preprocessing import MinMaxScaler
from model import *

parser = argparse.ArgumentParser('Lab4-2')
parser.add_argument('--sim_time', type=int, default=72) # 模拟时间，单位为h
parser.add_argument('--batch_size', type=int, default=16)
parser.add_argument('--niters', type=int, default=500)
parser.add_argument('--test_freq', type=int, default=1)
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--know_len', type=int, default=24*5*6) # 通过待预测t+1时刻之前know_len个信号来预测下一信号
parser.add_argument('--sample_slide_step', type=int, default=24*7*6) # 通过原始信号滑窗制作数据集时的滑窗步长
parser.add_argument('--pre_len', type=int, default=24*2*6) # 预测t时刻之后的pre_len个信号，单步预测就是1
parser.add_argument('--hidden_size', type=int, default=64)
parser.add_argument('--model_folder_dir', type=str, default='model') # 最佳模型存储路径
parser.add_argument('--tensorboard_dir', type=str, default='')
parser.add_argument('--plot_save_dir', type=str, default='png') # 测试图片储存路径

args = parser.parse_args()

class GRUNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, n_layers):
        super(GRUNet, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.gru_cell = nn.GRUCell(input_dim, hidden_dim)
        self.fc = nn.Linear(hidden_dim * 105, output_dim)

    def forward(self, x):
        h0 = torch.zeros(x.size(0), self.hidden_dim).to(device)
        h1 = torch.zeros(x.size(0), self.hidden_dim).to(device)
        for i in range(x.size(1)):
            h0 = self.gru_cell(x[:, i, :], h0)
            h1 = torch.cat((h1, h0), dim=1)
        out = self.fc(h1)
        return out

def visualize(test_y, pred_y, epoch, test_loss):
    # 清除当前图形的所有轴，但不关闭窗口，因此可以继续用于其他绘图。
    plt.clf()
    plt.xlabel('t')
    plt.ylabel('T(degC)')
    plt.plot(test_y.cpu().numpy()[:, 1], 'g-', linewidth=0.5)
    plt.plot(pred_y.cpu().numpy()[:, 1], 'b--', linewidth=0.5)
    # plt.ylim(0, 500)
    plt.title('{:05d} | {:.1f}'.format(epoch, test_loss))
    plt.tight_layout()
    plt.xticks(range(0, len(test_y.cpu().numpy()[:, 1]), 1000)) # set x-axis ticks
    plt.savefig(args.plot_save_dir + '/{:04d}'.format(epoch))
    plt.draw()
    

if __name__ == '__main__':
    torch.set_default_dtype(torch.float64)
    device = args.device
    data = pd.read_csv('jena_climate_2009_2016.csv')
    climate_origin_data = data[['p (mbar)', 'T (degC)', 'Tpot (K)', 'Tdew (degC)', 'rh (%)']].values.astype(np.float32)
    climate_origin_data = torch.tensor(climate_origin_data)
    scaler = MinMaxScaler(feature_range=(0, 1))
    climate_data = torch.tensor(scaler.fit_transform(climate_origin_data)).to(device) # torch.Size([420551, 5])
    
    know_len = args.know_len
    sample_slide_step = args.sample_slide_step
    pre_len = args.pre_len
    # x中的每个元素是：已知的过去的know_len个信号
    x = climate_data.unfold(dimension=0, size=know_len, step=sample_slide_step) 
    # y中的每个元素是：x对应位置的第know_len+pre_len个信号，即待预测信号
    y = climate_data[know_len:,:].unfold(dimension=0, size=pre_len, step=sample_slide_step) # y: torch.Size([417, 5, 288])
    x = x[:y.size(0), :]        # x: torch.Size([417, 5, 720])
    # print ('y:', y.size(), 'x:', x.size())
    # exit()
    
    # 6年(312周)训练，2年测试
    train_test_split_num = 312
    train_x = x[:train_test_split_num]
    train_y = y[:train_test_split_num]
    test_x = x[train_test_split_num:].permute(0, 2, 1)
    test_y = y[train_test_split_num:] # torch.Size([105, 5, 288])
    scaler.fit(climate_origin_data)
    test_y_plot = test_y.transpose(1, 2).contiguous().view(test_x.size(0) * 288, 5) # (test_len, 5)不含归一化，用来画图
    test_y_plot = torch.tensor(scaler.inverse_transform(test_y_plot.cpu())) 
    
    # 创建DataLoader
    batch_size = args.batch_size
    train_dataset = TensorDataset(train_x, train_y)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    # 定义模型
    model = GRUNet(5, args.hidden_size, 288*5, 1).to(device)

    # 定义损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    #训练模型
    for epoch in range(args.niters):
        for batch_x, batch_y in train_loader:
            # 前向传播
            batch_x = batch_x.permute(0, 2, 1).to(device) #[256, 720, 5]
            pred_y_1d = model(batch_x).to(device)
            # print ('batch_x.size:', batch_x.size(), 'pred_y_1d:', pred_y_1d.size())
            pred_y = pred_y_1d.view(batch_x.size(0), 5, 288)
            loss = criterion(pred_y, batch_y)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()    
            
        scheduler.step(loss)
        print(f'Epoch: {epoch}, Loss: {loss.item()}')
        

        # 测试模型
        if epoch % args.test_freq == 0:
            with torch.no_grad():
                test_x = test_x.to(device)
                # model = torch.load('./model/GRU.pth')
                pred_y_1d = model(test_x).to(device)
                pred_y = pred_y_1d.view(test_x.size(0), 5, 288)
                # print ('pred_y:', pred_y.size(), 'test_y:', test_y.size())
                test_loss = criterion(pred_y, test_y)
                print(f'Test Total Loss: {test_loss.item()}')
                scaler.fit(climate_origin_data)
                pred_y = pred_y.transpose(1, 2).contiguous().view(test_x.size(0) * 288, 5) # (test_len, 5)
                pred_y = torch.tensor(scaler.inverse_transform(pred_y.cpu())) 
                visualize(test_y_plot, pred_y, epoch, test_loss)
    
    torch.save(model, args.model_folder_dir + '/climate_GRU.pth')