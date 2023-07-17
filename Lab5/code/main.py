import argparse
import random
import mat4py
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from torch import autograd
from model import *
from draw import *

parser = argparse.ArgumentParser('Lab5')
parser.add_argument('--data_dir', type=str, default='points.mat')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--model', default='GAN')  # GAN/ WGAN/ WGAN-GP
parser.add_argument('--niters', type=int, default=1000)
parser.add_argument('--batch_size', default=512)
parser.add_argument('--seed', default=22)
parser.add_argument('--CLAMP', default=0.1) # WGAN权重截断参数
parser.add_argument('--test_freq', type=int, default=5)
parser.add_argument('--plot_result_dir', default='plot_result')
parser.add_argument('--model_folder_dir', type=str, default='best_model') # 最佳模型存储路径
parser.add_argument('--tensorboard_dir', type=str, default='')

args = parser.parse_args(['--niters', '200', '--model', 'GAN', '--tensorboard_dir', ''])


class Points(Dataset):
    def __init__(self):
        self.data = mat4py.loadmat(args.data_dir)['xx']

    def __getitem__(self, idx):
        xy = torch.tensor(np.array(self.data[idx])).to(torch.float32)
        return xy

    def __len__(self):
        return len(self.data)
    
    
def gradient_penalty(D, x_real,  x_fake, batchsz, args):
    t = torch.rand(batchsz, 1).to(args.device)
    t = t.expand_as(x_real)
    mid = t * x_real + (1 - t) *  x_fake
    mid.requires_grad_()
    pred = D(mid)
    grads = autograd.grad(outputs=pred, inputs=mid,
                          grad_outputs=torch.ones_like(pred),
                          create_graph=True, 
                          retain_graph=True,
                          only_inputs=True)[0]
    gp = torch.pow(grads.norm(2, dim=1) - 1, 2).mean()
    return gp


if __name__ == '__main__':
    global loss_D, loss_G
    device = args.device
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 数据读入
    print('Loading points dataset...')
    dataset = Points()
    train_loader = DataLoader(dataset, shuffle=True, batch_size=args.batch_size)
    print('Data loaded!')

    # 初始化模型
    G = Generator().to(device)
    D = Discriminator().to(device)
    if args.model == 'GAN':
        optim_G = torch.optim.Adam(G.parameters(), lr=5e-4, betas=((0.5, 0.999)))
        optim_D = torch.optim.Adam(D.parameters(), lr=5e-4, betas=((0.5, 0.999)))
    elif args.model == 'WGAN':
        optim_G = torch.optim.RMSprop(G.parameters(), lr=3e-4)
        optim_D = torch.optim.RMSprop(D.parameters(), lr=3e-4)
    elif args.model == 'WGAN-GP':
        optim_G = torch.optim.Adam(G.parameters(), lr=5e-4, betas=((0.0, 0.9)))
        optim_D = torch.optim.Adam(D.parameters(), lr=5e-4, betas=((0.0, 0.9)))
    else:
        raise ValueError(f'{args.model} not supported\n')
    # scheduler = ReduceLROnPlateau(optim_G, 'min')
    # scheduler = ReduceLROnPlateau(optim_D, 'min')
    writer = SummaryWriter(args.tensorboard_dir)
    
    # 训练
    print('Start training...')
    all_loss = []
    for epoch in range(1, args.niters + 1):
        for data in train_loader:
            # 优化判别器
            x_real = data.to(device)
            batchsz = x_real.shape[0]
            pred_real = D(x_real)

            z = torch.randn(batchsz, 10).to(device)
            x_fake = G(z).detach()
            pred_fake = D(x_fake)

            # 判别器损失反向传播，越接近0越小，判别效果越好
            loss_D = - (torch.log(pred_real) + torch.log(1. - pred_fake)).mean()
            if args.model == 'WGAN':
                for p in D.parameters():
                    # print(p.data)
                    p.data.clamp_(-args.CLAMP, args.CLAMP)
            if args.model == 'WGAN-GP':
                loss_D += 0.2 * gradient_penalty(D, x_real, x_fake.detach(), batchsz, args)
            optim_D.zero_grad()
            loss_D.backward()
            optim_D.step()
            # scheduler.step(loss_D)

            # 优化生成器
            z = torch.randn(args.batch_size, 10).to(device)
            x_fake = G(z)
            pred_fake = D(x_fake)
            # 生成器损失越接近负无穷越小，判别效果越好
            loss_G = torch.log(1. - pred_fake).mean()
            optim_G.zero_grad()
            loss_G.backward()
            optim_G.step()
            # scheduler.step(loss_G)

        if epoch % args.test_freq == 0:
            print('epoch %d/%d: Discriminator loss: %.3f, Generator loss: %.3f'
                  % (epoch, args.niters, loss_D.item(), loss_G.item()))
            writer.add_scalar('loss_D', loss_D.item(), epoch)
            writer.add_scalar('loss_G', loss_G.item(), epoch)
            input = torch.randn(1000, 10).to(device)
            output = G(input)
            output = output.to('cpu').detach()
            xy = np.array(output)
            draw_scatter(args, D, xy, epoch)
             
    writer.close()
    torch.save(D, args.model_folder_dir + '/' + args.model +'_D.pth')
    torch.save(G, args.model_folder_dir + '/' + args.model +'_G.pth')