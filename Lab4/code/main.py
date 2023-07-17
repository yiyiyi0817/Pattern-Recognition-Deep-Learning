import argparse
import random
import time
import os
from torch import nn, optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tensorboardX import SummaryWriter
from sklearn.metrics import classification_report
from dataset import *
from model import *

parser = argparse.ArgumentParser('Lab4-1')
parser.add_argument('--data_dir', type=str, default='online_shopping_10_cats.csv')
parser.add_argument('--seed', type=int, default=20) # 数据集划分随机种子
parser.add_argument('--nn', type=str, default='rnn') # 通过序号选择用哪一个神经网络，用来对比效果
parser.add_argument('--input-size', type=int, default=128) 
parser.add_argument('--hidden-size', type=int, default=64)
parser.add_argument('--device', default="cuda:0")
parser.add_argument('--niters', type=int, default=10) # 训练的epoch数
parser.add_argument('--model_folder_dir', type=str, default='model') # 最佳模型存储路径
parser.add_argument('--tensorboard_dir', type=str, default='')

args = parser.parse_args(args=['--niters', '1', '--nn', 'Bi-LSTM', '--tensorboard_dir', ''])

shopping_names = ['书籍', '平板', '手机', '水果', '洗发水', '热水器', '蒙牛', '衣服', '计算机', '酒店']  # 全部类别

if __name__ == '__main__':
    device = args.device
    seed = args.seed
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # 数据读入
    print('Loading shopping dataset...')
    train_loader, val_loader, test_loader = load_shopping_data(args)
    output_size = 10
    print("Data loaded!")
    
    #选择模型
    model_name = args.nn
    if args.nn == 'Bi-LSTM':
        model = LSTM(args, output_size, bidirectional=True).to(device)
    else:
        model_name = globals()[model_name]
        model = model_name(args, output_size).to(device)
    print('Model loaded!')
    
    # 初始化方法
    writer = SummaryWriter(args.tensorboard_dir)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min')
    
    # 训练
    for epoch in range(1, args.niters + 1):
        train_loss = 0.0 #整个epoch所有batch的loss
        for idx, data in enumerate(train_loader):
            label, sentence = data['cat'].to(device), data['review'].to(device)
            # print(sentence.shape)
            if len(sentence.shape) < 3:
                sentence = sentence[None]  # expand for batchsz
            # print(sentence.shape)  # 1, words, 128
            output = model(sentence)
            optimizer.zero_grad()

            loss = criterion(output, label)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        train_loss = train_loss / len(train_loader.dataset)  
        scheduler.step(loss)
        print('epoch %d\ttraining loss: %.3f\t' % (epoch, train_loss), end='')
        writer.add_scalar('train_loss', train_loss, epoch)
        
        # 验证
        with torch.no_grad():
            best_accuracy = 0.0
            accuracy = 0.0
            vid_loss = 0
            correct = 0
            for idx, data in enumerate(val_loader):
                label, sentence = data['cat'].to(device), data['review'].to(device)
                # print(sentence.shape)
                if len(sentence.shape) < 3:
                    sentence = sentence[None]  # expand for batchsz
                # print(sentence.shape)  # 1, words, 128
                output = model(sentence)
                correct += (label == torch.argmax(output, dim=1)).sum().item()
                loss = criterion(output, label)
                vid_loss += loss.item()
        
            vid_loss = vid_loss / len(val_loader.dataset)  
            writer.add_scalar('vid_loss', vid_loss, epoch)
            accuracy = correct / len(val_loader.dataset)  
            print('vid loss: %.3f\taccuracy:%.3f\n' % (vid_loss, accuracy), end='')
            if accuracy > best_accuracy:
                best_accur = accuracy
                torch.save(model, args.model_folder_dir + '/' + args.nn +'.pth')
            writer.close()
            writer.add_scalar('accuracy', accuracy, epoch)
        
    # 测试
    with torch.no_grad():
        accuracy = 0.0
        test_loss = 0
        correct = 0
        true = []
        preds = []
        model = torch.load(args.model_folder_dir + '/' + args.nn +'.pth')
        for idx, data in enumerate(test_loader):
            label, sentence = data['cat'].to(device), data['review'].to(device)
            # print(sentence.shape)
            if len(sentence.shape) < 3:
                sentence = sentence[None]  # expand for batchsz
            # print(sentence.shape)  # 1, words, 128
            output = model(sentence)
            correct += (label == torch.argmax(output, dim=1)).sum().item()
            pred = torch.argmax(output, dim=1)
            pred, label = pred.to('cpu').item(), label.to('cpu').item()
            preds.append(pred)
            true.append(label)

        accuracy = correct / len(test_loader.dataset)  
        result = classification_report(true, preds, target_names=shopping_names)
        print (result)

        