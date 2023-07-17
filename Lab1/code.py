import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, TensorDataset
from torch.optim.lr_scheduler import ReduceLROnPlateau
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt


parser = argparse.ArgumentParser('MNIST_MLP')
parser.add_argument('--batch_size', type=int, default=20)
parser.add_argument('--PCA_dim', type=int, default=784) # 通过序号选择用哪一个多层感知，用来对比效果
parser.add_argument('--nn', type=str, default='0') # 通过序号选择用哪一个多层感知，用来对比效果
parser.add_argument('--gpu', type=int, default=0) # -1时为CPU
parser.add_argument('--niters', type=int, default=20) # 训练的epoch数
args = parser.parse_args()

device = torch.device('cpu' if args.gpu == -1 else 'cuda:' + str(args.gpu))


class MLP_0(nn.Module):
    # 2隐藏层+1输出层：PCA_dim --> 500 --> 500 --> 10
    def __init__(self, input_dim = 28 * 28, hidden_num = 500):
        super(MLP_0, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_num),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_num, hidden_num),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_num, 10)
            )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    def forward(self, x):
        out = self.net(x)
        return F.softmax(out, dim=1)


class MLP_1(nn.Module):
    # 7隐藏层+1输出层：PCA_dim --> 500 --> 250 --> 125 --> 65 --> 30 --> 15 --> 15 --> 10
    def __init__(self, input_dim=28 * 28, hidden_num = 300):
        super(MLP_1, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 250),
            nn.ReLU(inplace=True),
            nn.Linear(250, 125),
            nn.ReLU(inplace=True),
            nn.Linear(125, 65),
            nn.ReLU(inplace=True),
            nn.Linear(65, 30),
            nn.ReLU(inplace=True),
            nn.Linear(30, 15),
            nn.ReLU(inplace=True),
            nn.Linear(15, 10),
            nn.ReLU(inplace=True),
            nn.Linear(10, 10),
            )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    def forward(self, x):
        out = self.net(x)
        return F.softmax(out, dim=1)


class MLP_2(nn.Module):
    # 4隐藏层+1输出层：PCA_dim --> 500 --> 300 --> 100 --> 100 --> 10
    def __init__(self, input_dim = 28 * 28):
        super(MLP_2, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 300),
            nn.ReLU(inplace=True),
            nn.Linear(300, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 100),
            nn.ReLU(inplace=True),
            nn.Linear(100, 10)
            )
        for m in self.net.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=0.1)
                nn.init.constant_(m.bias, val=0)
    def forward(self, x):
        out = self.net(x)
        return F.softmax(out, dim=1)


if __name__ == '__main__':
    # 定义数据转换器
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载 MNIST 训练集和测试集
    train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
    test_data = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

    # 获取训练集和测试集的图像数据和标签数据
    train_images = []
    train_labels = []
    for image, label in train_data:
        train_images.append(image.view(-1))
        train_labels.append(label)
    train_images = torch.stack(train_images).to(device)
    train_labels = torch.tensor(train_labels).to(device)

    test_images = []
    test_labels = []
    for image, label in test_data:
        test_images.append(image.view(-1))
        test_labels.append(label)
    test_images = torch.stack(test_images).to(device)
    test_labels = torch.tensor(test_labels).to(device)
    print ('Data Loaded.')

    # 使用 PCA 对图像数据进行降维
    pca = PCA(n_components=args.PCA_dim)
    train_images_pca = pca.fit_transform(train_images.cpu())
    test_images_pca = pca.transform(test_images.cpu())
    print ('PCA Finished.')

    # 定义训练集和测试集的 DataLoader
    batch_size = args.batch_size
    train_dataset = TensorDataset(torch.tensor(train_images_pca).to(device), train_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    test_dataset = TensorDataset(torch.tensor(test_images_pca).to(device), test_labels)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建(选择)多层感知机模型
    model_name = 'MLP_' + args.nn
    model_name = globals()[model_name]
    model = model_name(args.PCA_dim).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())
    scheduler = ReduceLROnPlateau(optimizer, 'min')

    # 训练
    loss_lst = []
    for epoch in range(args.niters):
        train_loss = 0.0
        for image, true_y in train_loader:  
            optimizer.zero_grad() 
            true_y = true_y.to(device)
            pred_y = model(image.float()).to(device)    # 得到预测值
            # print ('true_y', true_y, 'pred_y:',pred_y)
            # exit()
            loss = criterion(pred_y, true_y)    # 计算两者的平均损失
            # 反向传播和优化
            loss.backward()
            optimizer.step() 
            train_loss += loss.item() * image.size(0)   # image.size(0) = batch_size
        train_loss = train_loss / len(train_loader.dataset)     # len(train_loader.dataset)=60000
        loss_lst.append(train_loss)
        scheduler.step(loss)
        print (epoch)
    
    plt.plot(loss_lst)
    plt.ylabel('Training Loss')
    plt.xlabel('Epoch')

    # 测试
    with torch.no_grad():
        test_loss = 0.0
        correct = 0
        total = 0
        for image, test_true_y in test_loader:
            pred_y = model(image.float()).to(device) 
            test_true_y = test_true_y.to(device)
            loss = criterion(pred_y, test_true_y).to(device) 
            test_loss += loss.item() * image.size(0) 
            total += image.size(0)
            # print ('\n\n', pred_y, torch.argmax(pred_y, dim=1))
            correct += (test_true_y == torch.argmax(pred_y, dim=1)).sum().item()
        test_loss = test_loss / len(test_loader.dataset)

        plt.title('Testing Loss: {:.4f}  Accuracy: {:.3f}  batch_size: {}'.format(test_loss, correct / total, '5000'))
        plt.show()
        
        torch.save(model, './model/MLP_1.pth')