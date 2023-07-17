import pandas as pd
import jieba
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from gensim.models.word2vec import Word2Vec

shopping_dic = {'书籍': 0, '平板': 1, '手机': 2, '水果': 3, '洗发水': 4, '热水器': 5, '蒙牛': 6, '衣服': 7, '计算机': 8, '酒店': 9} 

class Shopping(Dataset):
    def __init__(self, my_list):
        self.cats = []    # 类别
        self.reviews = []
        for i in range(len(my_list)):
            self.cats.append(my_list[i]['cat'])
            self.reviews.append(my_list[i]['review'])

    def __getitem__(self, idx):
        return {'cat': self.cats[idx], 'review': torch.tensor(np.array(self.reviews[idx]))}

    def __len__(self):
        return len(self.cats)

def load_shopping_data(args):
    df = pd.read_csv(args.data_dir)
    val_list = []
    test_list = []
    train_list = []
    reviews = []
    cats = []
    for index, row in df.iterrows():
        if not isinstance(row['review'], str): # 检查是否是字符串类型,不是就跳过这次循环
            continue
        cats.append(shopping_dic[row['cat']])
        reviews.append(row['review'])
    tokens = [jieba.lcut(i) for i in reviews]  # 分词，返回词的列表

    model = Word2Vec(tokens, min_count=1, hs=1, window=3, vector_size=args.input_size) # 返回Word2Vec对象
    reviews_vector = [[model.wv[word] for word in sentence] for sentence in tokens]  # 返回词向量列表

    for i in range(62773):
        if i % 5 == 4:
            val_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        elif i % 5 == 0:
            test_list.append({'cat': cats[i], 'review': reviews_vector[i]})
        else:
            train_list.append({'cat': cats[i], 'review': reviews_vector[i]})

    # 使用动态长度，每句句子长度不同，batch-size只能为1
    train_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)
    val_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)
    test_loader = DataLoader(Shopping(train_list), shuffle=True, batch_size=1)

    return train_loader, val_loader, test_loader


