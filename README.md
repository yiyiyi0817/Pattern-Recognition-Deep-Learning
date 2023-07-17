# Pattern-Recognition-Deep-Learning  
哈工大2023春模式识别与深度学习实验作业  
HIT Spring 2023 Semester Pattern Recognition & Deep Learning lab assignments  
  
  
实验1：深度学习框架熟悉  
使用 PyTorch 实现 MLP，并在 MNIST 数据集上验证。  
  
Lab1: Familiarity with the deep learning framework  
The MLP is implemented using PyTorch and verified on MNIST dataset.  
  

  
实验2：卷积神经网络实现  
基于 PyTorch 实现 AlexNet 结构,在Caltech101 数据集上进行验证,并使用tensorboard 进行训练数据可视化。  
  
Lab2: Convolutional neural network(CNN) implementation  
AlexNet structure is implemented based on PyTorch, verified on Caltech101 dataset, and tensorboard is used for training data visualization.    
  

  
实验4：循环神经网络  
1. 利用 Pytorch 自己实现 RNN、GRU、LSTM 和 Bi-LSTM。不可直接调用nn.RNN(), nn.GRU(), nn.LSTM()。  
2. 利用上述四种结构进行文本多分类（60%）计算测试结果的准确率、召回率和 F1 值；对比分析四种结构的实验结果。  
3. 任选上述一种结构进行温度预测（40%）使用五天的温度值预测出未来两天的温度值；给出与真实值的平均误差和中位误差。  
  
Lab4: Recurrent neural network(RNN)  
1. Implement RNN, GRU, LSTM, and Bi-LSTM by yourself using Pytorch. You cannot call nn.RNN(), nn.GRU(), nn.LSTM() directly.  
2. Use the above four structures for text multi-classification (60%) to calculate the accuracy, recall and F1 value of the test results; The experimental results of the four structures were compared and analyzed.  
3. Choose one of the above structures for temperature prediction (40%). Use the temperature value of five days to predict the temperature value of the next two days; The mean and median errors from the true values are given.  

  
  
实验5：生成式对抗网络  
1. 基于 Pytorch 实现 GAN、WGAN、WGAN-GP。拟合给定分布（分布由 points.mat给出）。要求可视化训练过程。  
2. 基于给定的 ProGAN 代码和模型，实现隐空间语义方向搜索的代码的 SeFa 部分，完善 genforce/sefa.py 中 TODO 部分。  
  
Lab5:Generative Adversarial Networks(GAN)  
1. Implement GAN, WGAN, WGAN-GP based on Pytorch. Fits the given distribution (the distribution is given by points.mat) Ask to visualize the training process.  
2. Based on the given ProGAN code and model, implement the sefa part of the code for semantic direction search in latent space, and improve the TODO part of genforce/ SEFA.py.  

  
  
算法说明、应用场景设定、实验结果等详见每部分的实验报告。小组合作完成的代码和报告并未上传。  
The algorithm description, application scenario setting, and experimental results are detailed in the experimental report of each section. The code and report that the team worked on were not uploaded.  
