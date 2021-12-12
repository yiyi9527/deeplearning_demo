import torch.optim.optimizer
import torchvision.datasets
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# from AlexNet_Model import *

# 准备数据集
# train_data = torchvision.datasets.FashionMNIST("../data",train=True,transform=torchvision.transforms.ToTensor(),download=True)

train_data = torchvision.datasets.FashionMNIST("../data",train=True,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)), torchvision.transforms.ToTensor()]),download=True)
print(train_data[0][0].shape)

# print(train_data[0][0])
# train_data[0][0].show()

test_data = torchvision.datasets.FashionMNIST("../data",train=False,transform=torchvision.transforms.Compose([torchvision.transforms.Resize((224,224)),torchvision.transforms.ToTensor()]),download=True)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("训练集的长度为：{}".format(train_data_size))
print("测试集的长度为：{}".format(test_data_size))

# 加载数据集
train_dataloader = DataLoader(train_data,batch_size=64)
test_dataloader = DataLoader(test_data,batch_size=64)

print(len(train_dataloader))

# 搭建神经网络
class AlexNet_Model(nn.Module):
    def __init__(self):
        super(AlexNet_Model, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,96,kernel_size=11,stride=4,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(96,256,kernel_size=5,padding=2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Conv2d(256,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,384,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(384,256,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2),
            nn.Flatten(),
            nn.Linear(6400,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,4096),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(4096,10)
        )

    def forward(self,x):
        x = self.model(x)
        return x

# 创建网络模型
alexnet = AlexNet_Model()

if torch.cuda.is_available():
    alexnet = alexnet.cuda()

# 损失函数
loss_fn = nn.CrossEntropyLoss()
loss_fn = loss_fn.cuda()

# 优化器
learning_rate = 1e-2
optimizer = torch.optim.SGD(alexnet.parameters(),lr=learning_rate)

# 设置训练网络的一些参数
# 记录训练的轮数
total_train_step = 0
# 记录测试的轮数
total_test_step = 0
# 训练的轮数
epoch = 10

# 添加tensorboard
writer = SummaryWriter("../logs_alexnet_train")

for i in range(epoch):
    print("----------第{}轮训练开始----------".format(i + 1))

    # 训练步骤开始
    alexnet.train()
    for data in train_dataloader:
        imgs,targets = data
        imgs = imgs.cuda()
        targets = targets.cuda()
        output = alexnet(imgs)
        loss = loss_fn(output,targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数：{}，Loss：{}".format(total_train_step,loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)

    # 测试步骤开始
    alexnet.eval()
    total_test_loss = 0
    total_accuracy = 0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets = data
            imgs = imgs.cuda()
            targets = targets.cuda()
            output = alexnet(imgs)
            loss = loss_fn(output,targets)
            total_test_loss = total_test_loss + loss.item()
            accuracy = (output.argmax(1) == targets).sum()
            total_accuracy = total_accuracy + accuracy

    print("整体测试集上的Loss：{}".format( total_test_loss))
    print("整体测试集上的正确率：{}".format(total_accuracy / test_data_size))
    writer.add_scalar("test_loss",total_test_loss,total_test_step)
    writer.add_scalar("test_accuracy",total_accuracy / test_data_size,total_test_step)
    total_test_step = total_test_step + 1

    torch.save(alexnet,"alexnet_{}.pth".format(i + 1))
    print("模型已保存")

writer.close()