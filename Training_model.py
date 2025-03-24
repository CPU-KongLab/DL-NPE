# import torch
from torch import nn
from torch import cat
from torch.utils.data import DataLoader
from CreateDataloader import LoadData
import torch.nn.functional as F
from torch import no_grad
from torch import float as fl
from torch import cuda
from torch.optim import SGD
from torch import save
from torch.cuda.amp import autocast as autocast


class InceptionA(nn.Module):
    def __init__(self, in_channels):
        super(InceptionA, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1))

        self.branch5x5_1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch5x5_2 = nn.Conv2d(16, 24, kernel_size=(5, 5), padding=2)
        #self.branch5x5_3 = nn.Conv2d(24, 32, kernel_size=(5, 5), padding=2)

        self.branch3x3_1 = nn.Conv2d(in_channels, 16, kernel_size=(1, 1))
        self.branch3x3_2 = nn.Conv2d(16, 24, kernel_size=(3, 3), padding=1)
        self.branch3x3_3 = nn.Conv2d(24, 24, kernel_size=(3, 3), padding=1)

        self.brach_pool = nn.Conv2d(in_channels, 24, kernel_size=(1, 1))

    def forward(self, x):
        branchx1 = self.branch1x1(x)

        branch5x5 = self.branch5x5_1(x)
        branch5x5 = self.branch5x5_2(branch5x5)
        branch3x3 = self.branch3x3_1(x)
        branch3x3 = self.branch3x3_2(branch3x3)
        branch3x3 = self.branch3x3_3(branch3x3)

        batch_pool = F.avg_pool2d(x, kernel_size=3, padding=1, stride=1)
        batch_pool = self.brach_pool(batch_pool)

        outputs = [branchx1, branch3x3, branch5x5, batch_pool]
        return cat(outputs, dim=1)


class NeuralNetwork2(nn.Module):
    def __init__(self, classification):
        super(NeuralNetwork2, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5), padding=2)
        self.incep1 = InceptionA(in_channels=32)
        self.conv2 = nn.Conv2d(88, 32, kernel_size=(5, 5), padding=2)
        self.incep2 = InceptionA(in_channels=32)
        self.conv3 = nn.Conv2d(88, 32, kernel_size=(3, 3), padding=1)
        self.incep3 = InceptionA(in_channels=32)
        self.conv4 = nn.Conv2d(88, 16, kernel_size=(3, 3), padding=1)
        self.mp = nn.MaxPool2d(2)
        self.l1 = nn.Linear(2736, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 64)
        self.l4 = nn.Linear(64, classification)

    def forward(self, x):
        in_size = x.size(0)
        x = F.relu(self.mp(self.conv1(x)))
        x = self.incep1(x)
        x = F.relu(self.mp(self.conv2(x)))
        x = self.incep2(x)
        x = F.relu(self.mp(self.conv3(x)))
        x = self.incep3(x)
        x = F.relu(self.mp(self.conv4(x)))
        x = x.view(in_size, -1)  # 展平
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.channels = channels
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=(3, 3), padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))

        return F.relu(x + y)


class NeuralNetwork1(nn.Module):
    def __init__(self, classification):
        super(NeuralNetwork1, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=False)
        self.BN1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=False)
        self.BN2 = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 64, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=False)
        self.BN3 = nn.BatchNorm2d(64)
        self.conv4 = nn.Conv2d(64, 16, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=False)
        self.BN4 = nn.BatchNorm2d(16)
        self.conv5 = nn.Conv2d(16, 16, kernel_size=(5, 5), padding=2, stride=(1, 1), bias=False)  # 新加
        self.BN5 = nn.BatchNorm2d(16)
        self.mp = nn.MaxPool2d(2)
        self.reblock1 = ResidualBlock(64)
        self.reblock2 = ResidualBlock(128)
        self.reblock3 = ResidualBlock(64)
        self.reblock4 = ResidualBlock(16)
        self.reblock5 = ResidualBlock(4)
        self.l1 = nn.Linear(2736, 512)
        # self.l2 = nn.Linear(1024, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 64)
        # self.l5 = nn.Linear(64, 32)
        self.l4 = nn.Linear(64, classification)
        # self.drop = nn.Dropout(p=0.5)  # 新加的

    def forward(self, x):
        in_size = x.size(0)
        x = self.mp(F.relu(self.BN1(self.conv1(x))))
        x = self.reblock1(x)
        x = self.reblock1(x)
        x = self.mp(F.relu(self.BN2(self.conv2(x))))
        x = self.reblock2(x)
        x = self.reblock2(x)
        x = self.mp(F.relu(self.BN3(self.conv3(x))))
        x = self.reblock3(x)
        x = self.reblock3(x)
        x = self.mp(F.relu(self.BN4(self.conv4(x))))
        x = self.reblock4(x)
        x = self.reblock4(x)
        x = self.mp(F.relu(self.BN5(self.conv5(x))))
        x = self.reblock5(x)
        x = self.reblock5(x)
        x = F.avg_pool2d(x, kernel_size=3, stride=1, padding=1)
        x = x.view(in_size, -1)
        x = F.relu(self.l1(x))
        #  = self.drop(x)  # 新加的
        x = F.relu(self.l2(x))
        # x = self.drop(x)  # 新加的
        x = F.relu(self.l3(x))
        # x = F.relu(self.l4(x))
        # x = F.relu(self.l5(x))
        x = self.l4(x)

        return x


class NeuralNetwork3(nn.Module):
    def __init__(self, classification):
        super(NeuralNetwork3, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=(5, 5))
        # self.BN1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 16, kernel_size=(5, 5))
        # self.BN2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 8, kernel_size=(5, 5))
        # self.BN2 = nn.BatchNorm2d(8)
        self.mp = nn.MaxPool2d(2)
        self.l1 = nn.Linear(4080, 512)
        self.l2 = nn.Linear(512, 256)
        self.l3 = nn.Linear(256, 64)
        self.l4 = nn.Linear(64, classification)

        self.drop = nn.Dropout(p=0.5)  # 新加的

    def forward(self, x):
        in_size = x.size(0)
        # x = self.mp(F.relu(self.BN1(self.conv1(x))))
        x = self.mp(F.relu(self.conv1(x))) # x = self.mp(F.relu(self.BN2(self.conv2(x))))
        # x = self.mp(F.relu(self.BN2(self.conv2(x))))
        x = self.mp(F.relu(self.conv2(x)))
        x = self.mp(F.relu(self.conv3(x)))
        # x = self.mp(F.relu(self.BN3(self.conv3(x))))


        x = x.view(in_size, -1)
        x = F.relu(self.l1(x))
        x = self.drop(x)  # 新加的
        x = F.relu(self.l2(x))
        x = self.drop(x)  # 新加的
        x = F.relu(self.l3(x))
        x = self.l4(x)
        return x


'''定义训练函数，需要'''


def train(dataloader, model, loss_fn, optimizer, batch_size):
    scaler = cuda.amp.GradScaler()
    size = len(dataloader.dataset)
    avg_total = 0.0
    # 从数据加载器中读取batch（一次读取多少张，即批次数），X(图片数据)，y（图片真实标签）。
    for batch, (X, y) in enumerate(dataloader):
        # 将数据存到显卡
        X, y = X.cuda(), y.cuda()

        with autocast():
            # 得到预测的结果pred
            pred = model(X)
            # 计算预测的误差
            # print(pred,y)
            loss = loss_fn(pred, y)
        avg_total = avg_total + loss.item()

        # 反向传播，更新模型参数
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        # optimizer.step()

        # 每训练100次，输出一次当前信息
        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            # print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")
            a = f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]"
            print(a)
    avg_loss = f"{(avg_total / int(len(dataloader.dataset) / batch_size)):>5f}"  # 修改
    print(avg_loss)
    return avg_loss


def test(dataloader, model, loss_fn):
    size = len(dataloader.dataset)
    print("size = ", size)
    # 将模型转为验证模式
    model.eval()
    # 初始化test_loss 和 correct， 用来统计每次的误差
    test_loss, correct = 0, 0
    # 测试时模型参数不用更新，所以no_gard()
    # 非训练， 推理期用到
    with no_grad():
        # 加载数据加载器，得到里面的X（图片数据）和y(真实标签）
        for X, y in dataloader:
            # 将数据转到GPU
            X, y = X.cuda(), y.cuda()
            # 将图片传入到模型当中就，得到预测的值pred
            pred = model(X)
            # 计算预测值pred和真实值y的差距
            test_loss += loss_fn(pred, y).item()
            # 统计预测正确的个数
            correct += (pred.argmax(1) == y).type(fl).sum().item()
    test_loss /= size
    correct /= size
    accuracy = f"{(100 * correct):>0.1f}"
    avg_loss = f"{test_loss:>8f}"
    print("correct = ", correct)
    b = "correct = ", correct
    print(f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n")
    c = f"Test Error: \n Accuracy: {(100 * correct):>0.1f}%, Avg loss: {test_loss:>8f} \n"
    # 增加数据写入功能
    return accuracy, avg_loss, b, c


def write_result(fileloc, epoch, trainloss, testloss, testaccuracy):
    with open(fileloc, "a") as f:
        data = "Epoch: " + str(epoch) + "\tTrainLoss " + str(trainloss) + "\tTestLoss " + str(
            testloss) + "\tTestAccuracy " + str(testaccuracy) + "\n"
        f.write(data)


if __name__ == '__main__':
# def Training_model(batch_size, ):
    batch_size = 32

    # # 给训练集和测试集分别创建一个数据集加载器
    train_data = LoadData(
        "C:/Users/Li/Desktop/CNNtest/program_test/dataset_2_corp/train.txt", True)
    valid_data = LoadData(
        "C:/Users/Li/Desktop/CNNtest/program_test/dataset_2_corp/test.txt", False)

    train_dataloader = DataLoader(dataset=train_data, num_workers=8, pin_memory=True, batch_size=batch_size,
                                  shuffle=True)
    test_dataloader = DataLoader(dataset=valid_data, num_workers=8, pin_memory=True, batch_size=batch_size)

    for X, y in test_dataloader:
        print("Shape of X [N, C, H, W]: ", X.shape)
        print("Shape of y: ", y.shape, y.dtype)
        break

    # 如果显卡可用，则用显卡进行训练
    device = "cuda" if cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    # 调用刚定义的模型，将模型转到GPU（如果可用）
    classification = 4
    model = NeuralNetwork1(classification).to(device)

    print(model)

    # 定义损失函数，计算相差多少，交叉熵，
    loss_fn = nn.CrossEntropyLoss()

    # 定义优化器，用来训练时候优化模型参数，随机梯度下降法
    # lR = 1 * 1e-4
    optimizer = SGD(model.parameters(), lr=1 * 1e-4, momentum=0.5, weight_decay=1 * 1e-5)  # 初始学习率
    for epoch in range(50):
        if epoch % 5 == 0:
            for parameter in optimizer.param_groups:
                parameter['lr'] *= 0.9

    # 一共训练5次
    epochs = 40
    best = 0.0
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        train_loss, a = train(train_dataloader, model, loss_fn, optimizer, batch_size)
        accuracy, avg_loss, b, c = test(test_dataloader, model, loss_fn)
        write_result(
            "C:/Users/Li/Desktop/CNNtest/program_test/dataset_2_corp/traindata.txt",
            t + 1, train_loss, avg_loss, accuracy)
        save(model.state_dict(),
                   'C:/Users/Li/Desktop/CNNtest/program_test/dataset_2_corp/model'"/resnet_epoch_" + str(
                       t + 1) + "_acc_" + str(accuracy) + ".pth")
