import os.path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from matplotlib import pyplot as plt
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import set_seed

from utils.dataset import get_shape, NpzSignalDataset, generate_mixed_signal_data, GanDataset
from utils.early_stop import EarlyStopping


# 生成器网络：输入振动信号，输出去噪后的信号
class Generator(nn.Module):
    def __init__(self, lstm_hidden_size=64, num_layers=16):
        super(Generator, self).__init__()

        num_outputs = get_shape()[2]  # 输入和输出的长度相同
        self.LSTM = nn.LSTM(num_outputs, lstm_hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        # 因为是双向 LSTM, 所以要乘2
        self.classifier = nn.ModuleList()
        input_channel = 1
        output_channel = 2
        mlp_input_size = lstm_hidden_size * 2 * output_channel
        for _ in range(num_layers if num_layers <= 8 else 8):
            self.classifier.append(
                nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, 3, 1, 1),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_channel),
                )
            )  # 每次卷积后, 通道数翻倍
            if output_channel == 256:
                mlp_input_size = mlp_input_size * 2
                input_channel = output_channel
                break
            input_channel = output_channel
            output_channel = output_channel * 2
            mlp_input_size = mlp_input_size * 2
        for _ in range(num_layers if num_layers <= 8 else 8):
            self.classifier.append(
                nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, 3, 1, 1),
                    nn.ReLU(),
                    nn.BatchNorm1d(output_channel),
                )
            )  # 每次卷积后，长度不变, 通道数减半
            if output_channel == 1:
                mlp_input_size = mlp_input_size // 2
                break
            input_channel = output_channel
            output_channel = output_channel // 2
            mlp_input_size = mlp_input_size // 2
        self.ffn = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size//2),
            nn.ReLU(),
            nn.BatchNorm1d(mlp_input_size//2),
            nn.Linear(mlp_input_size//2, num_outputs)
        )

    def forward(self, x):
        lstm_hidden_states, _ = self.LSTM(x)  # shape: (batch_size, seq_len, lstm_hidden_size*2)
        for layer in self.classifier:
            lstm_hidden_states = layer(lstm_hidden_states)
        lstm_hidden_states = lstm_hidden_states.reshape(lstm_hidden_states.size(0), -1)
        # 归一化
        # lstm_hidden_states = (lstm_hidden_states - lstm_hidden_states.mean(dim=1, keepdim=True)) / (
        #         lstm_hidden_states.std(dim=1, keepdim=True) + 1e-8)
        return self.ffn(lstm_hidden_states)


# 修改判别器，让它专门判断信号是否带有噪声
class Discriminator(nn.Module):
    def __init__(self, input_size=get_shape()[2]):
        super(Discriminator, self).__init__()
        self.noise_classifier = nn.Sequential(
            nn.Linear(input_size, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.reshape(x.size(0), -1)
        # if x.size(1) != get_shape()[2]:
        #     raise ValueError('输入的信号长度不符合要求')
        return self.noise_classifier(x)


# 训练过程
def train_cgan(dataset, num_epochs=50, batch_size=1024, discriminator=None, generator=None, g_criterion=None,
               d_criterion=None,log_dir='../run/0000000', ckpt_path='generator.pth',
               optimizer_D=None, optimizer_G=None, patience=10):
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=str(num_epochs), flush_secs=5)  # 用于记录训练过程
    early_stopping = EarlyStopping(log_dir, patience=patience, verbose=True)  # 早停


    train_dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, drop_last=True)


    # 记录模型结构
    writer.add_graph(generator, torch.randn(batch_size, 1, 128).cuda())
    writer.add_graph(discriminator, torch.randn(batch_size, 128).cuda())

    for epoch in range(num_epochs):
        loss = 0
        min_loss = float('inf')
        for i, (inputs, labels) in tqdm(enumerate(train_dataloader), total=len(train_dataloader)):

            inputs, labels = inputs.cuda().float(), labels.cuda().float()
            # === 训练判别器 ===
            optimizer_D.zero_grad()
            real_labels = torch.ones(inputs.size(0), 1).cuda()  # 真实数据标签，表示是噪声信号
            fake_labels = torch.zeros(inputs.size(0), 1).cuda()  # 生成数据标签，表示不是噪声信号

            # 判别器在真实数据上的损失
            real_outputs = discriminator(inputs).reshape(-1, 1)
            d_loss_real = d_criterion(real_outputs, real_labels)

            # 生成去除噪声的信号
            generated_signals = generator(inputs)

            # 判别器在生成数据上的损失
            fake_outputs = discriminator(generated_signals.detach())
            d_loss_fake = d_criterion(fake_outputs, fake_labels)  # 判别器希望能够区分生成的信号和真实信号

            # 总判别器损失
            d_loss = d_loss_real + d_loss_fake
            d_loss.backward()  # 这里无需 retain_graph，因为不需要对生成器计算图反向传播
            optimizer_D.step()

            # === 训练生成器 ===
            optimizer_G.zero_grad()
            # 生成信号尽可能欺骗判别器
            generated_signals = generator(inputs)
            outputs = discriminator(generated_signals)

            # 生成器的损失包含两部分：
            # 1. 欺骗判别器，使其无法是否去噪
            noise_confusion_loss = -torch.log(torch.clamp(2.0 * outputs * (1 - outputs), min=1e-8))  # 最大化判别器的不确定性
            # 2. 保持信号的其他特征不变
            labels = labels.reshape(inputs.size(0), -1)
            reconstruction_loss = g_criterion(generated_signals, labels)

            g_loss = noise_confusion_loss.mean() + 0.1 * reconstruction_loss
            g_loss.backward()
            optimizer_G.step()

            if i % 10 == 0:  # 每10个batch记录一次,需要看到生成器和判别器的损失值
                writer.add_scalars('Loss', {'g_loss': g_loss.item(), 'd_loss': d_loss.item()},
                                   epoch * len(train_dataloader) + i)
                # GAN的训练过程中，生成器（Generator）和判别器（Discriminator）的损失值是监控训练的一个常见方式：
                #
                # 生成器损失：生成器的目标是欺骗判别器，生成器损失越低，表明生成的样本越能欺骗判别器。
                # 判别器损失：判别器的目标是区分真实样本和生成样本，判别器损失越低，说明它能更好地区分两者。
                # 判断标准：
                #
                # 如果生成器和判别器的损失值都很高，说明两者训练不足，生成的样本质量差。
                # 如果生成器损失较低，判别器损失较高，说明生成器开始成功欺骗判别器，这可能是训练趋向平衡的标志。
                # 过拟合：如果判别器的损失一直很低，生成器的损失很高，说明判别器过于强大，生成器难以学习。
            loss += (g_loss.item() + d_loss.item()) / 2
        loss /= len(train_dataloader)

        if min_loss > loss:
            min_loss = loss
            torch.save(generator.state_dict(), os.path.join(log_dir, ckpt_path))
            tqdm.write("Save model at epoch {}".format(epoch))
            tqdm.write("Current loss: {}".format(min_loss))
        writer.add_scalar('Loss/epoch', loss, epoch)
        early_stopping(loss, generator)
        if early_stopping.early_stop:
            tqdm.write("Early stopping")

            torch.save(generator.state_dict(), os.path.join(log_dir, 'last.pth'))
            break
    writer.close()

def gan_predict(generator, dataset, model_path="generator.pth", device='cuda' if torch.cuda.is_available() else 'cpu'):
    generator.load_state_dict(torch.load(model_path, weights_only=True))
    generator.to(device)
    generator.eval()
    test_dataloader = DataLoader(dataset, batch_size=256, shuffle=False, pin_memory=True)
    res = []
    with torch.no_grad():
        for i, (data, target) in tqdm(enumerate(test_dataloader), total=len(test_dataloader)):
            data = data.float()
            data, target = data.to(device), target.to(device)
            output = generator(data)
            res.extend(output.cpu())
    # 调整为3个维度，中间的维度为1
    return np.array(res).reshape(-1, 1, get_shape()[2])


if __name__ == '__main__':

    set_seed(42)  # 设置种子
    batch_size = 2048  # 批大小
    epoch = 300  # 训练轮数
    # dataSet = NpzSignalDataset('../data/AI1/', signal_size=get_shape()[-1])
    # pure_signals, noise_signals = generate_mixed_signal_data(dataSet.data)  # 生成噪声信号作为输入，纯净信号作为label
    # dataSet = GanDataset(noise_signals, pure_signals)
    shape = get_shape()[1:]
    # 定义损失函数和优化器
    config = {
        'lstm_hidden_size': 128,
        'num_layers': 16,
    }
    generator = Generator(**config).cuda()
    discriminator = Discriminator(input_size=shape[0] * shape[1]).cuda()
    g_criterion = nn.MSELoss()  # 生成器损失
    d_criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer_G = torch.optim.AdamW(generator.parameters(), lr=1e-2, weight_decay=1e-4)  # 优化器
    optimizer_D = torch.optim.AdamW(discriminator.parameters(), lr=1e-2, weight_decay=1e-4)  # 优化器

    torch.autograd.set_detect_anomaly(True)

    # # 假设 dataloaders 已经准备好
    # train_cgan(dataSet, num_epochs=epoch, batch_size=batch_size, discriminator=discriminator, generator=generator,
    #            g_criterion=g_criterion, d_criterion=d_criterion, optimizer_D=optimizer_D, optimizer_G=optimizer_G,
    #            patience=epoch//10, log_dir='../run/1121/gan14', ckpt_path='generator.pth')
    # 测试
    dataSet = NpzSignalDataset('../data/wuxi_a4/', signal_size=get_shape()[-1])
    # 每个label选择一个样本， 保存index、signal
    indexes = []
    for i in range(8):
        indexes.append(np.random.choice(np.where(dataSet.target == i)[0]))
    noisy_signals = dataSet.data[indexes]

    path = '../run/1121/gan14/'
    model_name = 'last.pth'

    dataSet.data = gan_predict(generator, dataSet, model_path=os.path.join(path, model_name))

    # 绘制8个处理后的信号
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))
    for i in range(8):
        ax = axes[i % 2, i // 2]
        # ax.plot((noisy_data[i][0]) / max(noisy_data[i][0]), label='noisy signal Normalized')  # 对其进行缩放
        ax.plot(noisy_signals[i][0], label='noisy signal') # 左边y轴
        # 右边y轴
        ax2 = ax.twinx()
        ax2.plot(dataSet.data[indexes[i]][0], label='denoised signal', color='orange')
        ax.set_title(f' signal {i + 1}, label {dataSet.target[indexes[i]]}')
    # 保存图像
    fig.tight_layout()
    fig.show()
    fig.savefig(os.path.join(path, 'denoised_signals.png'))


