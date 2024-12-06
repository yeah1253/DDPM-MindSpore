import os.path

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sympy import pprint
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader, TensorDataset
from transformers import set_seed

from model.ddpm import build_network
from model.model_configs import configs
from utils.dataset import get_shape, NpzSignalDataset
from utils.early_stop import EarlyStopping


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.shrinkage = Shrinkage(out_channels, gap_size=(1))
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(out_channels * BasicBlock.expansion),
            self.shrinkage
        )
        # shortcut
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):

         return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
        # a = self.residual_function(x),
        # b = self.shortcut(x),
        # c = a+b
        # return c


class Shrinkage(nn.Module):
    def __init__(self, channel, gap_size):
        super(Shrinkage, self).__init__()
        self.gap = nn.AdaptiveAvgPool1d(gap_size)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel),
            nn.BatchNorm1d(channel),
            nn.ReLU(inplace=True),
            nn.Linear(channel, channel),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x_raw = x
        x = torch.abs(x)
        x_abs = x
        x = self.gap(x)
        x = torch.flatten(x, 1)
        average = torch.mean(x, dim=1, keepdim=True)  #CS
        # average = x    #CW
        x = self.fc(x)
        x = torch.mul(average, x)
        x = x.unsqueeze(2)
        # soft thresholding
        sub = x_abs - x
        zeros = sub - sub
        n_sub = torch.max(sub, zeros)
        x = torch.mul(torch.sign(x_raw), n_sub)
        return x


class RSNet(nn.Module):

    def __init__(self, block, num_block, classifier):
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(64),
            nn.ReLU(inplace=True))
        # we use a different inputsize than the original paper
        # so conv2_x's stride is 1
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool1d((1))
        self.fc = nn.Linear(512 * block.expansion, get_shape()[-1])

        self.classifier = classifier

    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make rsnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual shrinkage block

        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer

        Return:
            return a rsnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        output = output.view(output.size(0), -1)
        output = self.fc(output)
        output = output.unsqueeze(1)
        return self.classifier(output)


def rsnet18(classifier=None):
    """ return a RSNet 18 object
    """
    return RSNet(BasicBlock, [2, 2, 2, 2], classifier=classifier)


def rsnet34(classifier=None):
    """ return a RSNet 34 object
    """
    return RSNet(BasicBlock, [3, 4, 6, 3], classifier=classifier)

def train_model(writer, config_id, train_loader, val_loader, log_dir, model_name, num_epochs=10, device='cpu'):
    classifier = build_network(configs[config_id])
    model = rsnet18(classifier)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(model.parameters(), lr=1e-3, weight_decay=1e-5)

    ealy_stop = EarlyStopping(log_dir, patience=30, verbose=True)
    for epoch in range(num_epochs):
        best_loss = float('inf')
        train_loss, train_acc = 0.0, 0.0
        val_loss, val_acc = 0.0, 0.0

        for i, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(device).float(), labels.to(device).long()  # 强制转换类型

            # 前向传播
            _, outputs = model(inputs)
            loss = criterion(outputs, labels)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _, predicted = torch.max(outputs.data, 1)
            train_loss += loss.item()
            train_acc += (predicted == labels).sum().item()

        # validate the model
        model.eval()
        with torch.no_grad():
            for i, (inputs, labels) in enumerate(val_loader):
                inputs, labels = inputs.to(device).float(), labels.to(device).long()
                _, outputs = model(inputs)

                loss = criterion(outputs, labels)

                val_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                val_acc += (predicted == labels).sum().item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader.dataset)
        val_loss /= len(val_loader)
        val_acc /= len(val_loader.dataset)
        writer.add_scalars('loss', {'train': train_loss, 'val': val_loss}, epoch)
        writer.add_scalars('acc', {'train': train_acc, 'val': val_acc}, epoch)

        ealy_stop(val_loss, model)
        if ealy_stop.early_stop:
            print("Early stopping")
            break

        if val_loss < best_loss:
            torch.save(model.state_dict(), os.path.join(log_dir, model_name))
            best_loss = val_loss
            print(f"Save model at epoch {epoch}, best_loss:{best_loss}")

def test_model(writer, config_id, test_loader, log_dir, model_name, device='cuda'):
    # 读取模型
    classifier = build_network(configs[config_id])
    model = rsnet18(classifier)
    model.load_state_dict(torch.load(os.path.join(log_dir, model_name)))
    model = model.to(device)
    model.eval()
    test_acc = 0.0
    features = []
    labels = []
    res = []
    with torch.no_grad(): # 不进行反向传播
        for i, (inputs, label) in enumerate(test_loader):
            inputs, label = inputs.to(device).float(), label.to(device).long()
            feature, outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            test_acc += (predicted == label).sum().item()
            features.append(feature.detach().cpu().numpy())
            labels.append(label.detach().cpu().numpy())
            res.append(predicted.detach().cpu().numpy())

    res = np.concatenate(res, axis=0)
    features = np.concatenate(features, axis=0)
    labels = np.concatenate(labels, axis=0)

    # 保存特征
    np.save(os.path.join(log_dir, 'res.npy'), res)
    np.save(os.path.join(log_dir, 'features.npy'), features)
    np.save(os.path.join(log_dir, 'labels.npy'), labels)


    test_acc /= len(test_loader.dataset)
    writer.add_scalar('test acc', test_acc)


if __name__=='__main__':
    batch_size = 512
    config_ids = [17, 16, 15, 14]
    task_type = 'DRSNCS'
    model_names = ['classify_cnn1d_mini_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_small_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_medium_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_big_best300_512batch_' + task_type + '.ckpt']
    log_dir = [
        '..\\run\\1122\\mini_DRSNCS',
        '..\\run\\1122\\small_DRSNCS',
        '..\\run\\1122\\medium_DRSNCS',
        '..\\run\\1122\\big_DRSNCS'
    ]
    dataset = NpzSignalDataset('..\\data\\wuxi_a4', signal_size=128)
    data, target = dataset.data, dataset.target
    n_epochs = 1000
    seed = 66
    set_seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(target)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_data, test_data, train_target, test_target = train_test_split(data, target, test_size=0.4, random_state=seed)
    test_data, val_data, test_target, val_target = train_test_split(test_data, test_target, test_size=0.5, random_state=seed)
    train_dataset = DataLoader(TensorDataset(torch.tensor(train_data), torch.tensor(train_target)), batch_size=batch_size, shuffle=True)
    val_dataset = DataLoader(TensorDataset(torch.tensor(val_data), torch.tensor(val_target)), batch_size=batch_size, shuffle=False)
    test_dataset = DataLoader(TensorDataset(torch.tensor(test_data), torch.tensor(test_target)), batch_size=batch_size, shuffle=False)

    for model_name, config_id, log_dir in zip(model_names, config_ids, log_dir):
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=str(n_epochs), flush_secs=5)
        train_model(writer, config_id, train_dataset, val_dataset, log_dir, model_name, num_epochs=n_epochs, device=device)

        test_model(writer, config_id, test_dataset, log_dir, model_name, device=device)