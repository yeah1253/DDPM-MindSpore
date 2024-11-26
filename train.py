import os

import cv2
import einops
import numpy as np
import torch
import torch.nn as nn
from matplotlib import pyplot as plt, rcParams
from networkx.algorithms.bipartite.basic import color
from sklearn.model_selection import train_test_split
from torch import tensor
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from model.ddpm import DDPM, build_network
import matplotlib.lines as mlines
from model.gan import Generator, gan_predict
from model.model_configs import configs
from model.signal_denoising_ddim import Signal_denoising
from model.vit import VisionTransformer
from utils.dataset import get_shape, get_signal_dataloader, tensor2signal, createFolder, GeneralFigures, \
    tensor2img, make_noise, Signals, KM_signal, WD_signal, NpzSignalDataset
from utils.early_stop import EarlyStopping
from utils.fft_plot import FFTPlot

batch_size = 512 * 8  # 最大化利用显存
_exp_name = "sample"


def config_read(config):
    batch_size = config['batch_size']
    n_epochs = config['n_epochs']
    device = config['device']
    learning_rate = config['lr']
    wd = config['weight_decay']
    seed = config['seed']
    shape = config['shape']
    num_classes = config['num_classes']
    return batch_size, n_epochs, device, learning_rate, wd, seed, shape, num_classes


# fix seed
def same_seeds(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def train_model(config, dataset):
    batch_size, n_epochs, device, learning_rate, wd, seed, shape, num_classes = config_read(config)
    same_seeds(seed)
    in_c, img_size, _ = shape
    model = VisionTransformer(img_size=img_size, in_c=in_c, num_classes=num_classes).to(device)
    stale = 0
    best_acc = 0
    patience = 300

    data, labels = [], []
    for d, l in dataset:
        data.append(d)
        labels.append(l)
    data = torch.stack(data)
    labels = torch.tensor(labels)
    writer = SummaryWriter(log_dir='run/04191518stft512_300', filename_suffix=str(n_epochs), flush_secs=5)
    train_data, valid_data, train_labels, valid_labels = train_test_split(data, labels, test_size=0.2)  # split dataset
    train_dataset = TensorDataset(train_data, train_labels)
    valid_dataset = TensorDataset(valid_data, valid_labels)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=wd)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(n_epochs):

        # ---------- Training ----------
        # Make sure the model is in train mode before training.
        model.train()

        # These are used to record information in training.
        train_loss = []
        train_accs = []

        for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{n_epochs}, training"):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()
            # print(imgs.shape,labels.shape)

            # Forward the data. (Make sure data and model are on the same device.)
            logits = model(imgs.to(device))

            # Calculate the cross-entropy loss.
            # We don't need to apply softmax before computing cross-entropy as it is done automatically.
            loss = criterion(logits, labels.to(device))

            # Gradients stored in the parameters in the previous step should be cleared out first.
            optimizer.zero_grad()

            # Compute the gradients for parameters.
            loss.backward()

            # Clip the gradient norms for stable training.
            grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

            # Update the parameters with computed gradients.
            optimizer.step()

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            train_loss.append(loss.item())
            train_accs.append(acc)

        train_loss = sum(train_loss) / len(train_loss)
        train_acc = sum(train_accs) / len(train_accs)
        # Print the information.
        # print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

        # ---------- Validation ----------
        # Make sure the model is in eval mode so that some modules like dropout are disabled and work normally.
        model.eval()

        # These are used to record information in validation.
        valid_loss = []
        valid_accs = []

        # Iterate the validation set by batches.
        for batch in tqdm(valid_loader, desc=f"Epoch {epoch + 1}/{n_epochs}, validation"):
            # A batch consists of image data and corresponding labels.
            imgs, labels = batch
            # imgs = imgs.half()

            # We don't need gradient in validation.
            # Using torch.no_grad() accelerates the forward process.
            with torch.no_grad():
                logits = model(imgs.to(device))

            # We can still compute the loss (but not the gradient).
            loss = criterion(logits, labels.to(device))

            # Compute the accuracy for current batch.
            acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

            # Record the loss and accuracy.
            valid_loss.append(loss.item())
            valid_accs.append(acc)
            # break

        # The average loss and accuracy for entire validation set is the average of the recorded values.
        valid_loss = sum(valid_loss) / len(valid_loss)
        valid_acc = sum(valid_accs) / len(valid_accs)

        # add to tensorboard
        # writer.add_scalar('train_loss', train_loss, epoch)
        # writer.add_scalar('train_acc', train_acc, epoch)
        # writer.add_scalar('valid_loss', valid_loss, epoch)
        # writer.add_scalar('valid_acc', valid_acc, epoch)
        writer.add_scalars('loss', {'train': train_loss, 'valid': valid_loss}, epoch)
        writer.add_scalars('acc', {'train': train_acc, 'valid': valid_acc}, epoch)
        # Print the information.
        # print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # update logs
        if valid_acc > best_acc:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(
                    f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
        else:
            with open(f"./{_exp_name}_log.txt", "a"):
                print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")

        # save models
        if valid_acc > best_acc:
            print(f"Best model found at epoch {epoch}, saving model")
            torch.save(model.state_dict(),
                       f"{_exp_name}_best.ckpt")  # only save best to prevent output memory exceed error
            best_acc = valid_acc
            stale = 0
        else:
            stale += 1
            if stale > patience:
                print(f"No improvment {patience} consecutive epochs, early stopping")
                break


def vit_train(dataset, config=None):
    # X_train, X_valid, y_train, y_valid= train_test_split(all_data_set)
    #  hyper parameters
    config = {
        "batch_size": 64,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # "device": "cpu",
        "n_epochs": 300,
        "lr": 0.001,
        "weight_decay": 1e-5,
        "seed": 66,
        "shape": (3, 256, 256),
        "num_classes": 8
    } if config is None else config
    train_model(config, dataset)


def plot_signal(signal, title, subplot_position):
    plt.subplot(subplot_position[0], subplot_position[1], subplot_position[2])
    plt.plot(tensor2signal(signal[0]))
    plt.title(title)
    plt.xlabel('time')
    plt.ylabel('amplitude')


def train_ddpm_step(ddpm: DDPM, net, device='cuda', ckpt_path='./model/model.pth',
                    path='./data', log_dir='./run/0000000', n_epochs=500):
    createFolder(log_dir)
    writer = SummaryWriter(log_dir=log_dir, filename_suffix=str(n_epochs), flush_secs=5)
    n_steps = ddpm.n_steps
    dataloader = get_signal_dataloader(path, batch_size, get_shape()[-1], slice_type='cut')
    net = net.to(device)  # 将网络放到GPU上, 并且将网络的参数类型设置为double
    loss_fn = nn.HuberLoss()  # 损失函数, 参数是huber loss的权重
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-2, weight_decay=1e-4)  # 优化器

    ealy_stop = EarlyStopping(log_dir, patience=n_epochs // 10, verbose=True)
    i = 0
    best_loss = np.inf  # 初始化最好的损失为无穷大
    for e in range(n_epochs):
        total_loss = 0

        for x, y in tqdm(dataloader, desc='Epoch {}'.format(e)):
            current_batch_size = x.shape[0]
            x = x.to(device).float()  # Convert x to float32， x是纯净信号
            t = torch.randint(0, n_steps, (current_batch_size,)).to(device)  # 生成一个0到n_steps之间的随机数
            # eps = torch.randn_like(x).to(device)  # 作用是生成一个与x同样shape的随机数，服从标准正态分布  # 生成一个噪声
            # TODO
            eps = make_noise(t).to(device)  # 生成一个噪声, 此处的x是纯净信号, t是一个随机数
            x_t = ddpm.sample_forward(x, t, eps)  # 将eps噪声与纯净信号叠加, 生成一个新的噪声混合信号, 包含噪声\纯净信号及其耦合的信息
            eps_theta = net(x_t, t.reshape(current_batch_size, 1))  # 去噪器, 其实是生成噪声, 生成的噪声是根据x_t+t生成的

            loss = loss_fn(eps, eps_theta)  # 计算损失,使得eps-eps_theta尽可能的接近x
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            # 信号diffusion的过程
            fig, ax = plt.subplots(2, 2, figsize=(10, 10))
            fig.tight_layout(h_pad=5, w_pad=5)

            plot_signal(x, 'original signal', (2, 2, 1))
            plot_signal(eps, 'noise', (2, 2, 2))
            plot_signal(x_t, 'signal with noise', (2, 2, 3))
            plot_signal(eps_theta, 'noise generated by denoiser', (2, 2, 4))
            writer.add_figure('signal process', plt.gcf(), i)
            writer.add_scalar('step loss', loss.item(), i)
            i += 1

        total_loss /= len(dataloader)

        writer.add_scalar('epochs loss', total_loss, e)
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(net.state_dict(), os.path.join(log_dir, ckpt_path))
        ealy_stop(total_loss, net)
        if ealy_stop.early_stop:
            print('Early stopping')
            break
    print('Done')


def sample_imgs(ddpm, net, output_path, n_sample=81, device='cuda', simple_var=True):
    net = net.to(device).eval()
    with torch.no_grad():
        shape = (n_sample, *get_shape())
        imgs = ddpm.sample_backward(shape, net, device=device, simple_var=simple_var).detach().cpu()
        imgs = (imgs + 1) / 2 * 255
        imgs = imgs.clamp(0, 255)
        imgs = einops.rearrange(imgs, '(b1 b2) c h w -> (b1 h) (b2 w) c',
                                b1=int(n_sample ** 0.5))
        imgs = tensor2img(imgs)
        cv2.imwrite(output_path, imgs)


def sample_signals(ddpm, net, n_sample=81, device='cuda', simple_var=True, input_path=None):
    net = net.to(device).eval()
    with (torch.no_grad()):
        in_signals = (n_sample, get_shape()[1], get_shape()[2]) \
            if input_path is None else \
            Signals(input_path, fs=5120, slice_length=512, slice_type='cut', add_noise=True)
        # 生成信号, /ddpm.n_steps没有道理
        # 随机抽取n_sample个idx
        if input_path is not None:
            idxes = np.random.choice(len(in_signals), n_sample, replace=False)
            in_signals, targets = in_signals[idxes]
        signals = ddpm.sample_backward(in_signals, net, device=device, simple_var=simple_var).detach().cpu().numpy()
        # if input_path is None:
        #     # 当signals的范围超过-1到1之间时，对signals的绝对值进行对数运算， 使得范围缩小
        #     # 记录信号的正负
        #     # signals = signals.clip(-1, 1)
        #     # 将信号的形状从(n_sample, 1, n_steps)转换为(n_sample, n_steps)
        #     # signals = signals.reshape(n_sample, -1)
        #     signals = np.where(np.abs(signals) > 10, np.sign(signals) * np.log10(np.abs(signals)), signals)
        #     signals = np.where(np.abs(signals) > 1, signals / 10, signals)
        # else:
        #     signals = np.where(np.abs(signals) > 1, signals / 10, signals)
        createFolder('work_dirs/original')
        createFolder('work_dirs/stft')
        createFolder('work_dirs/wf')
        for i in tqdm(range(n_sample), desc='sample signals'):
            fft = FFTPlot(signals[i][0], 'signal {}'.format(i + 1))
            fft.saveOriginal('work_dirs/original')
            fft.saveSTFT('work_dirs/stft')
            fft.saveWaveform('work_dirs/wf')


def vit_predict(model_path, data, device='cuda'):
    net = VisionTransformer(img_size=256, in_c=3, num_classes=8).to(device)
    net.load_state_dict(torch.load(model_path))
    net.eval()
    res = []
    with torch.no_grad():
        for d in tqdm(data, desc=f'predicting {model_path}'):
            data = d.unsqueeze(0).to(device)
            logits = net(data)
            res.append(logits.argmax(dim=-1).item())
    return res


def predict(device='cuda'):
    classify_model_path = './model/stft_512_vit_sample_best300_64batch.ckpt'
    stft_data = GeneralFigures('./work_dirs/stft', (3, 256, 256))
    stft_res = vit_predict(classify_model_path, stft_data.data, device=device)
    wf_data = GeneralFigures('./work_dirs/wf', (3, 256, 256))
    classify_model_path = './model/wf_512_vit_sample_best300_64batch.ckpt'
    wf_res = vit_predict(classify_model_path, wf_data.data, device=device)
    # 预测矩阵
    stft_res = np.array(stft_res).reshape(1000, 1)
    wf_res = np.array(wf_res).reshape(1000, 1)
    compare = stft_res == wf_res
    res = np.concatenate((stft_res, wf_res, compare), axis=1)
    # 统计预测结果
    same = np.sum(compare)
    percent = same / 1000
    # 将结果写入csv
    np.savetxt('work_dirs/result.csv', res, delimiter=',', fmt='%d')
    print(f'预测结果：{same}个相同，占比{percent}')


def train_ddpm(device, model_name, data_path, config_id, log_dir, n_epochs=500):
    n_steps = 2000
    config = configs[config_id]
    net = build_network(config, n_steps)
    sd_ddim = Signal_denoising(device, n_steps)
    train_ddpm_step(sd_ddim, net, device=device, ckpt_path=model_name, path=data_path,
                    log_dir=log_dir,
                    n_epochs=n_epochs)


def cnn_train(writer, net, train_dataloader, val_dataloader, device, ckpt_path, log_dir,
              n_epochs):  # 训练cnn1d分类模型
    createFolder(log_dir)
    net = net.to(device).float()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(net.parameters(), 1e-3, weight_decay=1e-5)
    ealy_stop = EarlyStopping(log_dir, patience=30, verbose=True)
    best_acc = 0
    train_acces = []
    train_losses = []
    val_acces = []
    val_losses = []
    for e in range(n_epochs):
        train_acc = 0
        train_loss = 0
        val_acc = 0
        val_loss = 0
        for x, y in tqdm(train_dataloader, desc='train Epoch {}'.format(e)):  # 训练
            x = x.to(device).float()
            y = y.to(device).long()
            _, logits = net(x)
            loss = loss_fn(logits, y)
            optimizer.zero_grad()
            loss.backward()  # 反向传播
            optimizer.step()  # 更新参数
            train_acc += (logits.argmax(dim=-1) == y).float().mean()
            train_loss += loss.item()
        net.eval()
        for x, y in tqdm(val_dataloader, desc='val Epoch {}'.format(e)):  # 测试
            x = x.to(device).float()
            y = y.to(device).long()
            _, logits = net(x)
            loss = loss_fn(logits, y)
            val_acc += (logits.argmax(dim=-1) == y).float().mean()
            val_loss += loss.item()
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(train_loss / len(train_dataloader))
        val_acces.append(val_acc / len(val_dataloader))
        val_losses.append(val_loss / len(val_dataloader))
        writer.add_scalars('acc', {'train': train_acces[-1], 'val': val_acces[-1]}, e)
        writer.add_scalars('loss', {'train': train_losses[-1], 'val': val_losses[-1]}, e)
        # 检测是否过拟合，如果过拟合则停止训练, Early Stop
        ealy_stop(val_losses[-1], net)
        if ealy_stop.early_stop:
            print('Early stopping')
            break
        if val_acc > best_acc:
            torch.save(net.state_dict(), os.path.join(log_dir, ckpt_path))
            best_acc = val_acc
    print('Done')


def train_classification_step(writer, device, model_name, config_id, log_dir, train_dataloader=None,
                              val_dataloader=None, n_epochs=500):
    config = configs[config_id]
    net = build_network(config)
    cnn_train(writer, net, train_dataloader=train_dataloader, val_dataloader=val_dataloader, device=device,
              ckpt_path=model_name, log_dir=log_dir, n_epochs=n_epochs)


def prepare_data(data_path='./data/wuxi_a4',
                 slice_length=get_shape()[-1],
                 slice_type='cut',
                 windows_ratio=0.5,
                 delete_labels=None,
                 add_noise=False,
                 denoising_properties=None,
                device = 'cuda' if torch.cuda.is_available() else 'cpu'
                 ):
    """
    准备数据，根据是否添加噪声，以及是否去噪，选择不同的数据集
    :param data_path: 数据集路径
    :param slice_length: 切片长度
    :param slice_type: 切片类型
    :param windows_ratio: 窗口比例
    :param delete_labels: 删除的标签
    :param add_noise: 是否添加噪声
    :param denoising_properties: 去噪属性
    :param device: 设备
    :return: 数据集，任务类型
    """
    # 根据是否添加噪声，以及是否去噪，选择不同的数据集， 以及绘制不同的图像，确定不同的任务类型
    if denoising_properties:
        add_noise = True
        # dataset = Signals(data_path, slice_length=slice_length, slice_type=slice_type,
        #                   add_noise=add_noise, windows_rate=windows_ratio, delete_labels=delete_labels)
        dataset = NpzSignalDataset(data_path, signal_size=slice_length)
        task_type = ' denoising'
        # 取出8个不同标签的信号，绘制信号的图
        indexes = []
        for i in range(8):
            indexes.append(np.where(dataset.target == i)[0][0])
        noisy_data = dataset.data[indexes].copy()

        denoising_methods = {
            'diffusion model': lambda: denoise_diffusion_model(dataset, denoising_properties, device),
            'kalman': lambda: KM_signal(dataset.data),
            'wavelet': lambda: WD_signal(dataset.data),
            'gan': lambda: denoise_gan(dataset, denoising_properties)
        }

        def denoise_diffusion_model(dataset, denoising_properties, device):
            n_steps = denoising_properties['n_steps']
            config_id = denoising_properties['config_id']
            root_dir = denoising_properties['root_dir']
            model_name = denoising_properties['model_name']
            config = configs[config_id]
            denoising_net = build_network(config, n_steps)
            sd_ddim = Signal_denoising(device, n_steps)
            denoising_net.load_state_dict(torch.load(os.path.join(root_dir, model_name), weights_only=True))
            denoising_net = denoising_net.to(device).eval()
            interval = 256
            for i in tqdm(range(0, len(dataset.data), interval), desc='denoising'):
                end = min(i + interval, len(dataset.data))
                cur = sd_ddim.sample_backward(tensor(dataset.data[i:end]).to(device).float(), denoising_net,
                                              device=device, simple_var=True).detach().cpu().numpy()
                dataset.data[i:end] = cur
            return dataset.data

        def denoise_gan(dataset, denoising_properties):
            config = {'lstm_hidden_size': 128, 'num_layers': 16}
            generator = Generator(**config).cuda()
            return gan_predict(generator, dataset, model_path=os.path.join(denoising_properties['root_dir'],
                                                                           denoising_properties['model_name']),
                               device='cuda')

        dataset.data = denoising_methods[denoising_properties['denoising method']]()      # 绘制8个处理后的信号

        fig, axes = plt.subplots(4, 4, figsize=(24, 12))  # 使用较小的 figsize
        # 画出16个信号，相邻两个子图分别是原始信号和去噪后的信号
        for i in range(8):
            # 绘制噪声信号
            ax_noisy = axes[i // 2, (i % 2) * 2]  # 每对信号占据两列中的第一列
            ax_noisy.plot(noisy_data[i][0], label='noisy')
            ax_noisy.set_title(f'Noisy{i}')
            # ax_noisy.legend()

            # 绘制去噪信号
            ax_denoised = axes[i // 2, (i % 2) * 2 + 1]  # 每对信号占据两列中的第二列
            ax_denoised.plot(dataset.data[indexes[i]][0], label='denoised',color='red')
            ax_denoised.set_title(f'Denoised{i}')
            # ax_denoised.legend()

        plt.tight_layout()
        # 保存图像
        plt.savefig(
            os.path.join('./work_dirs/classify', denoising_properties['denoising method'] + task_type + '_signal.png')
        )
    else:
        # dataset = Signals(data_path, slice_length=slice_length, slice_type=slice_type,
        #                   add_noise=add_noise, windows_rate=windows_ratio, delete_labels=delete_labels)
        dataset = NpzSignalDataset(data_path, signal_size=slice_length)
        task_type = 'pure' if not add_noise else 'noisy'
        # 取出8个不同标签信号，绘制信号的图
        indexes = []
        for i in range(8):
            indexes.append(np.where(dataset.target == i)[0][0])
        fig, axes = plt.subplots(2, 4, figsize=(12, 6))
        fig.tight_layout(h_pad=5, w_pad=5)
        for i in range(8):
            ax = axes[i // 4, i % 4]
            ax.plot(dataset.data[indexes[i]][0], label='signal')
            ax.set_title(f'label {dataset.target[indexes[i]]}')
        fig.tight_layout()
        plt.savefig(os.path.join('./work_dirs/classify', task_type + '_signal.png'))
    print('data prepared')
    return dataset, task_type


def test_classification(writer, device, model, config_id, log_dir, test_dataloader):
    config = configs[config_id]
    net = build_network(config)
    net.load_state_dict(torch.load(os.path.join(log_dir, model)))
    net = net.to(device).eval()
    with torch.no_grad():
        acc = 0
        best_feature = [] # 存储特征, 用于可视化
        labels = []
        for x, y in tqdm(test_dataloader, desc='validating'):
            x = x.to(device).float()
            y = y.to(device).long()
            feature, logits = net(x)
            acc += (logits.argmax(dim=-1) == y).float().mean()
            best_feature.append(feature.cpu().numpy())
            labels.append(y.cpu().numpy())
        acc /= len(test_dataloader)
        # 存储特征与标签
        features = np.concatenate(best_feature, axis=0)
        labels = np.concatenate(labels, axis=0)
        np.save(os.path.join(log_dir, 'features.npy'), features)
        np.save(os.path.join(log_dir, 'labels.npy'), labels)
        writer.add_scalar('test acc', acc.item())
        print(f'accuracy: {acc.item()}')
        return acc.item()


def train_classification(log_dirs, ds_config, add_noise=False, denoising_properties=None, batch_size=512):
    """
    训练分类模型, 可以选择是否添加噪声，以及是否去噪
    :param log_dirs: 保存模型的路径，tensorboard的log路径，
    :param ds_config: 数据集配置
    :param add_noise: 是否添加噪声
    :param denoising_properties: 去噪属性
    """
    os.makedirs('work_dirs', exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    n_epochs = 1000
    # 根据是否添加噪声，以及是否去噪，选择不同的数据集， 以及绘制不同的图像，确定不同的任务类型
    dataset, task_type = prepare_data(**ds_config, add_noise=add_noise, denoising_properties=denoising_properties)
    data = dataset.data
    target = dataset.target
    seed = 66
    torch.manual_seed(seed)

    # 随机选择2000个样本
    # np.random.seed(seed)
    # idxes = np.random.choice(len(data), 5 * 1024, replace=False)
    # data, target = data[idxes], target[idxes]

    # 为了保证每次的结果一致，设置随机种子,打乱数据
    np.random.seed(seed)
    np.random.shuffle(data)
    np.random.seed(seed)
    np.random.shuffle(target)
    # 准备数据，选择不同的模型，以及不同的log路径，训练分类模型
    train_data, test_data, train_labels, test_labels = train_test_split(data, target, test_size=0.4)
    test_data, val_data, test_labels, val_labels = train_test_split(test_data, test_labels, test_size=0.5)
    train_dataloader = DataLoader(TensorDataset(tensor(train_data), tensor(train_labels)),
                                  batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(TensorDataset(tensor(test_data), tensor(test_labels)),
                                 batch_size=batch_size, shuffle=False)
    val_dataloader = DataLoader(TensorDataset(tensor(val_data), tensor(val_labels)),
                                batch_size=batch_size, shuffle=False)
    config_ids = [17, 16, 15, 14]
    model_name = ['classify_cnn1d_mini_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_small_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_medium_best300_512batch_' + task_type + '.ckpt',
                  'classify_cnn1d_big_best300_512batch_' + task_type + '.ckpt']
    for model, config_id, log_dir in zip(model_name, config_ids, log_dirs):
        writer = SummaryWriter(log_dir=log_dir, filename_suffix=str(n_epochs), flush_secs=5)
        train_classification_step(writer, device, model, config_id, log_dir, train_dataloader=train_dataloader,
                                  val_dataloader=val_dataloader, n_epochs=n_epochs)
        test_classification(writer, device, model, config_id, log_dir, test_dataloader)


def train_sd_ddim(data_path='./data/', device="cuda" if torch.cuda.is_available() else "cpu"):
    model_names = ['reduce_noise_model_bi_lstm_big_huber_loss_power_snr.pth',
                   'reduce_noise_model_bi_lstm_medium_huber_loss_power_snr.pth',
                   'reduce_noise_model_bi_lstm_small_huber_loss_power_snr.pth']
    log_dirs = ['./run/1116_open_biglstm_seqlen128', './run/1116_open_mediumlstm_seqlen128', './run/1116_open_smalllstm_seqlen128']
    config_ids = [11, 12, 13]

    for model_name, config_id, log_dir in zip(model_names, config_ids, log_dirs):
        train_ddpm(device, model_name, data_path, config_id, log_dir, 300)


if __name__ == '__main__':
    # 字体大小
    # plt.rcParams.update({'font.size': 8})
    rcParams['font.sans-serif'] = ['SimSun']  # Chinese font
    rcParams['font.family'] = 'sans-serif'  # Set Chinese and other fonts to sans-serif
    rcParams['font.serif'] = ['Times New Roman']  # English font
    rcParams['axes.unicode_minus'] = False  # Solve the problem of displaying minus sign
    # train_sd_ddim(data_path='./data/AI1/')
    batch_size = 2048
    d_p = {
        'n_steps': 2000,  # ddpm的步数
        'config_id': 11,  # 11, 12, 13, 分别对应大中小
        'root_dir': './run/1116_open_biglstm_seqlen128',  # 保存模型的路径
        'model_name': 'reduce_noise_model_bi_lstm_big_huber_loss_power_snr.pth',  # 模型名称
        'denoising method': 'diffusion model'  # 去噪方法, sdddim, kalman, wavelet
    }
    dataset_config = {
        'data_path': './data/wuxi_a4/',
        'slice_length': 128,
        'slice_type': 'window',  # 'cut','window'
        'windows_ratio': 0.05,
    }
    prepare_data(**dataset_config, add_noise=True, denoising_properties=d_p)  # 准备数据，添加噪声，以及去噪
    d_p['denoising method'] = 'kalman'
    prepare_data(**dataset_config, add_noise=True, denoising_properties=d_p)  # 准备数据，添加噪声，以及去噪
    d_p['denoising method'] = 'wavelet'
    prepare_data(**dataset_config, add_noise=True, denoising_properties=d_p)  # 准备数据，添加噪声，以及去噪
    d_p['denoising method'] = 'gan'
    d_p['root_dir'] = './run/1121/gan14'
    d_p['model_name'] = 'generator.pth'
    prepare_data(**dataset_config, add_noise=True, denoising_properties=d_p)  # 准备数据，添加噪声，以及去噪

    # train_classification(
    #     log_dirs=['./run/11212100/mini_gan', './run/11212100/small_gan', './run/11212100/medium_gan', './run/11212100/big_gan'],
    #     ds_config=dataset_config, batch_size=batch_size
    # )  # 训练分类模型， 输入为原始受干扰信号经过wavelet去噪后的信号

    # prepare_data(add_noise=False)  # 准备数据，不添加噪声
    # prepare_data(add_noise=True)  # 准备数据，添加噪声
    # del_labels = ['Aligned', 'Parallel', 'Unbalance']  # 删除的标签,
    # 一共有8个标签,分别是 Aligned, Bearing, Bowed, Broken, Normal, Parallel, SWF, Unbalance
    # 删除后剩下5个标签，分别是 Bearing, Bowed, Broken, Normal, SWF

    # train_classification(
    #     log_dirs=['./run/11181525/mini_n', './run/11181525/small_n', './run/11181525/medium_n', './run/11181525/big_n'],
    #     ds_config=dataset_config, batch_size=batch_size
    # )  # 训练分类模型， 输入为原始受干扰信号
    # train_classification(
    #     log_dirs=['./run/11181525/mini_dn', './run/11181525/small_dn', './run/11181525/medium_dn', './run/11181525/big_dn']
    #     , denoising_properties=d_p, ds_config=dataset_config, batch_size=batch_size
    # )  # 训练分类模型， 输入为原始受干扰信号经过sd_ddim去噪后的信号
    # d_p['denoising method'] = 'kalman'
    # train_classification(
    #     log_dirs=['./run/11181525/mini_kal', './run/11181525/small_kal', './run/11181525/medium_kal', './run/11181525/big_kal'],
    #     denoising_properties=d_p,
    #     ds_config=dataset_config, batch_size=batch_size
    # )  # 训练分类模型， 输入为原始受干扰信号经过kalman去噪后的信号
    # d_p['denoising method'] = 'wavelet'
    # train_classification(
    #     log_dirs=['./run/11181525/mini_wav', './run/11181525/small_wav', './run/11181525/medium_wav', './run/11181525/big_wav'],
    #     ds_config=dataset_config, batch_size=batch_size
    # )  # 训练分类模型， 输入为原始受干扰信号经过wavelet去噪后的信号
    # dataset_config['data_path'] = './data/wuxi_a3/'
    # train_classification(
    #     log_dirs=['./run/1116mini_o', './run/1116small_o', './run/1116medium_o', './run/1116big_o'],
    #     ds_config=dataset_config, add_noise=False, batch_size=batch_size
    # )  # 训练分类模型， 输入为实验室条件下纯净信号