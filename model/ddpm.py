import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.dataset import get_shape


class PositionalEncoding(nn.Module):

    def __init__(self, max_seq_len: int, d_model: int):
        super().__init__()

        # Assume d_model is an even number for convenience
        assert d_model % 2 == 0

        pe = torch.zeros(max_seq_len, d_model)
        i_seq = torch.linspace(0, max_seq_len - 1, max_seq_len)
        j_seq = torch.linspace(0, d_model - 2, d_model // 2)
        pos, two_i = torch.meshgrid(i_seq, j_seq)
        pe_2i = torch.sin(pos / 10000 ** (two_i / d_model))
        pe_2i_1 = torch.cos(pos / 10000 ** (two_i / d_model))
        pe = torch.stack((pe_2i, pe_2i_1), 2).reshape(max_seq_len, d_model)

        self.embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding.weight.data = pe
        self.embedding.requires_grad_(False)

    def forward(self, t):
        return self.embedding(t)


class BiLSTM(nn.Module):
    def __init__(self, n_steps, lstm_hidden_size, pe_dim, num_layers):
        super().__init__()
        num_outputs = get_shape()[2]  # 输入和输出的长度相同
        self.pe = PositionalEncoding(n_steps, pe_dim)
        self.LSTM = nn.LSTM(num_outputs, lstm_hidden_size,
                            num_layers=num_layers, batch_first=True,
                            bidirectional=True)
        self.relu = nn.ReLU()
        # 因为是双向 LSTM, 所以要乘2
        self.classifier = nn.ModuleList()
        input_channel = pe_dim
        output_channel = pe_dim // 2
        mlp_input_size = lstm_hidden_size * 2 * pe_dim
        for _ in range(num_layers // 2 if num_layers <= 8 else 4):
            self.classifier.append(
                nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, 3, 1, 1),
                    nn.BatchNorm1d(output_channel),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                    nn.MaxPool1d(2)
                )
            )  # 每次卷积后，长度减半, 通道数减半
            if output_channel == 4:
                mlp_input_size = mlp_input_size // 4
                input_channel = output_channel
                break
            input_channel = output_channel
            output_channel = output_channel // 2
            mlp_input_size = mlp_input_size // 4
        output_channel = input_channel * 2
        for _ in range(num_layers // 2 if num_layers <= 8 else 4):
            self.classifier.append(
                nn.Sequential(
                    nn.Conv1d(input_channel, output_channel, 3, 1, 1),
                    nn.BatchNorm1d(output_channel),
                    nn.ReLU(),
                    nn.Dropout(0.5),
                )
            )  # 每次卷积后，长度不变, 通道数翻倍
            input_channel = output_channel
            output_channel = output_channel * 2
            mlp_input_size = mlp_input_size * 2
        self.ffn = nn.Sequential(
            nn.Linear(mlp_input_size, mlp_input_size // 16),
            nn.ReLU(),
            nn.Linear(mlp_input_size // 16, num_outputs)
        )

    def forward(self, x, t):

        pe = self.pe(t).reshape(t.shape[0], -1, 1)  # 位置编码
        x = x + pe  # shape: (batch_size, pe_dim, num_outputs)

        lstm_hidden_states, _ = self.LSTM(x)  # shape: (batch_size, pe_dim, lstm_hidden_size * 2)
        # LSTM 的最后一个时刻的隐藏状态, 即句向量
        for m_x in self.classifier:
            lstm_hidden_states = m_x(lstm_hidden_states)
        lstm_hidden_states = lstm_hidden_states.flatten(1)
        # shape: (batch, lstm_hidden_size * 2 * num_outputs // 2**num_layers)
        logits = self.ffn(lstm_hidden_states)

        return logits.view(-1, get_shape()[1], get_shape()[2])


class ResidualBlock(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.actvation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv2d(in_c, out_c, 1),
                                          nn.BatchNorm2d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x


class ResidualBlock1D(nn.Module):

    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.bn1 = nn.BatchNorm1d(out_c)
        self.actvation1 = nn.ReLU()
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.bn2 = nn.BatchNorm1d(out_c)
        self.actvation2 = nn.ReLU()
        if in_c != out_c:
            self.shortcut = nn.Sequential(nn.Conv1d(in_c, out_c, 1),
                                          nn.BatchNorm1d(out_c))
        else:
            self.shortcut = nn.Identity()

    def forward(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.actvation1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += self.shortcut(input)
        x = self.actvation2(x)
        return x


class ConvNet1D(nn.Module):

    def __init__(self,
                 n_steps,
                 intermediate_channels=None,
                 pe_dim=10,
                 insert_t_to_all_layers=False):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = [10, 20, 40]
        C, H, W = get_shape()  # 一维信号的channel, height, width, 当输入信号时，channel是1，形状为1，1，length
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock1D(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            prev_channel = channel
        self.output_layer = nn.Conv1d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1)
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x


class ConvNet1DClassify(nn.Module):
    def __init__(self, intermediate_channels=None, out_dim=10):
        """
        初始化1D卷积网络分类器。

        参数:
        - intermediate_channels (list): 每个卷积层的通道数。
        - out_dim (int): 输出的类别数。
        """
        super(ConvNet1DClassify, self).__init__()
        if intermediate_channels is None:
            intermediate_channels = [10, 20, 40]
        C, H, W = get_shape()  # 假设输入信号的channel为1，height为1，width为length

        self.cnn1d_blocks = nn.ModuleList()
        prev_channel = H
        for channel in intermediate_channels:
            block = nn.Sequential(
                nn.Conv1d(prev_channel, channel, 3, 1, 1),
                nn.BatchNorm1d(channel),
                nn.ReLU(),
                nn.MaxPool1d(2) if W > 64 else nn.Identity(),  # 池化层，只有当W > 64时应用
                nn.Dropout(0.5)  # Dropout层
            )
            self.cnn1d_blocks.append(block)
            prev_channel = channel
            if W > 64:
                W //= 2

        self.fces = nn.Sequential(
            nn.Linear(W * prev_channel, W * prev_channel // 8),
            nn.ReLU(),
            nn.Linear(W * prev_channel // 8, out_dim)
        )

    def forward(self, x):
        """
        前向传播函数。

        参数:
        - x (torch.Tensor): 输入张量，形状为 (batch_size, channels, length)。

        返回:
        - torch.Tensor: 输出张量，形状为 (batch_size, out_dim)。
        """
        for m_x in self.cnn1d_blocks:
            x = m_x(x)
        return x.flatten(1), self.fces(x.flatten(1))  # 输出feature和分类结果


class ConvNet(nn.Module):

    def __init__(self,
                 n_steps,
                 intermediate_channels=None,
                 pe_dim=10,
                 insert_t_to_all_layers=False):
        super().__init__()
        if intermediate_channels is None:
            intermediate_channels = [10, 20, 40]
        C, H, W = get_shape()  # 1, 28, 28;图片的channel, height, width, 当输入信号时，channel是1，形状为1，1，length
        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.pe_linears = nn.ModuleList()
        self.all_t = insert_t_to_all_layers
        if not insert_t_to_all_layers:
            self.pe_linears.append(nn.Linear(pe_dim, C))

        self.residual_blocks = nn.ModuleList()
        prev_channel = C
        for channel in intermediate_channels:
            self.residual_blocks.append(ResidualBlock(prev_channel, channel))
            if insert_t_to_all_layers:
                self.pe_linears.append(nn.Linear(pe_dim, prev_channel))
            else:
                self.pe_linears.append(None)
            prev_channel = channel
        self.output_layer = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        for m_x, m_t in zip(self.residual_blocks, self.pe_linears):
            if m_t is not None:
                pe = m_t(t).reshape(n, -1, 1, 1)
                x = x + pe
            x = m_x(x)
        x = self.output_layer(x)
        return x


class UnetBlock(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm(shape)
        self.conv1 = nn.Conv2d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class UNetBlock1D(nn.Module):

    def __init__(self, shape, in_c, out_c, residual=False):
        super().__init__()
        self.ln = nn.LayerNorm([in_c, shape[2]])
        self.conv1 = nn.Conv1d(in_c, out_c, 3, 1, 1)
        self.conv2 = nn.Conv1d(out_c, out_c, 3, 1, 1)
        self.activation = nn.ReLU()
        self.residual = residual
        if residual:
            if in_c == out_c:
                self.residual_conv = nn.Identity()
            else:
                self.residual_conv = nn.Conv1d(in_c, out_c, 1)

    def forward(self, x):
        out = self.ln(x)
        out = self.conv1(out)
        out = self.activation(out)
        out = self.conv2(out)
        if self.residual:
            out += self.residual_conv(x)
        out = self.activation(out)
        return out


class UNet(nn.Module):

    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False) -> None:
        super().__init__()
        C, H, W = get_shape()
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UnetBlock((prev_channel, cH, cW),
                              prev_channel,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))
            self.downs.append(nn.Conv2d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UnetBlock((prev_channel, Hs[-1], Ws[-1]),
                      prev_channel,
                      channel,
                      residual=residual),
            UnetBlock((channel, Hs[-1], Ws[-1]),
                      channel,
                      channel,
                      residual=residual),
        )
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose2d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UnetBlock((channel * 2, cH, cW),
                              channel * 2,
                              channel,
                              residual=residual),
                    UnetBlock((channel, cH, cW),
                              channel,
                              channel,
                              residual=residual)))

            prev_channel = channel

        self.conv_out = nn.Conv2d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1, 1)
        x = self.mid(x + pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1, 1)
            x = up(x)

            pad_x = encoder_out.shape[2] - x.shape[2]
            pad_y = encoder_out.shape[3] - x.shape[3]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x


class UNet1D(nn.Module):

    def __init__(self,
                 n_steps,
                 channels=[10, 20, 40, 80],
                 pe_dim=10,
                 residual=False,
                 ) -> None:
        super().__init__()
        C, H, W = get_shape()
        layers = len(channels)
        Hs = [H]
        Ws = [W]
        cH = H
        cW = W
        for _ in range(layers - 1):
            cH //= 2
            cW //= 2
            Hs.append(cH)
            Ws.append(cW)

        self.pe = PositionalEncoding(n_steps, pe_dim)

        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.pe_linears_en = nn.ModuleList()
        self.pe_linears_de = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        prev_channel = C
        for channel, cH, cW in zip(channels[0:-1], Hs[0:-1], Ws[0:-1]):
            self.pe_linears_en.append(
                nn.Sequential(nn.Linear(pe_dim, prev_channel), nn.ReLU(),
                              nn.Linear(prev_channel, prev_channel)))
            self.encoders.append(
                nn.Sequential(
                    UNetBlock1D((prev_channel, cH, cW),
                                prev_channel,
                                channel,
                                residual=residual),
                    UNetBlock1D((channel, cH, cW),
                                channel,
                                channel,
                                residual=residual)))
            self.downs.append(nn.Conv1d(channel, channel, 2, 2))
            prev_channel = channel

        self.pe_mid = nn.Linear(pe_dim, prev_channel)
        channel = channels[-1]
        self.mid = nn.Sequential(
            UNetBlock1D((prev_channel, Hs[-1], Ws[-1]),
                        prev_channel,
                        channel,
                        residual=residual),
            UNetBlock1D((channel, Hs[-1], Ws[-1]),
                        channel,
                        channel,
                        residual=residual),
        )
        prev_channel = channel
        for channel, cH, cW in zip(channels[-2::-1], Hs[-2::-1], Ws[-2::-1]):
            self.pe_linears_de.append(nn.Linear(pe_dim, prev_channel))
            self.ups.append(nn.ConvTranspose1d(prev_channel, channel, 2, 2))
            self.decoders.append(
                nn.Sequential(
                    UNetBlock1D((channel * 2, cH, cW),
                                channel * 2,
                                channel,
                                residual=residual),
                    UNetBlock1D((channel, cH, cW),
                                channel,
                                channel,
                                residual=residual)))

            prev_channel = channel

        self.conv_out = nn.Conv1d(prev_channel, C, 3, 1, 1)

    def forward(self, x, t):
        n = t.shape[0]
        t = self.pe(t)
        encoder_outs = []
        for pe_linear, encoder, down in zip(self.pe_linears_en, self.encoders,
                                            self.downs):
            pe = pe_linear(t).reshape(n, -1, 1)
            x = encoder(x + pe)
            encoder_outs.append(x)
            x = down(x)
        pe = self.pe_mid(t).reshape(n, -1, 1)
        x = self.mid(x + pe)
        for pe_linear, decoder, up, encoder_out in zip(self.pe_linears_de,
                                                       self.decoders, self.ups,
                                                       encoder_outs[::-1]):
            pe = pe_linear(t).reshape(n, -1, 1)
            x = up(x)

            pad_x = 0
            pad_y = encoder_out.shape[2] - x.shape[2]
            x = F.pad(x, (pad_x // 2, pad_x - pad_x // 2, pad_y // 2,
                          pad_y - pad_y // 2))
            x = torch.cat((encoder_out, x), dim=1)
            x = decoder(x + pe)
        x = self.conv_out(x)
        return x


class DDPM:

    # n_steps 就是论文里的 T
    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        betas = torch.linspace(min_beta, max_beta, n_steps).to(device)
        alphas = 1 - betas
        alpha_bars = torch.empty_like(alphas)
        product = 1
        for i, alpha in enumerate(alphas):
            product *= alpha
            alpha_bars[i] = product
        self.betas = betas
        self.n_steps = n_steps
        self.alphas = alphas
        self.alpha_bars = alpha_bars
        alpha_prev = torch.empty_like(alpha_bars)
        alpha_prev[1:] = alpha_bars[0:n_steps - 1]
        alpha_prev[0] = 1
        self.coef1 = torch.sqrt(alphas) * (1 - alpha_prev) / (1 - alpha_bars)
        self.coef2 = torch.sqrt(alpha_prev) * self.betas / (1 - alpha_bars)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1, 1)
        if eps is None:
            eps = torch.randn_like(x)
        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        return res

    def sample_backward(self, img_shape, net, device, simple_var=True,
                        clip_x0=True):
        x = torch.randn(img_shape).to(device)
        net = net.to(device)
        for t in range(self.n_steps - 1, -1, -1):
            x = self.sample_backward_step(x, t, net, simple_var, clip_x0)
        return x

    def sample_backward_step(self, x_t, t, net, simple_var=True, clip_x0=True):
        n = x_t.shape[0]
        t_tensor = torch.tensor([t] * n,
                                dtype=torch.long).to(x_t.device).unsqueeze(1)
        eps = net(x_t, t_tensor)

        if t == 0:
            noise = 0
        else:
            if simple_var:
                var = self.betas[t]
            else:
                var = (1 - self.alpha_bars[t - 1]) / (
                        1 - self.alpha_bars[t]) * self.betas[t]
            noise = torch.randn_like(x_t)
            noise *= torch.sqrt(var)

        if clip_x0:
            x_0 = (x_t - torch.sqrt(1 - self.alpha_bars[t]) *
                   eps) / torch.sqrt(self.alpha_bars[t])
            x_0 = torch.clip(x_0, -1, 1)
            mean = self.coef1[t] * x_t + self.coef2[t] * x_0
        else:
            mean = (x_t -
                    (1 - self.alphas[t]) / torch.sqrt(1 - self.alpha_bars[t]) *
                    eps) / torch.sqrt(self.alphas[t])

        x_t = mean + noise

        return x_t


def build_network(config: dict, n_steps=None):
    network_type = config.pop('type')
    network_mapping = {
        'ConvNet': ConvNet,
        'UNet': UNet,
        'ConvNet1D': ConvNet1D,
        'UNet1D': UNet1D,
        'BiLSTM': BiLSTM,
        'ConvNet1DClassify': ConvNet1DClassify
    }

    network_cls = network_mapping.get(network_type)

    network = network_cls(n_steps, **config) if n_steps is not None else network_cls(**config)
    config['type'] = network_type  # Restore the original config
    return network
