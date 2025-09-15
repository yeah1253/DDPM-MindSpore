import torch
from torch import tensor
from tqdm import tqdm

from model.ddim import DDIM
from utils.dataset import make_noise


class Signal_denoising(DDIM):

    def __init__(self,
                 device,
                 n_steps: int,
                 min_beta: float = 0.0001,
                 max_beta: float = 0.02):
        super().__init__(device, n_steps, min_beta, max_beta)

    def sample_forward(self, x, t, eps=None):
        alpha_bar = self.alpha_bars[t].reshape(-1, 1, 1)  # alpha_bar是一个标量,作用是控制噪声的强度,与t的关系是线性的
        # 计算x(32,1,512)中每一条信号(1,1,512)的power,得到(32,)
        x_power = torch.mean(x ** 2, dim=(1, 2))
        # 将x_power的维度变为(32,1,512)
        if eps is None:
            eps = make_noise(t)  # 生成噪声,此时的x是纯净的信号
        # 计算真实的噪声, x_power的维度是(32,)，eps的维度是(32,1,512), x_power的每行开根号和eps相应的行相乘
        for i in range(x_power.shape[0]):
            eps[i] = torch.sqrt(x_power[i]) * eps[i]  # 此时的噪声是真实的噪声

        res = eps * torch.sqrt(1 - alpha_bar) + torch.sqrt(alpha_bar) * x
        # res = eps + x
        # 生成带噪声的信号,信噪比sqrt((1-alpha_bar)/alpha_bar)
        return res

    def sample_backward(self,
                        original_signal,
                        net,
                        device,
                        simple_var=True,
                        ddim_step=20,
                        eta=1):
        ts = torch.linspace(self.n_steps, 0,
                            (ddim_step + 1)).to(device).to(torch.long)
        x = original_signal  # 输入原始带噪声信号，开始去噪
        batch_size = len(x)
        net = net.to(device)

        x = tensor(x).to(device)
        for i in range(1, ddim_step + 1):
            cur_t = ts[i - 1] - 1
            prev_t = ts[i] - 1

            ab_cur = self.alpha_bars[cur_t]
            ab_prev = self.alpha_bars[prev_t] if prev_t >= 0 else 1

            t_tensor = torch.tensor([cur_t] * batch_size,
                                    dtype=torch.long).to(device).unsqueeze(1)
            eps = net(x, t_tensor)
            var = eta * (1 - ab_prev) / (1 - ab_cur) * (1 - ab_cur / ab_prev)
            noise = make_noise(cur_t).to(device)

            first_term = (ab_prev / ab_cur) ** 0.5 * x
            second_term = ((1 - ab_prev - var) ** 0.5 -
                           (ab_prev * (1 - ab_cur) / ab_cur) ** 0.5) * eps
            if simple_var:
                third_term = (1 - ab_cur / ab_prev) ** 0.5 * noise
            else:
                third_term = var ** 0.5 * noise
            x = first_term + second_term + third_term

            # 保持x的均值为0左右
            x = x - torch.mean(x)
            x = x / torch.max(torch.abs(x))  # 归一化

        return x
