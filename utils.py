# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import random
import re
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch import distributions as pyd
from torch.distributions.utils import _standard_normal


class eval_mode:
    '''
    将模型切换为评估模式
    '''
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data.copy_(tau * param.data +
                                (1 - tau) * target_param.data)


def to_torch(xs, device):
    return tuple(torch.as_tensor(x, device=device) for x in xs)


def weight_init(m):
    '''
    权重初始化

    偏置设置为0
    权重设置为 
    '''
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data) # 创建正交矩阵来初始化神经网络的权重
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data, gain)
        if hasattr(m.bias, 'data'):
            m.bias.data.fill_(0.0)


class Until:
    def __init__(self, until, action_repeat=1):
        self._until = until
        self._action_repeat = action_repeat

    def __call__(self, step):
        # 如果self._until为None，表示没有限制，总是返回True
        if self._until is None:
            return True
        # 将总的时间步数转换为动作时间步数，因为每个动作会被重复执行多次
        until = self._until // self._action_repeat
        return step < until # 如果当前的动作时间步数小于设定的限制，则返回True，否则返回False


class Every:
    def __init__(self, every, action_repeat=1):
        self._every = every
        self._action_repeat = action_repeat

    def __call__(self, step):
        if self._every is None:
            return False
        every = self._every // self._action_repeat
        if step % every == 0:
            return True
        return False


class Timer:
    def __init__(self):
        self._start_time = time.time()
        self._last_time = time.time()

    def reset(self):
        elapsed_time = time.time() - self._last_time # 记录从上次重置到现在的时间
        self._last_time = time.time() # 更新上次重置的时间为当前时间
        total_time = time.time() - self._start_time # 记录从计时器开始到现在的总时间
        return elapsed_time, total_time

    def total_time(self):
        return time.time() - self._start_time


class TruncatedNormal(pyd.Normal):
    '''
    截断正太分布
    '''
    def __init__(self, loc, scale, low=-1.0, high=1.0, eps=1e-6):
        '''
        loc: 均值
        scale: 方差的缩放倍数
        low,hight: 截断的范围
        eps： 防止边界值的小范围
        '''
        super().__init__(loc, scale, validate_args=False)
        self.low = low
        self.high = high
        self.eps = eps

    def _clamp(self, x):
        # 硬裁剪到指定的范围
        clamped_x = torch.clamp(x, self.low + self.eps, self.high - self.eps)
        # 梯度技巧：保持前向传播的裁剪结果，但梯度仍然流过原始值
        # 因为使用了clamp，所以直接用无法保证梯度的传递
        x = x - x.detach() + clamped_x.detach()
        return x

    def sample(self, clip=None, sample_shape=torch.Size()):
        '''
        clip: 是否进行方差进行裁剪，防止采样过大
        '''
        shape = self._extended_shape(sample_shape) # 控制采样的batch，也是为了能够控制采样的batch
        # 生成标准的正态分布噪声
        eps = _standard_normal(shape,
                               dtype=self.loc.dtype,
                               device=self.loc.device)
        eps *= self.scale # 缩放噪声
        if clip is not None:
            # 进行额外的方差裁剪
            eps = torch.clamp(eps, -clip, clip)
        # 将随机噪声的值加入到均值中
        x = self.loc + eps
        return self._clamp(x)


def schedule(schdl, step):
    '''
    函数用于实现动态参数调度，在强化学习训练过程中根据训练步数动态调整超参数。
    '''
    try:
        return float(schdl) # schedule("0.1", step)  # 返回 0.1，不随步数变化
    except ValueError:
        # 线性调度 - linear(init, final, duration) schedule("linear(1.0, 0.1, 10000)", step)
        match = re.match(r'linear\((.+),(.+),(.+)\)', schdl)
        if match:
            init, final, duration = [float(g) for g in match.groups()]
            mix = np.clip(step / duration, 0.0, 1.0)
            return (1.0 - mix) * init + mix * final
        # 分段线性调度 - step_linear(init, final1, duration1, final2, duration2)
        # schedule("step_linear(1.0, 0.3, 5000, 0.05, 15000)", step)
        match = re.match(r'step_linear\((.+),(.+),(.+),(.+),(.+)\)', schdl)
        if match:
            init, final1, duration1, final2, duration2 = [
                float(g) for g in match.groups()
            ]
            if step <= duration1:
                mix = np.clip(step / duration1, 0.0, 1.0)
                return (1.0 - mix) * init + mix * final1
            else:
                mix = np.clip((step - duration1) / duration2, 0.0, 1.0)
                return (1.0 - mix) * final1 + mix * final2
    raise NotImplementedError(schdl)
