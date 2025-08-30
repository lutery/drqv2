# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import hydra
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils


class RandomShiftsAug(nn.Module):
    '''
    随机偏移裁剪
    '''

    def __init__(self, pad):
        super().__init__()
        self.pad = pad # 填充的像素数

    def forward(self, x):
        n, c, h, w = x.size()
        assert h == w
        padding = tuple([self.pad] * 4)
        x = F.pad(x, padding, 'replicate')
        eps = 1.0 / (h + 2 * self.pad)
        arange = torch.linspace(-1.0 + eps,
                                1.0 - eps,
                                h + 2 * self.pad,
                                device=x.device,
                                dtype=x.dtype)[:h]
        arange = arange.unsqueeze(0).repeat(h, 1).unsqueeze(2)
        base_grid = torch.cat([arange, arange.transpose(1, 0)], dim=2)
        base_grid = base_grid.unsqueeze(0).repeat(n, 1, 1, 1)

        shift = torch.randint(0,
                              2 * self.pad + 1,
                              size=(n, 1, 1, 2),
                              device=x.device,
                              dtype=x.dtype)
        shift *= 2.0 / (h + 2 * self.pad)

        grid = base_grid + shift
        return F.grid_sample(x,
                             grid,
                             padding_mode='zeros',
                             align_corners=False)


class Encoder(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()

        assert len(obs_shape) == 3
        self.repr_dim = 32 * 35 * 35

        # 构建网络，结果采用的是每个just conv2d + relu
        # 经过convnet后，图像尺寸将下降 一半
        self.convnet = nn.Sequential(nn.Conv2d(obs_shape[0], 32, 3, stride=2), # 尺寸减半
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), # 尺寸不变
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), # 尺寸不变
                                     nn.ReLU(), nn.Conv2d(32, 32, 3, stride=1), # 尺寸不变
                                     nn.ReLU())

        self.apply(utils.weight_init)

    def forward(self, obs):
        obs = obs / 255.0 - 0.5 # 归一化到 [-0.5, 0.5]
        h = self.convnet(obs) # [B, 32, 35, 35]
        h = h.view(h.shape[0], -1) # 展平 [B, 32*35*35]
        return h


class Actor(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        '''
        repr_dim: 编码器输出的特征维度
        action_shape: 动作空间的形状 (action_dim,)
        feature_dim: 特征层的维度
        hidden_dim: 隐藏层的维度
        该类实现了 DrQ-v2 算法中的策略网络
        '''
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        self.policy = nn.Sequential(nn.Linear(feature_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, hidden_dim),
                                    nn.ReLU(inplace=True),
                                    nn.Linear(hidden_dim, action_shape[0]))

        self.apply(utils.weight_init)

    def forward(self, obs, std):
        # 提取特征
        h = self.trunk(obs)

        mu = self.policy(h) 
        mu = torch.tanh(mu) # 预测动作的均值
        std = torch.ones_like(mu) * std #todo 标准差是外面传进来的？

        dist = utils.TruncatedNormal(mu, std)
        return dist # 返回截断正态分布


class Critic(nn.Module):
    def __init__(self, repr_dim, action_shape, feature_dim, hidden_dim):
        '''
        repr_dim: 编码器输出的特征维度
        action_shape: 动作空间的形状 (action_dim,)
        feature_dim: 特征层的维度
        hidden_dim: 隐藏层的维度

        预测的是两个 Q 值
        '''
        super().__init__()

        self.trunk = nn.Sequential(nn.Linear(repr_dim, feature_dim),
                                   nn.LayerNorm(feature_dim), nn.Tanh())

        # 结合环境的特征和动作，输入到两个独立的Q网络中
        self.Q1 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.Q2 = nn.Sequential(
            nn.Linear(feature_dim + action_shape[0], hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True), nn.Linear(hidden_dim, 1))

        self.apply(utils.weight_init)

    def forward(self, obs, action):
        h = self.trunk(obs)
        h_action = torch.cat([h, action], dim=-1)
        q1 = self.Q1(h_action)
        q2 = self.Q2(h_action)

        return q1, q2


class DrQV2Agent:
    def __init__(self, obs_shape, action_shape, device, lr, feature_dim,
                 hidden_dim, critic_target_tau, num_expl_steps,
                 update_every_steps, stddev_schedule, stddev_clip, use_tb):
        '''
        obs_shape: 观察空间的形状 (C, H, W)
        action_shape: 动作空间的形状 (action_dim,)
        device: 计算设备 (CPU 或 GPU)
        lr: 学习率
        feature_dim: 编码器输出的特征维度
        hidden_dim: 隐藏层的维度
        critic_target_tau: 软更新目标网络的系数
        num_expl_steps: 探索阶段的步数 todo
        update_every_steps: 每隔多少步更新一次网络
        stddev_schedule: 动作噪声的标准差调度 todo
        stddev_clip: 动作噪声的裁剪范围
        use_tb: 是否使用 TensorBoard 记录训练过程
        该类实现了 DrQ-v2 强化学习算法的主体逻辑，包括网络的构建、动作选择、网络更新等功能
        该类使用了数据增强技术来提高训练的稳定性和性能
        该类包含了一个编码器、一个策略网络和一个双重 Q 网络
        该类使用了目标网络来稳定 Q 值的估计
        该类使用了 Adam 优化器来更新网络参数
        该类支持在训练过程中记录各种指标，以便于分析和调试
        该类可以在训练和评估模式之间切换
        该类可以根据当前的训练步数动态调整动作噪声的标准
        '''
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.update_every_steps = update_every_steps
        self.use_tb = use_tb
        self.num_expl_steps = num_expl_steps
        self.stddev_schedule = stddev_schedule
        self.stddev_clip = stddev_clip # 控制方差的大小范围，防止方差偏差太大导致采样的动作太离谱

        # models
        self.encoder = Encoder(obs_shape).to(device)
        self.actor = Actor(self.encoder.repr_dim, action_shape, feature_dim,
                           hidden_dim).to(device)

        # 构建价值网络和目标价值网络
        self.critic = Critic(self.encoder.repr_dim, action_shape, feature_dim,
                             hidden_dim).to(device)
        self.critic_target = Critic(self.encoder.repr_dim, action_shape,
                                    feature_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # optimizers
        self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # data augmentation 又是一个随机裁剪
        self.aug = RandomShiftsAug(pad=4)

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, step, eval_mode):
        '''
        obs: 观察 (C, H, W)
        step: 当前的训练步数
        eval_mode: 是否处于评估模式
        '''
        obs = torch.as_tensor(obs, device=self.device)
        obs = self.encoder(obs.unsqueeze(0))
        stddev = utils.schedule(self.stddev_schedule, step) # 根据step动态调整探索噪声
        dist = self.actor(obs, stddev) # 预测动作的分布
        if eval_mode:
            action = dist.mean # 如果是验证模式，直接取均值
        else:
            action = dist.sample(clip=None) # 训练模式，采样动作
            if step < self.num_expl_steps:
                action.uniform_(-1.0, 1.0) # 如果在探索阶段，强制随机探索，也就说这里还是随机采样？
        return action.cpu().numpy()[0]

    def update_critic(self, obs, action, reward, discount, next_obs, step):
        '''
        更新评价模型
        obs: 当前状态的特征表示 [B, repr_dim]
        action: 当前动作 [B, action_dim]
        reward: 当前奖励 [B, 1]
        discount: 折扣因子 [B, 1]
        next_obs: 下一个状态的特征表示 [B, repr_dim]
        step: 当前的训练步数
        该函数实现了 DrQ-v2 算法中评价网络的更新逻辑
        该函数首先计算目标 Q 值，然后计算当前 Q 值的损失，并通过反向传播更新网络参数
        该函数使用了目标网络来计算目标 Q 值，以提高训练的稳定性
        该函数还更新了编码器的参数，以便更好地提取状态特征
        该函数返回了若干指标，以便于在训练过程中进行监控和分析
        该函数使用了数据增强技术来提高训练的稳定性和性能
        该函数使用了均方误损失来度量 Q 值的误差
        该函数使用了双重 Q 学习来减轻过估计偏差
        该函数使用了 Adam 优化器来更新网络参数
        该函数支持在训练过程中记录各种指标，以便于分析和调试
        '''
        metrics = dict()

        with torch.no_grad():
            stddev = utils.schedule(self.stddev_schedule, step)
            dist = self.actor(next_obs, stddev) # 预测下一个状态的动作分布
            next_action = dist.sample(clip=self.stddev_clip) # 采样下一个动作
            target_Q1, target_Q2 = self.critic_target(next_obs, next_action) # 计算目标Q值
            target_V = torch.min(target_Q1, target_Q2) # 双重Q学习，取较小值
            target_Q = reward + (discount * target_V) # 计算目标Q值

        Q1, Q2 = self.critic(obs, action) # 计算当前Q值，真实的obs和action
        critic_loss = F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q) # 预测的两个Q值都要和目标Q值计算损失

        if self.use_tb:
            metrics['critic_target_q'] = target_Q.mean().item()
            metrics['critic_q1'] = Q1.mean().item()
            metrics['critic_q2'] = Q2.mean().item()
            metrics['critic_loss'] = critic_loss.item()

        # optimize encoder and critic
        self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        self.encoder_opt.step()

        return metrics

    def update_actor(self, obs, step):
        '''
        obs: 这里obs采用了detach，因为这里不去更新encoder
        '''
        metrics = dict()

        stddev = utils.schedule(self.stddev_schedule, step) # 感觉是不是drqv2的特征里面，std是手动指定的？
        dist = self.actor(obs, stddev) # 获取动作的分布
        action = dist.sample(clip=self.stddev_clip) # 采样动作
        log_prob = dist.log_prob(action).sum(-1, keepdim=True) # 将动作概率的乘积转换为相加，并保持dim，比如 保持维度: (batch_size,) → (batch_size, 1)，但是这里并没有使用，仅作为记录，监控算法的稳定性
        Q1, Q2 = self.critic(obs, action) # 将当前的obs，和预测的动作传递得到评价
        Q = torch.min(Q1, Q2) # 取最小的Q值

        actor_loss = -Q.mean() # 得到动作损失

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics['actor_loss'] = actor_loss.item()
            metrics['actor_logprob'] = log_prob.mean().item()
            metrics['actor_ent'] = dist.entropy().sum(dim=-1).mean().item()

        return metrics

    def update(self, replay_iter, step):
        metrics = dict()

        if step % self.update_every_steps != 0:
            # 如果不到更新间隔，则不进行更新
            return metrics

        batch = next(replay_iter) # 从经验回放缓冲区中采样一个批次的数据
        obs, action, reward, discount, next_obs = utils.to_torch(
            batch, self.device) # 解包batch，并转换为torch

        # augment
        obs = self.aug(obs.float()) # 先将obs转换为float类型，再进行数据增强
        next_obs = self.aug(next_obs.float()) # 同上
        # encode
        obs = self.encoder(obs) # 提取特征
        with torch.no_grad():
            next_obs = self.encoder(next_obs) # 提取特征

        if self.use_tb:
            metrics['batch_reward'] = reward.mean().item() # 记录奖励的均值，并记录到metrics中

        # update critic
        metrics.update(
            self.update_critic(obs, action, reward, discount, next_obs, step))

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update critic target
        utils.soft_update_params(self.critic, self.critic_target,
                                 self.critic_target_tau)

        return metrics
