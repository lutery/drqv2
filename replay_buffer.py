# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import datetime
import io
import random
import traceback
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset


def episode_len(episode):
    # subtract -1 because the dummy first transition
    return next(iter(episode.values())).shape[0] - 1


def save_episode(episode, fn):
    '''
    episode: 包含一个完整游戏过程的数据，包含观察、动作、奖励、折扣等，是一个字典数据类型
    fn: 游戏存储的文件路径
    '''
    with io.BytesIO() as bs:
        np.savez_compressed(bs, **episode) # 将数据先压缩保存到内存中
        bs.seek(0)
        with fn.open('wb') as f:
            f.write(bs.read()) # 然后将数据写入到文件中


def load_episode(fn):
    with fn.open('rb') as f:
        episode = np.load(f)
        episode = {k: episode[k] for k in episode.keys()}
        return episode


class ReplayBufferStorage:
    '''
    重放缓冲区，负责存储和管理采集的数据，按连续的episode存储
    '''

    def __init__(self, data_specs, replay_dir):
        '''
        data_specs:
                # create replay buffer
                (self.train_env.observation_spec(),
                self.train_env.action_spec(),
                specs.Array((1,), np.float32, 'reward'),
                specs.Array((1,), np.float32, 'discount'))
        '''
        self._data_specs = data_specs
        self._replay_dir = replay_dir
        replay_dir.mkdir(exist_ok=True)
        # 存储采集的数据
        # 外层是一个字典，键是数据的名称，值是一个列表，存储该名称对应的数据
        # 列表存储的都是numpy数组，向量
        self._current_episode = defaultdict(list)
        self._preload()

    def __len__(self):
        return self._num_transitions # 返回已经存储的总采集数据量

    def add(self, time_step):
        # 遍历所有环境的数据规格
        for spec in self._data_specs:
            # 提取对应的数据
            value = time_step[spec.name]
            if np.isscalar(value):
                # 如果是标量，则转换为数组
                value = np.full(spec.shape, value, spec.dtype)
            assert spec.shape == value.shape and spec.dtype == value.dtype
            # 将数据添加到当前的episode中
            self._current_episode[spec.name].append(value)
        if time_step.last():
            # 如果是最后一个时间步，则将当前的episode存储到文件中
            # 外层是一个字典，内层是一个numpy array
            episode = dict()
            for spec in self._data_specs:
                value = self._current_episode[spec.name]
                episode[spec.name] = np.array(value, spec.dtype)
            self._current_episode = defaultdict(list) # 重置当前的episode
            self._store_episode(episode) # 将episode存储到文件中

    def _preload(self):
        self._num_episodes = 0 # 记录已经存储的episode数量
        self._num_transitions = 0
        for fn in self._replay_dir.glob('*.npz'):
            _, _, eps_len = fn.stem.split('_')
            self._num_episodes += 1
            self._num_transitions += int(eps_len)

    def _store_episode(self, episode):
        '''
        episode: 包含一个完整游戏过程的数据，包含观察、动作、奖励、折扣等，是一个字典数据类型
        '''
        eps_idx = self._num_episodes
        eps_len = episode_len(episode) # 计算episode的长度
        self._num_episodes += 1 # 更新episode数量
        self._num_transitions += eps_len
        ts = datetime.datetime.now().strftime('%Y%m%dT%H%M%S') # 获取当前时间
        eps_fn = f'{ts}_{eps_idx}_{eps_len}.npz' # 构造文件名，看起来是每一个episode一个文件
        save_episode(episode, self._replay_dir / eps_fn)


class ReplayBuffer(IterableDataset):
    def __init__(self, replay_dir, max_size, num_workers, nstep, discount,
                 fetch_every, save_snapshot):
        '''
        replay_dir: 重放缓冲区的存储路径
        max_size: 重放缓冲区的最大容量
        num_workers: 用于数据加载的工作线程数量
        nstep: 多步回报的步数
        discount: 折扣因子
        fetch_every: 每隔多少次采样尝试从存储目录中获取新的数据
        save_snapshot: 是否保存采集的原始数据文件
        '''
        self._replay_dir = replay_dir
        self._size = 0
        self._max_size = max_size
        self._num_workers = max(1, num_workers)
        self._episode_fns = []
        self._episodes = dict()
        self._nstep = nstep
        self._discount = discount
        self._fetch_every = fetch_every
        self._samples_since_last_fetch = fetch_every
        self._save_snapshot = save_snapshot

    def _sample_episode(self):
        eps_fn = random.choice(self._episode_fns)
        return self._episodes[eps_fn]

    def _store_episode(self, eps_fn):
        try:
            episode = load_episode(eps_fn)
        except:
            return False
        eps_len = episode_len(episode)
        while eps_len + self._size > self._max_size:
            early_eps_fn = self._episode_fns.pop(0)
            early_eps = self._episodes.pop(early_eps_fn)
            self._size -= episode_len(early_eps)
            early_eps_fn.unlink(missing_ok=True)
        self._episode_fns.append(eps_fn)
        self._episode_fns.sort()
        self._episodes[eps_fn] = episode
        self._size += eps_len

        if not self._save_snapshot:
            eps_fn.unlink(missing_ok=True)
        return True

    def _try_fetch(self):
        if self._samples_since_last_fetch < self._fetch_every:
            return
        self._samples_since_last_fetch = 0
        try:
            worker_id = torch.utils.data.get_worker_info().id
        except:
            worker_id = 0
        eps_fns = sorted(self._replay_dir.glob('*.npz'), reverse=True)
        fetched_size = 0
        for eps_fn in eps_fns:
            eps_idx, eps_len = [int(x) for x in eps_fn.stem.split('_')[1:]]
            if eps_idx % self._num_workers != worker_id:
                continue
            if eps_fn in self._episodes.keys():
                break
            if fetched_size + eps_len > self._max_size:
                break
            fetched_size += eps_len
            if not self._store_episode(eps_fn):
                break

    def _sample(self):
        try:
            self._try_fetch()
        except:
            traceback.print_exc()
        self._samples_since_last_fetch += 1
        episode = self._sample_episode()
        # add +1 for the first dummy transition
        idx = np.random.randint(0, episode_len(episode) - self._nstep + 1) + 1
        obs = episode['observation'][idx - 1]
        action = episode['action'][idx]
        next_obs = episode['observation'][idx + self._nstep - 1]
        reward = np.zeros_like(episode['reward'][idx])
        discount = np.ones_like(episode['discount'][idx])
        for i in range(self._nstep):
            step_reward = episode['reward'][idx + i]
            reward += discount * step_reward
            discount *= episode['discount'][idx + i] * self._discount
        return (obs, action, reward, discount, next_obs)

    def __iter__(self):
        while True:
            yield self._sample()


def _worker_init_fn(worker_id):
    seed = np.random.get_state()[1][0] + worker_id
    np.random.seed(seed)
    random.seed(seed)


def make_replay_loader(replay_dir, max_size, batch_size, num_workers,
                       save_snapshot, nstep, discount):
    '''
    todo 

    replay_dir: 重放缓冲区的存储路径
    max_size: 重放缓冲区的最大容量
    batch_size: 每个批次的样本数量
    num_workers: 用于数据加载的工作线程数量
    save_snapshot: 是否保存采集的原始数据文件
    nstep: 多步回报的步数
    discount: 折扣因子
    该函数创建并返回一个数据加载器，用于从重放缓冲区中采样数据
    该数据加载器是一个可迭代对象，可以在训练过程中使用
    该数据加载器会自动处理多线程数据加载和随机采样
    该数据加载器会定期从重放缓冲区中获取新的数据
    该数据加载器会根据指定的nstep和discount计算多步回报
    该数据加载器会返回一个批次的数据，包含观察、动作、奖励、折扣和下一个观察
    该数据加载器适用于强化学习中的经验回放
    '''
    max_size_per_worker = max_size // max(1, num_workers)

    iterable = ReplayBuffer(replay_dir,
                            max_size_per_worker,
                            num_workers,
                            nstep,
                            discount,
                            fetch_every=1000,
                            save_snapshot=save_snapshot)

    loader = torch.utils.data.DataLoader(iterable,
                                         batch_size=batch_size,
                                         num_workers=num_workers,
                                         pin_memory=True, # 将数据固定在内存中，加速从 CPU 到 GPU 的数据传输。因为有可能因为分页内存地址数据存储到硬盘中
                                         worker_init_fn=_worker_init_fn) # 为每个数据加载工作线程设置独立的随机种子，确保多线程采样的随机性
    return loader
