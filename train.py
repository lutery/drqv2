# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import warnings
warnings.filterwarnings('ignore', category=DeprecationWarning)

import os
os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from pathlib import Path

import hydra
import numpy as np
import torch
from dm_env import specs

import dmc
import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    '''
    obs_spec: 环境的观察空间规格
    action_spec: 环境的动作空间规格
    cfg: agent 的配置参数
    '''
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape
    return hydra.utils.instantiate(cfg) # drqv2.DrQV2Agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.agent = make_agent(self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                self.cfg.agent)
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0 # 记录训练了多少个完整的游戏过程

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        self.train_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                  self.cfg.action_repeat, self.cfg.seed)
        self.eval_env = dmc.make(self.cfg.task_name, self.cfg.frame_stack,
                                 self.cfg.action_repeat, self.cfg.seed)
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # 构建重放缓冲区
        self.replay_storage = ReplayBufferStorage(data_specs,
                                                  self.work_dir / 'buffer')

        self.replay_loader = make_replay_loader(
            self.work_dir / 'buffer', self.cfg.replay_buffer_size,
            self.cfg.batch_size, self.cfg.replay_buffer_num_workers,
            self.cfg.save_snapshot, self.cfg.nstep, self.cfg.discount)
        self._replay_iter = None

        # 构建视频记录器，记录验证时的视频
        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None)
        # 记录训练过程中的视频
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None)


    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        '''
        返回游戏经过的总帧数
        '''
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval(self):
        step, episode, total_reward = 0, 0, 0
        # self.cfg.num_eval_episodes： 评估时进行多少个完整的游戏过程
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0)) # 记录第一帧
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    action = self.agent.act(time_step.observation,
                                            self.global_step,
                                            eval_mode=True)
                time_step = self.eval_env.step(action) # 执行动作
                self.video_recorder.record(self.eval_env) # 将当前帧渲染为图像，添加到视频中
                total_reward += time_step.reward
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode) # 验证过程中每个游戏过程的平均奖励
            log('episode_length', step * self.cfg.action_repeat / episode) # 验证过程中每个游戏过程的平均帧数
            log('episode', self.global_episode) # 训练了多少个完整的游戏过程
            log('step', self.global_step) # 训练了多少个动作

    def train(self):
        '''
        开始进行训练
        '''
        # predicates todo 这是做啥的？
        train_until_step = utils.Until(self.cfg.num_train_frames, # 总训练帧数 作用: 控制整个训练过程的总长度
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames, # 随机探索帧数 作用: 在训练初期进行纯随机探索，收集初始经验，只有超过这个帧数才开始更新网络
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames, # 评估频率 作用: 控制多久进行一次模型评估
                                      self.cfg.action_repeat)

        episode_step, episode_reward = 0, 0 # episode_step 记录当前游戏过程的步数，episode_reward 记录当前游戏过程的总奖励
        time_step = self.train_env.reset()
        self.replay_storage.add(time_step)
        self.train_video_recorder.init(time_step.observation)
        metrics = None
        while train_until_step(self.global_step): # 限制在总训练帧数内
            if time_step.last(): 
                # 判断当前时间步是否为一个游戏生命周期的结束
                self._global_episode += 1
                self.train_video_recorder.save(f'{self.global_frame}.mp4') # 如果本轮游戏结束，保存训练视频
                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats 
                    # elapsed_time - 从上次记录到现在经过的时间
                    # total_time - 从训练开始到现在的总时间
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat # 当前游戏过程的总帧数，因为每个动作会被重复执行多次，也会经过多帧渲染
                    with self.logger.log_and_dump_ctx(self.global_frame,
                                                      ty='train') as log:
                        # 记录本轮训练的各种指标
                        log('fps', episode_frame / elapsed_time)
                        log('total_time', total_time)
                        log('episode_reward', episode_reward)
                        log('episode_length', episode_frame)
                        log('episode', self.global_episode)
                        log('buffer_size', len(self.replay_storage))
                        log('step', self.global_step)

                # reset env
                time_step = self.train_env.reset() # 重置环境，开始新的游戏过程
                self.replay_storage.add(time_step) # 将重置后的初始时间步添加到重放缓冲区
                self.train_video_recorder.init(time_step.observation) # 初始化训练视频记录器
                # try to save snapshot
                if self.cfg.save_snapshot:
                    # 是否保存训练状态的快照
                    self.save_snapshot()
                episode_step = 0 # 重置当前游戏过程的步数
                episode_reward = 0 # 重置当前游戏过程的总奖励

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                # 验证模型
                self.eval()

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        self.global_step,
                                        eval_mode=False) # 预测动作

            # try to update the agent
            if not seed_until_step(self.global_step):
                # 只有超过 seed 阶段才开始网络更新
                metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # take env step
            time_step = self.train_env.step(action) # 执行动作，获得下一个时间步
            episode_reward += time_step.reward # 记录当前游戏过程的总奖励
            self.replay_storage.add(time_step) # 将当前时间步添加到重放缓冲区
            self.train_video_recorder.record(time_step.observation) # 记录当前帧
            episode_step += 1
            self._global_step += 1

    def save_snapshot(self):
        # 创建训练状态的快照文件
        snapshot = self.work_dir / 'snapshot.pt'
        # 记录要保存的成员变量
        keys_to_save = ['agent', 'timer', '_global_step', '_global_episode']
        # 获取要保存的成员变量的值，并保存到文件中
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / 'snapshot.pt'
        with snapshot.open('rb') as f:
            payload = torch.load(f) # 加载保存的训练状态
        for k, v in payload.items(): # 并将数据恢复到当前的训练状态中
            self.__dict__[k] = v


@hydra.main(config_path='cfgs', config_name='config')
def main(cfg):
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt' # 恢复训练
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()