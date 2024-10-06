import warnings

warnings.filterwarnings('ignore', category=DeprecationWarning)

import os

os.environ['MKL_SERVICE_FORCE_INTEL'] = '1'
os.environ['MUJOCO_GL'] = 'egl'

from env_helper import make_envs
from pathlib import Path
from env.env_list import no_video_eval_list, save_image_eval_list, plot_prediction_list

import hydra
import numpy as np
import torch
import wandb
from dm_env import specs

import utils
from logger import Logger
from replay_buffer import ReplayBufferStorage, make_replay_loader
from video import TrainVideoRecorder, VideoRecorder

torch.backends.cudnn.benchmark = True

from utils import make_agent


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f'workspace: {self.work_dir}')

        self.cfg = cfg

        if "diayn" in self.cfg.agent.name:  # or self.cfg.agent.name == "cic":
            self.use_diayn = True

        else:
            self.use_diayn = False
            # For other methods, we currently don't supoort multi-env training
            assert self.cfg.n_env == 1

        utils.set_seed_everywhere(cfg.seed)

        # create logger
        if cfg.use_wandb:
            project = cfg.domain + "_" + cfg.agent.name
            self.run = wandb.init(project=project, group=cfg.exp_group, name=cfg.experiment)

        if self.use_diayn:
            cfg.discount = 0.95

        self.logger = Logger(self.work_dir,
                             use_tb=cfg.use_tb,
                             use_wandb=cfg.use_wandb)
        self.train_env, self.eval_env, self.use_gym = make_envs(cfg)

        # create agent
        self.agent = make_agent(cfg.obs_type,
                                self.train_env.observation_spec(),
                                self.train_env.action_spec(),
                                cfg.num_seed_frames // cfg.action_repeat,
                                cfg,
                                cfg.agent)

        if self.cfg.agent.name in ["dusdi_diayn"]:
            from omegaconf import OmegaConf, open_dict
            with open_dict(cfg):
                cfg.agent.skill_channel = self.agent.diayn_skill_channel
                cfg.agent.diayn_skill_channel = self.agent.diayn_skill_channel

        # get meta specs
        meta_specs = self.agent.get_meta_specs()
        # create replay buffer
        data_specs = (self.train_env.observation_spec(),
                      self.train_env.action_spec(),
                      specs.Array((1,), np.float32, 'reward'),
                      specs.Array((1,), np.float32, 'discount'))

        # create data storage
        self.replay_storage = ReplayBufferStorage(data_specs, meta_specs,
                                                  self.work_dir / 'buffer', cfg.n_env, cfg)

        # create replay buffer
        self.replay_loader = make_replay_loader(self.replay_storage,
                                                cfg.replay_buffer_size,
                                                cfg.batch_size,
                                                cfg.replay_buffer_num_workers,
                                                False, cfg.nstep, cfg.discount)
        self._replay_iter = None

        # create video recorders
        self.video_recorder = VideoRecorder(
            self.work_dir if cfg.save_video else None,
            camera_id=0,
            use_wandb=self.cfg.use_wandb)
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if cfg.save_train_video else None,
            camera_id=0,
            use_wandb=self.cfg.use_wandb)

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    @property
    def replay_iter(self):
        if self._replay_iter is None:
            self._replay_iter = iter(self.replay_loader)
        return self._replay_iter

    def eval_diayn(self):
        if self.cfg.num_eval_episodes <= 0:
            return

        step, episode, total_reward = 0, 0, 0
        # This function is customized for diayn & multi-diayn
        if self.cfg.domain in no_video_eval_list:
            assert self.cfg.save_video is False
            self.save_dir = self.work_dir / 'eval_video'
            self.save_dir.mkdir(exist_ok=True)

        obs_list = []
        z_list = []

        for z in range(self.cfg.num_eval_episodes):
            z = self.agent.get_random_skill()
            meta = self.agent.get_meta_from_skill(z, num_envs=1)
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env)
            ep_step = 0
            cut_step = self.cfg.agent.update_skill_every_step

            while not time_step.last() and ep_step < cut_step:
                with torch.no_grad(), utils.eval_mode(self.agent):
                    input_obs = time_step.observation
                    action = self.agent.act(input_obs,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                action = action.flatten()
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward.mean()
                step += 1
                ep_step += 1
                obs_list.append(time_step.observation)
                z_list.append(z)

            episode += 1
            if isinstance(z, np.ndarray):
                z = "".join(str(a) for a in z)
            self.video_recorder.save(f'{self.global_frame}_skill_{z}.gif')

            if self.cfg.domain in save_image_eval_list:
                file_name = f'step={self.global_frame}_skill={z}.png'
                path = self.save_dir / file_name
                self.eval_env.save_traj(path)

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)


    def eval(self):
        step, episode, total_reward = 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        if self.cfg.num_eval_episodes == 0:
            return

        meta = self.agent.init_meta()
        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(self.agent):
                    input_obs = time_step.observation
                    action = self.agent.act(input_obs,
                                            meta,
                                            self.global_step,
                                            eval_mode=True)
                action = action.flatten()
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                total_reward += time_step.reward.mean()
                step += 1

            episode += 1
            self.video_recorder.save(f'{self.global_frame}.mp4')

        with self.logger.log_and_dump_ctx(self.global_frame, ty='eval') as log:
            log('episode_reward', total_reward / episode)
            log('episode_length', step * self.cfg.action_repeat / episode)
            log('episode', self.global_episode)
            log('step', self.global_step)

    def train(self):
        # predicates
        train_until_step = utils.Until(self.cfg.num_train_frames,
                                       self.cfg.action_repeat)
        seed_until_step = utils.Until(self.cfg.num_seed_frames,
                                      self.cfg.action_repeat)
        eval_every_step = utils.Every(self.cfg.eval_every_frames,
                                      self.cfg.action_repeat)
        n_env = self.cfg.n_env
        episode_step, episode_reward = 0, 0
        time_step = self.train_env.reset()
        if self.use_diayn:
            meta = self.agent.init_meta(n_env)
        else:
            meta = self.agent.init_meta()
        self.replay_storage.add(time_step, meta)
        self.train_video_recorder.init(time_step.observation)
        metrics = None

        n_updates = self.cfg.n_env // self.cfg.update_ratio

        while train_until_step(self.global_step):
            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log('eval_total_time', self.timer.total_time(),
                                self.global_frame)
                if self.use_diayn:
                    self.eval_diayn()
                else:
                    self.eval()

            if self.use_diayn:
                meta = self.agent.update_meta(meta, episode_step // n_env, time_step, n_env, self.global_frame)
            else:
                meta = self.agent.update_meta(meta, episode_step, time_step)

            # sample action
            with torch.no_grad(), utils.eval_mode(self.agent):
                action = self.agent.act(time_step.observation,
                                        meta,
                                        self.global_step,
                                        eval_mode=False)

            # take env step
            if self.cfg.n_env == 1:
                action = action.flatten()
            time_step = self.train_env.step(action)
            episode_reward += time_step.reward.mean()
            episode_step += n_env
            self._global_step += n_env

            if time_step.last():
                self._global_episode += n_env
                self.train_video_recorder.save(f'{self.global_frame}.mp4')

                if self.use_gym and n_env > 1:
                    # Handle automatic reset by vectorized environment
                    info = self.train_env.info
                    terminal_obs = np.array([info[i]["terminal_observation"] for i in range(n_env)])
                    terminal_time_step = time_step._replace(observation=terminal_obs)
                    self.replay_storage.add(terminal_time_step, meta)
                    # Create new timestep
                    time_step = self.train_env.generate_reset_observation(time_step.observation)
                    meta = self.agent.init_meta(n_env)
                else:
                    self.replay_storage.add(time_step, meta)
                    time_step = self.train_env.reset()
                    if self.use_diayn:
                        meta = self.agent.init_meta(n_env)
                    else:
                        meta = self.agent.init_meta()

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    if self.global_frame % self.agent.update_every_steps == 0:
                        with self.logger.log_and_dump_ctx(self.global_frame,
                                                          ty='train') as log:
                            log('fps', episode_frame / elapsed_time)
                            log('total_time', total_time)
                            log('episode_reward', episode_reward)
                            log('episode_length', episode_frame)
                            log('episode', self.global_episode)
                            log('buffer_size', len(self.replay_storage))
                            log('step', self.global_step)

                episode_step = 0
                episode_reward = 0

            self.replay_storage.add(time_step, meta)
            self.train_video_recorder.record(time_step.observation)

            # try to update the agent
            if not seed_until_step(self.global_step):
                for i in range(n_updates):
                    metrics = self.agent.update(self.replay_iter, self.global_step)
                self.logger.log_metrics(metrics, self.global_frame, ty='train')

            # try to save snapshot
            if self.global_frame in self.cfg.snapshots:
                self.save_agent()

        if self.cfg.use_wandb:
            self.run.finish()

    def save_snapshot(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        snapshot = snapshot_dir / f'snapshot_{self.global_frame}.pt'
        keys_to_save = ['agent', '_global_step', '_global_episode']
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open('wb') as f:
            torch.save(payload, f)

    def save_agent(self):
        snapshot_dir = self.work_dir / Path(self.cfg.snapshot_dir)
        snapshot_dir.mkdir(exist_ok=True, parents=True)
        actor = snapshot_dir / f'actor_{self.global_frame}.pt'
        with actor.open('wb') as f:
            torch.save(self.agent.actor.state_dict(), f)
        critic = snapshot_dir / f'critic_{self.global_frame}.pt'
        with critic.open('wb') as f:
            torch.save(self.agent.critic.state_dict(), f)
        if self.use_diayn and self.cfg.agent.name != "diayn" and self.agent.diayn_skill_channel > 0:
            discriminator = snapshot_dir / f'discriminator_{self.global_frame}.pt'
            with discriminator.open('wb') as f:
                torch.save(self.agent.diayn.state_dict(), f)

            if self.cfg.agent.anti:
                anti = snapshot_dir / f'anti_{self.global_frame}.pt'
                with anti.open('wb') as f:
                    torch.save(self.agent.anti_diayn.state_dict(), f)


@hydra.main(config_path='.', config_name='pretrain')
def main(cfg):
    from pretrain import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
