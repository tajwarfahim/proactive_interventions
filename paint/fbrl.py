import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from omegaconf import OmegaConf
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import copy
import env_loader
import hydra
import numpy as np
import torch
import utils

from dm_env import specs
from logger import Logger
from buffers.replay_buffer import ReplayBufferStorage, make_replay_loader
from buffers.simple_replay_buffer import SimpleReplayBuffer
from buffers.balanced_replay_buffer import BalancedReplayBuffer
from video import TrainVideoRecorder, VideoRecorder
from agents import SACAgent
from backend.timestep import ExtendedTimeStep, StepType

torch.backends.cudnn.benchmark = True


def make_agent(obs_spec, action_spec, cfg):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape

    return SACAgent(
        obs_shape=cfg.obs_shape,
        action_shape=cfg.action_shape,
        device=cfg.device,
        lr=cfg.lr,
        feature_dim=cfg.feature_dim,
        hidden_dim=cfg.hidden_dim,
        critic_target_tau=cfg.critic_target_tau,
        reward_scale_factor=cfg.reward_scale_factor,
        use_tb=cfg.use_tb,
        from_vision=cfg.from_vision,
    )


class Workspace:
    def __init__(self, cfg):
        self.work_dir = Path.cwd()
        print(f"workspace: {self.work_dir}")

        with open(self.work_dir / "config.yaml", "w") as fp:
            OmegaConf.save(config=cfg, f=fp.name)

        self.cfg = cfg
        utils.set_seed_everywhere(cfg.seed)
        self.device = torch.device(cfg.device)
        self.setup()

        self.forward_agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.forward_agent,
        )
        self.backward_agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.backward_agent,
        )
        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0  # how many episodes have been run

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
        # create envs
        (
            self.train_env,
            self.eval_env,
            self.reset_states,
            self.goal_states,
            self.forward_demos,
            self.backward_demos,
        ) = env_loader.make(
            self.cfg.env_name,
            self.cfg.frame_stack,
            self.cfg.action_repeat,
            0,
            self.cfg.train_horizon,
            self.cfg.eval_horizon,
            self.cfg.reward_type,
            self.cfg.seed,
        )
        # create replay buffer
        data_specs = (
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            specs.Array((1,), np.float32, "reward"),
            specs.Array((1,), np.float32, "discount"),
            specs.Array((1,), np.float32, "is_stuck"),
        )
        self.obs_dim = data_specs[0].shape[0] // 2

        if self.cfg.forward_agent.balanced_buffer:
            print("\nUsing forward balanced buffer.\n")
            fraction_generator = lambda tstep: utils.generate_fraction(
                step=tstep,
                initial_fraction=self.cfg.forward_agent.initial_fraction,
                final_fraction=self.cfg.forward_agent.final_fraction,
                final_timestep=self.cfg.forward_agent.final_timestep,
            )
            self.replay_storage_f = BalancedReplayBuffer(
                data_specs,
                self.cfg.forward_agent.replay_buffer_size,
                fraction_generator,
                self.cfg.forward_agent.batch_size,
                self.work_dir / "forward_buffer",
                self.cfg.forward_agent.discount,
                filter_transitions=True,
                with_replacement=self.cfg.forward_agent.with_replacement,
            )
        else:
            self.replay_storage_f = SimpleReplayBuffer(
                data_specs=data_specs,
                max_size=self.cfg.forward_agent.replay_buffer_size,
                batch_size=self.cfg.forward_agent.batch_size,
                replay_dir=self.work_dir / "forward_buffer",
                discount=self.cfg.forward_agent.discount,
                filter_transitions=True,
                with_replacement=self.cfg.forward_agent.with_replacement,
            )

        if self.cfg.backward_agent.balanced_buffer:
            print("\nUsing backward balanced buffer.\n")
            fraction_generator = lambda tstep: utils.generate_fraction(
                step=tstep,
                initial_fraction=self.cfg.backward_agent.initial_fraction,
                final_fraction=self.cfg.backward_agent.final_fraction,
                final_timestep=self.cfg.backward_agent.final_timestep,
            )
            self.replay_storage_b = BalancedReplayBuffer(
                data_specs,
                self.cfg.backward_agent.replay_buffer_size,
                fraction_generator,
                self.cfg.backward_agent.batch_size,
                self.work_dir / "forward_buffer",
                self.cfg.backward_agent.discount,
                filter_transitions=True,
                with_replacement=self.cfg.backward_agent.with_replacement,
            )
        else:
            self.replay_storage_b = SimpleReplayBuffer(
                data_specs=data_specs,
                max_size=self.cfg.backward_agent.replay_buffer_size,
                batch_size=self.cfg.backward_agent.batch_size,
                replay_dir=self.work_dir / "backward_buffer",
                discount=self.cfg.backward_agent.discount,
                filter_transitions=True,
                with_replacement=self.cfg.backward_agent.with_replacement,
            )

        self._forward_iter, self._backward_iter = None, None

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = TrainVideoRecorder(
            self.work_dir if self.cfg.save_train_video else None
        )

        # recording metrics for EARL
        np.save(self.work_dir / "eval_interval.npy", self.cfg.eval_every_frames)
        try:
            self.deployed_policy_eval = np.load(
                self.work_dir / "deployed_eval.npy"
            ).tolist()
        except:
            self.deployed_policy_eval = []

    @property
    def forward_iter(self):
        if self._forward_iter is None:
            self._forward_iter = iter(self.replay_storage_f)
        return self._forward_iter

    @property
    def backward_iter(self):
        if self._backward_iter is None:
            self._backward_iter = iter(self.replay_storage_b)
        return self._backward_iter

    def sample_goal_state(self):
        return self.goal_states[np.random.randint(0, self.goal_states.shape[0])]

    def sample_reset_state(self):
        return self.reset_states[np.random.randint(0, self.reset_states.shape[0])]

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def eval(self, eval_agent):
        steps, episode, total_reward, episode_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)

        while eval_until_episode(episode):
            time_step = self.eval_env.reset()
            self.video_recorder.init(self.eval_env, enabled=(episode == 0))
            episode_step, completed_successfully = 0, 0
            while not time_step.last():
                with torch.no_grad(), utils.eval_mode(eval_agent):
                    action = eval_agent.act(
                        time_step.observation.astype("float32"),
                        uniform_action=False,
                        eval_mode=True,
                    )
                time_step = self.eval_env.step(action)
                self.video_recorder.record(self.eval_env)
                if hasattr(
                    self.eval_env, "is_successful"
                ) and self.eval_env.is_successful(time_step.observation):
                    completed_successfully = 1

                total_reward += time_step.reward
                episode_step += 1
                steps += 1

            episode += 1
            episode_success += completed_successfully
            self.video_recorder.save(f"{self.global_frame}.mp4")

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("success_avg", episode_success / episode)
            log("episode_length", steps * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)

        # EARL deployed policy evaluation
        self.deployed_policy_eval.append(episode_success / episode)
        np.save(self.work_dir / "deployed_eval.npy", self.deployed_policy_eval)

    def replace_goal_np(self, obs, goal):
        return np.concatenate([obs[: self.obs_dim], goal], axis=0).astype("float32")

    def choose_demos(self, demos):
        end_indices = np.where(demos["rewards"] > 0)[0]
        num_demos = min(self.cfg.num_demos, len(end_indices))
        end_index = end_indices[num_demos - 1] + 1

        for key in demos.keys():
            demos[key] = demos[key][:end_index]

        return demos

    def train(self):
        # predicates
        train_until_step = utils.Until(
            self.cfg.num_train_frames, self.cfg.action_repeat
        )
        seed_until_step = utils.Until(self.cfg.num_seed_frames, self.cfg.action_repeat)
        eval_every_step = utils.Every(
            self.cfg.eval_every_frames, self.cfg.action_repeat
        )

        time_step = self.train_env.reset()
        # dummy action to be used for first transition
        init_action = time_step.action

        if self.forward_demos is not None:
            self.forward_demos = self.choose_demos(demos=self.forward_demos)
            self.replay_storage_f.add_offline_data(self.forward_demos, init_action)
            print("# forward demo timesteps: ", len(self.replay_storage_f))
        if self.backward_demos is not None:
            self.backward_demos = self.choose_demos(demos=self.backward_demos)
            self.replay_storage_b.add_offline_data(self.backward_demos, init_action)
            print("# backward demo timesteps: ", len(self.replay_storage_b))

        cur_policy = "forward"
        cur_goal = time_step.observation[self.obs_dim:]
        cur_agent = self.forward_agent
        cur_buffer = self.replay_storage_f
        cur_iter = self.forward_iter
        switch_policy = False
        abort_early = False
        
        cur_buffer.add(time_step)
        # next action needs to be computed before hand to check for switching
        with torch.no_grad(), utils.eval_mode(cur_agent):
            action = cur_agent.act(
                time_step.observation.astype("float32"),
                uniform_action=seed_until_step(self.global_step),
                eval_mode=False,
            )

        if self.cfg.forward_agent.from_vision:
            self.train_video_recorder.init(time_step.observation)

        metrics = None
        episode_step, episode_reward, num_interventions = 0, 0, 0
        while train_until_step(self.global_step):

            if switch_policy:
                # pretend episode ends when the policy switches
                self._global_episode += 1
                if self.cfg.forward_agent.from_vision:
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")

                if cur_policy == "forward":
                    cur_policy = "backward"
                    cur_goal = self.sample_reset_state()
                    cur_agent = self.backward_agent
                    cur_buffer = self.replay_storage_b
                    cur_iter = self.backward_iter

                elif cur_policy == "backward":
                    cur_policy = "forward"
                    cur_goal = self.sample_goal_state()
                    cur_agent = self.forward_agent
                    cur_buffer = self.replay_storage_f
                    cur_iter = self.forward_iter

                print('cur_policy: {}'.format(cur_policy))
                # add time step as a new time step to the current buffer
                if self.cfg.replace_goal:
                    updated_obs = self.replace_goal_np(time_step.observation, cur_goal)
                else:
                    updated_obs = time_step.observation

                time_step = ExtendedTimeStep(
                    observation=updated_obs,
                    step_type=StepType.FIRST,
                    action=init_action,
                    reward=0.0,
                    discount=time_step.discount,
                    is_stuck=time_step.is_stuck,
                )
                cur_buffer.add(time_step)
                with torch.no_grad(), utils.eval_mode(cur_agent):
                    action = cur_agent.act(
                        time_step.observation.astype("float32"),
                        uniform_action=seed_until_step(self.global_step),
                        eval_mode=False,
                    )

                # wait until all the metrics schema is populated
                if metrics is not None:
                    # log stats
                    elapsed_time, total_time = self.timer.reset()
                    episode_frame = episode_step * self.cfg.action_repeat
                    with self.logger.log_and_dump_ctx(
                        self.global_frame, ty="train"
                    ) as log:
                        log("fps", episode_frame / elapsed_time)
                        log("total_time", total_time)
                        log("episode_reward", episode_reward)
                        log("episode_length", episode_frame)
                        log("episode", self.global_episode)
                        log("forward_buffer_size", len(self.replay_storage_f))
                        log("backward_buffer_size", len(self.replay_storage_b))
                        log("step", self.global_step)
                        log("num_interventions", num_interventions)

                # save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step, episode_reward = 0, 0

                # disable for the next global step
                switch_policy = False

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval(self.forward_agent)

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = cur_agent.update(
                    cur_agent.transition_tuple(cur_iter), self.global_step
                )
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # NOTE: take env step with the already computed action
            time_step = self.train_env.step(action)

            if self.cfg.replace_goal:
                updated_obs = self.replace_goal_np(time_step.observation, cur_goal)
                # we assume the ability to query the reward for task goals provided by the environment
                updated_reward = float(self.train_env.compute_reward(obs=updated_obs))
            else:
                updated_obs = time_step.observation
                updated_reward = time_step.reward
            
            new_time_step = time_step._replace(observation=updated_obs,
                                               reward=updated_reward)
            episode_reward += updated_reward
            # NOTE: compute the next action to decide whether to switch!
            with torch.no_grad(), utils.eval_mode(cur_agent):
                action = cur_agent.act(
                    new_time_step.observation.astype("float32"),
                    uniform_action=seed_until_step(self.global_step),
                    eval_mode=False,
                )

            updated_step_type = new_time_step.step_type
            # all switching logic
            if cur_policy == 'forward':
                if (episode_step + 1) % self.cfg.policy_switch_frequency == 0 or self.train_env.is_successful(updated_obs):
                    print('policy switch at {}, f->b'.format(self.global_step))
                    switch_policy = True
                    updated_step_type = StepType.LAST
                else:
                    # the backward policy only knows reset goals!
                    back_observation = self.replace_goal_np(updated_obs, self.sample_reset_state())
                    # delaying this computation into a separate block
                    back_agent_Q_val = self.backward_agent.get_Q_value(
                                        obs=utils.numpy_to_torch(data=back_observation, device=self.device),
                                        action=utils.numpy_to_torch(data=action, device=self.device),)
                    if back_agent_Q_val < self.cfg.switch_policy_threshold:
                        print('policy switch at {}, f->b'.format(self.global_step))
                        print('back_Q_val: {}'.format(back_agent_Q_val))
                        switch_policy = True
                        updated_step_type = StepType.LAST

            elif cur_policy == 'backward':
                if (episode_step + 1) % self.cfg.policy_switch_frequency == 0:
                    print('policy switch at {}, b->f'.format(self.global_step))
                    switch_policy = True
                    updated_step_type = StepType.LAST
                    
                    # hard reset based on whether backward policy was successful
                    abort_early = not self.train_env.is_successful(updated_obs)
                    if abort_early:
                        print('aborting: {}'.format(updated_obs))

            new_time_step = new_time_step._replace(step_type=updated_step_type)
            cur_buffer.add(new_time_step)

            if self.cfg.forward_agent.from_vision:
                self.train_video_recorder.record(new_time_step.observation)

            if time_step.last() or abort_early:
                print("hard reset")
                abort_early = False
                time_step = self.train_env.reset()
                cur_policy = "backward"
                switch_policy = True
                num_interventions += 1
            else:
                time_step = new_time_step

            self._global_step += 1
            episode_step += 1

    def save_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        keys_to_save = ["agent", "timer", "_global_step", "_global_episode"]
        payload = {k: self.__dict__[k] for k in keys_to_save}
        with snapshot.open("wb") as f:
            torch.save(payload, f)

    def load_snapshot(self):
        snapshot = self.work_dir / "snapshot.pt"
        with snapshot.open("rb") as f:
            payload = torch.load(f)
        for k, v in payload.items():
            self.__dict__[k] = v


@hydra.main(config_path="cfgs", config_name="fbrl")
def main(cfg):
    root_dir = Path.cwd()
    workspace = Workspace(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
