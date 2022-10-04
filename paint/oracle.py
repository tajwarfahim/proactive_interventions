import warnings

warnings.filterwarnings("ignore", category=DeprecationWarning)

from omegaconf import OmegaConf
import os

os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MUJOCO_GL"] = "egl"

from pathlib import Path

import env_loader
import hydra
import numpy as np
import torch
import random
import utils

from bsuite.utils.gym_wrapper import space2spec
from gym import spaces
from dm_env import specs
from logger import Logger
from buffers.replay_buffer import ReplayBufferStorage, make_replay_loader
from buffers.simple_replay_buffer import SimpleReplayBuffer
from buffers.balanced_replay_buffer import BalancedReplayBuffer
from video import TrainVideoRecorder, VideoRecorder
from agents import (
    SACAgent,
    SafeSACAgent,
    SafeSACAgentStuckDiscrim,
    SafetyCriticSACAgent,
    BCAgent,
)
from backend.timestep import (
    ExtendedTimeStep,
    relabel_timesteps_with_correct_stuck_info,
)

torch.backends.cudnn.benchmark = True


def make_agent(
    obs_spec,
    action_spec,
    cfg,
    use_stuck_detector,
    use_safety_critic,
    stuck_reward,
    epsilon,
    stuck_discrim,
    behavior_clone,
):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape

    agent_kwargs = {
        "obs_shape": cfg.obs_shape,
        "action_shape": cfg.action_shape,
        "device": cfg.device,
        "lr": cfg.lr,
        "feature_dim": cfg.feature_dim,
        "hidden_dim": cfg.hidden_dim,
        "critic_target_tau": cfg.critic_target_tau,
        "reward_scale_factor": cfg.reward_scale_factor,
        "use_tb": cfg.use_tb,
        "from_vision": cfg.from_vision,
    }

    if use_safety_critic:
        agent_kwargs["epsilon"] = epsilon
        agent_class = SafetyCriticSACAgent
    elif behavior_clone:
        agent_class = BCAgent
    elif use_stuck_detector:
        agent_kwargs["stuck_reward"] = stuck_reward
        if stuck_discrim:
            agent_kwargs["stuck_discrim_hidden_size"] = cfg.stuck_discrim_hidden_size
            agent_kwargs["stuck_discrim_unsupervised"] = cfg.stuck_discrim_unsupervised
            agent_class = SafeSACAgentStuckDiscrim
        else:
            agent_class = SafeSACAgent
    else:
        agent_class = SACAgent

    print("\nPrinting agent kwargs: \n")
    for key in agent_kwargs:
        print(key, ": ", agent_kwargs[key])
    print("\n")

    print(agent_class)

    return agent_class(**agent_kwargs)


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

        use_stuck_detector = (
            self.cfg.use_stuck_oracle_for_Q
            or self.cfg.use_stuck_buffer_for_Q
            or self.cfg.use_stuck_discrim_for_Q
            or self.cfg.use_stuck_discrim_for_term
        )

        self.agent = make_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.agent,
            use_stuck_detector,
            self.cfg.use_safety_critic,
            self.cfg.r_min / (1.0 - self.cfg.discount),
            self.cfg.epsilon,
            self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_Q or self.cfg.use_stuck_discrim_for_term,
            self.cfg.behavior_clone,
        )

        if self.cfg.use_stuck_oracle_for_Q:
            self.agent.set_stuck_state_detector(
                stuck_state_detector=self.train_env.is_stuck_state,
                domain="numpy",
            )
        elif self.cfg.use_stuck_discrim_for_Q:

            def stuck_discrim_probs(obs):
                return torch.sigmoid(self.agent.stuck_discriminator(obs))

            self.agent.set_stuck_state_detector(
                stuck_state_detector=stuck_discrim_probs,
                domain="torch",
            )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0  # how many episodes have been run
        self._num_interventions = 0
        self._num_stuck_labels = 0

    def setup(self):
        # create logger
        self.logger = Logger(self.work_dir, use_tb=self.cfg.use_tb)
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
            self.cfg.default_stuck_value,
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
        if self.cfg.simple_buffer:
            print("\nUsing simple buffer..\n")
            self.replay_storage_f = SimpleReplayBuffer(
                data_specs,
                self.cfg.replay_buffer_size,
                self.cfg.batch_size,
                self.work_dir / "forward_buffer",
                self.cfg.discount,
                filter_transitions=True,
                with_replacement=self.cfg.with_replacement,
            )
            if self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_term or self.cfg.use_stuck_discrim_for_Q:
                if self.cfg.stuck_discriminator.unsupervised:
                    observation_space = spaces.Box(
                        np.repeat(data_specs[0].minimum, 2),
                        np.repeat(data_specs[0].maximum, 2),
                    )
                    data_specs = (
                        space2spec(observation_space, name="observation"),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, "reward"),
                        specs.Array((1,), np.float32, "discount"),
                        specs.Array((1,), np.float32, "is_stuck"),
                    )
                self.replay_storage_pos = SimpleReplayBuffer(
                    data_specs=data_specs,
                    max_size=self.cfg.stuck_discriminator.positive_buffer_size,
                    batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                    replay_dir=self.work_dir / "buffer_pos",
                    filter_transitions=False,
                    with_replacement=self.cfg.stuck_discriminator.with_replacement,
                )

                self.replay_storage_neg = SimpleReplayBuffer(
                    data_specs=data_specs,
                    max_size=self.cfg.stuck_discriminator.negative_buffer_size,
                    batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                    replay_dir=self.work_dir / "buffer_neg",
                    filter_transitions=False,
                    with_replacement=self.cfg.stuck_discriminator.with_replacement,
                )

        elif self.cfg.balanced_buffer:
            print("\nUsing balanced buffer.\n")
            fraction_generator = lambda tstep: utils.generate_fraction(
                step=tstep,
                initial_fraction=self.cfg.initial_fraction,
                final_fraction=self.cfg.final_fraction,
                final_timestep=self.cfg.final_timestep,
            )
            self.replay_storage_f = BalancedReplayBuffer(
                data_specs,
                self.cfg.replay_buffer_size,
                fraction_generator,
                self.cfg.batch_size,
                self.work_dir / "forward_buffer",
                self.cfg.discount,
                filter_transitions=True,
                with_replacement=self.cfg.with_replacement,
            )
            if self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_term or self.cfg.use_stuck_discrim_for_Q:
                if self.cfg.stuck_discriminator.unsupervised:
                    observation_space = spaces.Box(
                        np.repeat(data_specs[0].minimum, 2),
                        np.repeat(data_specs[0].maximum, 2),
                    )
                    data_specs = (
                        space2spec(observation_space, name="observation"),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, "reward"),
                        specs.Array((1,), np.float32, "discount"),
                        specs.Array((1,), np.float32, "is_stuck"),
                    )
                self.replay_storage_pos = SimpleReplayBuffer(
                    data_specs=data_specs,
                    max_size=self.cfg.stuck_discriminator.positive_buffer_size,
                    batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                    replay_dir=self.work_dir / "buffer_pos",
                    filter_transitions=False,
                    with_replacement=self.cfg.with_replacement,
                )

                self.replay_storage_neg = SimpleReplayBuffer(
                    data_specs=data_specs,
                    max_size=self.cfg.stuck_discriminator.negative_buffer_size,
                    batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                    replay_dir=self.work_dir / "buffer_neg",
                    filter_transitions=False,
                    with_replacement=self.cfg.with_replacement,
                )

        else:
            self.replay_storage_f = ReplayBufferStorage(
                data_specs,
                self.work_dir / "forward_buffer",
            )

            self.forward_loader = make_replay_loader(
                self.work_dir / "forward_buffer",
                self.cfg.replay_buffer_size,
                self.cfg.batch_size,
                self.cfg.replay_buffer_num_workers,
                self.cfg.save_snapshot,
                self.cfg.nstep,
                self.cfg.discount,
            )

            if self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_term or self.cfg.use_stuck_discrim_for_Q:
                if self.cfg.stuck_discriminator.unsupervised:
                    observation_space = spaces.Box(
                        np.repeat(data_specs[0].minimum, 2),
                        np.repeat(data_specs[0].maximum, 2),
                    )
                    data_specs = (
                        space2spec(observation_space, name="observation"),
                        self.train_env.action_spec(),
                        specs.Array((1,), np.float32, "reward"),
                        specs.Array((1,), np.float32, "discount"),
                        specs.Array((1,), np.float32, "is_stuck"),
                    )
                self.replay_storage_pos = ReplayBufferStorage(
                    data_specs, self.work_dir / "buffer_pos"
                )

                self.replay_storage_neg = ReplayBufferStorage(
                    data_specs, self.work_dir / "buffer_neg"
                )

                # batches are balanced to have same number of positive and negative data points
                self.pos_loader = make_replay_loader(
                    self.work_dir / "buffer_pos",
                    self.cfg.stuck_discriminator.positive_buffer_size,
                    self.cfg.stuck_discriminator.batch_size // 2,
                    self.cfg.replay_buffer_num_workers,
                    self.cfg.save_snapshot,
                )
                self.neg_loader = make_replay_loader(
                    self.work_dir / "buffer_neg",
                    self.cfg.stuck_discriminator.negative_buffer_size,
                    self.cfg.stuck_discriminator.batch_size // 2,
                    self.cfg.replay_buffer_num_workers,
                    self.cfg.save_snapshot,
                )

        self._forward_iter, self._pos_iter, self._neg_iter = None, None, None

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
    def pos_iter(self):
        if self._pos_iter is None:
            if self.cfg.simple_buffer or self.cfg.balanced_buffer:
                self._pos_iter = iter(self.replay_storage_pos)
            else:
                self._pos_iter = iter(self.pos_loader)
        return self._pos_iter

    @property
    def neg_iter(self):
        if self._neg_iter is None:
            if self.cfg.simple_buffer or self.cfg.balanced_buffer:
                self._neg_iter = iter(self.replay_storage_neg)
            else:
                self._neg_iter = iter(self.neg_loader)
        return self._neg_iter

    @property
    def forward_iter(self):
        if self._forward_iter is None:
            if self.cfg.simple_buffer or self.cfg.balanced_buffer:
                self._forward_iter = iter(self.replay_storage_f)
            else:
                self._forward_iter = iter(self.forward_loader)
        return self._forward_iter

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def state_value(self, observation):
        Q_value_arr = []

        with torch.no_grad(), utils.eval_mode(self.agent):
            for i in range(self.cfg.num_action_samples):
                action = self.agent.act(
                    observation.astype("float32"),
                    uniform_action=False,
                    eval_mode=False,
                )

                Q_value = self.agent.get_Q_value(
                    obs=utils.numpy_to_torch(data=observation, device=self.device).float(),
                    action=utils.numpy_to_torch(data=action, device=self.device).float(),
                )

                Q_value_arr.append(Q_value.item())

        value = np.mean(Q_value_arr)
        return value

    def get_stuck_prob(self, time_step, prev_time_step):
        if self.cfg.stuck_discriminator.unsupervised:
            obs = np.concatenate((
                prev_time_step.observation,
                time_step.observation,
            ))
        else:
            obs = time_step.observation

        stuck_prob = torch.sigmoid(
            self.agent.stuck_discriminator(
                utils.numpy_to_torch(
                    data=obs, device=self.device
                )
            )
        )

        return stuck_prob

    def should_abort_early(self, time_step, prev_time_step):
        abort_early = False

        if self.global_step > self.cfg.num_seed_frames:
            if self.cfg.use_Q_value_for_term:
                value = self.state_value(observation=time_step.observation)
                abort_early = value < self.cfg.early_abort_threshold

            elif self.cfg.use_stuck_discrim_for_term:
                stuck_prob = self.get_stuck_prob(time_step=time_step, prev_time_step=prev_time_step)
                abort_early = stuck_prob > self.cfg.early_abort_threshold

            elif self.cfg.use_oracle_for_term:
                abort_early = self.train_env.is_stuck_state(time_step.observation)

            elif self.cfg.use_initial_value_for_term:
                # if multiple start states (continual learning), use the one in use now
                if len(self.reset_states.shape) > 1:
                    value_start = self.state_value(
                        observation=self.train_env.get_curr_start_state()
                    )
                else:
                    value_start = self.state_value(observation=self.reset_states)
                value_curr = self.state_value(observation=time_step.observation)

                abort_early = (
                    value_start > value_curr + self.cfg.early_abort_threshold
                )

        if torch.is_tensor(abort_early):
            abort_early = abort_early.item()

        if abort_early:
            self.explore_until_step = utils.Until(
                self.global_step + self.cfg.num_explore_steps,
                self.cfg.action_repeat,
            )

        return abort_early

    def should_terminate(self):
        return not self.explore_until_step(self.global_step)

    def eval(self, eval_agent):
        steps, episode, total_reward, episode_success = 0, 0, 0, 0
        eval_until_episode = utils.Until(self.cfg.num_eval_episodes)
        eval_num_stuck_states, eval_num_stuck_episodes = 0, 0
        stuck = False

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

                stuck = self.train_env.is_stuck_state(time_step.observation) or stuck
                eval_num_stuck_states += int(stuck)

                # early stopping
                if time_step.reward == 1.0:
                    break

            eval_num_stuck_episodes += int(stuck)
            stuck = False
            episode += 1
            episode_success += completed_successfully
            self.video_recorder.save(f"{self.global_frame}.mp4")

        self.logger.log(
            "eval_num_stuck_states", eval_num_stuck_states, self.global_frame
        )
        self.logger.log(
            "eval_num_stuck_episodes", eval_num_stuck_episodes, self.global_frame
        )

        with self.logger.log_and_dump_ctx(self.global_frame, ty="eval") as log:
            log("episode_reward", total_reward / episode)
            log("success_avg", episode_success / episode)
            log("episode_length", steps * self.cfg.action_repeat / episode)
            log("episode", self.global_episode)
            log("step", self.global_step)
            log("num_interventions", self._num_interventions)

        # EARL deployed policy evaluation
        self.deployed_policy_eval.append(episode_success / episode)
        np.save(self.work_dir / "deployed_eval.npy", self.deployed_policy_eval)

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

        self._stuck_discrim_loss = 0.0

        time_step = self.train_env.reset()
        episode_time_steps = [time_step]
        dummy_action = time_step.action

        if self.forward_demos is not None:
            self.forward_demos = self.choose_demos(demos=self.forward_demos)

            self.replay_storage_f.add_offline_data(self.forward_demos, dummy_action)
            if not self.cfg.stuck_discriminator.unsupervised and (
                self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_Q or self.cfg.use_stuck_discrim_for_term
            ):
                self.replay_storage_neg.add_offline_data(
                    self.forward_demos, dummy_action
                )

            print("Number of demo timesteps: ", len(self.replay_storage_f))

        cur_agent = self.agent
        cur_buffer = self.replay_storage_f
        cur_iter = self.forward_iter

        time_step = time_step._replace(is_stuck=0.0)
        cur_buffer.add(time_step)

        if self.cfg.agent.from_vision:
            self.train_video_recorder.init(time_step.observation)

        metrics = None

        if self.cfg.behavior_clone:
            while train_until_step(self.global_step):
                # train actor with behavior cloning
                metrics = cur_agent.update(
                    cur_agent.transition_tuple(cur_iter), self.global_step
                )
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

                # try to evaluate
                if eval_every_step(self.global_step):
                    self.logger.log(
                        "eval_total_time", self.timer.total_time(), self.global_frame
                    )
                    self.eval(self.agent)

                self._global_step += 1
            return

        episode_step, episode_reward = 0, 0
        train_num_stuck_states, train_num_stuck_episodes = 0, 0
        abort_early = False
        window = self.cfg.stuck_discriminator.unsupervised_window

        while train_until_step(self.global_step):
            if time_step.last():
                abort_early = False
                self._global_episode += 1
                time_step = self.train_env.reset()
                cur_buffer.add(time_step)

                if self.cfg.agent.from_vision:
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")

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
                        log("step", self.global_step)
                        if "stuck_discriminator_loss" not in metrics:
                            log("stuck_discriminator_loss", self._stuck_discrim_loss)

                # try to save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()

                (
                    modified_time_steps,
                    num_steps,
                ) = relabel_timesteps_with_correct_stuck_info(
                    episode_time_steps=episode_time_steps,
                    train_env=self.train_env,
                )

                stuck_state_info = []
                for i in range(len(modified_time_steps)):
                    idx = len(cur_buffer) - len(modified_time_steps) + i
                    cur_buffer.replace(idx=idx, time_step=modified_time_steps[i])
                    stuck_state_info.append(int(modified_time_steps[i].is_stuck))

                    if (
                        self.cfg.use_stuck_discrim_label
                        or self.cfg.use_stuck_discrim_for_term
                        or self.cfg.use_stuck_discrim_for_Q
                    ):
                        if self.cfg.stuck_discriminator.unsupervised:
                            end = min(i + window + 1, len(modified_time_steps))
                            for j in range(i + 1, end):
                                next_obs = modified_time_steps[j].observation

                                # (s_i, s_i+j)
                                neg_obs = np.concatenate((
                                    modified_time_steps[i].observation, next_obs,
                                ))
                                neg_time_step = ExtendedTimeStep(
                                    observation=neg_obs,
                                    step_type=modified_time_steps[i].step_type,
                                    action=modified_time_steps[i].action,
                                    reward=modified_time_steps[i].reward,
                                    discount=modified_time_steps[i].discount,
                                    is_stuck=modified_time_steps[i].is_stuck,
                                )
                                self.replay_storage_neg.add(neg_time_step)

                                # (s_i+j, s_i)
                                pos_obs = np.concatenate((
                                    next_obs, modified_time_steps[i].observation,
                                ))
                                pos_time_step = ExtendedTimeStep(
                                    observation=pos_obs,
                                    step_type=modified_time_steps[i].step_type,
                                    action=modified_time_steps[i].action,
                                    reward=modified_time_steps[i].reward,
                                    discount=modified_time_steps[i].discount,
                                    is_stuck=modified_time_steps[i].is_stuck,
                                )
                                self.replay_storage_pos.add(pos_time_step)
                        else:
                            if stuck_state_info[i] == 1:
                                self.replay_storage_pos.add(modified_time_steps[i])
                            else:
                                self.replay_storage_neg.add(modified_time_steps[i])

                train_num_stuck_states += int(np.sum(stuck_state_info))
                train_num_stuck_episodes += int(np.sum(stuck_state_info) > 0.0)
                self._num_stuck_labels += num_steps

                episode_time_steps = []
                episode_step, episode_reward = 0, 0

                if self.cfg.use_stuck_ratio and not self.cfg.use_stuck_discrim_for_term:
                    self.train_env.update_default_stuck_value(
                        default_stuck_value=cur_buffer.stuck_ratio,
                    )

                if self.cfg.stuck_discriminator.train_offline and (
                    self.cfg.use_stuck_discrim_label
                    or self.cfg.use_stuck_discrim_for_Q
                    or self.cfg.use_stuck_discrim_for_term
                ):
                    self.agent.reset_stuck_discrim()
                    if (
                        len(self.replay_storage_pos) > 0
                        and len(self.replay_storage_neg) > 0
                    ):
                        for k in range(10000):
                            metrics = self.agent.update_stuck_discriminator(
                                self.pos_iter, self.neg_iter
                            )
                            if k % 1000 == 0:
                                print('Iteration {}: {}'.format(k, metrics['stuck_discriminator_loss']))

                        self.logger.log_metrics(metrics, self.global_frame, ty="train")
                        self._stuck_discrim_loss = metrics['stuck_discriminator_loss']

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval(self.agent)

                self.logger.log(
                    "train_num_stuck_states", train_num_stuck_states, self.global_frame
                )
                self.logger.log(
                    "train_num_stuck_episodes",
                    train_num_stuck_episodes,
                    self.global_frame,
                )
                self.logger.log(
                    "train_calls_to_stuck_function",
                    self._num_stuck_labels,
                    self.global_frame,
                )

            # sample action
            with torch.no_grad(), utils.eval_mode(cur_agent):
                uniform_action = seed_until_step(self.global_step) or (
                    abort_early and not self.explore_until_step(self.global_step)
                )
                action = cur_agent.act(
                    time_step.observation.astype("float32"),
                    uniform_action=uniform_action,
                    eval_mode=False,
                )

            # try to update the agent
            if not seed_until_step(self.global_step):
                metrics = cur_agent.update(
                    cur_agent.transition_tuple(cur_iter), self.global_step
                )
                self.logger.log_metrics(metrics, self.global_frame, ty="train")

                if not self.cfg.stuck_discriminator.train_offline and (
                    self.cfg.use_stuck_discrim_label
                    or self.cfg.use_stuck_discrim_for_Q
                    or self.cfg.use_stuck_discrim_for_term
                ):
                    if (
                        self._global_step % self.cfg.stuck_discriminator.train_interval
                        == 0
                        and len(self.replay_storage_pos) > 0
                        and len(self.replay_storage_neg) > 0
                    ):
                        for k in range(
                            self.cfg.stuck_discriminator.train_steps_per_iteration
                        ):
                            metrics = self.agent.update_stuck_discriminator(
                                self.pos_iter, self.neg_iter
                            )
                        self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            prev_time_step = time_step
            time_step = self.train_env.step(action)
            if self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_term:
                time_step = time_step._replace(is_stuck=-1.0)

            if not abort_early:
                abort_early = self.should_abort_early(time_step=time_step, prev_time_step=prev_time_step)

            # TODO: early termination only works for goal-conditioned + sparse reward environment
            # also, we assume for the "Goal" state, when reward > 0, is_stuck = 0
            # part of the general assumptions that start and goal states are reversible
            terminate = abort_early and self.should_terminate()

            if time_step.last() or (time_step.reward == 1.0 and self.cfg.reset_at_success) or terminate:
                time_step = ExtendedTimeStep(
                    observation=time_step.observation,
                    step_type=2,
                    action=action,
                    reward=time_step.reward,
                    discount=time_step.discount,
                    is_stuck=0.0 if time_step.reward == 1.0 else 1.0,
                )

                self._num_interventions += 1

            episode_reward += time_step.reward
            cur_buffer.add(time_step)
            episode_time_steps.append(time_step)

            if self.cfg.agent.from_vision:
                self.train_video_recorder.record(time_step.observation)

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


@hydra.main(config_path="cfgs", config_name="oracle")
def main(cfg):
    from oracle import Workspace as W

    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / "snapshot.pt"
    if snapshot.exists():
        print(f"resuming: {snapshot}")
        workspace.load_snapshot()
    workspace.train()


if __name__ == "__main__":
    main()
