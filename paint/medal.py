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
    MEDALBackwardAgent,
    SafeMEDALAgent,
    BCAgent,
)
from backend.timestep import (
    ExtendedTimeStep,
    relabel_timesteps_with_correct_stuck_info,
)

torch.backends.cudnn.benchmark = True


def make_forward_agent(
    obs_spec,
    action_spec,
    cfg,
    use_stuck_detector,
    use_safety_critic,
    stuck_reward,
    epsilon,
    stuck_discrim,
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


def make_backward_agent(
    obs_spec,
    action_spec,
    cfg,
    use_stuck_detector,
    use_safety_critic,
    stuck_reward,
    epsilon,
    stuck_discrim,
):
    cfg.obs_shape = obs_spec.shape
    cfg.action_shape = action_spec.shape

    agent_kwargs = {
        "discrim_hidden_size": cfg.discrim_hidden_size,
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
        agent_class = SafetyCriticMEDALAgent
    elif use_stuck_detector:
        agent_kwargs["stuck_reward"] = stuck_reward
        if stuck_discrim:
            agent_kwargs["stuck_discrim_unsupervised"] = cfg.stuck_discrim_unsupervised
        agent_class = SafeMEDALAgent
    else:
        agent_class = MEDALBackwardAgent

    print("\nPrinting backward agent kwargs: \n")
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

        self.forward_agent = make_forward_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.forward_agent,
            use_stuck_detector,
            self.cfg.use_safety_critic,
            self.cfg.r_min / (1.0 - self.cfg.discount),
            self.cfg.epsilon,
            self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_Q or self.cfg.use_stuck_discrim_for_term,
        )
        self.backward_agent = make_backward_agent(
            self.train_env.observation_spec(),
            self.train_env.action_spec(),
            self.cfg.backward_agent,
            use_stuck_detector,
            self.cfg.use_safety_critic,
            self.cfg.r_min / (1.0 - self.cfg.discount),
            self.cfg.epsilon,
            self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_Q or self.cfg.use_stuck_discrim_for_term,
        )

        if self.cfg.use_stuck_oracle_for_Q:
            self.forward_agent.set_stuck_state_detector(
                stuck_state_detector=self.train_env.is_stuck_state,
                domain="numpy",
            )
            self.backward_agent.set_stuck_state_detector(
                stuck_state_detector=self.train_env.is_stuck_state,
                domain="numpy",
            )
        elif self.cfg.use_stuck_discrim_for_Q:

            def stuck_discrim_probs(obs):
                return torch.sigmoid(self.forward_agent.stuck_discriminator(obs))

            self.forward_agent.set_stuck_state_detector(
                stuck_state_detector=stuck_discrim_probs,
                domain="torch",
            )
            self.backward_agent.set_stuck_state_detector(
                stuck_state_detector=stuck_discrim_probs,
                domain="torch",
            )

        self.timer = utils.Timer()
        self._global_step = 0
        self._global_episode = 0  # how many episodes have been run
        self._virtual_episode = 0
        self._num_interventions = 0
        self._num_stuck_labels = 0

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

        self.backward_pos = SimpleReplayBuffer(
            data_specs=data_specs,
            max_size=self.cfg.discriminator.positive_buffer_size,
            batch_size=self.cfg.discriminator.batch_size // 2,
            replay_dir=self.work_dir / "backward_buffer_pos",
            filter_transitions=False,
            with_replacement=self.cfg.discriminator.with_replacement,
        )

        self.backward_neg = SimpleReplayBuffer(
            data_specs=data_specs,
            max_size=self.cfg.discriminator.negative_buffer_size,
            batch_size=self.cfg.discriminator.batch_size // 2,
            replay_dir=self.work_dir / "backward_buffer_neg",
            filter_transitions=False,
            with_replacement=self.cfg.discriminator.with_replacement,
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
            self.stuck_replay_storage_pos = SimpleReplayBuffer(
                data_specs=data_specs,
                max_size=self.cfg.stuck_discriminator.positive_buffer_size,
                batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                replay_dir=self.work_dir / "buffer_pos",
                filter_transitions=False,
                with_replacement=self.cfg.stuck_discriminator.with_replacement,
            )

            self.stuck_replay_storage_neg = SimpleReplayBuffer(
                data_specs=data_specs,
                max_size=self.cfg.stuck_discriminator.negative_buffer_size,
                batch_size=self.cfg.stuck_discriminator.batch_size // 2,
                replay_dir=self.work_dir / "buffer_neg",
                filter_transitions=False,
                with_replacement=self.cfg.stuck_discriminator.with_replacement,
            )

        (
            self._forward_iter,
            self._backward_iter,
            self._pos_iter,
            self._neg_iter,
            self._stuck_pos_iter,
            self._stuck_neg_iter,
        ) = (
            None,
            None,
            None,
            None,
            None,
            None,
        )

        self.video_recorder = VideoRecorder(
            self.work_dir if self.cfg.save_video else None
        )
        self.train_video_recorder = VideoRecorder(
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
            self._pos_iter = iter(self.backward_pos)
        return self._pos_iter

    @property
    def neg_iter(self):
        if self._neg_iter is None:
            self._neg_iter = iter(self.backward_neg)
        return self._neg_iter

    @property
    def stuck_pos_iter(self):
        if self._stuck_pos_iter is None:
            self._stuck_pos_iter = iter(self.stuck_replay_storage_pos)
        return self._stuck_pos_iter

    @property
    def stuck_neg_iter(self):
        if self._stuck_neg_iter is None:
            self._stuck_neg_iter = iter(self.stuck_replay_storage_neg)
        return self._stuck_neg_iter

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

    @property
    def global_step(self):
        return self._global_step

    @property
    def global_episode(self):
        return self._global_episode

    @property
    def global_frame(self):
        return self.global_step * self.cfg.action_repeat

    def state_value(self, agent, observation):
        Q_value_arr = []

        with torch.no_grad(), utils.eval_mode(agent):
            for i in range(self.cfg.num_action_samples):
                action = agent.act(
                    observation.astype("float32"),
                    uniform_action=False,
                    eval_mode=False,
                )

                Q_value = agent.get_Q_value(
                    obs=utils.numpy_to_torch(data=observation, device=self.device),
                    action=utils.numpy_to_torch(data=action, device=self.device),
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
            self.forward_agent.stuck_discriminator(
                utils.numpy_to_torch(data=obs, device=self.device)
            )
        )
        return stuck_prob

    def should_switch_policies(self, time_step, agent):
        if self.global_step <= self.cfg.num_seed_frames:
            return self.global_step % self.cfg.policy_switch_frequency == 0

        if self.cfg.use_Q_value_for_term:
            value = self.state_value(agent=agent, observation=time_step.observation)
            if value < self.cfg.switch_policy_threshold:
                return True

        elif self.cfg.use_stuck_discrim_for_term:
            # shared discriminator between forward and backward agents
            if (
                self.get_stuck_prob(time_step=time_step)
                > self.cfg.switch_policy_threshold
            ):
                return True

        return False

    def should_abort_early(self, agent, time_step, prev_time_step):
        abort_early = False
        if self.global_step < self.cfg.num_seed_frames:
            # early termination during initial seed frames
            abort_early = (self.global_step + 1) == self.cfg.num_seed_frames
        else:
            if self.cfg.use_Q_value_for_term:
                value = self.state_value(
                    agent=agent,
                    observation=time_step.observation,
                )
                abort_early = value < self.cfg.early_abort_threshold

            elif self.cfg.use_stuck_discrim_for_term:
                stuck_prob = self.get_stuck_prob(
                    time_step=time_step,
                    prev_time_step=prev_time_step,
                )
                abort_early = stuck_prob > self.cfg.early_abort_threshold

            elif self.cfg.use_oracle_for_term:
                abort_early = self.train_env.is_stuck_state(time_step.observation)

            elif self.cfg.use_initial_value_for_term:
                initial_state = np.concatenate((
                    self.reset_states[0],
                    time_step.observation[6:],
                )).astype(np.float32)
                value_start = self.state_value(agent=agent, observation=initial_state)
                value_curr = self.state_value(agent=agent, observation=time_step.observation)

                abort_early = (
                    value_start > value_curr + self.cfg.early_abort_threshold
                )

        if torch.is_tensor(abort_early):
            abort_early = abort_early.item()

        if abort_early:
            num_explore_steps = 0 if self.global_step < self.cfg.num_seed_frames else self.cfg.num_explore_steps
            self.explore_until_step = utils.Until(
                self.global_step + num_explore_steps,
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

    def replace_goal_np(self, obs, goal):
        return np.concatenate([obs[: self.obs_dim], goal], axis=0).astype("float32")

    # rewards will be incorrect after relabeling
    def get_relabeled_demos(self, demos, goal):
        new_demos = copy.deepcopy(demos)
        new_demos["observations"][:, self.obs_dim :] = goal.copy()
        new_demos["next_observations"][:, self.obs_dim :] = goal.copy()

        return new_demos

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

        prev_time_step = time_step = self.train_env.reset()
        episode_time_steps = [time_step]
        init_action = time_step.action

        self._stuck_discrim_loss = 0.0

        # policy(timestep) = 1 --> we are running forward policy at timestep t
        # policy(timestep) = 0 --> we are running backward policy at timestep t
        policy_at_episode_steps = [1]

        self.forward_demos = self.choose_demos(demos=self.forward_demos)
        self.replay_storage_f.add_offline_data(self.forward_demos, init_action)
        print("# forward demo timesteps: ", len(self.replay_storage_f))

        # backward buffer gets both forward/backward demos
        if self.backward_demos is not None:
            self.backward_demos = self.choose_demos(demos=self.backward_demos)
            if self.cfg.replace_goal:
                for goal in self.goal_states:
                    self.replay_storage_b.add_offline_data(
                        self.get_relabeled_demos(self.backward_demos, goal), init_action
                    )
            else:
                self.replay_storage_b.add_offline_data(self.backward_demos, init_action)

        print("# backward demo timesteps: ", len(self.replay_storage_b))

        # the positive dataset for MEDAL discriminator
        self.backward_pos.add_offline_data(self.forward_demos, init_action)

        # the negative dataset for the stuck discriminator
        if not self.cfg.forward_agent.stuck_discrim_unsupervised and (
            self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_Q or self.cfg.use_stuck_discrim_for_term
        ):
            self.stuck_replay_storage_neg.add_offline_data(
                self.forward_demos, init_action,
            )
            self.stuck_replay_storage_neg.add_offline_data(
                self.backward_demos, init_action,
            )

            for _ in range(self.cfg.num_stuck_state_samples):
                stuck_state = self.train_env.sample_stuck_state()
                stuck_time_step = ExtendedTimeStep(
                    observation=stuck_state,
                    step_type=0,
                    action=init_action,
                    reward=0.0,
                    discount=time_step.discount,
                    is_stuck=1.0,
                )
                self.stuck_replay_storage_pos.add(stuck_time_step)
            print('Stuck pos size: ', len(self.stuck_replay_storage_pos))

        cur_policy = "forward"
        cur_goal = time_step.observation[self.obs_dim :]
        cur_agent = self.forward_agent
        cur_buffer = self.replay_storage_f
        cur_iter = self.forward_iter
        switch_policy = False
        abort_early = False
        terminate = False
        window = self.cfg.stuck_discriminator.unsupervised_window

        cur_buffer.add(time_step)
        if self.cfg.forward_agent.from_vision:
            self.train_video_recorder.init(time_step.observation)
        self.train_video_recorder.init(self.train_env, enabled=False)

        metrics = None
        episode_step, episode_reward = 0, 0
        train_num_stuck_states, train_num_stuck_episodes = 0, 0

        while train_until_step(self.global_step):

            if switch_policy and not abort_early:
                # pretend episode ends when the policy switches
                self._virtual_episode += 1
                if self.cfg.forward_agent.from_vision:
                    self.train_video_recorder.save(f"{self.global_frame}.mp4")

                if cur_policy == "forward":
                    cur_policy = "backward"
                    cur_goal = self.sample_goal_state()
                    cur_agent = self.backward_agent
                    cur_buffer = self.replay_storage_b
                    cur_iter = self.backward_iter
                elif cur_policy == "backward":
                    # NOTE: do not change the goal when switching from backward to forward policy
                    cur_policy = "forward"
                    cur_agent = self.forward_agent
                    cur_buffer = self.replay_storage_f
                    cur_iter = self.forward_iter

                # add time step as a new time step to the current buffer
                if self.cfg.replace_goal:
                    updated_obs = self.replace_goal_np(time_step.observation, cur_goal)
                else:
                    updated_obs = time_step.observation

                time_step = ExtendedTimeStep(
                    observation=updated_obs,
                    step_type=0,
                    action=init_action,
                    reward=0.0,
                    discount=time_step.discount,
                    is_stuck=time_step.is_stuck,
                )

                cur_buffer.add(time_step)
                episode_time_steps.append(time_step)
                if cur_policy == "backward":
                    self.backward_neg.add(time_step)
                    policy_at_episode_steps.append(0)
                else:
                    policy_at_episode_steps.append(1)

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
                        log("backward_pos", len(self.backward_pos))
                        log("backward_neg", len(self.backward_neg))
                        log("step", self.global_step)
                        if "stuck_discriminator_loss" not in metrics:
                            log("stuck_discriminator_loss", self._stuck_discrim_loss)

                # save snapshot
                if self.cfg.save_snapshot:
                    self.save_snapshot()
                episode_step, episode_reward = 0, 0

                # disable for the next global step
                switch_policy = False

            if abort_early and not terminate:
                # start exploration steps
                time_step = ExtendedTimeStep(
                    observation=time_step.observation,
                    step_type=0,
                    action=init_action,
                    reward=0.0,
                    discount=time_step.discount,
                    is_stuck=time_step.is_stuck,
                )
                cur_buffer.add(time_step)
                episode_time_steps.append(time_step)
                if cur_policy == "backward":
                    self.backward_neg.add(time_step)
                    policy_at_episode_steps.append(0)
                else:
                    policy_at_episode_steps.append(1)

            # hard reset only if the actual environment returned done or termination condition met
            if time_step.last() or terminate:
                print("hard reset")
                abort_early = False
                episode_step = 0
                self._num_interventions += 1
                self._global_episode += 1

                if self.cfg.save_train_video:
                    self.train_video_recorder.save(
                        f"episode_{self._global_episode}.mp4"
                    )

                time_step = self.train_env.reset()
                cur_goal = time_step.observation[self.obs_dim :]
                cur_policy = "forward"
                cur_agent = self.forward_agent
                cur_buffer = self.replay_storage_f
                cur_iter = self.forward_iter

                if self.cfg.save_train_video:
                    self.train_video_recorder.init(
                        self.train_env,
                        enabled=(self._global_episode % self.cfg.train_video_save_freq == 0)
                    )

                (
                    modified_time_steps,
                    num_steps,
                ) = relabel_timesteps_with_correct_stuck_info(
                    episode_time_steps=episode_time_steps,
                    train_env=self.train_env,
                )

                assert len(policy_at_episode_steps) == len(episode_time_steps)

                num_forward_policy_steps = np.sum(policy_at_episode_steps)
                num_backward_policy_steps = (
                    len(policy_at_episode_steps) - num_forward_policy_steps
                )
                forward_buffer_index = (
                    len(self.replay_storage_f) - num_forward_policy_steps
                )
                backward_buffer_index = (
                    len(self.replay_storage_b) - num_backward_policy_steps
                )

                stuck_state_info = []
                for i in range(len(modified_time_steps)):
                    stuck_state_info.append(int(modified_time_steps[i].is_stuck))

                    if policy_at_episode_steps[i] == 1:
                        self.replay_storage_f.replace(
                            idx=forward_buffer_index,
                            time_step=modified_time_steps[i],
                        )
                        forward_buffer_index += 1
                    else:
                        self.replay_storage_b.replace(
                            idx=backward_buffer_index,
                            time_step=modified_time_steps[i],
                        )
                        backward_buffer_index += 1

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
                                self.stuck_replay_storage_neg.add(neg_time_step)

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
                                self.stuck_replay_storage_pos.add(pos_time_step)
                        else:
                            if stuck_state_info[i] == 1:
                                self.stuck_replay_storage_pos.add(modified_time_steps[i])
                            else:
                                self.stuck_replay_storage_neg.add(modified_time_steps[i])
                print('Stuck pos size: ', len(self.stuck_replay_storage_pos))
                print('Stuck neg size: ', len(self.stuck_replay_storage_neg))

                train_num_stuck_states += int(np.sum(stuck_state_info))
                train_num_stuck_episodes += int(np.sum(stuck_state_info) > 0.0)
                self._num_stuck_labels += num_steps

                cur_buffer.add(time_step)
                episode_time_steps = [time_step]
                policy_at_episode_steps = [1]

                if self.cfg.use_stuck_ratio:
                    stuck_ratio = self.replay_storage_f.stuck_ratio * len(
                        self.replay_storage_f
                    ) + self.replay_storage_b.stuck_ratio * len(self.replay_storage_b)
                    stuck_ratio = float(stuck_ratio) / (
                        len(self.replay_storage_f) + len(self.replay_storage_b)
                    )
                    self.train_env.update_default_stuck_value(
                        default_stuck_value=stuck_ratio,
                    )

                if (
                    self.cfg.use_stuck_discrim_label
                    or self.cfg.use_stuck_discrim_for_Q
                    or self.cfg.use_stuck_discrim_for_term
                ):
                    self.forward_agent.reset_stuck_discrim()
                    if (
                        len(self.stuck_replay_storage_pos) > 0
                        and len(self.stuck_replay_storage_neg) > 0
                    ):
                        for k in range(self.cfg.stuck_discriminator.train_steps_per_iteration):
                            metrics = self.forward_agent.update_stuck_discriminator(
                                self.stuck_pos_iter, self.stuck_neg_iter
                            )
                            if k % 1000 == 0:
                                print('Iteration {}: {}'.format(k, metrics['stuck_discriminator_loss']))

                        self.logger.log_metrics(metrics, self.global_frame, ty="train")
                        self._stuck_discrim_loss = metrics['stuck_discriminator_loss']
                    self.backward_agent.set_stuck_discriminator(self.forward_agent.stuck_discriminator)

            # try to evaluate
            if eval_every_step(self.global_step):
                self.logger.log(
                    "eval_total_time", self.timer.total_time(), self.global_frame
                )
                self.eval(self.forward_agent)

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

                if self._global_step % self.cfg.discriminator.train_interval == 0:
                    for k in range(self.cfg.discriminator.train_steps_per_iteration):
                        metrics = self.backward_agent.update_discriminator(
                            self.pos_iter, self.neg_iter
                        )
                    self.logger.log_metrics(metrics, self.global_frame, ty="train")

            # take env step
            prev_time_step = time_step
            time_step = self.train_env.step(action)

            if self.cfg.use_stuck_discrim_label or self.cfg.use_stuck_discrim_for_term:
                time_step = time_step._replace(is_stuck=-1.0)

            self.train_video_recorder.record(self.train_env)
            if self.cfg.replace_goal:
                updated_obs = self.replace_goal_np(time_step.observation, cur_goal)
                # we assume the ability to query the reward for task goals provided by the environment
                updated_reward = float(self.train_env.compute_reward(obs=updated_obs))
            else:
                updated_obs = time_step.observation
                updated_reward = time_step.reward
            updated_step_type = time_step.step_type
            # switch between backward and forward policies
            should_switch_policy = (
                not abort_early
                and self.cfg.switch_policy_at_condition
                and self.should_switch_policies(time_step=time_step, agent=cur_agent)
            )
            if (
                (cur_policy == "forward" and self.train_env.is_successful(updated_obs))
                or (
                    (episode_step + 1) % self.cfg.policy_switch_frequency == 0
                    and not abort_early
                )
                or should_switch_policy
            ):
                switch_policy = True
                updated_step_type = 2

            if not abort_early:
                abort_early = self.should_abort_early(
                    agent=cur_agent,
                    time_step=time_step,
                    prev_time_step=prev_time_step,
                )
                if abort_early:
                    updated_step_type = 2

            terminate = abort_early and self.should_terminate()

            if terminate:
                updated_step_type = 2

            new_time_step = ExtendedTimeStep(
                observation=updated_obs,
                step_type=updated_step_type,
                action=action,
                reward=updated_reward,
                discount=time_step.discount,
                is_stuck=time_step.is_stuck,
            )
            episode_reward += new_time_step.reward
            cur_buffer.add(new_time_step)
            episode_time_steps.append(new_time_step)

            if cur_policy == "backward":
                self.backward_neg.add(new_time_step)
                policy_at_episode_steps.append(0)

            else:
                policy_at_episode_steps.append(1)

            if self.cfg.forward_agent.from_vision:
                self.train_video_recorder.record(new_time_step.observation)

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


@hydra.main(config_path="cfgs", config_name="medal")
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
