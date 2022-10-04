import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from networks import (
    RandomShiftsAug,
    Encoder,
    DDPGActor,
    SACActor,
    BCActor,
    Critic,
)


class SACAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        reward_scale_factor,
        use_tb,
        from_vision,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.critic_target_tau = critic_target_tau
        self.reward_scale_factor = reward_scale_factor
        self.use_tb = use_tb
        self.from_vision = from_vision
        # Changed log_std_bounds from [-10, 2] -> [-20, 2]
        self.log_std_bounds = [-20, 2]
        # Changed self.init_temperature to 1.0
        self.init_temperature = 1.0

        # models
        self.create_networks()
        self.create_dual_variables()

        # Changed target entropy from -dim(A) -> -dim(A)/2
        self.target_entropy = -action_shape[0] / 2.0

        # optimizers
        self.create_optimizers()

        self.train()
        self.critic_target.train()

    def create_dual_variables(self):
        self.log_alpha = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_alpha.requires_grad = True

    def create_optimizers(self):
        self.log_alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=self.lr)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=self.lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=self.lr)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=self.lr)
            # data augmentation
            self.aug = RandomShiftsAug(pad=4)

    def make_critic(self):
        critic = Critic(
            self.model_repr_dim,
            self.action_shape,
            self.feature_dim,
            self.hidden_dim,
        ).to(self.device)

        return critic

    def create_networks(self):
        if self.from_vision:
            self.encoder = Encoder(self.obs_shape).to(self.device)
            self.model_repr_dim = self.encoder.repr_dim
        else:
            self.model_repr_dim = self.obs_shape[0]

        self.actor = SACActor(
            self.model_repr_dim,
            self.action_shape,
            self.feature_dim,
            self.hidden_dim,
            self.log_std_bounds,
        ).to(self.device)

        self.critic = self.make_critic()
        self.critic_target = self.make_critic()
        self.critic_target.load_state_dict(self.critic.state_dict())

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)
        self.critic.train(training)

    def act(self, obs, uniform_action=False, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        if self.from_vision:
            obs = self.encoder(obs.unsqueeze(0))

        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()

        if uniform_action:
            action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()

    def get_Q_value(self, obs, action):
        with torch.no_grad():
            target_Q1, target_Q2 = self.critic_target(obs, action)
            Q_value = torch.min(target_Q1, target_Q2)

        return Q_value

    def calculate_critic_target_Q(self, next_obs, reward, not_done, discount, is_stuck):
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)
            target_V = self.get_Q_value(obs=next_obs, action=next_action)
            target_V -= self.alpha.detach() * log_prob
            target_Q = self.reward_scale_factor * reward + (
                discount * target_V * not_done.unsqueeze(1)
            )

        return target_Q

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, not_done, is_stuck
    ):
        metrics = dict()
        target_Q = self.calculate_critic_target_Q(
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )
        Q1, Q2 = self.critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def calculate_actor_loss(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        actor_loss = -Q + (self.alpha.detach() * log_prob)
        return actor_loss.mean()

    def update_actor(self, obs, step):
        metrics = dict()

        actor_loss = self.calculate_actor_loss(obs=obs)

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def update_alpha(self, obs, step):
        metrics = dict()

        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha * (-log_prob - self.target_entropy).detach()).mean()
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

        if self.use_tb:
            metrics["actor_logprob"] = log_prob.mean().item()
            metrics["alpha_loss"] = alpha_loss
            metrics["alpha_value"] = self.alpha
        return metrics

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        ) = utils.to_torch(batch, self.device)

        return (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        )

    def update(self, trans_tuple, step):
        metrics = dict()

        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        ) = trans_tuple

        not_done = next_step_type.clone()
        not_done[not_done < 2] = 1
        not_done[not_done == 2] = 0

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                step,
                not_done,
                is_stuck,
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        return metrics


class SafetyCriticSACAgent(SACAgent):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        reward_scale_factor,
        use_tb,
        from_vision,
        epsilon,
    ):
        self.epsilon = epsilon

        super(SafetyCriticSACAgent, self).__init__(
            obs_shape,
            action_shape,
            device,
            lr,
            feature_dim,
            hidden_dim,
            critic_target_tau,
            reward_scale_factor,
            use_tb,
            from_vision,
        )

        self.safety_critic_target.train()

        self.stuck_state_detector = None
        self.stuck_state_detector_domain = None

    def create_networks(self):
        super(SafetyCriticSACAgent, self).create_networks()
        self.safety_critic = self.make_critic()
        self.safety_critic_target = self.make_critic()
        self.safety_critic_target.load_state_dict(self.safety_critic.state_dict())

    def create_dual_variables(self):
        super(SafetyCriticSACAgent, self).create_dual_variables()

        # additional dual variable
        self.log_lambda = torch.tensor(np.log(self.init_temperature)).to(self.device)
        self.log_lambda.requires_grad = True

    @property
    def _lambda(self):
        return self.log_lambda.exp()

    def train(self, training=True):
        super(SafetyCriticSACAgent, self).train(training)
        self.safety_critic.train(training)

    def create_optimizers(self):
        super(SafetyCriticSACAgent, self).create_optimizers()

        # additional optimizers
        self.log_lambda_optimizer = torch.optim.Adam([self.log_lambda], lr=self.lr)

        self.safety_critic_opt = torch.optim.Adam(
            self.safety_critic.parameters(), lr=self.lr
        )

    def set_stuck_state_detector(self, stuck_state_detector, domain):
        self.stuck_state_detector = stuck_state_detector
        self.stuck_state_detector_domain = domain

    def compute_reward_for_safety_critic(self, next_obs):
        if self.stuck_state_detector is None:
            return None

        if self.stuck_state_detector_domain == "numpy":
            next_obs = next_obs.detach().cpu().numpy()

        is_stuck = self.stuck_state_detector(next_obs)

        if self.stuck_state_detector_domain == "numpy":
            is_stuck = torch.from_numpy(is_stuck).to(self.device)

        return is_stuck

    def update_safety_critic(
        self, obs, action, discount, next_obs, step, not_done, is_stuck
    ):
        assert self.stuck_state_detector is not None
        assert self.stuck_state_detector_domain is not None

        metrics = dict()
        with torch.no_grad():
            dist = self.actor(next_obs)
            next_action = dist.rsample()
            log_prob = dist.log_prob(next_action).sum(-1, keepdim=True)

            target_Q1, target_Q2 = self.safety_critic_target(next_obs, next_action)
            target_V = torch.min(target_Q1, target_Q2)

            # reward -> I(s')
            reward = self.compute_reward_for_safety_critic(next_obs=next_obs)
            if reward is None:
                reward = is_stuck

            reward = reward.type(target_V.dtype).reshape(-1, 1)

            target_Q_safe = reward + discount * target_V * not_done.unsqueeze(1)

        Q1_safe, Q2_safe = self.safety_critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss_safe = 0.5 * (
            F.mse_loss(Q1_safe, target_Q_safe) + F.mse_loss(Q2_safe, target_Q_safe)
        )

        if self.use_tb:
            metrics["critic_target_q_safe"] = target_Q_safe.mean().item()
            metrics["critic_q1_safe"] = Q1_safe.mean().item()
            metrics["critic_q2_safe"] = Q2_safe.mean().item()
            metrics["critic_loss_safe"] = critic_loss_safe.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)

        self.safety_critic_opt.zero_grad(set_to_none=True)
        critic_loss_safe.backward()
        self.safety_critic_opt.step()

        return metrics

    def calculate_actor_loss(self, obs):
        dist = self.actor(obs)
        action = dist.rsample()
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        Q1, Q2 = self.critic(obs, action)
        Q = torch.min(Q1, Q2)

        Q1_safe, Q2_safe = self.safety_critic(obs, action)
        Q_safe = torch.min(Q1_safe, Q2_safe)

        actor_loss = (
            -Q + (self.alpha.detach() * log_prob) + self._lambda.detach() * Q_safe
        )

        return actor_loss.mean()

    def update_lambda(self, obs, action):
        metrics = dict()

        Q1_safe, Q2_safe = self.safety_critic(obs, action)
        Q_safe = torch.min(Q1_safe, Q2_safe)

        self.log_lambda_optimizer.zero_grad()
        lambda_loss = (self._lambda * (self.epsilon - Q_safe).detach()).mean()
        lambda_loss.backward()
        self.log_lambda_optimizer.step()

        if self.use_tb:
            metrics["lambda_loss"] = lambda_loss
            metrics["lambda_value"] = self._lambda
        return metrics

    def update(self, trans_tuple, step):
        metrics = dict()

        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        ) = trans_tuple

        not_done = next_step_type.clone()
        not_done[not_done < 2] = 1
        not_done[not_done == 2] = 0

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update critic
        metrics.update(
            self.update_critic(
                obs,
                action,
                reward,
                discount,
                next_obs,
                step,
                not_done,
                is_stuck,
            )
        )

        # update safety critic
        metrics.update(
            self.update_safety_critic(
                obs, action, discount, next_obs, step, not_done, is_stuck
            )
        )

        # update actor
        metrics.update(self.update_actor(obs.detach(), step))

        # update alpha
        metrics.update(self.update_alpha(obs.detach(), step))

        # update lambda
        metrics.update(self.update_lambda(obs, action))

        # update critic target
        utils.soft_update_params(
            self.critic, self.critic_target, self.critic_target_tau
        )

        # update safety critic target
        utils.soft_update_params(
            self.safety_critic, self.safety_critic_target, self.critic_target_tau
        )

        return metrics


class SafeSACAgent(SACAgent):
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        reward_scale_factor,
        use_tb,
        from_vision,
        stuck_reward,
    ):

        super(SafeSACAgent, self).__init__(
            obs_shape,
            action_shape,
            device,
            lr,
            feature_dim,
            hidden_dim,
            critic_target_tau,
            reward_scale_factor,
            use_tb,
            from_vision,
        )

        self.stuck_state_detector = None
        self.stuck_state_detector_domain = None

        # stuck_reward = R_min / (1 - gamma) from our algorithm
        self.stuck_reward = stuck_reward

    def set_stuck_state_detector(self, stuck_state_detector, domain):
        self.stuck_state_detector = stuck_state_detector
        self.stuck_state_detector_domain = domain

    def calculate_critic_target_Q(self, next_obs, reward, not_done, discount, is_stuck):
        target_Q = super(SafeSACAgent, self).calculate_critic_target_Q(
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )

        if self.stuck_state_detector is not None:
            if self.stuck_state_detector_domain == "numpy":
                next_obs = next_obs.detach().cpu().numpy()

            is_stuck = self.stuck_state_detector(next_obs)

            if self.stuck_state_detector_domain == "numpy":
                is_stuck = torch.from_numpy(is_stuck).to(self.device)

        is_stuck = is_stuck.type(target_Q.dtype).reshape(-1, 1)
        target_Q = is_stuck * self.stuck_reward + (1.0 - is_stuck) * target_Q

        return target_Q


class SafeSACAgentStuckDiscrim(SafeSACAgent):
    def __init__(
        self,
        *agent_args,
        stuck_discrim_unsupervised=False,
        stuck_discrim_hidden_size=128,
        stuck_discrim_lr=1e-3,
        stuck_mixup=False,
        **agent_kwargs
    ):

        super(SafeSACAgentStuckDiscrim, self).__init__(**agent_kwargs)
        self.stuck_discrim_unsupervised = stuck_discrim_unsupervised
        self.stuck_discrim_hidden_size = stuck_discrim_hidden_size
        self.stuck_discrim_lr = stuck_discrim_lr
        self.stuck_mixup = stuck_mixup

        self.input_shape = input_shape = self.obs_shape[0] * (stuck_discrim_unsupervised + 1)
        self.stuck_discriminator = nn.Sequential(
            nn.Linear(input_shape, stuck_discrim_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(stuck_discrim_hidden_size, stuck_discrim_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(stuck_discrim_hidden_size, 1),
        ).to(self.device)

        self.stuck_discrim_opt = torch.optim.Adam(
            self.stuck_discriminator.parameters(), lr=stuck_discrim_lr
        )

    def reset_stuck_discrim(self):
        self.stuck_discriminator = nn.Sequential(
            nn.Linear(self.input_shape, self.stuck_discrim_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.stuck_discrim_hidden_size, self.stuck_discrim_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(self.stuck_discrim_hidden_size, 1),
        ).to(self.device)

        self.stuck_discrim_opt = torch.optim.Adam(
            self.stuck_discriminator.parameters(), lr=self.stuck_discrim_lr
        )

    def calculate_critic_target_Q(self, obs, next_obs, reward, not_done, discount, is_stuck):
        target_Q = super(SafeSACAgent, self).calculate_critic_target_Q(
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )

        if self.stuck_state_detector is not None:
            if self.stuck_state_detector_domain == "numpy":
                next_obs = next_obs.detach().cpu().numpy()

            with torch.no_grad():
                if self.stuck_discrim_unsupervised:
                    is_stuck = self.stuck_state_detector(
                        torch.cat((obs, next_obs), axis=-1)
                    )
                    is_stuck = torch.clip(is_stuck, 0.5, 1.0)
                    is_stuck = 2.0 * (is_stuck - 0.5)
                else:
                    is_stuck = self.stuck_state_detector(next_obs)

            if self.stuck_state_detector_domain == "numpy":
                is_stuck = torch.from_numpy(is_stuck).to(self.device)

            is_stuck = is_stuck.type(target_Q.dtype).reshape(-1, 1)
        else:
            is_stuck = is_stuck.type(target_Q.dtype).reshape(-1, 1)
            with torch.no_grad():
                if self.stuck_discrim_unsupervised:
                    is_stuck_preds = torch.sigmoid(self.stuck_discriminator(
                        torch.cat((obs, next_obs), -1)
                    ))
                    is_stuck_preds = torch.clip(is_stuck_preds, 0.5, 1.0)
                    is_stuck_preds = 2.0 * (is_stuck_preds - 0.5)
                else:
                    is_stuck_preds = torch.sigmoid(self.stuck_discriminator(next_obs))
                is_stuck_preds = is_stuck_preds.reshape(-1, 1)

            is_stuck = torch.where(is_stuck == -1.0, is_stuck_preds, is_stuck)

        target_Q = is_stuck * self.stuck_reward + (1.0 - is_stuck) * target_Q

        return target_Q

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, not_done, is_stuck
    ):
        metrics = dict()
        target_Q = self.calculate_critic_target_Q(
            obs=obs,
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )
        Q1, Q2 = self.critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics

    def update_stuck_discriminator(self, pos_replay_iter, neg_replay_iter):
        if self.from_vision:
            print("update_stuck_discrim does not support vision")
            exit()

        metrics = dict()

        batch_pos = next(pos_replay_iter)
        obs_pos = utils.to_torch(batch_pos, self.device)[0]
        num_pos = obs_pos.shape[0]

        batch_neg = next(neg_replay_iter)
        obs_neg = utils.to_torch(batch_neg, self.device)[0]
        num_neg = obs_neg.shape[0]

        if self.stuck_mixup:
            alpha = 1.0
            beta_dist = torch.distributions.beta.Beta(
                torch.tensor([alpha]), torch.tensor([alpha])
            )

            l = beta_dist.sample([num_pos + num_neg])
            mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(
                self.device
            )
            disc_inputs = torch.cat((obs_pos, obs_neg), 0)

            ridxs = torch.randperm(num_pos + num_neg)
            perm_labels = labels[ridxs]
            perm_disc_inputs = disc_inputs[ridxs]

            images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
            labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
        else:
            images = torch.cat((obs_pos, obs_neg), 0)
            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(
                self.device
            )

        loss = torch.nn.BCELoss()
        m = nn.Sigmoid()
        discrim_loss = loss(m(self.stuck_discriminator(images)), labels)

        self.stuck_discrim_opt.zero_grad(set_to_none=True)
        discrim_loss.backward()
        self.stuck_discrim_opt.step()

        if self.use_tb:
            metrics["stuck_discriminator_loss"] = discrim_loss.item()

        return metrics


class MEDALBackwardAgent(SACAgent):
    def __init__(
        self,
        *agent_args,
        discrim_hidden_size=128,
        discrim_lr=3e-4,
        mixup=True,
        discrim_eps=1e-10,
        **agent_kwargs
    ):

        super(MEDALBackwardAgent, self).__init__(**agent_kwargs)
        self.discrim_hidden_size = discrim_hidden_size
        self.discrim_lr = discrim_lr
        self.discrim_eps = discrim_eps
        self.mixup = mixup
        self.discriminator = nn.Sequential(
            nn.Linear(self.obs_shape[0], discrim_hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(discrim_hidden_size, 1),
        ).to(self.device)

        self.discrim_opt = torch.optim.Adam(
            self.discriminator.parameters(), lr=discrim_lr
        )

    def update_discriminator(self, pos_replay_iter, neg_replay_iter):
        if self.from_vision:
            print("update_discrim does not support vision")
            exit()

        metrics = dict()

        batch_pos = next(pos_replay_iter)
        obs_pos = utils.to_torch(batch_pos, self.device)[0]
        num_pos = obs_pos.shape[0]

        batch_neg = next(neg_replay_iter)
        obs_neg = utils.to_torch(batch_neg, self.device)[0]
        num_neg = obs_neg.shape[0]

        if self.mixup:
            alpha = 1.0
            beta_dist = torch.distributions.beta.Beta(
                torch.tensor([alpha]), torch.tensor([alpha])
            )

            l = beta_dist.sample([num_pos + num_neg])
            mixup_coef = torch.reshape(l, (num_pos + num_neg, 1)).to(self.device)

            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(
                self.device
            )
            disc_inputs = torch.cat((obs_pos, obs_neg), 0)

            ridxs = torch.randperm(num_pos + num_neg)
            perm_labels = labels[ridxs]
            perm_disc_inputs = disc_inputs[ridxs]

            images = disc_inputs * mixup_coef + perm_disc_inputs * (1 - mixup_coef)
            labels = labels * mixup_coef + perm_labels * (1 - mixup_coef)
        else:
            images = torch.cat((obs_pos, obs_neg), 0)
            labels = torch.cat((torch.ones(num_pos, 1), torch.zeros(num_neg, 1)), 0).to(
                self.device
            )

        loss = torch.nn.BCELoss()
        m = nn.Sigmoid()
        discrim_loss = loss(m(self.discriminator(images)), labels)

        self.discrim_opt.zero_grad(set_to_none=True)
        discrim_loss.backward()
        self.discrim_opt.step()

        if self.use_tb:
            metrics["discriminator_loss"] = discrim_loss.item()

        return metrics

    def compute_reward(self, obs):
        actual_reward = -torch.log(
            1 - torch.sigmoid(self.discriminator(obs)) + self.discrim_eps
        )
        return actual_reward

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        ) = utils.to_torch(batch, self.device)

        return (
            obs,
            action,
            self.compute_reward(next_obs).detach(),
            discount,
            next_obs,
            step_type,
            next_step_type,
            is_stuck,
        )


class SafeMEDALAgent(MEDALBackwardAgent):
    def __init__(
        self,
        stuck_reward,
        stuck_discrim_unsupervised=False,
        **agent_kwargs,
    ):
        super(SafeMEDALAgent, self).__init__(**agent_kwargs)

        self.stuck_state_detector = None
        self.stuck_state_detector_domain = None

        # stuck_reward = R_min / (1 - gamma) from our algorithm
        self.stuck_reward = stuck_reward
        self.stuck_discrim_unsupervised = stuck_discrim_unsupervised

    def set_stuck_state_detector(self, stuck_state_detector, domain):
        self.stuck_state_detector = stuck_state_detector
        self.stuck_state_detector_domain = domain

    def set_stuck_discriminator(self, stuck_discriminator):
        self.stuck_discriminator = stuck_discriminator

    def calculate_critic_target_Q(self, obs, next_obs, reward, not_done, discount, is_stuck):
        target_Q = super(SafeMEDALAgent, self).calculate_critic_target_Q(
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )

        if self.stuck_state_detector is not None:
            if self.stuck_state_detector_domain == "numpy":
                next_obs = next_obs.detach().cpu().numpy()

            with torch.no_grad():
                if self.stuck_discrim_unsupervised:
                    is_stuck = self.stuck_state_detector(
                        torch.cat((obs, next_obs), -1)
                    )
                else:
                    is_stuck = self.stuck_state_detector(next_obs)

            if self.stuck_state_detector_domain == "numpy":
                is_stuck = torch.from_numpy(is_stuck).to(self.device)
        else:
            is_stuck = is_stuck.type(target_Q.dtype).reshape(-1, 1)
            with torch.no_grad():
                if self.stuck_discrim_unsupervised:
                    is_stuck_preds = torch.sigmoid(self.stuck_discriminator(
                        torch.cat((obs, next_obs), -1)
                    ))
                else:
                    is_stuck_preds = torch.sigmoid(self.stuck_discriminator(next_obs))
                is_stuck_preds = is_stuck_preds.reshape(-1, 1)

            is_stuck = torch.where(is_stuck == -1.0, is_stuck_preds, is_stuck)

        is_stuck = is_stuck.type(target_Q.dtype).reshape(-1, 1)
        target_Q = is_stuck * self.stuck_reward + (1.0 - is_stuck) * target_Q

        return target_Q

    def update_critic(
        self, obs, action, reward, discount, next_obs, step, not_done, is_stuck
    ):
        metrics = dict()
        target_Q = self.calculate_critic_target_Q(
            obs=obs,
            next_obs=next_obs,
            reward=reward,
            not_done=not_done,
            discount=discount,
            is_stuck=is_stuck,
        )
        Q1, Q2 = self.critic(obs, action)
        # scaled the loss by 0.5, might have some effect initially
        critic_loss = 0.5 * (F.mse_loss(Q1, target_Q) + F.mse_loss(Q2, target_Q))

        if self.use_tb:
            metrics["critic_target_q"] = target_Q.mean().item()
            metrics["critic_q1"] = Q1.mean().item()
            metrics["critic_q2"] = Q2.mean().item()
            metrics["critic_loss"] = critic_loss.item()

        # optimize encoder and critic
        if self.from_vision:
            self.encoder_opt.zero_grad(set_to_none=True)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()
        if self.from_vision:
            self.encoder_opt.step()

        return metrics


class BCAgent:
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        lr,
        feature_dim,
        hidden_dim,
        critic_target_tau,
        reward_scale_factor,
        use_tb,
        from_vision,
    ):
        self.obs_shape = obs_shape
        self.action_shape = action_shape
        self.lr = lr
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.device = device
        self.use_tb = use_tb
        self.from_vision = from_vision
        # Changed log_std_bounds from [-10, 2] -> [-20, 2]
        self.log_std_bounds = [-20, 2]
        # Changed self.init_temperature to 1.0
        self.init_temperature = 1.0

        # models
        if self.from_vision:
            self.encoder = Encoder(obs_shape).to(device)
            model_repr_dim = self.encoder.repr_dim
        else:
            model_repr_dim = obs_shape[0]

        self.actor = BCActor(
            model_repr_dim,
            action_shape,
            feature_dim,
            hidden_dim,
        ).to(device)

        # optimizers
        if self.from_vision:
            self.encoder_opt = torch.optim.Adam(self.encoder.parameters(), lr=lr)
            # data augmentation
            self.aug = RandomShiftsAug(pad=4)

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)

        self.train()

    def train(self, training=True):
        self.training = training
        if self.from_vision:
            self.encoder.train(training)
        self.actor.train(training)

    def act(self, obs, uniform_action=False, eval_mode=False):
        obs = torch.as_tensor(obs, device=self.device)
        if self.from_vision:
            obs = self.encoder(obs.unsqueeze(0))

        dist = self.actor(obs)
        if eval_mode:
            action = dist.mean
        else:
            action = dist.sample()

        if uniform_action:
            action.uniform_(-1.0, 1.0)

        return action.cpu().numpy()

    def update_actor(self, obs, action):
        metrics = dict()

        dist = self.actor(obs)
        log_prob = dist.log_prob(action).sum(-1, keepdim=True)

        actor_loss = -log_prob
        actor_loss = actor_loss.mean()

        # optimize actor
        self.actor_opt.zero_grad(set_to_none=True)
        actor_loss.backward()
        self.actor_opt.step()

        if self.use_tb:
            metrics["actor_loss"] = actor_loss.item()

        return metrics

    def transition_tuple(self, replay_iter):
        batch = next(replay_iter)
        (
            obs,
            action,
            reward,
            discount,
            next_obs,
            step_type,
            next_step_type,
        ) = utils.to_torch(batch, self.device)

        return (obs, action, reward, discount, next_obs, step_type, next_step_type)

    def update(self, trans_tuple, step):
        metrics = dict()

        obs, action, reward, discount, next_obs, step_type, next_step_type = trans_tuple

        not_done = next_step_type.clone()
        not_done[not_done < 2] = 1
        not_done[not_done == 2] = 0

        # augment
        if self.from_vision:
            obs = self.aug(obs.float())
            next_obs = self.aug(next_obs.float())
            # encode
            obs = self.encoder(obs)
            with torch.no_grad():
                next_obs = self.encoder(next_obs)

        if self.use_tb:
            metrics["batch_reward"] = reward.mean().item()

        # update actor
        metrics.update(self.update_actor(obs.detach(), action.detach()))

        return metrics
