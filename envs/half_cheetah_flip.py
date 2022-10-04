import numpy as np

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv


goals = [3., 4., 5., 6., 7., 8.]
initial_states = np.array([[*([0.] * 18), goal] for goal in goals])
goal_states = np.array([[*([0.] * 17), goal, goal] for goal in goals])


class HalfCheetahFlipEnv(HalfCheetahEnv):
	def __init__(
		self,
		*args,
		reward_type="sparse",
		success_threshold=0.1,
		goal_switch_freq=500,
	):
		self._reward_type = reward_type
		self._success_threshold = success_threshold
		self._max_goal_velocity = np.max(goals)
		self._goal_switch_freq = goal_switch_freq
		self._set_goal_velocity()
		self._episode_step = 0

		super(HalfCheetahFlipEnv, self).__init__(*args)

	def reset(self):
		self._episode_step = 0
		obs = super(HalfCheetahFlipEnv, self).reset()
		return np.concatenate((obs, [0.0, self._set_goal_velocity()]))

	def step(self, action):
		obs, _, done, info = super(HalfCheetahFlipEnv, self).step(action)

		x_velocity = info["x_velocity"]
		obs = np.concatenate((obs, [x_velocity, self._goal_velocity]))

		reward = self.compute_reward(obs, action)

		if (self._episode_step + 1) % self._goal_switch_freq == 0:
			self._set_goal_velocity()

		self._episode_step += 1
		return obs, reward, done, info

	def compute_reward(self, obs, action):
		if self._reward_type == "sparse":
			reward = float(self.is_successful(obs))
		elif self._reward_type == "dense":
			velocity_diff = np.abs(obs[-2] - obs[-1])
			reward = 0.95 * (self._max_goal_velocity - velocity_diff) / self._max_goal_velocity
			reward += 0.05 * (6. - np.sum(np.square(action))) / 6.
		else:
			raise NotImplementedError
		return reward

	def _set_goal_velocity(self, goal_velocity=None):
		if goal_velocity is not None:
			self._goal_velocity = goal_velocity
		else:
			self._goal_velocity = self.np_random.choice(goals)
		return self._goal_velocity

	def is_successful(self, obs):
		return np.abs(obs[-2] - obs[-1]) < self._success_threshold

	def is_stuck_state(self, obs):
		if len(obs.shape) == 2:
			result = np.abs(np.arctan2(np.sin(obs[:, 1]), np.cos(obs[:, 1]))) > (2/3)*np.pi
		elif len(obs.shape) == 1:
			result = np.abs(np.arctan2(np.sin(obs[1]), np.cos(obs[1]))) > (2/3)*np.pi
		else:
			raise ValueError("Given obs shape not supported.")
		return result

	def get_init_states(self):
		return initial_states

	def get_goal_states(self):
		return goal_states

if __name__ == "__main__":
	env = HalfCheetahFlipEnv()
	env.reset()

	for _ in range(1000):
		obs, _, _, _ = env.step(env.action_space.sample() * 10.0)
		if env.is_stuck_state(obs):
			img = env.sim.render(width=4096, height=4096)
			break
	import imageio
	import cv2

	img = cv2.flip(img, 0)
	imageio.imwrite('cheetah_flip.png', img)
