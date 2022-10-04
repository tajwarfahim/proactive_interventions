# imports from other packages
import gym
from gym import spaces, core
import numpy as np
import random
from matplotlib import pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

plt.rcParams["figure.dpi"] = 50

# imports from our code
from maze_layouts import maze_layouts


class ParticleMazeEnv(core.Env):
    def __init__(
        self,
        grid_name,
        dense=False,
        init_index=None,
        goal_index=None,
        dt=0.1,
        num_collision_steps=10,
        include_stuck_indicator=False,
        indicator_noise=0.0,
        randomize_initial_position=False,
    ):
        self.dense = dense
        self.init_index = init_index
        self.goal_index = goal_index

        self.dt = dt
        self.num_collision_steps = num_collision_steps

        self.include_stuck_indicator = include_stuck_indicator
        self.indicator_noise = indicator_noise
        self.randomize_initial_position = randomize_initial_position

        self.in_wall = False

        if self.include_stuck_indicator:
            self.observation_space = spaces.Box(
                np.array([-1.0, -1.0, 0.0], dtype=np.float32),
                np.array([1.0, 1.0, 1.0], dtype=np.float32),
            )
        else:
            self.observation_space = spaces.Box(
                np.array([-1.0, -1.0], dtype=np.float32),
                np.array([1.0, 1.0], dtype=np.float32),
            )
        self.action_space = spaces.Box(
            np.array([-1.0, -1.0], dtype=np.float32),
            np.array([1.0, 1.0], dtype=np.float32),
        )

        self.x = np.zeros(2)
        self.goal = np.zeros(2)
        self.reset_grid(grid_name)
        self.traj = []

    def step(self, action):
        """
        Action is a clipped dx. Must respect collision with walls.
        """

        # Action movement and collision detection
        action = np.clip(action, -1, 1)
        ddt = self.dt / self.num_collision_steps

        for _ in range(self.num_collision_steps):
            x_new = self.x + action * ddt
            ind = self.get_index(x_new)

            if self.in_wall and not self.grid[ind[0], ind[1]]:
                break
            else:
                self.x = x_new

        self.x = np.clip(self.x, -1, 1)
        ind = self.get_index(self.x)
        self.in_wall = self.grid[ind[0], ind[1]]

        rewards = self.get_rew()

        self.traj.append(self.x)

        info = {
            "dist_to_goal": np.linalg.norm(self.x - self.goal),
            "is_successful": 1 if np.linalg.norm(self.x - self.goal) < 0.1 else 0,
        }
        return self.get_obs(), rewards, False, info

    def get_random_pos(self, indices):
        return self.get_coords(indices[np.random.randint(indices.shape[0])])

    def reset_agent(self, mode=None):
        """
        Reset the agent's position (should be used rarely in lifelong
        learning).
        """
        if self.start_ind is not None and not self.randomize_initial_position:
            # Spawn the agent at the start state
            self.x = self.get_coords(self.start_ind)
        else:
            # Spawn the agent not too close to the goal
            self.x = self.get_random_pos(self.grid_free_index)
            while np.sum(np.square(self.x - self.goal)) < 0.5:
                self.x = self.get_random_pos(self.grid_free_index)
        self.initial_state = self.x.copy()

    def reset_grid(self, grid_name):
        """
        Changes the current grid layout, i.e. the walls of the maze. The
        agent's position is not reset, unless it would be placed inside of a
        wall by the change, in which case it spawns in the set start position.
        """
        self.grid = maze_layouts[grid_name]
        self.grid = self.grid.replace("\n", "")
        self.grid_size = int(np.sqrt(len(self.grid)))

        GS = [self.grid_size, self.grid_size]

        self.grid_chars = (np.array(list(self.grid)) != "S").reshape(GS)
        self.start_ind = np.argwhere(self.grid_chars == False)

        # Check if there is a specified start location S
        if self.init_index is not None:
            self.start_ind = np.array(self.init_index)
        elif self.start_ind.shape[0] > 0:
            self.start_ind = self.start_ind[0]
        else:
            self.start_ind = None

        # Get the goal location
        self.grid_chars = (np.array(list(self.grid)) != "G").reshape(GS)
        self.goal_ind = np.argwhere(self.grid_chars == False)

        if self.goal_index is not None:
            self.goal_ind = np.array(self.goal_index)
        else:
            self.goal_ind = self.goal_ind[0]
        self.goal = self.get_coords(self.goal_ind)

        self.grid = self.grid.replace("S", " ")
        self.grid = self.grid.replace("G", " ")

        self.grid = (np.array(list(self.grid)) != " ").reshape(GS)
        self.grid_wall_index = np.argwhere(self.grid == True)
        self.grid_free_index = np.argwhere(self.grid != True)

        # Reset the agent only if it is stuck in the wall
        cur_ind = self.get_index(self.x)
        if self.grid[cur_ind[0], cur_ind[1]]:
            self.reset()

    def reset(self):
        """
        Resets agent to initial state.
        """
        self.in_wall = False
        self.traj = []
        self.max_dist = np.linalg.norm(self.x - self.goal)

        self.reset_agent()
        return self.get_obs()

    def get_obs(self):
        """
        Observation is the coordinates of the agent and the goal.
        """
        if self.include_stuck_indicator:
            stuck = self.is_stuck_state(self.x)
            # if np.random.rand() < self.indicator_noise:
            #     stuck = not stuck
            stuck += np.random.normal(loc=float(stuck), scale=0.2)
            return np.concatenate(
                [
                    self.x.astype(np.float32),
                    np.array([stuck]).astype(np.float32),
                ]
            )
        else:
            return self.x.astype(np.float32)

    def get_rew(self):
        """
        Reward for the agent, based on the current state. Environment supports
        dense and sparse reward variants.
        """
        if self.dense:
            return (self.max_dist - np.linalg.norm(self.x - self.goal)) / self.max_dist
        else:
            return (
                1
                if np.linalg.norm(self.x - self.goal) < 0.1 and not self.in_wall
                else 0
            )

    def get_coords(self, index):
        """
        Convert indices of grid into coordinates.
        """
        return ((index + 0.5) / self.grid_size) * 2 - 1

    def get_index(self, coords):
        """
        Convert coordinates to indices of grid.
        """
        return np.clip(
            (((coords + 1) * 0.5) * (self.grid_size)) + 0, 0, self.grid_size - 1
        ).astype(np.int8)

    def is_successful(self, obs):
        return (
            1 if np.linalg.norm(obs[:2] - self.goal) < 0.1 and not self.in_wall else 0
        )

    def is_initial_state(self, obs):
        return np.all(self.get_index(self.initial_state) == self.get_index(obs))

    def is_stuck_state(self, obs):
        if self.include_stuck_indicator:
            assert obs.shape[-1] <= 3 and len(obs.shape) <= 2
        else:
            assert obs.shape[-1] == 2 and len(obs.shape) <= 2

        # dealing with batches
        if len(obs.shape) == 2:
            ind = self.get_index(coords=obs[:, :2])
            assert ind.shape == obs[:, :2].shape
            result = self.grid[ind[:, 0], ind[:, 1]]

        # dealing with single observation
        elif len(obs.shape) == 1:
            ind = self.get_index(coords=obs[:2])
            assert ind.shape == obs[:2].shape
            result = self.grid[ind[0], ind[1]]

        else:
            raise ValueError("Given obs shape not supported.")

        return result

    def render(self, mode="rgb_array"):
        grid = self.grid.copy().astype(np.float32)
        fig, ax = plt.subplots(figsize=(grid.shape[0], grid.shape[1]))
        ax.text(
            self.start_ind[0],
            self.start_ind[1],
            "S",
            va="center",
            ha="center",
            fontsize=30,
            color="white",
        )
        ax.text(
            self.goal_ind[0],
            self.goal_ind[1],
            "G",
            va="center",
            ha="center",
            fontsize=30,
            color="white",
        )

        width, height = fig.get_size_inches() * fig.get_dpi()
        self.width = int(width)
        self.height = int(height)
        canvas = FigureCanvas(fig)

        grid[self.start_ind[0], self.start_ind[1]] = 0.5
        grid[self.goal_ind[0], self.goal_ind[1]] = 0.5
        ax.imshow(grid, cmap=plt.cm.get_cmap("binary"))
        plt.axis("off")

        if len(self.traj) != 0:
            xs = np.array(self.traj)
            coords = np.clip(
                (((xs + 1) * 0.5) * (self.grid_size)) - 0.5, 0, self.grid_size - 1
            )
            plt.scatter(coords[:, 1], coords[:, 0], s=50)

        canvas.draw()
        image = np.frombuffer(canvas.tostring_rgb(), dtype="uint8").reshape(
            self.height, self.width, 3
        )
        plt.close()

        return image
