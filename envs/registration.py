import gym


ENVIRONMENT_SPECS = (
    *[
        {
            "id": "maze-config{}".format(i),
            "entry_point": "sticky_wall_env.maze:ParticleMazeEnv",
            "kwargs": {"grid_name": str(maze_id)},
        }
        for i, maze_id in enumerate(range(1, 11))
    ],
    *[
        {
            "id": "maze-walls-{}".format(name),
            "entry_point": "sticky_wall_env.maze:ParticleMazeEnv",
            "kwargs": {"grid_name": "walls-{}".format(name)},
        }
        for name in ["000", "001", "010", "011", "100", "101", "110", "111"]
    ],
    *[
        {
            "id": "maze-walls-{}-gn1".format(name),
            "entry_point": "sticky_wall_env.maze:ParticleMazeEnv",
            "kwargs": {"grid_name": "walls-{}".format(name), "include_stuck_indicator": True, "indicator_noise": 1.0},
        }
        for name in ["000", "001", "010", "011", "100", "101", "110", "111"]
    ],
    {
        "id": "half_cheetah_flip",
        "entry_point": "half_cheetah_flip:HalfCheetahFlipEnv",
        "kwargs": {"reward_type": "dense"}
    },
)


def register_environments():
    for env in ENVIRONMENT_SPECS:
        gym.register(**env)

    gym_ids = tuple(environment_spec["id"] for environment_spec in ENVIRONMENT_SPECS)

    return
