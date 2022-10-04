import earl_benchmark
import sticky_wall_env
import gym
import numpy as np

from backend.wrappers import (
    ActionRepeatWrapper,
    ObsActionDTypeWrapper,
    ExtendedTimeStepWrapper,
    ActionScaleWrapper,
    DMEnvFromGymWrapper,
)


def add_wrappers(env, action_repeat, default_stuck_value):
    wrapped_env = DMEnvFromGymWrapper(env)
    wrapped_env = ObsActionDTypeWrapper(wrapped_env, np.float32, np.float32)
    wrapped_env = ActionRepeatWrapper(wrapped_env, action_repeat)
    wrapped_env = ActionScaleWrapper(wrapped_env, minimum=-1.0, maximum=+1.0)
    wrapped_env = ExtendedTimeStepWrapper(wrapped_env, default_stuck_value)

    return wrapped_env


def make(
    name, frame_stack, action_repeat, default_stuck_value, train_horizon, eval_horizon, reward_type, seed,
):
    env_loader = earl_benchmark.EARLEnvs(
        name,
        reward_type=reward_type,
        reset_train_env_at_goal=False,
        setup_as_lifelong_learning=False,
        train_horizon=train_horizon,
        eval_horizon=eval_horizon,
    )

    train_env, eval_env = env_loader.get_envs()
    reset_states = env_loader.get_initial_states()
    reset_state_shape = reset_states.shape[1:]
    goal_states = env_loader.get_goal_states()
    if env_loader.has_demos():
        forward_demos, backward_demos = env_loader.get_demonstrations()
    else:
        forward_demos, backward_demos = None, None

    # add wrappers
    train_env = add_wrappers(
        env=train_env,
        action_repeat=action_repeat,
        default_stuck_value=default_stuck_value,
    )
    eval_env = add_wrappers(
        env=eval_env,
        action_repeat=action_repeat,
        default_stuck_value=default_stuck_value,
    )

    return train_env, eval_env, reset_states, goal_states, forward_demos, backward_demos
