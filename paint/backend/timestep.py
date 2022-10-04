import enum
from typing import Any, NamedTuple


class StepType(enum.IntEnum):
    """Defines the status of a `TimeStep` within a sequence."""

    # Denotes the first `TimeStep` in a sequence.
    FIRST = 0
    # Denotes any `TimeStep` in a sequence that is not FIRST or LAST.
    MID = 1
    # Denotes the last `TimeStep` in a sequence.
    LAST = 2

    def first(self) -> bool:
        return self is StepType.FIRST

    def mid(self) -> bool:
        return self is StepType.MID

    def last(self) -> bool:
        return self is StepType.LAST


class ExtendedTimeStep(NamedTuple):
    step_type: Any
    reward: Any
    discount: Any
    observation: Any
    action: Any
    is_stuck: Any

    def first(self):
        return self.step_type == StepType.FIRST

    def mid(self):
        return self.step_type == StepType.MID

    def last(self):
        return self.step_type == StepType.LAST

    def __getitem__(self, attr):
        return getattr(self, attr)


def get_first_stuck_state(episode_time_steps, train_env):
    low = 0
    high = len(episode_time_steps) - 1
    found = False
    num_steps = 0

    while low < high:
        num_steps += 1
        mid = low + (high - low) // 2
        obs = episode_time_steps[mid].observation

        if train_env.is_stuck_state(obs=obs):
            high = mid
            found = True
        else:
            low = mid + 1

    if found:
        return low, num_steps
    else:
        return None, num_steps


def relabel_timesteps_with_correct_stuck_info(episode_time_steps, train_env):
    first_stuck_state, num_steps = get_first_stuck_state(
        episode_time_steps=episode_time_steps,
        train_env=train_env,
    )

    modified_episode_time_steps = []
    for i in range(len(episode_time_steps)):
        is_stuck_info = 0.0

        # time_step.is_stuck -> whether next state, s', is stuck or not
        if first_stuck_state is not None and i >= first_stuck_state - 1:
            is_stuck_info = 1.0

        modified_time_step = ExtendedTimeStep(
            observation=episode_time_steps[i].observation,
            step_type=episode_time_steps[i].step_type,
            action=episode_time_steps[i].action,
            reward=episode_time_steps[i].reward,
            discount=episode_time_steps[i].discount,
            is_stuck=is_stuck_info,
        )

        modified_episode_time_steps.append(modified_time_step)

    return modified_episode_time_steps, num_steps
