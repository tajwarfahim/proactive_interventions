# general package import
import gym
import pickle
import os
import numpy as np

# import from our maze env code
from registration import register_environments

register_environments()


# assume start point is already in the partial path
def generate_partial_path(
    env, obs, target_point, num_steps, noise, tol, stopping_criterion, steps
):
    observations, actions, next_observations, rewards, terminals, infos = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    mid = env.get_coords(target_point)

    while np.linalg.norm(obs - mid) > tol and steps < num_steps:
        act = (mid - obs) / np.linalg.norm(mid - obs)
        noisy_act = act + np.random.normal(scale=noise, size=2)
        obs, rew, done, info = env.step(noisy_act)
        steps += 1

        next_observations.append(obs)
        observations.append(obs)
        actions.append(noisy_act)
        rewards.append(rew)
        infos.append(info)

        if stopping_criterion is not None:
            done = stopping_criterion(rew)
            terminals.append(done)
            if done:
                break
        else:
            terminals.append(done)

    return observations, actions, next_observations, rewards, terminals, infos, steps


def save_demonstrations(
    observations, actions, next_observations, rewards, terminals, infos, save_dir
):
    total_steps = len(observations)
    lsts = [next_observations, actions, rewards, terminals, infos]
    for lst in lsts:
        assert len(lst) == total_steps

    observations = np.array(observations, dtype=np.float32).reshape(total_steps, -1)
    next_observations = np.array(next_observations, dtype=np.float32).reshape(
        total_steps, -1
    )
    actions = np.array(actions, dtype=np.float32).reshape(total_steps, -1)
    rewards = np.array(rewards, dtype=np.float32).reshape(total_steps, -1)
    terminals = np.array(terminals).reshape(total_steps, -1)
    infos = np.array(infos)

    data = {
        "observations": observations,
        "actions": actions,
        "rewards": rewards,
        "terminals": terminals,
        "next_observations": next_observations,
        "infos": infos,
    }

    if save_dir is not None:
        demo_save_file = os.path.join(save_dir, "demo_data.pkl")
        with open(demo_save_file, "wb") as handle:
            pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def generate_demos(
    env_name,
    num_demos,
    num_steps,
    noise,
    tol,
    traj_points,
    save_dir=None,
    return_sample_demo=False,
):
    env = gym.make(env_name)

    # we are working in the sparse reward setting
    assert not env.dense

    observations, actions, next_observations, rewards, terminals, infos = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    traj_points.append(env.goal_ind)

    for demo_index in range(num_demos):
        obs = env.reset()
        observations.append(obs)
        steps = 0

        for i in range(len(traj_points)):
            obs = observations[-1]

            stopping_criterion = None
            if i == len(traj_points) - 1:
                stopping_criterion = lambda x: x > 0

            (
                p_observations,
                p_actions,
                p_next_obs,
                p_rewards,
                p_terminals,
                p_infos,
                steps,
            ) = generate_partial_path(
                env=env,
                obs=obs,
                target_point=np.array(traj_points[i]),
                num_steps=num_steps,
                noise=noise,
                tol=tol,
                stopping_criterion=stopping_criterion,
                steps=steps,
            )

            observations += p_observations
            actions += p_actions
            next_observations += p_next_obs
            rewards += p_rewards
            terminals += p_terminals
            infos += p_infos

        observations = observations[:-1]

    save_demonstrations(
        observations=observations,
        actions=actions,
        next_observations=next_observations,
        rewards=rewards,
        terminals=terminals,
        infos=infos,
        save_dir=save_dir,
    )

    if return_sample_demo:
        return env.render()
