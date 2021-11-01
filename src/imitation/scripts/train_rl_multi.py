"""Uses RL to train a policy for multiple ground-truth rewards from scratch,
saving rollouts and policy.

This can be used:
    1. To train a policy on a ground-truth reward function, as a source of
       synthetic "expert" demonstrations to train IRL or imitation learning
       algorithms.
    2. To train a policy on a learned reward function, to solve a task or
       as a way of evaluating the quality of the learned reward function.
"""

import logging
import os
import os.path as osp
from typing import Mapping, Optional

import sacred.run
from sacred.observers import FileStorageObserver
from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize

from imitation.data import rollout, wrappers
from imitation.policies import serialize
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import load_reward
from imitation.scripts.common import common, rl, train
from imitation.scripts.config.train_rl_multi import train_rl_multi_ex


# Reward weights and config settings corresponding to the `highway-fast` env. 
DEFAULT_ENV_CONFIG = dict(
    collision_reward=-1.,
    # right_lane_reward=0.,
    # high_speed_reward=0.,
    simulation_frequency=5,
    lanes_count=3,
    vehicles_count=20,
    duration=30,  # [s]
    ego_spacing=1.5,
)


@train_rl_multi_ex.main
def train_rl_multi(
    *,
    _run: sacred.run.Run,
    _seed: int,
    total_timesteps: int,
    normalize: bool,
    normalize_kwargs: dict,
    reward_type: Optional[str],
    reward_path: Optional[str],
    rollout_save_final: bool,
    rollout_save_n_timesteps: Optional[int],
    rollout_save_n_episodes: Optional[int],
    policy_save_interval: int,
    policy_save_final: bool,
) -> Mapping[str, float]:
    """Trains an expert policy from scratch and saves the rollouts and policy.
      Does this repeatedly for multiple reward instantiations, saving each policy
      separately, but combines the rollouts into a single pickle file.

    Checkpoints:
      At applicable training steps `step` (where step is either an integer or
      "final"):

        - Policies are saved to `{log_dir}/policies/{step}/`.
        - Rollouts are saved to `{log_dir}/rollouts/{step}.pkl`.

    Args:
        total_timesteps: Number of training timesteps in `model.learn()`.
        normalize: If True, then rescale observations and reward.
        normalize_kwargs: kwargs for `VecNormalize`.
        reward_type: If provided, then load the serialized reward of this type,
            wrapping the environment in this reward. This is useful to test
            whether a reward model transfers. For more information, see
            `imitation.rewards.serialize.load_reward`.
        reward_path: A specifier, such as a path to a file on disk, used by
            reward_type to load the reward model. For more information, see
            `imitation.rewards.serialize.load_reward`.
        rollout_save_final: If True, then save rollouts right after training is
            finished.
        rollout_save_n_timesteps: The minimum number of timesteps saved for every
            policy. Could be more than `rollout_save_n_timesteps` because
            trajectories are saved by episode rather than by transition.
            Must set exactly one of `rollout_save_n_timesteps`
            and `rollout_save_n_episodes`.
        rollout_save_n_episodes: The number of episodes saved for every
            policy. Must set exactly one of `rollout_save_n_timesteps` and
            `rollout_save_n_episodes`.
        policy_save_interval: The number of training updates between in between
            intermediate rollout saves. If the argument is nonpositive, then
            don't save intermediate updates.
        policy_save_final: If True, then save the policy right after training is
            finished.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    custom_logger, log_dir = common.setup_logging()
    rollout_dir = osp.join(log_dir, "rollouts")
    policy_dir = osp.join(log_dir, "policies")
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    # TODO(jon): make this a parameter.
    coll_rewards = [-1., 0.]  # N.B. `collision_reward` is only defined on the range [-1, 0].

    rollout_stats = []
    rl_algos = []
    for it, cr in enumerate(coll_rewards):
        print("======================")
        print("  Processing iter: {}".format(it))
        print("======================")

        env_config_kwargs = DEFAULT_ENV_CONFIG
        env_config_kwargs["collision_reward"] = cr

        venv = common.make_venv(
            post_wrappers=[lambda env, idx: wrappers.RolloutInfoWrapper(env)],
            env_config_kwargs=env_config_kwargs,
        )
        callback_objs = []

        if reward_type is not None:
            reward_fn = load_reward(reward_type, reward_path, venv)
            venv = RewardVecEnvWrapper(venv, reward_fn)
            callback_objs.append(venv.make_log_callback())
            logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

        vec_normalize = None
        if normalize:
            venv = vec_normalize = VecNormalize(venv, **normalize_kwargs)

        if policy_save_interval > 0:
            save_policy_callback = serialize.SavePolicyCallback(policy_dir, vec_normalize)
            save_policy_callback = callbacks.EveryNTimesteps(
                policy_save_interval,
                save_policy_callback,
            )
            callback_objs.append(save_policy_callback)
        callback = callbacks.CallbackList(callback_objs)

        rl_algo = rl.make_rl_algo(venv)
        rl_algo.set_logger(custom_logger)
        rl_algo.learn(total_timesteps, callback=callback)

        # Save final artifacts after training is complete.
        # N.B. Here, we save the policies separately, and the rollouts together. 
        if policy_save_final:
            output_dir = os.path.join(policy_dir, "final_iter_"+str(it))
            serialize.save_stable_model(output_dir, rl_algo, vec_normalize)

        rollout_stats.append(train.eval_policy(rl_algo, venv))
        rl_algos.append(rl_algo)

    if rollout_save_final:
        save_path = osp.join(rollout_dir, "final.pkl")
        rollout.rollout_multi_and_save(
            save_path,
            rl_algos,
            venv,
            rollout_save_n_timesteps,
            rollout_save_n_episodes
        )

    # Final evaluation of expert policy.
    return rollout_stats


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_rl_multi"))
    train_rl_multi_ex.observers.append(observer)
    train_rl_multi_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
