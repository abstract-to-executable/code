import os

import gym
import numpy as np
import sapien.core as sapien
import tqdm
import os.path as osp
import skilltranslation.envs
# from skilltranslation.agents.teacher import TeacherMagicPolicy

from skilltranslation.planner.boxpusherteacher import BoxPusherReacherPlanner, BoxPusherTaskPlanner
from paper_rl.common.rollout import Rollout
from stable_baselines3.common.vec_env import SubprocVecEnv

save_folder = osp.join("./data_reach_v0", "learned_trajectories")
# save_folder = osp.join("./data", "trajectories")
if not osp.isdir(save_folder):
    os.mkdir(save_folder)
if __name__ == "__main__":
    
    # collect reach
    n_envs = 1
    rollout = Rollout()
    for traj_id in range(4):
        env_kwargs = dict(
            # actual env
            balls=1,
            magic_control=False,
            obs_mode="dense",
            reward_type="trajectory",
            control_type="2D-continuous",
            offscreen_only=True,
            # traj env
            max_ep_len_factor=2,
            max_trajectory_skip_steps=15,
            trajectories=[traj_id],
            max_trajectory_length=150,
            data_dir="./data_reach_v0/trajectories",
            # dense_obs_only=True,
        )
        def policy(obs):
            final_agent_pos = obs["teacher_frames"][-1][2:4]
            agent_pos = obs["observation"][2:4]
            a = final_agent_pos - agent_pos
            return a
            acts = []
            # print(obs["observation"].shape)
            # print(obs["teacher_frames"].shape)
            for env_idx in range(n_envs):
                # print(obs["teacher_frames"][env_idx])
                final_agent_pos = obs["teacher_frames"][env_idx][-1][2:4]
                agent_pos = obs["observation"][env_idx][2:4]
                a = final_agent_pos - agent_pos
                acts.append(a)
            acts = np.stack(acts)
            return acts
        env = gym.make("BoxPusherTrajectory-v0", **env_kwargs)
        env.seed(0)
        obs = env.reset()
        done = False
        observations, actions, rewards = [obs["observation"]], [], []
        ep_len = 0
        while not done:
            o = obs["observation"]
            a = policy(obs)
            obs, r, done, _ = env.step(a)

            observations.append(obs["observation"])
            actions.append(a)
            ep_len += 1
            if ep_len > 19:
                break

        actions = np.stack(actions)
        observations = np.stack(observations)

        # for idx, traj in enumerate(trajs):
        np.save(
            f"{save_folder}/{traj_id}_traj.npy",
            dict(
                observations=observations,
                actions=actions,
            ),
        )