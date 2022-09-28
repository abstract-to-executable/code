import os
import pickle

import gym
import numpy as np
import sapien.core as sapien
from skilltranslation.envs.trajectory_env import TrajectoryEnv
import torch
import tqdm
import os.path as osp
import skilltranslation.envs

from tqdm import tqdm
from skilltranslation.planner.boxpusherteacher import BoxPusherReacherPlanner, BoxPusherTaskPlanner
from paper_rl.common.rollout import Rollout
from stable_baselines3.common.vec_env import SubprocVecEnv
from paper_rl.architecture.ac.mlp import MLPActorCritic
save_folder = osp.join("./data_v1_new", "learned_trajectories")
# save_folder = osp.join("./data", "trajectories")
if not osp.isdir(save_folder):
    os.mkdir(save_folder)

if __name__ == "__main__":

    cpu = 4
    max_ep_len = 100
    def make_env(traj_id):
        def _init():
            import skilltranslation.envs.boxpusher.traj_env
            env = gym.make(
                "BoxPusherTrajectory-v0",
                obs_mode="dense",
                reward_type="trajectory",
                control_type="2D-continuous",
                balls=1,
                magic_control=False,
                max_trajectory_skip_steps=15,
                trajectories=[traj_id],
                max_trajectory_length=100,
                data_dir="./data_v1_new/trajectories",
                dense_obs_only=True,
                fixed_max_ep_len=max_ep_len,
                max_stray_dist=3e-1,
            )
            env.seed(traj_id)
            return env

        return _init

    
    all_trajs = []
    # pbar = tqdm(range(4000))


    # models = [
    #     "workspace/solveall_fixed_teacher_0_rewscale/models/model_1000.pt",
    #     # 600 - 0.813 700 - 0.859 800 - 0.9 900 - 0.895 1000 - 0.918 1100 0.909
    #     "workspace/solveall_fixed_teacher_1_rewscale/models/model_1100.pt",
    #     # 1000 - 0.856 1100 - 0.877
    #     "workspace/solveall_fixed_teacher_2_rewscalev2/models/model_720.pt",
    #     # 200 - 0.823 600 - 0.962 700 - 0.973 710 - 0.97 720 - 0.977
    #     "workspace/solveall_fixed_teacher_3_rewscale/models/model_900.pt"
    #     # 600 0.872 700 0.92, 800 0.901 900 - 0.921 910 - 0.909 1000 - 0.887 1100 - 0.901 1200 -0.878
    # ]
    models = [
        
    ]

    env: TrajectoryEnv = make_env(1)()
    torch.manual_seed(0)
    np.random.seed(0)
    device=torch.device("cpu")
    model = MLPActorCritic(
       env.observation_space, env.action_space, hidden_sizes=[128] * 4, log_std_scale=-0.5
    ).to(device)
    
    trajs = []
    successes = np.zeros(4000, dtype=bool)
    for a_type in range(0,4):
        ckpt = torch.load(models[a_type])
        model.load_state_dict(ckpt["model_state_dict"])
        print(f"=== loaded model {models[a_type]}===")
        for i in tqdm(range(1000)):
            traj_id = i + a_type*1000
            env: TrajectoryEnv = make_env(traj_id)()
            env.seed(traj_id)
            done=False
            actions, rewards = [], []
            o = env.reset()
            observations = [o]
            ep_len = 0
            success = False
            while not done:
                with torch.no_grad():
                    a = model.act(torch.as_tensor(o,dtype=torch.float32), deterministic=True)
                    actions.append(a)
                n_o, r, done, _ = env.step(a)
                rewards.append(r)
                observations.append(n_o)
                ep_len += 1
                o = n_o
                if ep_len >= max(env.traj_len*1.25, env.traj_len+10):
                    break
                if done:
                    success = True
                    break
            observations = np.stack(observations)
            rewards = np.stack(rewards)
            actions = np.stack(actions)
            traj = dict(observations=observations, actions=actions, rewards=rewards)
            trajs.append(traj)
            successes[traj_id] = success
        print("===")
        print("Success rate", successes.mean())
    data = dict(student=dict())
    for idx, traj in tqdm(enumerate(trajs), total=len(trajs)):
        if successes[i]: data["student"][idx] = traj
    with open("clean_learned.pkl", "wb") as f:
        pickle.dump(data)