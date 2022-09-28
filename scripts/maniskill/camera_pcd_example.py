import gym
from multiprocessing import Pool
from pathlib import Path
import os.path as osp
import pickle
import numpy as np
from omegaconf import OmegaConf

try:
    import mani_skill.env
    from mani_skill.env.open_cabinet_door_drawer import OpenCabinetDrawerMagicEnv
    from mani_skill.utils.visualization import visualize_point_cloud

except:
    print("#" * 15, "no Maniskill 1", "#" * 15, )
from tqdm import tqdm
from skilltranslation.utils.animate import animate


drawer_idx=[
    1000,
    1004,
    1005,
    # 1013 , BAD
    1016,
    1021,
    1024,
    1027,
    1032,
    1033,
    # 1035 , BAD
    1038,
    1040,
    1044,
    1045,
    1052,
    1054,
    # 1056 , BAD
    1061,
    1063,
    1066,
    1067,
    1076,
    1079,
    1082
]


def process_and_visualize():
    env=gym.make('OpenCabinetDrawer-v0')
    env.set_env_mode(obs_mode='pointcloud', reward_type='sparse')
    env.reset(level=0)
    for i in range(5):
        env.step(np.zeros(13))

    # Process pcd
    obs = env.get_obs()
    #print(env.observation_space)
    assert type(obs) == dict
    print(obs.keys())
    # dense obs contains qpos, qvel, and gt bbox.(same as before)
    dense_obs = obs['agent']

    # raw pointcloud observation from the environment
    pcd = obs['pointcloud']['xyz']

    cam = env.cameras
    print(cam[0].get_model_matrix())
    print(cam[0].get_projection_matrix())
    #print(cam[0].get_intrinsic_matrix())
    #print(pcd)
    print(pcd.shape)
    mins, maxs = env.get_aabb_for_min_x(env.target_link)
    print( env.get_aabb_for_min_x(env.target_link))
    from transforms3d.quaternions import quat2mat
    mat = quat2mat([0.908009, 0.0445405, 0.10001, -0.404393])
    print(mat)
    T = np.zeros((4,4))
    T[:3, :3] = mat
    T[:3, 3] = np.array([-0.9, 1, 0.8])
    T[3,3] = 1
    print('sample', pcd[:10])
    print('t', T)
    print('mi', mins)
    print('ma', maxs)
    for i in range(10):
        p = pcd[i]
        pp = np.hstack([p,1])
        res = T @ pp
        print('res', res)
   #visualize_point_cloud(pcd)
    #visualize_point_cloud(processed_pcd)

def gt_pcd():
    env=gym.make('OpenCabinetDrawerMagic-v0')
    env.set_env_mode(obs_mode='custom', reward_type='sparse')
    env.reset(level=0)
    for i in range(5):
        env.step(np.zeros(5))

    # Process pcd
    obs = env.get_obs()
    print(len(obs))
    o3d = obs[-300:].reshape(100,3)
    print('aabb_gt',env.get_aabb_for_min_x(env.target_link))
    max_x, min_x = -np.inf, np.inf
    max_y, min_y = -np.inf, np.inf
    max_z, min_z = -np.inf, np.inf
    for point in o3d:
        x,y,z = point
        max_x=max(x,max_x)
        max_y=max(y,max_y)
        max_z=max(z,max_z)

        min_x = min(x,min_x)
        min_y = min(y,min_y)
        min_z = min(z,min_z)

    print('aabb from o3d', [min_x,min_y,min_z], [max_x, max_y, max_z])

if __name__ == '__main__':
    # process_and_visualize()
    gt_pcd()