import os.path as osp
import sys
from h5py import File
import numpy as np
def convert_maniskill_trajectory(traj, use_qpos=True):
    """
    Parameters
    ----------
    use_qpos : bool
        - if true, encodes target information about object as simply the qpos, velocity and the target qpos.
        - if false, uses the rest of the observation (for state mode, this is pose.)
    """
    observations = np.vstack([np.array(traj['obs']), np.array(traj['next_obs'][-1])])
    actions = np.array(traj['actions'])
    teacher_xyz = (observations[:, -38:-35] + observations[:, -35:-32]) / 2
    if use_qpos:
        info_q = np.vstack(
            [np.array(traj['info_qpos']), np.array(traj['info_qvel']), np.array(traj['info_target_qpos'])]
        ).T
        info_q = np.vstack([info_q, info_q[-1]])
        teacher = np.hstack([teacher_xyz, info_q])
    else:
        teacher = np.hstack([observations[:, :-38], teacher_xyz])
    
    
    return dict(observations=observations, actions=actions, teacher=teacher, env_states=np.array(traj["env_states"]), env_levels=np.array(traj["env_levels"]))

def convert_maniskill_dataset(h5_file_path, use_qpos=True):
    f = File(h5_file_path, "r")
    dataset = dict(student={}, teacher={})
    for traj_id in f.keys():
        new_traj = convert_maniskill_trajectory(f[traj_id], use_qpos=use_qpos)
        dataset["student"][traj_id] = dict(observations=new_traj["observations"], actions=new_traj["actions"])
        dataset["teacher"][traj_id] = dict(observations=new_traj["teacher"], env_states=new_traj["env_states"], env_levels=new_traj["env_levels"])
    return dataset

if __name__ == "__main__":
    import pickle
    # dataset = convert_maniskill_dataset(sys.argv[1])
    # with open(osp.splitext(sys.argv[1])[0] + ".pkl", "wb") as f:
    #     pickle.dump(dataset, f)

    import os
    from tqdm import tqdm
    use_qpos=False

    
    full_dataset = {"student":{}, "teacher":{}}
    for f in os.walk(sys.argv[1]):
        dirname = f[0]
        for file in tqdm(f[2]):
            if osp.splitext(file)[1] == ".h5":
                env_name = osp.basename(osp.splitext(file)[0])
                # print(env_name)
                file_path=osp.join(dirname, file)
                # print(file_path)
                dataset = convert_maniskill_dataset(file_path, use_qpos=use_qpos)
                
                for traj_id in dataset["student"]:
                    formatted_traj_id = f"{env_name}_{traj_id}"
                    full_dataset["student"][formatted_traj_id] = dataset["student"][traj_id]
                    full_dataset["teacher"][formatted_traj_id] = dataset["teacher"][traj_id]
    dataset_name = osp.splitext(sys.argv[1])[0]
    if not use_qpos:
        dataset_name = dataset_name + "_noqpos"
    with open(dataset_name + ".pkl", "wb") as f:
        pickle.dump(full_dataset, f)
    
    a = list(full_dataset['student'].keys())
    np.random.shuffle(a)
    train_length = int(len(a) * 0.8)
    val_length = len(a) - train_length
    np.save(dataset_name + "_train_ids.npy", a[:train_length])
    np.save(dataset_name + "_val_ids.npy", a[train_length:])