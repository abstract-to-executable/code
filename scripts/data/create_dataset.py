import numpy as np
import os.path as osp
dataset = {}
dataset["student"] = {}
dataset["teacher"] = {}
train_ids = []
val_ids = []
np.random.seed(0)
for traj_id in range(4000):
    learned_traj_path = f"data_v1/learned_trajectories/{traj_id}_traj.npy"
    if osp.exists(learned_traj_path):
        t1 = np.load(f"data_v1/trajectories/{traj_id}_traj.npy",allow_pickle=True).reshape(1)[0]
        dataset["teacher"][traj_id] = t1
        s1 = np.load(learned_traj_path, allow_pickle=True).reshape(1)[0]
        s1["observations"] = s1["observations"][:, :6] # remove the farthest step observation for now
        dataset["student"][traj_id] = s1

all_ids = list(dataset["student"].keys())
np.random.shuffle(all_ids)
train_size = int(len(all_ids) * 0.75)
val_size = len(all_ids) - train_size

train_ids = all_ids[:train_size]
val_ids = all_ids[train_size:]
np.save("data_v1/dataset.npy", dataset)
np.save("data_v1/train_ids.npy", train_ids)
np.save("data_v1/val_ids.npy", val_ids)