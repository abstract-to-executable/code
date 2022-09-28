"""
python 'scripts/blockstack/collect_visual_dataset.py' \
    traj=datasets/blockstack_v2/dataset.pkl traj_id=0 cfg=cfgs/blockstacking/watch_translation.yml save_dir=datasets/blockstack_visual
"""

from pathlib import Path
import pickle
import numpy as np
from omegaconf import OmegaConf
from skilltranslation.data.teacherstudent import TeacherStudentDataset
from skilltranslation.data.utils import MinMaxScaler
from skilltranslation.models.translation.lstm import LSTM
import gym
from paper_rl.cfg import parse
from skilltranslation.models.translation.mlp_id import MLPTranslationID, MLPTranslationIDTeacherStudentActorCritic
import torch
from skilltranslation.models.translation.translation_transformer import (
    TranslationTransformerGPT2,
)
from skilltranslation.utils.animate import animate
import os.path as osp
import skilltranslation.envs.boxpusher.traj_env
# import skilltranslation.envs.maniskill.traj_env
import skilltranslation.envs.maze.traj_env

def main(cfg):
    gym.logger.set_level(100)
    env_cfg = cfg.env_cfg
    model, traj_actions = None, None
    
    save_dir = cfg.save_dir
    if cfg.model is not None:
        ckpt = torch.load(cfg.model, map_location=cfg.device)

        model_cfg = ckpt["cfg"]["model_cfg"]

        env_cfg.stack_size = model_cfg["stack_size"]
        env_cfg.max_trajectory_length = model_cfg["max_teacher_length"]
        env_cfg.trajectory_sample_skip_steps = model_cfg["trajectory_sample_skip_steps"]
        device = torch.device(cfg.device)
        model_cls = TranslationTransformerGPT2
        if model_cfg["type"] == "LSTM":
            model_cls = LSTM
        elif model_cfg["type"] == "MLPID":
            model_cls = MLPTranslationID
        model = model_cls.load_from_checkpoint(ckpt, device=device)
        model.eval()
        actions_scaler = None
        print("###MODEL STEP: ", ckpt["stats"]["train/Epoch"])
        if "actions_scaler" in ckpt.keys():
            actions_scaler = MinMaxScaler()
            actions_scaler.min = torch.as_tensor(ckpt["actions_scaler"]["min"], dtype=torch.float32, device=device)
            actions_scaler.max = torch.as_tensor(ckpt["actions_scaler"]["max"], dtype=torch.float32, device=device)
    else:
        assert "traj" in cfg # require a working trajectory
        with open(cfg.traj, "rb") as f:
            data = pickle.load(f)
            if "student" in data.keys():
                traj_actions = data['student'][str(cfg.traj_id)]["actions"]
            else:
                traj_actions = data['actions']
        env_cfg.stack_size = 1
        env_cfg.max_trajectory_length = 1000 #exp_cfg.max_ep_len
    env_cfg.trajectories = [cfg.traj_id]
    traj_id = cfg.traj_id
    done = False
    
    env = gym.make(cfg.env, show_goal_visuals=False, **env_cfg)
    env.seed(0)
    obs = env.reset()
    # if "trajectory_sample_skip_steps" in env_cfg:
    # env.draw_teacher_trajectory(skip=0)

    def obs_to_tensor(o):
        for k in o:
            o[k] = torch.as_tensor(o[k], device=device)
            if o[k].dtype == torch.float64:
                o[k] = o[k].float()
        return o

   
    ep_len=0
    attns = []

    ep_ret = 0
    success = False
    # viewer = env.render("human")
    # viewer.paused=True
    rgb_imgs = []
    depth_imgs = []
    seg_imgs = []
    cam_int, cam_ext = None, None
    gt_poses = []
    while ep_len < 10000:
        for k in obs:
            if not isinstance(obs[k], int) and not isinstance(obs[k], float):
                obs[k] = obs[k][None, :]
            else:
                new_k = torch.zeros((1, 1))
                new_k[0,0] = obs[k]
                obs[k] = new_k
        with torch.no_grad():
            if model is not None:
                obs=obs_to_tensor(obs)
                if "traj_id" in obs:
                    obs["traj_id"] = obs["traj_id"].long()
                if cfg.save_attn:
                    a, attn = model.step(obs, output_attentions=True)
                    attns.append(attn)
                else:
                    a = model.step(obs)
                a = a.cpu().numpy()[0]
            if traj_actions is not None:
                if ep_len >= len(traj_actions):
                    a = np.zeros_like(traj_actions[0])
                else:
                    a = traj_actions[min(ep_len, len(traj_actions)-1)]
            vis = env.render("state_visual")
            rgb = vis["rgb"]
            depth = vis["depth"]
            cam_int = vis["camera_intrinsic"]
            cam_ext = vis["camera_extrinsic_world_frame"]
            seg = vis['actor_seg']
            poses = {}
            for i, b in enumerate(env.env.env.blocks + env.env.env.completed_blocks):
                pose = b.get_pose()
                if np.linalg.norm(pose.p) > 5: continue
                pose = np.hstack([pose.p, pose.q])
                poses[b.get_id()] = pose # + 2 offset by id since 1 is background, 2 is first block
            gt_poses.append(poses)
            rgb_imgs.append(rgb)
            depth_imgs.append(depth)
            seg_imgs.append(seg)
            # env.render("human")

            # import pdb;pdb.set_trace()        
        obs, reward, done, info = env.step(a)
        ep_ret += reward
        # print(ep_len, reward)
        ep_len += 1
        if ep_len >= env_cfg.fixed_max_ep_len:
            break
        if ep_len >= len(traj_actions):
            break
    print(f"episode return: {ep_ret}, success: {success}")

    # process data and save as scene
    datas = []
    for i in range(len(rgb_imgs)):
        rgb = rgb_imgs[i]
        depth = depth_imgs[i]
        # dict mapping block id (seg id) to pose
        pose = gt_poses[i]
        seg = seg_imgs[i]
        # save_path = osp.join(save_path, f"{traj_id}-{i}.pkl")
        # with open(save_path, "wb") as f:
        #     pickle.dump({
        #         "rgb": rgb, "depth": depth, "pose": pose, "seg":seg
        #     })
        datas.append(
            {
                "rgb": rgb, "depth": depth, "pose": pose, "seg": seg
            }
        )
    Path(save_dir).mkdir(parents=True,exist_ok=True)
    dataset = dict(
        cam_ext=cam_ext,
        cam_int=cam_int,
        datas=datas
    )
    with open(osp.join(save_dir, f"{traj_id}.pkl"), 'wb') as f:
        pickle.dump(dataset, f)
    # with open(, "wb"):
    #     pickle.dump(datas, f)



    env.close()

if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]

    cfg = parse.parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/watch_translation.yml"), cfg_path=custom_cfg)
    parse.clean_and_transform(cfg)
    main(cfg)