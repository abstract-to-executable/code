import gym
import numpy as np
from omegaconf import OmegaConf
import os.path as osp
from paper_rl.cfg.parse import clean_and_transform, parse_cfg

from skilltranslation.utils.animate import animate
import torch
from skilltranslation.models.translation.lstm import LSTM
from skilltranslation.models.translation.translation_transformer import TranslationTransformerGPT2
import skilltranslation.envs

def main(cfg):
    ckpt = torch.load(cfg.model, map_location=cfg.device)
    model_cfg = ckpt["cfg"]["model_cfg"]
    env_cfg = cfg.env_cfg
    env_cfg.stack_size = model_cfg["stack_size"]
    env_cfg.max_trajectory_length = model_cfg["max_teacher_length"]
    env_cfg.trajectory_sample_skip_steps = model_cfg["trajectory_sample_skip_steps"]
    # env_cfg.fixed_max_ep_len = 128 #exp_cfg.max_ep_len
    device = torch.device(cfg.device)
    model_cls = TranslationTransformerGPT2
    if model_cfg["type"] == "LSTM":
        model_cls = LSTM
    model = model_cls.load_from_checkpoint(ckpt, device=device)
    use_teacher=not cfg.ignore_teacher
    save_traj_path=None
    if "save_traj_path" in cfg:
        save_traj_path=cfg["save_traj_path"]
    device = torch.device(cfg.device)
    model.eval()
    def unsqueeze_dict(o):
        for k in o:
            o[k] = o[k].unsqueeze(0)
        return o

    def obs_to_tensor(o):
        tensor_o = {}
        for k in o:
            tensor_o[k] = torch.as_tensor(o[k], device=device)
            if tensor_o[k].dtype == torch.float64:
                tensor_o[k] = tensor_o[k].float()
        return tensor_o
    def policy(o):
        with torch.no_grad():
            o = obs_to_tensor(o)
            o = unsqueeze_dict(o)
            if not use_teacher:
                o["teacher_attn_mask"][:] = False
                o["teacher_frames"] = o["teacher_frames"] * 0
            # print(o["observation"])
            a = model.step(o)
            a = model.actions_scaler.untransform(a).cpu().numpy()[0]
        # print(a.shape, a)
        return a
    def format_trajectory(t_obs, t_act, t_rew, t_info):
        ep_len = len(t_act)
        t_info = t_info[-1]
        return dict(
            returns=np.sum(t_rew),
            traj_match=t_info["stats"]["farthest_traj_match_frac"],
            match_left=t_info["traj_len"] - t_info["stats"]["farthest_traj_match"] - 1,
            traj_len=t_info["traj_len"],
            ep_len=ep_len,
            success=ep_len < env_cfg.fixed_max_ep_len,
            traj_id=t_info["traj_id"]
        )


    # planner = BoxPusherOneDirection()
    # planner = BoxPusherDrawingPlanner()
    if "BoxPusher" in cfg.env:
        from skilltranslation.planner.boxpusherteacher import BoxPusherTaskPlanner
        from skilltranslation.envs.boxpusher.env import BoxPusherEnv
        planner = BoxPusherTaskPlanner()
        planner.set_type(0)
        if "offscreen_only" in env_cfg:
            offscreen_only = env_cfg["offscreen_only"]
        else:
            offscreen_only = False
        planning_env = BoxPusherEnv(
            offscreen_only=offscreen_only,
            **env_cfg.planner_cfg.env_cfg
        )
    elif "BlockStack" in cfg.env:
        from mani_skill2.utils.wrappers import ManiSkillActionWrapper, NormalizeActionWrapper
        from skilltranslation.planner.blockstackplanner import BlockStackPlanner
        from skilltranslation.envs.blockstacking.env import BlockStackMagicPandaEnv
        planner = BlockStackPlanner(replan_threshold=1e-1)
        planning_env = BlockStackMagicPandaEnv(obs_mode="state_dict", goal=env_cfg.goal, num_blocks=1)
        planning_env = ManiSkillActionWrapper(planning_env)
        planning_env = NormalizeActionWrapper(planning_env)
    elif "OpenDrawer" in cfg.env:
        from skilltranslation.planner.opendrawerplanner import OpenDrawerPlanner
        from mani_skill.env.open_cabinet_door_drawer import OpenCabinetDrawerMagicEnv_CabinetSelection
        planner = OpenDrawerPlanner(replan_threshold=1e-2)
        # we only need planning env to get details about the object used
        planning_env = OpenCabinetDrawerMagicEnv_CabinetSelection()
        planning_env.set_env_mode(obs_mode=env_cfg.obs_mode, reward_type='sparse')
        planning_env.reset()
    planning_env.seed(0)
    planning_env.reset()

    env_cfg = OmegaConf.to_container(cfg.env_cfg)
   
    env_cfg["planner_cfg"]["planner"] = planner
    env_cfg["planner_cfg"]["planning_env"] = planning_env
    env_cfg["trajectories"] = [cfg.traj_id]
    env = gym.make(cfg.env, **env_cfg)
    
    env.seed(0)
    o = env.reset()
    trajectory = dict(
        states=[o["observation"]],
        observations=[o],
        actions=[]
    )
    def save_traj_so_far():
        nonlocal trajectory
        if save_traj_path is not None:
            print(f"saving trajectory to {save_traj_path}")
            import pickle
            with open(save_traj_path, "wb") as f:
                pickle.dump(trajectory, f)
            
    save_video_path = None
    if cfg.save_video_path is not None:
        save_video_path  = cfg.save_video_path
    else:
        if "BlockStack" in cfg.env or "OpenDrawer" in cfg.env:
            viewer = env.render()
            # import pdb;pdb.set_trace();
            viewer.paused=True
        else:
            env.render()
            env.viewer.paused=True
    imgs = []
    d = False
    ep_len=0
    while not d:
        env.draw_teacher_trajectory(skip=0)
        if save_video_path is not None:
            if "ManiSkill" or "OpenDrawer" in cfg.env:
                img = env.render(mode="color_image")["world"]["rgb"]
                img = img * 255
            else:
                img = env.render("rgb_array")
            imgs.append(img)
        else:
            env.render()
            pass
        a=policy(o)
        o, r, d, i = env.step(a)
        # trajectory["observations"].append(obs)
        trajectory["states"].append(o["observation"])
        trajectory["actions"].append(a)
        print(f"Step: {ep_len}")
        ep_len += 1
        save_traj_so_far()
        if i['replanned']:
            
            trajectory = dict(
                states=[],
                observations=[],
                actions=[]
            )
        # import pdb; pdb.set_trace()
        if d: 
            if i["task_complete"]:
                print("Success")
            else:
                print("failed")
            break

    
    if save_video_path is not None:
        animate(imgs, save_video_path, fps=16)
    exit()
if __name__ == "__main__":
    base_cfgs_path = osp.join(osp.dirname(__file__), "../cfgs")
    cli_conf = OmegaConf.from_cli()
    custom_cfg = None
    if "cfg" in cli_conf: custom_cfg = cli_conf["cfg"]

    cfg = parse_cfg(default_cfg_path=osp.join(base_cfgs_path, "defaults/plan_translate.yml"), cfg_path=custom_cfg)

    # convenience for common parameters when using CLI
    for k in ["state_dims", "act_dims", "teacher_dims"]:
        if k in cfg:
            cfg.model_cfg[k] = cfg[k]
    clean_and_transform(cfg)
    main(cfg)