## All Watching scripts


## Watch test sets using pre-generated abstract trajectories ##

# replace trajectories_dataset with ... dataset_<tower|pyramid>-<n> and env_cfg.goal=<tower|pyramid>-<n>

# State based
python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    env_cfg.show_goal_visuals=True \
    traj_id=1

# RGBD input

python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
    traj_id=1

# FAILED [25,12,5]
## watch test sets using planner generated abstract trajectories ##
python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    env_cfg.show_goal_visuals=True traj_id=1

# optionally add env_cfg.intervene_count=<n> to add a number of interventions where a block is teleported off the stack elsewhere