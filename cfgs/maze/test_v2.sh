python scripts/eval_translation.py cfg=cfgs/maze/eval_maze_v2.yml \
    model=results/online/maze_final_v2_8/transformer/5330/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset="datasets/maze_v2/couch_6_corridorrange_12_30/dataset_teacher.pkl" device=cuda


python scripts/eval_translation.py cfg=cfgs/maze/eval_couch_4_corridorrange_12_20.yml \
    model=results/online/maze_final_iclr/mlp_subgoal/9577/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_teacher.pkl" device=cuda \
    env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True