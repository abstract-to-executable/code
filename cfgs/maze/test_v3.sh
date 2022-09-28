python scripts/eval_translation.py cfg=cfgs/maze/eval_maze_v2.yml \
    model=results/online/maze_final_v2_8/transformer/5330/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset="datasets/maze_v2/couch_6_corridorrange_12_30/dataset_teacher.pkl" device=cuda
