# Train TR^2-GP2

python scripts/train_translation_online.py \
    cfg=cfgs/maze/train_translation_online_scratch_v3.yml restart_training=False \
    logging_cfg.exp_name=maze/transformer_v3 \
    env_cfg.trajectories="datasets/maze_v3/couch_5_corridorrange_12_24/dataset_train_ids_1200.npy" \
    env_cfg.trajectories_dataset="datasets/maze_v3/couch_5_corridorrange_12_24/dataset_teacher.pkl" \
    exp_cfg.n_envs=1 \
    model_cfg.trajectory_sample_skip_steps=4 model_cfg.max_teacher_length=20 exp_cfg.seed=1

# Watch it
python scripts/watch_translation.py cfg=cfgs/maze/watch_translation_v3.yml \
    model=results/online/maze_final_v2_8/transformer/5330/models/best_train_EpRet.pt traj_id=2

python scripts/eval_translation.py cfg=cfgs/maze/eval_couch_4_corridorrange_12_20.yml \
    model=results/online/maze/transformer_v2_6_skip8/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset="datasets/maze_v2/couch_6_corridorrange_12_30/dataset_teacher.pkl" device=cuda

