# Train TR^2-GP2

python scripts/train_translation_online.py \
    cfg=cfgs/maze/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=maze/transformer \
    env_cfg.trajectories="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_train_ids_2400.npy" \
    env_cfg.trajectories_dataset="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_teacher.pkl" \
    exp_cfg.n_envs=16 \
    model_cfg.trajectory_sample_skip_steps=3 model_cfg.max_teacher_length=50 exp_cfg.seed=0

# Watch it
python scripts/watch_translation.py cfg=cfgs/maze/watch_translation_v1.yml \
    model=results/online/maze/transformer/models/best_train_EpRet.pt traj_id=2

python scripts/eval_translation.py cfg=cfgs/maze/eval_maze_v1.yml \
    model=results/online/maze/transformer/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_teacher.pkl" device=cuda



# Train SGC
python scripts/train_translation_online.py \
    cfg=cfgs/maze/train_translation_online_scratch_iclr.yml restart_training=True \
    logging_cfg.exp_name=maze/mlp_4_12_20 \
    env_cfg.trajectories="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_train_ids_2400.npy" \
    env_cfg.trajectories_dataset="datasets/maze_iclr/couch_4_corridorrange_12_20/dataset_teacher.pkl" \
    exp_cfg.n_envs=16 \
    model_cfg.trajectory_sample_skip_steps=3 model_cfg.max_teacher_length=40 exp_cfg.seed=0 model_cfg.state_dims=17 \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True

python scripts/watch_translation.py cfg=cfgs/maze/watch_translation.yml \
    model=results/online/maze/transformer_3_12_20_s0/models/best_train_EpRet.pt traj_id=2 env_cfg.sub_goal_nstep=5 env_cfg.sub_goals=True