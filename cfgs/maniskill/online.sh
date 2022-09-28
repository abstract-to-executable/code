
# train
python scripts/train_translation_online.py \
    cfg=cfgs/maniskill/train_translation_online_scratch_v1.yml restart_training=False \
    logging_cfg.exp_name=maniskill_opendrawer/s_1 \
    env_cfg.trajectories="datasets/maniskill/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/maniskill/dataset.pkl" \
    exp_cfg.n_envs=10 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=22 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense"

python scripts/train_translation_online.py \
    cfg=cfgs/maniskill/train_translation_online_scratch_v1_pcd.yml restart_training=False \
    logging_cfg.exp_name=maniskill_opendrawer/s_1pcd \
    env_cfg.trajectories="datasets/maniskill_pcd/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/maniskill_pcd/dataset.pkl" \
    exp_cfg.n_envs=4 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=2000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=32 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense"

# watch model
python scripts/watch_translation.py cfg=cfgs/maniskill/watch_translation.yml \
    model=results/online/maniskill_opendrawer_v3_final/5670/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id="1000-3" 

# watch ground truth
python scripts/watch_translation.py cfg=cfgs/maniskill/watch_translation.yml \
    traj=datasets/maniskill_opendrawer/dataset.pkl \
    env_cfg.reward_type="lcs_dense" traj_id=2394 

# evaluate model
python scripts/eval_translation.py cfg=cfgs/maniskill/eval_translation.yml \
    model=results/online/maniskill_opendrawer_v3_final/5670/models/best_train_EpRet.pt

# Subgoal mlp

python scripts/train_translation_online.py \
    cfg=cfgs/maniskill/train_translation_online_scratch_v1.yml \
    logging_cfg.wandb=False device=cuda \
    logging_cfg.exp_name=maniskill_opendrawer_v3_final/mlp_subgoal/5670 model_cfg.state_dims=48 model_cfg.act_dims=13 \
    env_cfg.trajectories="datasets/maniskill_v3/dataset_open_train_ids.npy" \
    env_cfg.trajectories_dataset="datasets/maniskill_v3/dataset_open.pkl" \
    exp_cfg.seed=5670 exp_cfg.log_std_scale=-0.5 env_cfg.give_traj_id=False \
    env_cfg.reward_type="lcs_dense" restart_training=False \
    model_cfg.state_embedding_activation=relu model_cfg.final_mlp_activation=relu \
    exp_cfg.batch_size=1024 exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 model_cfg.trajectory_sample_skip_steps=1 \
    model_cfg.final_mlp_hidden_sizes="(128, 128)" model_cfg.state_embedding_hidden_sizes="(64,)" \
    exp_cfg.accumulate_grads=False exp_cfg.pi_lr=0.0003 logging_cfg.wandb_cfg.group=maniskill_opendrawer_v3_final \
    env_cfg.env_rew_weight=0.02 model_cfg.teacher_timestep_embeddings=True \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    exp_cfg.target_kl=0.15 exp_cfg.update_iters=3 model_cfg.max_teacher_length=32 exp_cfg.epochs=20000 \
    env_cfg.sub_goals=True