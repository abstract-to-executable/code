python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1_visual.yml restart_training=True \
    logging_cfg.exp_name=blockstack_v4_translation/visual_s_3dense \
    env_cfg.trajectories="datasets/blockstack_v4/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstack_v4/dataset.pkl" \
    exp_cfg.n_envs=16 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=28 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense2"