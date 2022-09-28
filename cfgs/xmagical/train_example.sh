python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation_final/gripper/3029 \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.update_iters=3 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=4 model_cfg.max_teacher_length=30 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Gripper model_cfg.state_dims=17 model_cfg.act_dims=3 exp_cfg.gamma=0.99 seed=3029

python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation_final/gripper/3029_ft \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.update_iters=3 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=4 model_cfg.max_teacher_length=30 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Gripper model_cfg.state_dims=17 model_cfg.act_dims=3 exp_cfg.gamma=0.99 seed=3029 \
    pretrained_ac_weights=results/online/xmagical_translation_final/gripper/3029/models/best_train_EpRet.pt \
    exp_cfg.accumulate_grads=True