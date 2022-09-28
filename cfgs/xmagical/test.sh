# collect dataset
python scripts/xmagical/collect_traj.py

# online train

python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation/shortstick/test3_valid \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=4 model_cfg.max_teacher_length=30 exp_cfg.gamma=0.99 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense"


python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation/test2_finetune \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=32 \
    model_cfg.state_embedding_hidden_sizes="(32,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense2" \
    pretrained_ac_weights=results/online/xmagical_translation/test2/models/best_train_EpRet.pt exp_cfg.accumulate_grads=True


# longstick

python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation/longstick/test2_valid_lcs_dense2 \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=10 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=48 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Longstick

# medium stick
python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation/mediumstick/test2_valid_lcs_dense2 \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=10 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=48 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Mediumstick

python scripts/train_translation_online.py \
    cfg=cfgs/xmagical/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=xmagical_translation/mediumstick/test2_valid_lcs_dense2_ft \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=10 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=48 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Mediumstick \
    pretrained_ac_weights=results/online/xmagical_translation/mediumstick/test2_valid_lcs_dense2/models/best_train_EpRet.pt \
    exp_cfg.accumulate_grads=True

    
# gripper
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
    logging_cfg.exp_name=xmagical_translation/gripper/test5_valid_lcs_dense2_ft \
    env_cfg.trajectories="datasets/xmagical/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/xmagical/dataset.pkl" \
    exp_cfg.n_envs=10 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=2 model_cfg.max_teacher_length=48 \
    env_cfg.reward_type="lcs_dense2" \
    env_cfg.embodiment=Gripper model_cfg.state_dims=17 model_cfg.act_dims=3 exp_cfg.gamma=0.995 \
    pretrained_ac_weights=results/online/xmagical_translation/gripper/test5_valid_lcs_dense2/models/best_train_EpRet.pt \
    exp_cfg.accumulate_grads=True
    