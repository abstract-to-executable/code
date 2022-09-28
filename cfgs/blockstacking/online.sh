#v4
python scripts/blockstack/collect_dataset_train.py save_path=datasets/blockstack_v4/dataset.pkl n=4000

python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=blockstack_v4_translation/test env_cfg.task_range=large \
    env_cfg.trajectories="datasets/blockstack_large/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstack_large/dataset.pkl" \
    exp_cfg.n_envs=16 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=1 model_cfg.max_teacher_length=60 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense2"


#v4
python scripts/blockstack/collect_dataset_train.py save_path=datasets/blockstack_v4/dataset.pkl n=4000

python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=blockstack_v4_translation/test \
    env_cfg.trajectories="datasets/blockstack_v4/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstack_v4/dataset.pkl" \
    exp_cfg.n_envs=16 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=1 model_cfg.max_teacher_length=55 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense2"

# fine tune
python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1.yml restart_training=True \
    logging_cfg.exp_name=blockstack_v4_translation/test_finetune \
    env_cfg.trajectories="datasets/blockstack_v4/dataset_train_ids.npy" env_cfg.trajectories_dataset="datasets/blockstack_v4/dataset.pkl" \
    exp_cfg.n_envs=16 device=cuda exp_cfg.batch_size=1024 exp_cfg.steps_per_epoch=20000 \
    model_cfg.trajectory_sample_skip_steps=1 model_cfg.max_teacher_length=55 \
    model_cfg.state_embedding_hidden_sizes="(64,)" model_cfg.final_mlp_hidden_sizes="(128,128)" \
    model_cfg.transformer_config.n_head=2 exp_cfg.target_kl=0.15 env_cfg.reward_type="lcs_dense2" \
    pretrained_ac_weights=results/online/blockstack_v4_translation/s_3_dense2/models/best_train_EpRet.pt exp_cfg.accumulate_grads=True

# watch model
python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v3_translation/s_1/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id=2394 

# watch ground truth
python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    traj=datasets/blockstack_v4/dataset.pkl \
    env_cfg.reward_type="lcs_dense" traj_id=2394 



# Train mlp with subgoals
python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1.yml \
    logging_cfg.wandb=False device=cuda \
    logging_cfg.exp_name=blockstack_v4_translation/final/mlp_subgoal/224 model_cfg.state_dims=42 model_cfg.act_dims=4 \
    env_cfg.trajectories="datasets/blockstack_v4/dataset_train_ids.npy" \
    env_cfg.trajectories_dataset="datasets/blockstack_v4/dataset.pkl" \
    exp_cfg.seed=224 exp_cfg.log_std_scale=-0.5 env_cfg.give_traj_id=False \
    env_cfg.reward_type="lcs_dense2" restart_training=False \
    model_cfg.state_embedding_activation=relu model_cfg.final_mlp_activation=relu \
    exp_cfg.batch_size=1024 exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 model_cfg.trajectory_sample_skip_steps=1 \
    model_cfg.final_mlp_hidden_sizes="(128,128)" model_cfg.state_embedding_hidden_sizes="(64,)" \
    env_cfg.max_world_state_stray_dist=0.03 \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    exp_cfg.accumulate_grads=False exp_cfg.pi_lr=0.0003 logging_cfg.wandb_cfg.group=blockstack_v4_final \
    exp_cfg.target_kl=0.15 exp_cfg.update_iters=3 model_cfg.max_teacher_length=55 exp_cfg.epochs=2000 \
    env_cfg.sub_goals=True restart_training=True

python scripts/train_translation_online.py \
    cfg=cfgs/blockstacking/train_translation_online_scratch_v1.yml \
    logging_cfg.wandb=False device=cuda \
    logging_cfg.exp_name=blockstack_v4_translation/final_finetune/mlp_subgoal/224 model_cfg.state_dims=42 model_cfg.act_dims=4 \
    env_cfg.trajectories="datasets/blockstack_v4/dataset_train_ids.npy" \
    env_cfg.trajectories_dataset="datasets/blockstack_v4/dataset.pkl" \
    exp_cfg.seed=224 exp_cfg.log_std_scale=-0.5 env_cfg.give_traj_id=False \
    env_cfg.reward_type="lcs_dense2" restart_training=False \
    model_cfg.state_embedding_activation=relu model_cfg.final_mlp_activation=relu \
    exp_cfg.batch_size=1024 exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 model_cfg.trajectory_sample_skip_steps=1 \
    model_cfg.final_mlp_hidden_sizes="(128,128)" model_cfg.state_embedding_hidden_sizes="(64,)" \
    env_cfg.max_world_state_stray_dist=0.03 \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    exp_cfg.accumulate_grads=False exp_cfg.pi_lr=0.0003 logging_cfg.wandb_cfg.group=blockstack_v4_final \
    exp_cfg.target_kl=0.15 exp_cfg.update_iters=3 model_cfg.max_teacher_length=55 exp_cfg.epochs=2000 \
    env_cfg.sub_goals=True restart_training=True \
    pretrained_ac_weights=results/online/blockstack_v4_translation/final/mlp_subgoal/224/models/best_train_EpRet.pt exp_cfg.accumulate_grads=True

python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final/mlp_subgoal/224/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id=2394 env_cfg.sub_goals=True

python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final/mlp_subgoal/224/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    env_cfg.show_goal_visuals=True env_cfg.sub_goals=True \
    traj_id=1
