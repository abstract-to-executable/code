python scripts/train_translation_online.py \
    cfg=cfgs/boxpusher/train_translation_online_scratch.yml restart_training=True \
    logging_cfg.exp_name=boxpusher_v2_translation_final/transformer/9252 \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.seed=9252

python scripts/train_translation_online.py \
    cfg=cfgs/boxpusher/train_translation_online_scratch.yml restart_training=True \
    logging_cfg.exp_name=boxpusher_v2_translation_final/transformer/5843 \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.seed=5843

python scripts/watch_translation.py cfg=cfgs/boxpusher/watch_translation.yml \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id=100

python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation.yml \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt

python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml \
    planner_cfg.min_student_execute_length=100 \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt

python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt



# plan and translate
python scripts/plan_translate.py cfg=cfgs/boxpusher/watch_translation_replan.yml \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id=49

# plan translate very long horizon
python scripts/plan_translate.py cfg=cfgs/boxpusher/watch_translation_obstacle.yml \
    model=results/online/boxpusher_v2_translation_final/transformer/9252/models/best_train_EpRet.pt \
    env_cfg.reward_type="lcs_dense" traj_id=49


# Boxpusher MLP
python scripts/train_translation_online.py \
    cfg=cfgs/boxpusher/train_translation_online_scratch.yml \
    logging_cfg.wandb=False device=cuda \
    logging_cfg.exp_name=boxpusher_v2_final/mlp_subgoal/2586 \
    env_cfg.trajectories="datasets/boxpusher_v2/dataset_train_ids.npy" \
    env_cfg.trajectories_dataset="datasets/boxpusher_v2/dataset.pkl" \
    exp_cfg.seed=2586 exp_cfg.log_std_scale=-0.5 env_cfg.give_traj_id=False \
    env_cfg.reward_type="lcs_dense" restart_training=False \
    model_cfg.state_embedding_activation=relu model_cfg.final_mlp_activation=relu \
    exp_cfg.batch_size=1024 exp_cfg.n_envs=20 exp_cfg.steps_per_epoch=20000 model_cfg.trajectory_sample_skip_steps=2 \
    model_cfg.final_mlp_hidden_sizes="(128, 128)" model_cfg.state_embedding_hidden_sizes="(32,)" \
    exp_cfg.accumulate_grads=False exp_cfg.pi_lr=0.0003 logging_cfg.wandb_cfg.group=boxpusher_v2_final \
    env_cfg.env_rew_weight=0.1 model_cfg.teacher_timestep_embeddings=True \
    model_cfg.type=MLPID model_cfg.stack_size=1 model_cfg.mlp_config.dropout=0.1 \
    env_cfg.exclude_target_state=False model_cfg.state_dims=10 \
    exp_cfg.target_kl=0.15 exp_cfg.update_iters=3 model_cfg.max_teacher_length=32 exp_cfg.epochs=2000 \
    env_cfg.speed_factor=0.5 exp_cfg.max_ep_len=200 env_cfg.sub_goals=True restart_training=True
