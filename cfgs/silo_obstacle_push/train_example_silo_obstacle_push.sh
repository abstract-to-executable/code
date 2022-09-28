python scripts/train_translation_online.py \
    cfg=cfgs/silo_obstacle_push/train_translation_online_scratch.yml restart_training=True \
    logging_cfg.exp_name=silo_obstacle_push/transformer/9252 \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.seed=9252
