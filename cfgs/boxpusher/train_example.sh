python scripts/train_translation_online.py \
    cfg=cfgs/boxpusher/train_translation_online_scratch.yml restart_training=True \
    logging_cfg.exp_name=boxpusher_v2/transformer/9252 \
    exp_cfg.n_envs=20 device=cuda exp_cfg.batch_size=1024 exp_cfg.seed=9252
