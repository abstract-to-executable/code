python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset_teacher_extracted.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 model_cfg.use_past_actions=False \
    logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline_teacher_extracted/transformer \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=False