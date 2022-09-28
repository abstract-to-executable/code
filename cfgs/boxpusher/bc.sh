# Train Transformer
python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 model_cfg.use_past_actions=False \
    logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/transformer \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=True

python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt

# on new dataset that is probably workin

python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset_fix.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_3074.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_3074.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 model_cfg.use_past_actions=False \
    logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/transformer_new \
    eval_cfg=cfgs/boxpusher/eval_translation_new.yml

# python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation.yml model=results/boxpusher_v1_offline/transformer_new/models/best_avg_return.pt

# python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/transformer_new/models/best_avg_return.pt



# Train LSTM
python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc_lstm.yml exp_cfg.steps=300000 \
    model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/lstm_prepend_dropout \
    model_cfg.lstm_config.dropout=0.1 model_cfg.prepend_student=True model_cfg.use_past_actions=False \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=True


# Train LSTM new better dataset
python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc_lstm.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset_fix.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_3074.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_3074.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/lstm_prepend_dropout_new \
    model_cfg.lstm_config.dropout=0.1 model_cfg.prepend_student=True model_cfg.use_past_actions=False \
    eval_cfg=cfgs/boxpusher/eval_translation_new.yml

# evaluate model
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation.yml model=results/boxpusher_v1_offline/lstm_prepend_dropout/models/best_avg_return.pt


# Train ConvNet model

python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc_convnet.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/convnet_prepend_student \
    model_cfg.convnet_config.dropout=0.1 model_cfg.convnet_config.kernel_size=11 model_cfg.prepend_student=False model_cfg.use_past_actions=False \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=True

# evaluate model
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation.yml model=results/boxpusher_v1_offline/convnet_prepend_student/models/best_avg_return.pt
