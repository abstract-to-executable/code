# Train Transformer
python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 model_cfg.use_past_actions=False \
    logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/transformer \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=False

# test without replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16

# test with replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt test_n=512 n_envs=16


# Train

python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc_lstm.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/lstm_prepend_dropout \
    model_cfg.lstm_config.dropout=0.1 model_cfg.prepend_student=True model_cfg.use_past_actions=False \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=True


# test without replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16

# test with replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/transformer/models/best_avg_return.pt test_n=512 n_envs=16


#CONVNET

python scripts/train_translation_bc.py \
    cfg=cfgs/boxpusher/boxpusher_bc_convnet.yml exp_cfg.steps=300000 \
    dataset=datasets/boxpusher_v1/dataset.pkl train_ids=datasets/boxpusher_v1/dataset_train_ids_1024.npy \
    val_ids=datasets/boxpusher_v1/dataset_val_ids_1024.npy model_cfg.state_dims=4 model_cfg.teacher_dims=4 model_cfg.act_dims=2 \
    device=cuda model_cfg.stack_size=20 logging_cfg.wandb=False logging_cfg.exp_name=boxpusher_v1_offline/convnet \
    model_cfg.convnet_config.dropout=0.1 model_cfg.prepend_student=True model_cfg.use_past_actions=False model_cfg.convnet_config.kernel_size=11 \
    eval_cfg=cfgs/boxpusher/eval_translation.yml restart_training=False


# test without replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/convnet/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/convnet/models/best_avg_return.pt \
    env_cfg.planner_cfg.max_plan_length=99999 env_cfg.planner_cfg.min_student_execute_length=99999 \
    test_n=512 n_envs=16

# test with replanning on box pusher and box pusher w/ obstacles
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan.yml model=results/boxpusher_v1_offline/convnet/models/best_avg_return.pt test_n=512 n_envs=16
python scripts/eval_translation.py cfg=cfgs/boxpusher/eval_translation_replan_obstacles.yml model=results/boxpusher_v1_offline/convnet/models/best_avg_return.pt test_n=512 n_envs=16
