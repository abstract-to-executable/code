python scripts/train_translation_online.py \
  cfg=cfgs/boxpusher/mlp.yml \
  env_cfg.trajectories=datasets/boxpusher_v2/ids_0.npy logging_cfg.exp_name=boxpusher_v2_rl/mlp_0 \
  restart_training=True exp_cfg.n_envs=10 device=cuda

python scripts/train_translation_online.py \
  cfg=cfgs/boxpusher/mlp.yml \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_1.npy logging_cfg.exp_name=boxpusher_v1_rl/mlp_1 \
  restart_training=True

python scripts/train_translation_online.py \
  cfg=cfgs/boxpusher/mlp.yml \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_2.npy logging_cfg.exp_name=boxpusher_v1_rl/mlp_2 \
  restart_training=True

python scripts/train_translation_online.py \
  cfg=cfgs/boxpusher/mlp.yml \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_3.npy logging_cfg.exp_name=boxpusher_v1_rl/mlp_3 \
  restart_training=True

python scripts/eval_translation.py \
  cfg=cfgs/boxpusher/eval_translation.yml \
  save_solved_trajectories=True \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_0.npy \
  env_cfg.exclude_target_state=False test_n=1024 model=results/boxpusher_v1_rl/mlp_0/models/latest.pt device=cuda

python scripts/eval_translation.py \
  cfg=cfgs/boxpusher/eval_translation.yml \
  save_solved_trajectories=True \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_1.npy \
  env_cfg.exclude_target_state=False test_n=1024 model=results/boxpusher_v1_rl/mlp_1/models/latest.pt device=cuda


python scripts/eval_translation.py \
  cfg=cfgs/boxpusher/eval_translation.yml \
  save_solved_trajectories=True \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_2.npy \
  env_cfg.exclude_target_state=False test_n=1024 model=results/boxpusher_v1_rl/mlp_2/models/latest.pt device=cuda

python scripts/eval_translation.py \
  cfg=cfgs/boxpusher/eval_translation.yml \
  save_solved_trajectories=True \
  env_cfg.trajectories=datasets/boxpusher_v1/ids_3.npy \
  env_cfg.exclude_target_state=False test_n=1024 model=results/boxpusher_v1_rl/mlp_3/models/latest.pt device=cuda