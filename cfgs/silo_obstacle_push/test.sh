python scripts/eval_translation.py cfg=cfgs/silo_obstacle_push/eval.yml \
   model=results/online/silo_obstacle_push/transformer/376_gamma_0.97_steps_10000/models/best_train_EpRet.pt 

python scripts/watch_translation.py cfg=cfgs/silo_obstacle_push/watch_translation.yml \
   model=results/online/silo_obstacle_push/transformer/376_gamma_0.97_steps_10000/models/best_train_EpRet.pt traj_id=6