python scripts/watch_translation.py cfg=cfgs/pick_and_place_silo/watch_translation.yml \
    model=results/online/pick_and_place_silo_v3/transformer/9179_1024_tte_True/models/best_train_EpRet.pt \
    traj_id=17

python scripts/eval_translation.py cfg=cfgs/pick_and_place_silo/eval_translation.yml \
    model=results/online/pick_and_place_silo_v3/transformer/9179_1024_tte_True/models/best_train_EpRet.pt