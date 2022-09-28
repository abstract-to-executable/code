model=$2
base=$1
for i in $(seq 1 4)
do
    echo "Testing on tower-4 intervening " $i
    max_ep_len=$((i*80+4*80))
    echo ${max_ep_len}
    python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation_replan.yml \
        env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-4.pkl \
        env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-4_train_ids.npy n_envs=16 test_n=32 env_cfg.fixed_max_ep_len=${max_ep_len} \
        env_cfg.partial_trajectories=False \
        env_cfg.goal=tower-4 \
        env_cfg.intervene_count=${i} \
        model=${model} \
        save_results_path=${base}.tower4intervenereplan-${i}.csv \
        # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
        device=cuda
done