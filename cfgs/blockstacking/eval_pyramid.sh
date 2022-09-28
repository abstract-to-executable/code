model=$2
base=$1
for i in $(seq 2 4)
do
    echo "Testing on pyramid " $i
    max_ep_len=$((i*150))
    echo ${max_ep_len}
    python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
        env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_pyramid-${i}.pkl \
        env_cfg.trajectories=datasets/blockstack_v4/dataset_pyramid-${i}_train_ids.npy n_envs=16 test_n=32 env_cfg.fixed_max_ep_len=${max_ep_len} \
        env_cfg.partial_trajectories=True \
        env_cfg.goal=pyramid-${i} \
        model=${model} \
        save_results_path=${base}.pyramid-${i}.csv \
        env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual device=cuda
done