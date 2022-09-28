# python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt


# # evaluate on test sets
# python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-4.pkl \
#     env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-4_train_ids.npy n_envs=8 test_n=32 env_cfg.fixed_max_ep_len=600 \
#     env_cfg.partial_trajectories=True env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
#     env_cfg.goal=tower-4 \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt

# python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl \
#     env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-6_train_ids.npy n_envs=8 test_n=32 env_cfg.fixed_max_ep_len=900 \
#     env_cfg.partial_trajectories=True env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
#     env_cfg.goal=tower-6 \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt

# python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-7.pkl \
#     env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-7_train_ids.npy n_envs=8 test_n=32 env_cfg.fixed_max_ep_len=1050 \
#     env_cfg.partial_trajectories=True env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
#     env_cfg.goal=tower-7 \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt

# python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-8.pkl \
#     env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-8_train_ids.npy n_envs=8 test_n=32 env_cfg.fixed_max_ep_len=1200 \
#     env_cfg.partial_trajectories=True env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
#     env_cfg.goal=tower-8 \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt

### Test tower envs
model=$2
base=$1
for i in $(seq 4 6)
do
    echo "Testing on tower " $i
    max_ep_len=$((i*60))
    echo ${max_ep_len}
    python scripts/eval_translation.py cfg=cfgs/blockstacking/eval_translation.yml \
        env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-${i}.pkl \
        env_cfg.trajectories=datasets/blockstack_v4/dataset_tower-${i}_train_ids.npy n_envs=16 test_n=32 env_cfg.fixed_max_ep_len=${max_ep_len} \
        env_cfg.partial_trajectories=True \
        env_cfg.goal=tower-${i} \
        model=${model} \
        save_results_path=${base}.tower-${i}.csv \
        # env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
        device=cuda
done