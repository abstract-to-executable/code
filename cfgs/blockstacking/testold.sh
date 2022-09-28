# python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
#     model=results/online/blockstack_v4_translation/s_3_dense2/models/best_train_EpRet.pt \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_pyramid-3.pkl env_cfg.goal=pyramid-3 \
#     env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
#     traj_id=1

# python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-8.pkl env_cfg.goal=tower-8 \
#     env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
#     traj_id=1

# python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
#     model=results/online/blockstack_v4_translation/final_finetune/224_finetune/models/best_train_EpRet.pt \
#     env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-8.pkl env_cfg.goal=tower-8 \
#     env_cfg.fixed_max_ep_len=800 env_cfg.controller=ee env_cfg.partial_trajectories=True \
#     env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
#     traj_id=1


python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/3103_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_pyramid-4.pkl env_cfg.goal=pyramid-4 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
    traj_id=1


python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual \
    traj_id=1


# replan
python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/3103_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-4.pkl env_cfg.goal=tower-4 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    env_cfg.show_goal_visuals=False env_cfg.obs_mode=state_visual traj_id=1


python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final/lstm/224/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset_tower-6.pkl env_cfg.goal=tower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.controller=ee env_cfg.partial_trajectories=True \
    traj_id=1


python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_large/dataset.pkl env_cfg.task_range=large \
    env_cfg.fixed_max_ep_len=8000 \
    traj_id=1

python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_v4/dataset.pkl env_cfg.task_range=small \
    env_cfg.fixed_max_ep_len=8000 \
    traj_id=1

# large


# meme builds
python scripts/blockstack/collect_dataset_test.py \
    save_path=datasets/blockstack_real/dataset_mc.pkl n=1 goal=minecraft_villagerhouse-25 render=True env_cfg.spawn_all_blocks=False

python scripts/watch_translation.py cfg=cfgs/blockstacking/watch_translation.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.trajectories_dataset=datasets/blockstack_real/dataset_mc.pkl env_cfg.goal=minecraft_villagerhouse-25 env_cfg.task_range=small \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=True \
    traj_id=0


# REAL WORLD

python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=realtower-6 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realtower-6-1.pkl env_cfg.reset_agent_after_trajectory=True

python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=realpyramid-2 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realpyramid-2-1.pkl env_cfg.reset_agent_after_trajectory=True

python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=realcustom_mc_scene_3-7 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realcustom_mc_scene_3-7.pkl env_cfg.reset_agent_after_trajectory=True

python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=realcustom_mc_scene_2-8 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realcustom_mc_scene_2-8.pkl env_cfg.reset_agent_after_trajectory=True


python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=realcustom_mc_scene_1-5 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realcustom_mc_scene_1-5.pkl env_cfg.reset_agent_after_trajectory=True


python scripts/plan_translate.py cfg=cfgs/blockstacking/watch_translation_replan.yml \
    model=results/online/blockstack_v4_translation/final_finetune/transformer/224_finetune/models/best_train_EpRet.pt \
    env_cfg.goal=custom_mc_scene_4-9 \
    env_cfg.fixed_max_ep_len=8000 env_cfg.partial_trajectories=False \
    traj_id=1 save_traj_path=realcustom_mc_scene_4-9.pkl env_cfg.reset_agent_after_trajectory=True